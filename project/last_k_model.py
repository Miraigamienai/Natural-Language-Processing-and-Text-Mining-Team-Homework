import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from load_data import DataLoader as DL
from utils import *

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

FEATURES = [
    "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "sex", "numberGame", "gamePlayerId", "gamePlayerOtherId",
    "scoreDiff"
]

PAD_TOKEN = 0
PATIENCE = 10000
MIN_DELTA = 0.05

INFERENCE = True
bidirectional = False
OUTCOME_MAXLEN = 30


def pad2d_to_k(a, k, pad_val=PAD_TOKEN):
    out = np.full((k, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), k)
    out[:T] = a[:T]
    return out, T


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, L):
        self._X = torch.tensor(X, dtype=torch.long)
        self._yA = torch.tensor(yA, dtype=torch.long)
        self._yP = torch.tensor(yP, dtype=torch.long)
        self._L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, i):
        return self._X[i], self._yA[i], self._yP[i], self._L[i]

    @property
    def X(self):
        return self._X.detach().cpu().numpy()

    @property
    def yA(self):
        return self._yA.detach().cpu().numpy()

    @property
    def yP(self):
        return self._yP.detach().cpu().numpy()

    @property
    def L(self):
        return self._L.detach().cpu().numpy()

    def subset(self, idx):
        return RallyDataset(self.X[idx], self.yA[idx], self.yP[idx], self.L[idx])


class InferDataset(Dataset):
    def __init__(self, data, encode_frame, k):
        self.X = []
        self.L = []
        self.rid = []

        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            Xg = encode_frame(g)

            hist = Xg[max(0, len(Xg) - k):]   # 最後 k 筆
            Xp, T = pad2d_to_k(hist, k)
            self.X.append(Xp)
            self.L.append(T)
            self.rid.append(int(rid))

        self.X = torch.tensor(np.array(self.X), dtype=torch.long)
        self.L = torch.tensor(np.array(self.L), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.L[i], self.rid[i]


class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])

        self.lstm = nn.LSTM(
            len(num_tokens_per_feature) * emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.drop = nn.Dropout(dropout)
        out_hidden = hidden * (2 if bidirectional else 1)

        self.act_head = nn.Linear(out_hidden, n_act)
        self.pt_head = nn.Linear(out_hidden, n_pt)

    def forward(self, X, lengths):
        es = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        o = self.drop(o)

        return self.act_head(o), self.pt_head(o)


def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad2d_last(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m)
    out[:T] = a[-T:]
    return out, T


def score(allA, allAp, allP, allPp):
    try:
        f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
        f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
    except Exception:
        f1A, f1P = 0.0, 0.0

    final = 0.5 * f1A + 0.5 * f1P
    return f1A, f1P, final


def main(args):
    loader = DL(args.val_size)

    train = loader.get_train()
    val = loader.get_val()
    test = loader.get_test()

    # mapping dictionary
    act_classes = np.sort(train["actionId"].unique())
    n_act = len(act_classes)
    act_id2idx = {v: i for i, v in enumerate(act_classes)}

    pt_classes = np.sort(train["pointId"].unique())
    n_pt = len(pt_classes)
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    MIN_FREQ = 5
    UNK = "<UNK>"
    cats = {}

    for c in FEATURES:
        s = train[c].astype(str)
        vc = s.value_counts()
        rare = vc[vc < MIN_FREQ].index
        s = s.where(~s.isin(rare), UNK)
        cat = pd.Categorical(s).categories.tolist()
        if UNK not in cat:
            cat.append(UNK)
        cats[c] = cat

    def encode_frame(df):
        outs = []
        for col in FEATURES:
            s = df[col].astype(str)
            s = s.where(s.isin(cats[col]), UNK)
            codes = pd.Categorical(s, categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    def inference(model, data, k):
        ds = InferDataset(data, encode_frame, k)
        dl = DataLoader(ds, batch_size=max(args.batch * 2, 128), shuffle=False)

        allRid, allAp, allPp = [], [], []
        model.eval()

        with torch.no_grad():
            for Xb, Lb, Rb in dl:
                Xb = Xb.to(device)
                Lb = Lb.to(device)

                la, lp = model(Xb, Lb)

                idx = torch.arange(Xb.size(0), device=device)
                last = Lb - 1

                la = la[idx, last]
                lp = lp[idx, last]

                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)

                allAp += a_pred.detach().cpu().tolist()
                allPp += p_pred.detach().cpu().tolist()
                allRid += Rb.tolist()

        return allRid, allAp, allPp

    def build_train_dataset(data, k):
        X_list, yA_list, yP_list, L_list = [], [], [], []

        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            if len(g) <= k:
                continue

            Xg = encode_frame(g)

            # 固定使用第 1 ~ k 筆，預測第 k+1 筆
            hist = Xg[:max(len(g),k)]
            Xp, T = pad2d_to_k(hist, k)

            X_list.append(Xp)
            L_list.append(T)
            yA_list.append(act_id2idx.get(g.loc[k, "actionId"], -1))
            yP_list.append(pt_id2idx.get(g.loc[k, "pointId"], -1))

        return RallyDataset(
            np.array(X_list, dtype=np.int64),
            np.array(yA_list, dtype=np.int64),
            np.array(yP_list, dtype=np.int64),
            np.array(L_list, dtype=np.int64)
        )

    def evaluate(model, dataset, device):
        model.eval()
        allA, allAp = [], []
        allP, allPp = [], []

        loader = DataLoader(dataset, batch_size=max(args.batch * 2, 128), shuffle=False)

        with torch.no_grad():
            for X, yA, yP, L in loader:
                X = X.to(device)
                yA = yA.to(device)
                yP = yP.to(device)
                L = L.to(device)

                la, lp = model(X, L)

                idx = torch.arange(X.size(0), device=device)
                last = L - 1

                la = la[idx, last]
                lp = lp[idx, last]

                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)

                mA = (yA != -1)
                mP = (yP != -1)

                allA += yA[mA].detach().cpu().tolist()
                allAp += a_pred[mA].detach().cpu().tolist()

                allP += yP[mP].detach().cpu().tolist()
                allPp += p_pred[mP].detach().cpu().tolist()

        return score(allA, allAp, allP, allPp)

    def train_fun(train_ds: RallyDataset, val_ds: RallyDataset, model, device):
        yA_tr = train_ds.yA
        yP_tr = train_ds.yP

        act_counts = np.bincount(yA_tr[yA_tr != -1].ravel(), minlength=n_act) + 1
        pt_counts = np.bincount(yP_tr[yP_tr != -1].ravel(), minlength=n_pt) + 1

        act_w = torch.tensor(1.0 / act_counts, dtype=torch.float32)
        act_w = act_w * (n_act / act_w.sum())

        pt_w = torch.tensor(1.0 / pt_counts, dtype=torch.float32)
        pt_w = pt_w * (n_pt / pt_w.sum())

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

        ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
        ce_point = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        best_final = 0.0
        counter = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            run_loss = 0.0

            for Xb, yAb, yPb, Lb in train_loader:
                Xb = Xb.to(device)
                yAb = yAb.to(device)
                yPb = yPb.to(device)
                Lb = Lb.to(device)

                opt.zero_grad()
                la, lp = model(Xb, Lb)

                idx = torch.arange(Xb.size(0), device=device)
                last = Lb - 1

                la = la[idx, last]
                lp = lp[idx, last]

                loss_a = ce_action(la, yAb)
                loss_p = ce_point(lp, yPb)

                loss = 0.5 * loss_a + 0.5 * loss_p
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                run_loss += loss.item() * Xb.size(0)

            tr_loss = run_loss / len(train_loader.dataset)
            t1_f1A, t1_f1P, t1_final = evaluate(model, val_ds, device)

            print(
                f"[Epoch {ep}/{args.epochs}] "
                f"train_loss={tr_loss:.4f} "
                f"F1_action={t1_f1A:.4f} F1_point={t1_f1P:.4f} "
                f"Final={t1_final:.4f}"
            )

            if t1_final > best_final + 1e-12:
                best_final = t1_final
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                print("Early stopping")
                break

    train_ds = build_train_dataset(train, args.k)
    val_ds = build_train_dataset(val, args.k)

    num_tokens_per_feature = [len(cats[c]) for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLSTM(
        num_tokens_per_feature,
        n_act,
        n_pt,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop
    ).to(device)

    train_fun(train_ds, val_ds, model, device)

    if INFERENCE:
        allRid, allAp, allPp = inference(model, test, args.k)

        action_pred = [int(act_classes[x]) for x in allAp]
        point_pred = [int(pt_classes[x]) for x in allPp]

        pred_df = pd.DataFrame({
            "rally_uid": allRid,
            "actionId": action_pred,
            "pointId": point_pred,
            "serverGetPoint": 0.5
        }).sort_values("rally_uid")

        pred_df.to_csv(args.out, index=False)
        print(f"Saved submission to: {args.out}")
        print(pred_df.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_CSV)
    ap.add_argument("--test", default=TEST_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--epochs", type=int, default=99)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=1)
    args, unknown = ap.parse_known_args()

    main(args)