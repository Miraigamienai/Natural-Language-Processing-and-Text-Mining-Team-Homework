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
MIN_FREQ = 5
UNK = "<UNK>"

INFERENCE = True


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP):
        self._X = torch.tensor(X, dtype=torch.long)
        self._yA = torch.tensor(yA, dtype=torch.long)
        self._yP = torch.tensor(yP, dtype=torch.long)

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, i):
        return self._X[i], self._yA[i], self._yP[i]

    @property
    def X(self):
        return self._X.detach().cpu().numpy()

    @property
    def yA(self):
        return self._yA.detach().cpu().numpy()

    @property
    def yP(self):
        return self._yP.detach().cpu().numpy()


class TabularNextDataset(Dataset):
    def __init__(self, X, rid):
        self._X = torch.tensor(X, dtype=torch.long)
        self._rid = rid

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, i):
        return self._X[i], self._rid[i]


class NextFeatureMLP(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=256, dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])

        in_dim = len(num_tokens_per_feature) * emb_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head = nn.Linear(hidden, n_pt)

    def forward(self, X):
        es = [emb(X[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        x = self.backbone(x)
        return self.act_head(x), self.pt_head(x)


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

    # label mapping
    act_classes = np.sort(train["actionId"].unique())
    n_act = len(act_classes)
    act_id2idx = {v: i for i, v in enumerate(act_classes)}

    pt_classes = np.sort(train["pointId"].unique())
    n_pt = len(pt_classes)
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    # category vocab for each feature
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

    def build_train_dataset(data):
        X_list, yA_list, yP_list = [], [], []

        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < 2:
                continue

            Xg = encode_frame(g)

            # 用第 t 筆 FEATURES，預測第 t+1 筆 actionId / pointId
            for t in range(len(g) - 1):
                X_list.append(Xg[t])
                yA_list.append(act_id2idx.get(g.loc[t + 1, "actionId"], -1))
                yP_list.append(pt_id2idx.get(g.loc[t + 1, "pointId"], -1))

        return RallyDataset(
            np.array(X_list, dtype=np.int64),
            np.array(yA_list, dtype=np.int64),
            np.array(yP_list, dtype=np.int64),
        )

    def build_infer_dataset(data):
        X_list = []
        rid_list = []

        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            Xg = encode_frame(g)

            # infer 時固定用最後一筆 FEATURES 預測下一筆
            X_list.append(Xg[-1])
            rid_list.append(list(map(int,rid)))

        return TabularNextDataset(
            np.array(X_list, dtype=np.int64),
            rid_list
        )

    def evaluate(model, dataset, device):
        model.eval()
        allA, allAp = [], []
        allP, allPp = [], []

        loader = DataLoader(dataset, batch_size=max(args.batch * 2, 128), shuffle=False)

        with torch.no_grad():
            for X, yA, yP in loader:
                X = X.to(device)
                yA = yA.to(device)
                yP = yP.to(device)

                la, lp = model(X)

                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)

                mA = (yA != -1)
                mP = (yP != -1)

                allA += yA[mA].detach().cpu().tolist()
                allAp += a_pred[mA].detach().cpu().tolist()

                allP += yP[mP].detach().cpu().tolist()
                allPp += p_pred[mP].detach().cpu().tolist()

        return score(allA, allAp, allP, allPp)

    def train_fun(train_ds, val_ds, model, device):
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

            for Xb, yAb, yPb in train_loader:
                Xb = Xb.to(device)
                yAb = yAb.to(device)
                yPb = yPb.to(device)

                opt.zero_grad()
                la, lp = model(Xb)

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

    def inference(model, data, device):
        ds = build_infer_dataset(data)
        dl = DataLoader(ds, batch_size=max(args.batch * 2, 128), shuffle=False)

        allRid, allAp, allPp = [], [], []
        model.eval()

        with torch.no_grad():
            for Xb, Rb in dl:
                Xb = Xb.to(device)

                la, lp = model(Xb)

                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)

                allAp += a_pred.detach().cpu().tolist()
                allPp += p_pred.detach().cpu().tolist()
                allRid += len(Rb)

        return allRid, allAp, allPp

    train_ds = build_train_dataset(train)
    val_ds = build_train_dataset(val)

    num_tokens_per_feature = [len(cats[c]) for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NextFeatureMLP(
        num_tokens_per_feature=num_tokens_per_feature,
        n_act=n_act,
        n_pt=n_pt,
        emb_dim=args.emb,
        hidden=args.hidden,
        dropout=args.drop
    ).to(device)

    train_fun(train_ds, val_ds, model, device)

    if INFERENCE:
        allRid, allAp, allPp = inference(model, test, device)

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
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.1)
    args, unknown = ap.parse_known_args()

    main(args)