import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import os

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

all = [
    "sex","handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId","scoreSelf","scoreOther","strikeNumber"
]
# remove = ["scoreSelf","scoreOther","sex","strikeNumber"]
remove = []
FEATURES = [f for f in all if f not in remove]

PAD_TOKEN = 0
PATIENCE = 10
INFERENCE = True
bidirectional = False
OUTCOME_MAXLEN = 5


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self._X = torch.tensor(X, dtype=torch.long)
        self._yA = torch.tensor(yA, dtype=torch.long)
        self._yP = torch.tensor(yP, dtype=torch.long)
        self._yR = torch.tensor(yR, dtype=torch.float32)
        self._L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self._X.shape[0]
    def __getitem__(self, i): return self._X[i], self._yA[i], self._yP[i], self._yR[i], self._L[i]
    @property
    def X(self): return self._X.detach().cpu().numpy()
    @property
    def yA(self): return self._yA.detach().cpu().numpy()
    @property
    def yP(self): return self._yP.detach().cpu().numpy()
    @property
    def yR(self): return self._yR.detach().cpu().numpy()
    @property
    def L(self): return self._L.detach().cpu().numpy()
    def subset(self, idx): return RallyDataset(self.X[idx], self.yA[idx], self.yP[idx], self.yR[idx], self.L[idx])


class OutcomeDataset(Dataset):
    def __init__(self, X, y, L):
        self._X = torch.tensor(X, dtype=torch.long)
        self._y = torch.tensor(y, dtype=torch.float32)
        self._L = torch.tensor(L, dtype=torch.long)
    def __len__(self): return self._X.shape[0]
    def __getitem__(self, i): return self._X[i], self._y[i], self._L[i]
    @property
    def X(self): return self._X.detach().cpu().numpy()
    @property
    def y(self): return self._y.detach().cpu().numpy()
    @property
    def L(self): return self._L.detach().cpu().numpy()
    def subset(self, idx): return OutcomeDataset(self.X[idx], self.y[idx], self.L[idx])


class InferDataset(Dataset):
    def __init__(self, data, encode_frame, maxlen):
        self.X, self.L, self.rid = [], [], []
        def pad2d_cap(a, m, pad_val=PAD_TOKEN):
            out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
            T = min(len(a), m)
            out[:T] = a[-T:]
            return out, T
        for rid, g in data.groupby("rally_uid"):
            Xg = encode_frame(g)
            Xp, T = pad2d_cap(Xg, maxlen)
            self.X.append(Xp)
            self.L.append(max(1, T))
            self.rid.append(int(rid))
        self.X = torch.tensor(np.array(self.X), dtype=torch.long)
        self.L = torch.tensor(self.L, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.L[i], self.rid[i]


class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(len(num_tokens_per_feature)*emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        out_hidden = hidden*(2 if bidirectional else 1)
        self.act_head = nn.Linear(out_hidden, n_act)
        self.pt_head  = nn.Linear(out_hidden, n_pt)
    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # o,_ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        h = h[-1]
        h = self.drop(h)
        return self.act_head(h), self.pt_head(h)


class RallyOutcomeModel(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(len(num_tokens_per_feature)*emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feat = self.drop(h[-1])
        return self.head(feat).squeeze(1)


def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out
def pad2d_last(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m)
    out[:T] = a[-T:]
    return out, T

def score(allA, allAp, allP, allPp, allR, allRp):
    try:
        f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
        f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
        auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
    except Exception:
        f1A, f1P, auc = 0.0, 0.0, 0.5
    final = 0.4*f1A + 0.4*f1P + 0.2*auc
    return f1A, f1P, auc, final

def eval_mtl(model, dataset, device):
    model.eval()
    allA, allAp, allP, allPp = [], [], [], []
    with torch.no_grad():
        for X, yA, yP, yR, L in DataLoader(dataset, batch_size=128, shuffle=False):
            X, yA, yP, L = X.to(device), yA.to(device), yP.to(device), L.to(device)
            la, lp = model(X, L)

            yA_flat = yA.view(-1).detach().cpu().numpy()
            yP_flat = yP.view(-1).detach().cpu().numpy()
            a_pred = la.argmax(-1).view(-1).detach().cpu().numpy()
            p_pred = lp.argmax(-1).view(-1).detach().cpu().numpy()

            mA = (yA_flat != -1)
            mP = (yP_flat != -1)
            allA += yA_flat[mA].tolist(); allAp += a_pred[mA].tolist()
            allP += yP_flat[mP].tolist(); allPp += p_pred[mP].tolist()
    return allA, allAp, allP, allPp

def t_minus_1_eval_mtl(model, dataset, device):
    model.eval()
    allA, allAp, allP, allPp = [], [], [], []
    with torch.no_grad():
        for X, yA, yP, yR, L in DataLoader(dataset, batch_size=64, shuffle=False):
            X, yA, yP, L = X.to(device), yA.to(device), yP.to(device), L.to(device)
            last_t2 = L - 2
            mask = (last_t2 >= 0)
            X, yA, yP, L, last_t2 = X[mask], yA[mask], yP[mask], L[mask], last_t2[mask]
            if X.size(0) == 0:
                continue

            idx = torch.arange(X.size(0), device=device)
            X = X.clone()
            X[idx, L-1, :] = 0
            la, lp = model(X, L)

            yA_flat = yA.view(-1).detach().cpu().numpy()
            yP_flat = yP.view(-1).detach().cpu().numpy()
            a_pred = la.argmax(-1).view(-1).detach().cpu().numpy()
            p_pred = lp.argmax(-1).view(-1).detach().cpu().numpy()

            mA = (yA_flat != -1)
            mP = (yP_flat != -1)
            allA += yA_flat[mA].tolist(); allAp += a_pred[mA].tolist()
            allP += yP_flat[mP].tolist(); allPp += p_pred[mP].tolist()
    return allA, allAp, allP, allPp

def eval_outcome(model, dataset, device):
    model.eval()
    allR, allRp = [], []
    with torch.no_grad():
        for X, y, L in DataLoader(dataset, batch_size=256, shuffle=False):
            X, y, L = X.to(device), y.to(device), L.to(device)
            logit = model(X, L)
            prob = torch.sigmoid(logit)
            allR += y.cpu().tolist()
            allRp += prob.cpu().tolist()
    return allR, allRp

def train_mtl(train_ds, val_ds, model, device, n_act, n_pt, args):
    yA_tr = train_ds.yA
    yP_tr = train_ds.yP
    act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
    act_w = torch.tensor(1.0/act_counts, dtype=torch.float32); act_w = act_w * (n_act/act_w.sum())
    pt_w  = torch.tensor(1.0/pt_counts,  dtype=torch.float32); pt_w  = pt_w  * (n_pt /pt_w.sum())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=max(args.batch*2,128), shuffle=False)
    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_state = None
    patience = PATIENCE
    counter = 0

    for ep in range(1, args.epochs+1):
        model.train(); run_loss = 0.0
        for Xb, yAb, yPb, yRb, Lb in train_loader:
            Xb, yAb, yPb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), Lb.to(device)
            opt.zero_grad()
            la, lp = model(Xb, Lb)
            loss = 0.5*ce_action(la.view(-1, la.size(-1)), yAb.view(-1)) + 0.5*ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * Xb.size(0)

        model.eval(); val_loss = 0.0
        allA, allAp, allP, allPp = [], [], [], []
        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb in val_loader:
                Xb, yAb, yPb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), Lb.to(device)
                la, lp = model(Xb, Lb)
                loss = 0.5*ce_action(la.view(-1, la.size(-1)), yAb.view(-1)) + 0.5*ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                val_loss += loss.item() * Xb.size(0)

                yA_flat = yAb.view(-1).cpu().numpy(); yP_flat = yPb.view(-1).cpu().numpy()
                a_pred = la.argmax(-1).view(-1).cpu().numpy(); p_pred = lp.argmax(-1).view(-1).cpu().numpy()
                mA = (yA_flat != -1); mP = (yP_flat != -1)
                allA += yA_flat[mA].tolist(); allAp += a_pred[mA].tolist()
                allP += yP_flat[mP].tolist(); allPp += p_pred[mP].tolist()

        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)
        f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
        f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f}")

        if va_loss < best_loss - 1e-5:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_outcome(train_ds, val_ds, model, device, args):
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=max(args.batch*2,128), shuffle=False)
    pos = float(train_ds.y.sum())
    neg = float(len(train_ds.y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_state = None
    patience = PATIENCE
    counter = 0

    for ep in range(1, args.epochs+1):
        model.train(); run_loss = 0.0
        for Xb, yb, Lb in train_loader:
            Xb, yb, Lb = Xb.to(device), yb.to(device), Lb.to(device)
            opt.zero_grad()
            logit = model(Xb, Lb)
            loss = criterion(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * Xb.size(0)

        model.eval(); allR, allRp = [], []
        with torch.no_grad():
            for Xb, yb, Lb in val_loader:
                Xb, yb, Lb = Xb.to(device), yb.to(device), Lb.to(device)
                logit = model(Xb, Lb)
                prob = torch.sigmoid(logit)
                allR += yb.cpu().tolist()
                allRp += prob.cpu().tolist()

        auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        tr_loss = run_loss / len(train_loader.dataset)
        print(f"[Epoch {ep}/{args.epochs}] outcome_loss={tr_loss:.4f} outcome_auc={auc:.4f}")

        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def inference_mtl(model, data, encode_frame, maxlen, device):
    ds = InferDataset(data, encode_frame, maxlen)
    dl = DataLoader(ds, batch_size=max(128, 2), shuffle=False)
    allRid, allAp, allPp = [], [], []
    model.eval()
    with torch.no_grad():
        for Xb, Lb, Rb in dl:
            Xb, Lb = Xb.to(device), Lb.to(device)
            la, lp = model(Xb, Lb)
            idx = torch.arange(Xb.size(0), device=device)
            last_t = Lb - 1
            a_pred = la[idx, last_t].argmax(-1)
            p_pred = lp[idx, last_t].argmax(-1)
            allAp += a_pred.cpu().tolist()
            allPp += p_pred.cpu().tolist()
            allRid += Rb.cpu().tolist()
    return allRid, allAp, allPp

def inference_outcome(model, data, encode_frame, maxlen, device):
    ds = InferDataset(data, encode_frame, maxlen)
    dl = DataLoader(ds, batch_size=max(128, 2), shuffle=False)
    allRid, allRp = [], []
    model.eval()
    with torch.no_grad():
        for Xb, Lb, Rb in dl:
            Xb, Lb = Xb.to(device), Lb.to(device)
            logit = model(Xb, Lb)
            prob = torch.sigmoid(logit)
            allRp += prob.cpu().tolist()
            allRid += Rb.cpu().tolist()
    return allRid, allRp

def main(args):
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])

    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}

    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}
    def encode_frame(df):
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)
    mtl_X_list, mtl_yA_list, mtl_yP_list, mtl_yR_list, mtl_L_list = [], [], [], [], []
    out_X_list, out_y_list, out_L_list = [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: 
            continue
        Xg = encode_frame(g)

        X = Xg[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)
        mtl_X_list.append(X); mtl_yA_list.append(yA); mtl_yP_list.append(yP); mtl_yR_list.append(int(g["serverGetPoint"].iloc[0])); mtl_L_list.append(len(X))
        
        cut = np.random.randint(0, len(Xg))
        if cut>0: Xg=Xg[:-cut]
        
        Xo, To = pad2d_last(Xg, OUTCOME_MAXLEN)
        out_X_list.append(Xo); out_y_list.append(int(g["serverGetPoint"].iloc[0])); out_L_list.append(To)

    MAXLEN = max(mtl_L_list)

    X_mtl  = np.stack([pad2d(s, MAXLEN) for s in mtl_X_list])
    yA_mtl = np.stack([pad1d(s, MAXLEN) for s in mtl_yA_list])
    yP_mtl = np.stack([pad1d(s, MAXLEN) for s in mtl_yP_list])
    yR_mtl = np.array(mtl_yR_list, dtype=np.float32)
    L_mtl  = np.array(mtl_L_list, dtype=np.int64)

    yA_mtl = np.vectorize(lambda x: act_id2idx.get(x, -1))(yA_mtl).astype(np.int64)
    yP_mtl = np.vectorize(lambda x: pt_id2idx.get(x, -1))(yP_mtl).astype(np.int64)

    X_out = np.stack(out_X_list)
    y_out = np.array(out_y_list, dtype=np.float32)
    L_out = np.array(out_L_list, dtype=np.int64)

    idx = np.arange(len(X_mtl))
    stratify = (yR_mtl > 0.5).astype(int)
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=SEED, stratify=stratify)

    mtl_train_ds = RallyDataset(X_mtl[tr_idx], yA_mtl[tr_idx], yP_mtl[tr_idx], yR_mtl[tr_idx], L_mtl[tr_idx])
    mtl_val_ds   = RallyDataset(X_mtl[va_idx], yA_mtl[va_idx], yP_mtl[va_idx], yR_mtl[va_idx], L_mtl[va_idx])

    out_train_ds = OutcomeDataset(X_out[tr_idx], y_out[tr_idx], L_out[tr_idx])
    out_val_ds   = OutcomeDataset(X_out[va_idx], y_out[va_idx], L_out[va_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tokens_per_feature = [len(cats[c]) for c in FEATURES]
    mtl_model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
    out_model  = RallyOutcomeModel(num_tokens_per_feature, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)

    mtl_model = train_mtl(mtl_train_ds, mtl_val_ds, mtl_model, device, n_act, n_pt, args)
    out_model = train_outcome(out_train_ds, out_val_ds, out_model, device, args)

    allA, allAp, allP, allPp = eval_mtl(mtl_model, mtl_val_ds, device)
    allR, allRp = eval_outcome(out_model, out_val_ds, device)
    f1A, f1P, auc, final = score(allA, allAp, allP, allPp, allR, allRp)
    print(f"VAL: F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

    tA, tAp, tP, tPp = t_minus_1_eval_mtl(mtl_model, mtl_val_ds, device)
    t_f1A, t_f1P, t_auc, t_final = score(tA, tAp, tP, tPp, allR, allRp)
    print(f"T-1 VAL: F1_action={t_f1A:.4f} F1_point={t_f1P:.4f} AUC={t_auc:.4f} Final~{t_final:.4f}")

    if INFERENCE:
        allRidA, allAp, allPp = inference_mtl(mtl_model, test, encode_frame, MAXLEN, device)
        allRidR, allRp = inference_outcome(out_model, test, encode_frame, OUTCOME_MAXLEN, device)

        action_pred = [int(act_classes[x]) for x in allAp]
        point_pred  = [int(pt_classes[x]) for x in allPp]

        pred_df = pd.DataFrame({
            "rally_uid": allRidA,
            "actionId": action_pred,
            "pointId": point_pred
        }).merge(
            pd.DataFrame({"rally_uid": allRidR, "serverGetPoint": allRp}),
            on="rally_uid",
            how="left"
        ).sort_values("rally_uid")

        pred_df.to_csv(args.out, index=False)
        print(f"Saved submission to: {args.out}")
        print(pred_df.head())


def get_base_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

BASE_DIR = get_base_dir()
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_CSV = os.path.join(DATASETS_DIR, "train.csv")
TEST_CSV = os.path.join(DATASETS_DIR, "test_new.csv")
OUT_CSV = os.path.join(BASE_DIR, "submission_lstm_baseline.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_CSV)
    ap.add_argument("--test", default=TEST_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.2)
    args, unknown = ap.parse_known_args()
    main(args)