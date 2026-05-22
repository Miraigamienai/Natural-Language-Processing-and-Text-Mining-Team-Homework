import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

all = [
    "sex","handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId","scoreSelf","scoreOther","strikeNumber"]
remove = ["scoreSelf","scoreOther","sex","strikeNumber","positionId"]

FEATURES = [f for f in all if f not in remove]

PAD_TOKEN = 0
PATIENCE = 15

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
    def X(self):
        return self._X.detach().cpu().numpy()
    @property
    def yA(self):
        return self._yA.detach().cpu().numpy()
    @property
    def yP(self):
        return self._yP.detach().cpu().numpy()
    @property
    def yR(self):
        return self._yR.detach().cpu().numpy()
    @property
    def L(self):
        return self._L.detach().cpu().numpy()
    def subset(self, idx):
        return RallyDataset(self.X[idx], self.yA[idx], self.yP[idx], self.yR[idx], self.L[idx])


class InferDataset(Dataset):
    def __init__(self, data, encode_frame, maxlen):
        self.X=[]
        self.L=[]
        self.rid=[]
        
        def pad2d_cap(a,m,pad_val=PAD_TOKEN):
            out=np.full((m,a.shape[1]),pad_val,dtype=np.int64)
            T=min(len(a),m)
            out[:T]=a[:T]
            return out,T

        for rid,g in data.groupby("rally_uid"):
            Xg=encode_frame(g)
            Xp,T=pad2d_cap(Xg,maxlen)
            self.X.append(Xp)
            self.L.append(max(1,T))
            self.rid.append(int(rid))

        self.X=torch.tensor(np.array(self.X),dtype=torch.long)
        self.L=torch.tensor(self.L,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,i):
        return self.X[i],self.L[i],self.rid[i]


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
        self.rly_head = nn.Linear(out_hidden, 1)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o,_ = self.lstm(packed)
        o,_ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        o = self.drop(o)
        # o  : (batch,T,hidden)
        # la : (batch,T,n_act)
        # lp : (batch,T,n_pt)
        # lr : (batch)
        return self.act_head(o), self.pt_head(o), self.rly_head(o).squeeze(-1)


def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out
def pad2d_last(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m); out[:T] = a[-T:]
    return out, T

def score(allA, allAp, allP, allPp, allR, allRp):
    try:
        f1A=f1_score(allA,allAp,average="macro") if len(allA) else 0.0
        f1P=f1_score(allP,allPp,average="macro") if len(allP) else 0.0
        auc=roc_auc_score(allR,allRp) if len(set(allR))>1 else 0.5
    except Exception: f1A,f1P,auc=0.0,0.0,0.5
    final=0.4*f1A+0.4*f1P+0.2*auc
    return f1A, f1P, auc, final

def main(args):
    # data
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])
    # train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    # test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)
    
    # mapping dictionary
    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
        
    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}
    def encode_frame(df):
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    def inference(model, data, maxlen):
        ds=InferDataset(data, encode_frame, maxlen)
        dl=DataLoader(ds, batch_size=max(args.batch*2,128), shuffle=False)
        allRid,allAp,allPp,allRp=[],[],[],[]
        model.eval()
        with torch.no_grad():
            for Xb,Lb,Rb in dl:
                Xb=Xb.to(device); Lb=Lb.to(device)
                la,lp,lr = model(Xb, Lb); last_t = Lb-1
                idx=torch.arange(Xb.size(0),device=device)
                la=la[idx,last_t]; lp=lp[idx,last_t]; lr=lr[idx,last_t]
                a_pred=la.argmax(-1); p_pred=lp.argmax(-1)
                r_pred=torch.sigmoid(lr)
                allAp+=a_pred.cpu().tolist(); allPp+=p_pred.cpu().tolist(); allRp+=r_pred.cpu().tolist()
                allRid+=Rb.tolist()
        return allRid,allAp,allPp,allRp
        
    #train
    def load_dataset_all(data):
        X_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], []
        for rid, g in data.groupby("rally_uid"):
            if len(g) < 2: continue
            # (timestamp, features)
            X = encode_frame(g)[:-1]
            yA = g["actionId"].values[1:].astype(np.int64)
            yP = g["pointId"].values[1:].astype(np.int64)
            yR = g["serverGetPoint"].values[1:].astype(np.int64)
            X_list.append(X); yA_list.append(yA); yP_list.append(yP); yR_list.append(yR) 
            L_list.append(len(X))

        # MAXLEN = int(np.percentile(L_all, 95))
        MAXLEN = max(L_list)
        # (batch, timestamp, features)
        X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
        yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
        yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
        yR_all = np.stack([pad1d(s, MAXLEN) for s in yR_list], dtype=np.float32)
        L_all  = np.array(L_list, dtype=np.int64)

        yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
        yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)
        return RallyDataset(X_all, yA_all, yP_all, yR_all, L_all), MAXLEN
    
    def data_split(data_all: RallyDataset):
        idx = np.arange(len(data_all))
        stratify = (
            (data_all.yR[:,0]>0.5).astype(str)
        )
        tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=SEED, stratify=stratify)
        train_ds = data_all.subset(tr_idx)
        val_ds   = data_all.subset(va_idx)
        return train_ds, val_ds
    
    def expand_dataset(ds):
        X2,yA2,yP2,yR2,L2=[],[],[],[],[]
        for X,yA,yP,yR,L in zip(ds.X,ds.yA,ds.yP,ds.yR,ds.L):
            for end in range(1,L+1):
                X2.append(X[:end])
                yA2.append(yA[:end])
                yP2.append(yP[:end])
                yR2.append(yR[:end])
                L2.append(end)
        MAX=max(L2)
        X2=np.stack([pad2d(x,MAX) for x in X2])
        yA2=np.stack([pad1d(x,MAX) for x in yA2])
        yP2=np.stack([pad1d(x,MAX) for x in yP2])
        yR2=np.stack([pad1d(x,MAX) for x in yR2])
        return RallyDataset(X2, yA2, yP2, yR2, L2)

    def train_fun(train_ds: RallyDataset, val_ds: RallyDataset, model, device):
        yA_tr = train_ds.yA
        yP_tr = train_ds.yP
        act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
        pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
        act_w = torch.tensor(1.0/act_counts, dtype=torch.float32); act_w = (act_w * (n_act/act_w.sum()))
        pt_w  = torch.tensor(1.0/pt_counts,  dtype=torch.float32); pt_w  = (pt_w  * (n_pt /pt_w.sum()))

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

        ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
        ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
        bce_rally = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        MIN_DELTA = 1e-5
        patience = PATIENCE
        best_loss = float("inf")
        counter = 0
        for ep in range(1, args.epochs+1):
            model.train(); run_loss=0.0
            for Xb,yAb,yPb,yRb,Lb in train_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                opt.zero_grad(); la,lp,lr = model(Xb,Lb)
                yRbm=(yRb!=-1)
                yRb=yRb[yRbm]; lr=lr[yRbm]
                loss = 0.4*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.4*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.2*bce_rally(lr,yRb)
                # loss = ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                run_loss += loss.item()*Xb.size(0)

            model.eval(); val_loss=0.0
            allA,allAp,allP,allPp,allR,allRp=[],[],[],[],[],[]
            with torch.no_grad():
                for Xb,yAb,yPb,yRb,Lb in val_loader:
                    Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                    la,lp,lr = model(Xb,Lb)
                    yRbm=(yRb!=-1)
                    yRb=yRb[yRbm]; lr=lr[yRbm]
                    loss = 0.4*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.4*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.2*bce_rally(lr,yRb)
                    val_loss += loss.item()*Xb.size(0)

                    allR+=yRb.detach().cpu().view(-1).tolist(); allRp+=torch.sigmoid(lr).detach().cpu().view(-1).tolist()
                    yA_flat=yAb.view(-1).detach().cpu().numpy(); yP_flat=yPb.view(-1).detach().cpu().numpy()
                    a_pred=la.argmax(-1).view(-1).detach().cpu().numpy(); p_pred=lp.argmax(-1).view(-1).detach().cpu().numpy()
                    mA=(yA_flat!=-1); mP=(yP_flat!=-1)
                    allA+=yA_flat[mA].tolist(); allAp+=a_pred[mA].tolist()
                    allP+=yP_flat[mP].tolist(); allPp+=p_pred[mP].tolist()

            tr_loss = run_loss/len(train_loader.dataset); va_loss=val_loss/len(val_loader.dataset)
            f1A, f1P, auc, final = score(allA, allAp, allP, allPp, allR, allRp)
            print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")
            if va_loss < best_loss - MIN_DELTA:
                best_loss = va_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping")
                break
    
    def t_minus_1_eval(model, dataset, device):
        model.eval()
        allA, allAp = [], []
        allP, allPp = [], []
        allR, allRp = [], []
        with torch.no_grad():
            for X, yA, yP, yR, L in DataLoader(dataset, batch_size=64):
                X, yA, yP, yR, L = X.to(device), yA.to(device), yP.to(device), yR.to(device), L.to(device)
                last_t2 = L-2
                mask = (last_t2 >= 0)
                X = X[mask]; yA = yA[mask]; yP = yP[mask]; yR = yR[mask]; L = L[mask]
                if X.size(0)==0:
                    continue
                last_t2 = last_t2[mask]
                last_t = L-1
                idx=torch.arange(X.size(0),device=device)
                X=X.clone()
                X[idx,last_t,:]=0
                la, lp, lr = model(X, last_t)
                la=la[idx,last_t2]; lp=lp[idx,last_t2]; lr=lr[idx,last_t2]
                yA=yA[idx,last_t2]; yP=yP[idx,last_t2]; yR=yR[idx,last_t2]
                a_pred=la.argmax(-1).view(-1).detach().cpu().numpy(); p_pred=lp.argmax(-1).view(-1).detach().cpu().numpy()
                r_pred=torch.sigmoid(lr).view(-1).detach().cpu().numpy()
                
                yA_flat=yA.view(-1).detach().cpu().numpy(); yP_flat=yP.view(-1).detach().cpu().numpy(); yR_flat=yR.view(-1).detach().cpu().numpy()
                mA=(yA_flat!=-1); mP=(yP_flat!=-1); mR=(yR_flat!=-1)
                
                allR+=yR_flat[mR].tolist(); allRp+=r_pred[mR].tolist()
                allA+=yA_flat[mA].tolist(); allAp+=a_pred[mA].tolist()
                allP+=yP_flat[mP].tolist(); allPp+=p_pred[mP].tolist()
        return score(allA, allAp, allP, allPp, allR, allRp)


    data_all, maxlen = load_dataset_all(train)
    train_ds, val_ds = data_split(data_all)
    # train_ds = expand_dataset(train_ds)
    num_tokens_per_feature = [len(cats[c]) for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
    train_fun(train_ds, val_ds, model, device)
    t1_f1A, t1_f1P, t1_auc, t1_final = t_minus_1_eval(model, train_ds, device)
    # t1_f1A, t1_f1P, t1_auc, t1_final = t_minus_1_eval(model, val_ds, device)
    print("T-1 VAL:", t1_f1A, t1_f1P, t1_auc, t1_final)
    
    if INFERENCE: 
        allRid,allAp,allPp,allRp = inference(model, test, maxlen)

        action_pred = [int(act_classes[x]) for x in allAp]
        point_pred  = [int(pt_classes[x]) for x in allPp]
        pred_df=pd.DataFrame({
            "rally_uid":allRid,
            "actionId":action_pred,
            "pointId":point_pred,
            "serverGetPoint":allRp
        }).sort_values("rally_uid")

        pred_df.to_csv(args.out, index=False); print(f"Saved submission to: {args.out}"); print(pred_df.head())



import os
def get_base_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

BASE_DIR = get_base_dir()
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
TRAIN_CSV = os.path.join(DATASETS_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASETS_DIR, 'test_new.csv')
OUT_CSV =  os.path.join(BASE_DIR, 'submission_lstm_baseline.csv')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_CSV)
    ap.add_argument("--test", default=TEST_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.1)
    args, unknown = ap.parse_known_args()

    main(args)
