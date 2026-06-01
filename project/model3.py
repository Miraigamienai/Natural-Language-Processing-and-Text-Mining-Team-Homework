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

FEATURES = [
    "handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId","strikeNumber", 
    "sex", "numberGame", "gamePlayerId", "gamePlayerOtherId",
    "scoreSelf", "scoreOther"]


PAD_TOKEN = 0
PATIENCE = 10
MIN_DELTA = 1e-7

INFERENCE = True
bidirectional = False
OUTCOME_MAXLEN = 5


class RallyDataset(Dataset):
    def __init__(self, X, Y, L):
        self._X = torch.tensor(X, dtype=torch.long)
        self._Y = torch.tensor(Y, dtype=torch.float32)
        self._L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self._X.shape[0]
    def __getitem__(self, i): return self._X[i], self._Y[i], self._L[i]
    @property
    def X(self): return self._X.detach().cpu().numpy()
    @property
    def Y(self): return self._Y.detach().cpu().numpy()
    @property
    def L(self): return self._L.detach().cpu().numpy()
    def subset(self, idx): return RallyDataset(self.X[idx], self.Y[idx], self.L[idx])

class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(len(num_tokens_per_feature)*emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        out_hidden = hidden*(2 if bidirectional else 1)
        self.pt_head  = nn.Linear(out_hidden, 1)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o,_ = self.lstm(packed)
        o,_ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        o = self.drop(o)
        return self.pt_head(o).squeeze(-1)


def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out
def pad2d_last(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m); out[:T] = a[-T:]
    return out, T

def score(allR, allRp):
    try:auc=roc_auc_score(allR, allRp) if len(set(allR))>1 else .5
    except Exception: auc=.5
    return auc

def main(args):
    def load_csv(filepath_or_buffer):
        result = pd.read_csv(filepath_or_buffer).sort_values(["rally_uid","strikeNumber"])
        # result["strikeNumber"] = result["strikeNumber"].clip(0, 40)
        return result
    
    train = load_csv(args.train)
    
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
        outs=[]
        for col in FEATURES:
            s = df[col].astype(str)
            s = s.where(s.isin(cats[col]),UNK)
            codes = (pd.Categorical(s,categories=cats[col]).codes+1)
            outs.append(np.asarray(codes,dtype=np.int64))
        return np.stack(outs,axis=1)
        
    #train
    def load_dataset_all(data):
        X_list, yP_list, L_list = [], [], []
        for rid, g in data.groupby("rally_uid"):
            if len(g) < 2: continue
            # (timestamp, features)
            X = encode_frame(g)[:-1]
            Y = g["serverGetPoint"].values[1:].astype(np.int64)
            X_list.append(X); yP_list.append(Y) 
            L_list.append(len(X))

        # MAXLEN = int(np.percentile(L_all, 95))
        MAXLEN = max(L_list)
        # (batch, timestamp, features)
        X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
        yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
        L_all  = np.array(L_list, dtype=np.float32)
        return RallyDataset(X_all, yP_all, L_all)

    def expand_dataset(ds, prefix=False, suffix=False):
        X2,yP2,L2=[],[],[]
        for X,yP,L in zip(ds.X,ds.Y,ds.L):
            if prefix:
                for end in range(1,L+1):
                    X2.append(X[:end])
                    yP2.append(yP[:end])
                    L2.append(end)
            if suffix:
                for start in range(L):
                    X2.append(X[start:])
                    yP2.append(yP[start:])
                    L2.append(L - start)
        MAX=max(L2)
        X2=np.stack([pad2d(x,MAX) for x in X2])
        yP2=np.stack([pad1d(x,MAX) for x in yP2])
        return RallyDataset(X2, yP2, L2)
    
    def data_split(data_all: RallyDataset):
        idx = np.arange(len(data_all))
        tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=SEED)
        train_ds = data_all.subset(tr_idx)
        val_ds   = data_all.subset(va_idx)
        return train_ds, val_ds

    def t_minus_k_eval(model, dataset, device):
        model.eval()
        allP, allPp = [], []
        with torch.no_grad():
            for X, yP, L in DataLoader(dataset, batch_size=args.batch):
                X, yP, L = X.to(device), yP.to(device), L.to(device)
                target = (torch.rand(L.size(), device=device) * L).long()
                
                idx=torch.arange(X.size(0),device=device)
                time_idx=torch.arange(X.size(1),device=device)[None,:]
                mask=(time_idx>(target[:,None]))
                X=X.clone()
                X[mask]=0
                lp= model(X, target+1)
                lp=lp[idx,target]
                yP=yP[idx,target]

                # p_pred=lp.argmax(-1).view(-1).detach().cpu().numpy()
                pred = torch.sigmoid(lp)
                pred=pred.cpu().numpy()
                
                yP_flat=yP.view(-1).detach().cpu().numpy()
                mP=(yP_flat!=-1)
                allP+=yP_flat[mP].tolist(); allPp+=pred[mP].tolist()
        return score(allP, allPp)

    def train_fun2(train_ds: RallyDataset, val_ds: RallyDataset, model, device):
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        bce = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        MIN_DELTA = 0.05
        best_final = 0
        counter = 0
        for ep in range(1, args.epochs+1):
            model.train(); run_loss=0.0
            for Xb,yPb,Lb in train_loader:
                Xb,yPb,Lb = Xb.to(device),yPb.to(device),Lb.to(device)
                opt.zero_grad()
                lp = model(Xb,Lb)
                yRbm=(yPb!=-1)
                loss=bce(lp[yRbm], yPb[yRbm])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                run_loss += loss.item()*Xb.size(0)

            model.eval()
            with torch.no_grad():
                val_k = t_minus_k_eval(model, val_ds, device)
                train_k = t_minus_k_eval(model, train_ds, device)

            tr_loss = run_loss/len(train_loader.dataset)
            print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_k={val_k:.4f} train_k={train_k:.4f}")
            if val_k > best_final - MIN_DELTA:
                best_final = max(val_k, best_final); counter = 0
            else: 
                counter += 1
            if counter >= PATIENCE:
                print("Early stopping")
                break

    def train_fun(train_ds: RallyDataset, val_ds: RallyDataset, model, device):
        train_loader = DataLoader(train_ds,batch_size=args.batch,shuffle=True)
        bce = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)

        MIN_DELTA = 0.05
        best_final = 0
        counter = 0

        for ep in range(1,args.epochs+1):
            model.train()
            run_loss=0
            for Xb,yPb,Lb in train_loader:
                Xb=Xb.to(device); yPb=yPb.to(device); Lb=Lb.to(device)
                B,T,F = Xb.shape
                target = (torch.rand(B,device=device)*Lb).long()
                idx=torch.arange(B,device=device)
                time_idx=torch.arange(T,device=device)[None,:]
                mask=(time_idx>target[:,None])
                Xcut=Xb.clone()
                Xcut[mask]=0
                opt.zero_grad()
                lp=model(Xcut,target+1)
                pred=lp[idx,target]
                gt=yPb[idx,target]
                loss=bce(pred,gt.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                run_loss += loss.item()*B
            model.eval()
            with torch.no_grad():
                val_k=t_minus_k_eval(model,val_ds,device)
                train_k=t_minus_k_eval(model,train_ds,device)

            tr_loss=run_loss/len(train_loader.dataset)

            print(
                f"[Epoch {ep}/{args.epochs}] "
                f"train_loss={tr_loss:.4f} "
                f"val_k={val_k:.4f} "
                f"train_k={train_k:.4f}"
            )

            if val_k > best_final-MIN_DELTA:
                best_final=max(val_k,best_final)
                counter=0
            else:
                counter+=1

            if counter>=PATIENCE:
                print("Early stopping")
                break
    

    data_all = load_dataset_all(train)
    train_ds, val_ds = data_split(data_all)
    # train_ds = train_ds.subset(torch.arange(end=3000))
    # train_ds = expand_dataset(train_ds,prefix=True,suffix=False)
    num_tokens_per_feature = [len(cats[c]) for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskLSTM(num_tokens_per_feature, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
    train_fun(train_ds, val_ds, model, device)
    

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
    ap.add_argument("--epochs", type=int, default=int(1e9))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.2)
    args, unknown = ap.parse_known_args()

    main(args)
