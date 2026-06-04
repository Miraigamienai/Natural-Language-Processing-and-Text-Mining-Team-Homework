import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from load_data import DL
from utils import *

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

FEATURES = [
    "handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId",
    "strikeNumber", "rally_id", "numberGame", "match",
    "sex", "numberGame", "gamePlayerId", "gamePlayerOtherId",
    "scoreDiff", "scoreSelf", "scoreOther", 
    "side", "player_pk_key"
]

FEATURES2 = ([
    "p1_winrate", "p2_winrate", "winrate_diff"] 
    + [f'action_freq_player_{i}' for i in range(19)]
    + [f'action_freq_other_player_{i}' for i in range(19)]
    + [f'spin_freq_player_{i}' for i in range(6)]
    + [f'spin_freq_other_player_{i}' for i in range(6)]
    + [f'point_freq_player_{i}' for i in range(10)]
    + [f'point_freq_other_player_{i}' for i in range(10)]
)

PAD_TOKEN = 0
PATIENCE = 10000
# MIN_DELTA = 1e-7
MIN_DELTA = 0.05
        
INFERENCE = True

ACT_IDX = 0
PT_IDX = 1
SPT_IDX = 2
        
def pad2d_to_k(a, k, pad_val=PAD_TOKEN):
    out = np.full((k, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), k)
    out[:T] = a[:T]
    return out, T

def pad2d_to_k_float(a, k, pad_val=PAD_TOKEN):
    out = np.full((k, a.shape[1]), pad_val, dtype=np.float32)
    T = min(len(a), k)
    out[:T] = a[:T]
    return out, T

class RallyDataset(Dataset):
    def __init__(self, X, valueX, Y_list, Y_type_list, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.valueX = torch.tensor(valueX, dtype=torch.float32)
        self.Y = [torch.tensor(y, dtype=y_type) for y,y_type in zip(Y_list,Y_type_list)]
        self.L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.valueX[i], [y[i] for y in self.Y], self.L[i]

class InferDataset(Dataset):
    def __init__(self, data, k, cats):
        self.X = []
        self.valueX = []
        self.L = []
        self.rid = []
        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            Xg, valueXg = encode_frame(g, FEATURES, FEATURES2, cats)

            hist = Xg[max(0, len(Xg) - k):]   # 最後 k 筆
            Xp, T = pad2d_to_k(hist, k)
            hist2 = valueXg[max(0, len(valueXg) - k):]   # 最後 k 筆
            valueXp, T2 = pad2d_to_k_float(hist2, k)
            assert T==T2

            self.X.append(Xp)
            self.valueX.append(valueXp)
            self.L.append(T)
            self.rid.append(int(rid))

        self.X = torch.tensor(np.array(self.X), dtype=torch.long)
        self.L = torch.tensor(np.array(self.L), dtype=torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.valueX[i], self.L[i], self.rid[i]

def pool(o, query, mask):
    score = torch.matmul(o, query)  # [B,T]
    score = score.masked_fill(~mask, -1e9)
    weight = torch.softmax(score, dim=1)
    return (weight.unsqueeze(-1) * o).sum(1)

class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(len(num_tokens_per_feature)*emb_dim+len(FEATURES2), hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head  = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)

        self.act_query = nn.Parameter(torch.randn(hidden))
        self.pt_query  = nn.Parameter(torch.randn(hidden))
        self.rly_query = nn.Parameter(torch.randn(hidden))

    def forward(self, X, valueX, lengths):
        # X: (B, T, F)
        # valueX: (B, T, F2)
        # X[:,:,i]: (B, T)
        # es: list(B, T, H)
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        # x: (B, T, H*F)
        x = torch.cat(es, dim=-1)
        x = torch.cat([x, valueX], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o,_ = self.lstm(packed)
        o,_ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        o = self.drop(o)
        B, T, F = o.shape

        # act_o =  self.drop(nn.functional.gelu(self.act_change_list[min(ACT_SIZE, T)-1](o[:,:min(ACT_SIZE, T)].view(B, -1))))
        # pt_o = self.drop(nn.functional.gelu(self.pt_change_list[min(PT_SIZE, T)-1](o[:,:min(PT_SIZE, T)].view(B, -1))))
        
        mask = (torch.arange(T, device=device)[None] < lengths[:,None])
        
        act_o = pool(o, self.act_query, mask)
        pt_o  = pool(o, self.pt_query, mask)
        rly_o = pool(o, self.rly_query, mask)

        la, lp, lr = self.act_head(act_o), self.pt_head(pt_o), self.rly_head(rly_o).squeeze(-1)
        # la, lp, lr = self.act_head(o), self.pt_head(o), self.rly_head(rly_o).squeeze(-1)
        # idx = torch.arange(B, device=device)
        # last = lengths - 1
        # la = la[idx, last]; lp = lp[idx, last]
        
        assert self.cnt_weights
        la *= self.cnt_weights['act_w']
        lp *= self.cnt_weights['pt_w']
        return la, lp, lr

def score(allA, allAp, allP, allPp, allR, allRp):
    try:
        f1A=f1_score(allA,allAp,average="macro") if len(allA) else 0.0
        f1P=f1_score(allP,allPp,average="macro") if len(allP) else 0.0
        auc=roc_auc_score(allR,allRp) if len(set(allR))>1 else 0.5
    except Exception: f1A,f1P,auc=0.0,0.0,0.5
    final=0.4*f1A+0.4*f1P+0.2*auc
    return f1A, f1P, auc, final

def main(args):
    loader = DL(with_test=True)

    train = loader.get_train2()
    val   = loader.get_test2()
    test  = loader.get_test()

    # mapping dictionary
    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}

    cats = get_categories(FEATURES, train)

    def inference(model, data, k):
        ds = InferDataset(data, k, cats)
        dl = DataLoader(ds, batch_size=max(args.batch * 2, 128), shuffle=False)

        allRid, allAp, allPp, allRp = [], [], [], []
        model.eval()

        with torch.no_grad():
            for Xb, valueXb, Lb, Rb in dl:
                Xb = Xb.to(device)
                valueXb = valueXb.to(device)
                Lb = Lb.to(device)

                la, lp, lr = model(Xb, valueXb, Lb)

                # assert cnt_weights
                # la *= cnt_weights['act_w']
                # lp *= cnt_weights['pt_w']
                
                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)
                r_pred = torch.sigmoid(lr)

                allAp += a_pred.detach().cpu().tolist()
                allPp += p_pred.detach().cpu().tolist()
                allRp += r_pred.detach().cpu().tolist()
                allRid += Rb.tolist()

        return allRid, allAp, allPp, allRp
        
    #train
    def build_train_dataset(data, k):
        X_list, L_list = [], []
        valueX_list = []
        Y_types = [torch.long, torch.long, torch.float32]
        Y_lists = [[] for _ in range(3)]
        for rid, g in data.groupby("rally_uid", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < 2:
                continue

            Xg, valueXg = encode_frame(g, FEATURES, FEATURES2, cats)
            
            for t in range(1, len(g)):
                hist = Xg[max(0, t - k):t]   # 最多 k 筆
                Xp, T = pad2d_to_k(hist, k)  # pad 到固定長度 k

                hist2 = valueXg[max(0, t - k):t]
                valueXp, T2 = pad2d_to_k_float(hist2, k)  # pad 到固定長度 k
                assert T == T2

                X_list.append(Xp)
                valueX_list.append(valueXp)
                L_list.append(T)
                
                Y_lists[ACT_IDX].append(act_id2idx.get(g.loc[t, "actionId"], -1))
                Y_lists[PT_IDX].append(pt_id2idx.get(g.loc[t, "pointId"], -1))
                Y_lists[SPT_IDX].append(float(g.loc[t, "serverGetPoint"]))
        return RallyDataset(np.stack(X_list), np.stack(valueX_list), Y_lists, Y_types, L_list)

    def evaluate(model, dataset, device):
        model.eval()

        allA, allAp = [], []
        allP, allPp = [], []
        allR, allRp = [], []

        loader = DataLoader(dataset, batch_size=max(args.batch * 2, 128), shuffle=False)

        with torch.no_grad():
            for X, valueXb, Yb, L in loader:
                X = X.to(device)
                valueXb = valueXb.to(device)
                yA = Yb[ACT_IDX].to(device)
                yP = Yb[PT_IDX].to(device)
                yR = Yb[SPT_IDX].to(device)
                L = L.to(device)

                la, lp, lr = model(X, valueXb, L)

                a_pred = la.argmax(-1)
                p_pred = lp.argmax(-1)
                # a_pred = torch.multinomial(torch.softmax(la, dim=-1), num_samples=1)
                # p_pred = torch.multinomial(torch.softmax(lp, dim=-1), num_samples=1)

                r_pred = torch.sigmoid(lr).view(-1)

                mA = (yA != -1)
                mP = (yP != -1)
                mR = (yR != -1)

                allA += yA[mA].detach().cpu().tolist()
                allAp += a_pred[mA].detach().cpu().tolist()

                allP += yP[mP].detach().cpu().tolist()
                allPp += p_pred[mP].detach().cpu().tolist()

                allR += yR[mR].detach().cpu().tolist()
                allRp += r_pred[mR].detach().cpu().tolist()

        return score(allA, allAp, allP, allPp, allR, allRp)
     
    def train_fun(train_ds: RallyDataset, val_ds: RallyDataset, model, device, ce_action, ce_point):
        bce_rally = nn.BCEWithLogitsLoss()        
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        best_final = 0.0
        counter = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            run_loss = 0.0
            for Xb, valueXb, Yb, Lb in train_loader:
                Xb = Xb.to(device)
                valueXb = valueXb.to(device)
                yAb = Yb[ACT_IDX].to(device)
                yPb = Yb[PT_IDX].to(device)
                yRb = Yb[SPT_IDX].to(device)
                Lb = Lb.to(device)

                opt.zero_grad()
                la, lp, lr = model(Xb, valueXb, Lb)

                loss_a = ce_action(la, yAb)
                loss_p = ce_point(lp, yPb)

                r_mask = (yRb != -1)
                if r_mask.any():
                    loss_r = bce_rally(lr[r_mask], yRb[r_mask])
                else:
                    loss_r = torch.tensor(0.0, device=device)

                loss = 0.4 * loss_a + 0.4 * loss_p + 0.2 * loss_r
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                run_loss += loss.item() * Xb.size(0)

            tr_loss = run_loss / len(train_loader.dataset)
            t1_f1A, t1_f1P, t1_auc, t1_final = evaluate(model, val_ds, device)

            print(
                f"[Epoch {ep}/{args.epochs}] "
                f"train_loss={tr_loss:.4f} "
                f"F1_action={t1_f1A:.4f} F1_point={t1_f1P:.4f} "
                f"AUC={t1_auc:.4f} Final~{t1_final:.4f}"
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
    
    act_idx = test['actionId'].map(act_id2idx)
    act_counts = np.bincount(act_idx, minlength=n_act) + 1
    act_w = torch.tensor(1.0 / act_counts, dtype=torch.float32).to(device)
    # act_w[[8,18]]=0
    # act_w[[8]]=0
    act_w /= sum(act_w)

    pt_idx = test['pointId'].map(pt_id2idx)
    pt_counts = np.bincount(pt_idx, minlength=n_pt) + 1
    pt_w = torch.tensor(1.0 / pt_counts, dtype=torch.float32).to(device)
    # pt_w[[0,3]]=0
    pt_w /= sum(pt_w)

    cnt_weights = {
        'act_w':torch.sqrt(torch.sqrt(torch.sqrt(torch.sqrt(torch.tensor(act_counts/sum(act_counts), dtype=torch.float32).to(device))))),
        'pt_w':torch.sqrt(torch.sqrt(torch.sqrt(torch.sqrt(torch.tensor(pt_counts/sum(pt_counts), dtype=torch.float32).to(device))))),
    }
    model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop).to(device)
    setattr(model, "cnt_weights", cnt_weights)
    
    # ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.sqrt(act_w.to(device)))
    # ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.sqrt(pt_w.to(device)))
    ce_action = nn.CrossEntropyLoss(ignore_index=-1)
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1)
    
    train_fun(train_ds, val_ds, model, device, ce_action, ce_point)
    
    if INFERENCE:
        allRid, allAp, allPp, allRp = inference(model, test, args.k)
        action_pred = [int(act_classes[x]) for x in allAp]
        point_pred  = [int(pt_classes[x]) for x in allPp]
        df_to_csv(allRid, action_pred, point_pred, allRp, OUT_CSV)
        allRid, allAp, allPp, allRp = inference(model, loader.get_test2(), args.k)
        action_pred = [int(act_classes[x]) for x in allAp]
        point_pred  = [int(pt_classes[x]) for x in allPp]
        df_to_csv(allRid, action_pred, point_pred, allRp, OUT2_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--epochs", type=int, default=int(1e9))
    ap.add_argument("--epochs", type=int, default=11)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=10)
    args, unknown = ap.parse_known_args()

    main(args)
