from utils import *
from sklearn.metrics import f1_score, roc_auc_score

pred  = pd.read_csv(OUT2_CSV).sort_values(['rally_uid'])
label = pd.read_csv(LABEL2_CSV).sort_values(['rally_uid'])

def score(allA, allAp, allP, allPp, allR, allRp):
    try:
        f1A=f1_score(allA,allAp,average="macro") if len(allA) else 0.0
        f1P=f1_score(allP,allPp,average="macro") if len(allP) else 0.0
        print(f1_score(allP,allPp,average=None))
        print(f1_score(allA,allAp,average=None))
        auc=roc_auc_score(allR,allRp) if len(set(allR))>1 else 0.5
    except Exception: f1A,f1P,auc=0.0,0.0,0.5
    final=0.4*f1A+0.4*f1P+0.2*auc
    return f1A, f1P, auc, final     


assert all(pred['rally_uid'] == label['rally_uid'])
s = score(label['actionId'], pred['actionId'],
      label['pointId'], pred['pointId'],
      label['serverGetPoint'], pred['serverGetPoint'])

print(s)

