import torch
import pandas as pd
import numpy as np
import os

def get_base_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

BASE_DIR = get_base_dir()
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
TRAIN_CSV = os.path.join(DATASETS_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASETS_DIR, 'test_new.csv')
OUT_CSV =  os.path.join(BASE_DIR, 'submission_lstm_baseline.csv')

SELF_DEFI_DIR = os.path.join(DATASETS_DIR, 'self_defi_data')
LABEL2_CSV = os.path.join(SELF_DEFI_DIR, 'label2.csv')
TEST2_CSV = os.path.join(SELF_DEFI_DIR, 'test2.csv')
OUT2_CSV =  os.path.join(SELF_DEFI_DIR, 'submission.csv')

SEED = 42
UNK = "<UNK>"
MIN_FREQ = 5
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_to_csv(rally_uids, actionIds, pointIds, serverGetPoints, out_path):
    assert all(map(lambda x: len(x)==len(rally_uids), [actionIds, pointIds, serverGetPoints]))
    serverGetPoints = list(map(lambda x: round(x, 2), serverGetPoints))
    pred_df = pd.DataFrame({
        "rally_uid": rally_uids,
        "actionId": actionIds,
        "pointId": pointIds,
        "serverGetPoint": serverGetPoints
    }).sort_values("rally_uid")
    pred_df.to_csv(out_path, index=False)


def encode_frame(df, features, cats):
    outs=[]
    for col in features:
        s = df[col].astype(str)
        s = s.where(s.isin(cats[col]),UNK)
        codes = (pd.Categorical(s,categories=cats[col]).codes+1)
        outs.append(np.asarray(codes,dtype=np.int64))
    assert all([all(x!=0) for x in outs])
    return np.stack(outs,axis=1)


def get_categories(features, df):
    cats = {}
    for c in features:
        s = df[c].astype(str)
        vc = s.value_counts()
        rare = vc[vc < MIN_FREQ].index
        s = s.where(~s.isin(rare), UNK)
        cat = pd.Categorical(s).categories.tolist()
        if UNK not in cat:
            cat.append(UNK)
        cats[c] = cat
    return cats