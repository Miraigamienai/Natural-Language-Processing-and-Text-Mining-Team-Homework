import os
def get_base_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()
BASE_DIR = get_base_dir()
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CSV_DIR = os.path.join(LOG_DIR, 'csv')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

import pandas as pd
import numpy as np
data_sub  = pd.read_csv(os.path.join(BASE_DIR, 'submission_lstm_baseline.csv')).sort_values(["rally_uid"])
data_true = pd.read_csv(os.path.join(DATASETS_DIR, 'Reference_Only_Old_Test_Data', 'test.csv')).sort_values(["rally_uid"])
data_pred = pd.read_csv(os.path.join(CSV_DIR, '0.3067041.csv')).sort_values(["rally_uid"])
data_pred = data_sub
data_true = data_true[["rally_uid","serverGetPoint"]]
data_pred = data_pred[["rally_uid","serverGetPoint"]]

merged = pd.merge(
    data_pred,
    data_true,
    on="rally_uid",
    how="inner",
    suffixes=("_pred","_true")
)
print(merged)
n = min(len(data_true), len(data_pred))
y_true = data_true.iloc[:n]
y_pred = data_pred.iloc[:n]
arr = np.stack([y_true, y_pred], axis=-1)


from sklearn.metrics import roc_auc_score
auc=roc_auc_score(
    merged["serverGetPoint_true"],
    merged["serverGetPoint_pred"]
)

print(f"AUC = {auc:.4f}")

print(merged[merged["serverGetPoint_true"].astype(bool) != (merged["serverGetPoint_pred"]>=0.5)])
