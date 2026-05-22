import os
import shutil

name = "0.2558294"

import os
def get_base_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()
BASE_DIR = get_base_dir()
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CODES_DIR = os.path.join(LOG_DIR, 'codes')
CSV_DIR = os.path.join(LOG_DIR, 'csv')
os.makedirs(CODES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

LOG_POS = os.path.join(CODES_DIR, f'{name}.log')

CSV_POS = os.path.join(CSV_DIR, f'{name}.csv')
# 複製 baseline code.py -> logs/codes/name.log
with open(os.path.join(BASE_DIR, 'baseline code.py'), 'r', encoding='utf-8') as f:
    with open(LOG_POS, 'w', encoding='utf-8') as f2:
        f2.write(f.read())

# 移動 submission_lstm_baseline.csv -> logs/csv/name.csv
shutil.move(
    os.path.join(BASE_DIR, 'submission_lstm_baseline.csv'),
    CSV_POS
)