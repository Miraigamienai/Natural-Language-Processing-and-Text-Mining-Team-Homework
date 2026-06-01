from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DL():
    train2 = None
    def __init__(self, *, val_size=None, with_test=False):
        self.test = self.read_csv(TEST_CSV)
        self.data = self.read_csv(TRAIN_CSV)

        self.test2  = self.read_csv(TEST2_CSV)
        self.label2 = self.read_csv(LABEL2_CSV)

        
        if with_test:
            df = self.read_csv(os.path.join(DATASETS_DIR, 'Reference_Only_Old_Test_Data', 'test.csv'))
            self.data = pd.concat([self.data, df])

            DL.train2 = self.data[~self.data['rally_uid'].isin(self.test2['rally_uid'])]
            
            assert not self.data.isna().values.any()
        
        if val_size:
            rally_lengths = []
            for r, g in self.data.groupby("rally_uid"):
                rally_lengths.append(len(g))
            median_len = np.median(rally_lengths)
            group_info=[]
            for r, g in self.data.groupby("rally_uid"):
                row = g.iloc[0]
                sex = row["sex"]
                length = len(g)
                length_high = int(length > median_len)
                server = row["serverGetPoint"]
                label = f"{sex}_{length_high}_{server}"
                group_info.append({
                    "rally_uid":r,
                    "label":label
                })
            group_df = pd.DataFrame(group_info)
            tr_rally, va_rally = train_test_split(
                group_df["rally_uid"],
                test_size=val_size,
                random_state=SEED,
                stratify=group_df["label"]
            )
            tr_rally=set(tr_rally); va_rally=set(va_rally)

            self.train = self.data[self.data["rally_uid"].isin(tr_rally)].reset_index(drop=True)
            self.val = self.data[self.data["rally_uid"].isin(va_rally)].reset_index(drop=True)

    def get_train(self):
        return self.train
    
    def get_val(self):
        return self.val

    def get_test(self):
        return self.test
    
    def get_test2(self):
        return self.test2
    
    def get_label2(self):
        return self.label2
    
    @classmethod
    def get_train2(cls):
        return cls.train2
    
    def read_csv(self, path):
        try:
            data = pd.read_csv(path).sort_values(["rally_uid","strikeNumber"])
            data["scoreDiff"] = data["scoreSelf"] - data["scoreOther"]
            position_map = {0: "無", 1: "正手", 2: "中間", 3: "反手", 4: "正手",
                            5: "中間", 6: "反手", 7: "正手", 8: "中間", 9: "反手"}
            data['side'] = data["pointId"].map(position_map)
        except Exception:
            data = pd.read_csv(path).sort_values(["rally_uid"])
        return data