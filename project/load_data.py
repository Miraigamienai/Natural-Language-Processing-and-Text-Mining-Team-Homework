from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

    
class DL():
    train2 = None
    def __init__(self, *, val_size=None, with_test=False):
        self.data = self.read_csv(TRAIN_CSV, cnt_win_rate=True)
        self.test = self.read_csv(TEST_CSV)

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
    
    def read_csv(self, path, cnt_win_rate=False):
        try:
            data = pd.read_csv(path).sort_values(["rally_uid","strikeNumber"])
            data["scoreDiff"] = data["scoreSelf"] - data["scoreOther"]
            position_map = {0: "無", 1: "正手", 2: "中間", 3: "反手", 4: "正手",
                            5: "中間", 6: "反手", 7: "正手", 8: "中間", 9: "反手"}
            data['side'] = data["pointId"].map(position_map)
            data['player_pk_key'] = np.maximum(data['gamePlayerId'], data['gamePlayerOtherId']).astype(str)\
                                    + '_' + np.minimum(data['gamePlayerId'], data['gamePlayerOtherId']).astype(str)
            # winrate
            if cnt_win_rate:
                rally_df = data.sort_values("strikeNumber").groupby("rally_uid").last().reset_index()
                df1 = rally_df.copy()
                df1["player"] = rally_df["gamePlayerOtherId"] #最後一拍的對方是得分
                df1["is_win"] = rally_df["serverGetPoint"]
                df2 = rally_df.copy()
                df2["player"] = rally_df["gamePlayerId"]
                df2["is_win"] = 1 - rally_df["serverGetPoint"]
                long_rally = pd.concat([df1, df2], ignore_index=True)
                self.train_player_winrate = long_rally.groupby("player")["is_win"].mean()

                self.action_freq = self.get_player_skill_freq(data, columns_name="actionId")
                self.spin_freq = self.get_player_skill_freq(data, columns_name="spinId")
                self.point_freq = self.get_player_skill_freq(data, columns_name="pointId")
            
            data["p1_winrate"] = data["gamePlayerId"].map(self.train_player_winrate).fillna(0.5)
            data["p2_winrate"] = data["gamePlayerOtherId"].map(self.train_player_winrate).fillna(0.5)
            data["winrate_diff"] = data["p1_winrate"] - data["p2_winrate"]

            data = self.make_freq_data(data, "action_freq")
            data = self.make_freq_data(data, "spin_freq")
            data = self.make_freq_data(data, "point_freq")

        except Exception:
            data = pd.read_csv(path).sort_values(["rally_uid"])
        return data
    
    
    def get_player_skill_freq(self, data, columns_name):
        freq = data.groupby(["gamePlayerId", columns_name]).size().unstack(fill_value=0)
        freq = freq.div(freq.sum(axis=1).replace(0, 1), axis=0)
        self.skill_freq = freq 
        return freq
        
    def make_freq_data(self, data, freq_name):
        freq = getattr(self, freq_name, None)
        if freq is not None:
            for col in freq.columns:
                data[f"{freq_name}_player_{col}"] = data["gamePlayerId"].map(freq[col]).fillna(0)
                data[f"{freq_name}_other_player_{col}"] = data["gamePlayerOtherId"].map(freq[col]).fillna(0)
        return data
