# NLP Project 桌球時序預測

## 小組隊員
- 112590059 傅啓碩 (100%)

## 安裝與使用
1. **安裝套件**

```bash
pip install -r requirements.txt
````

2. **執行程式**

```bash
python "baseline code.py"
```

## 說明
**檔案架構**

```text
project/
│
├── datasets/
│   └── self_defi_data/
│       └── label2.csv
│       └── submission.csv #生成結果(測試用)
│       └── test2.csv
│   └── sample_submission.csv #生成結果(繳交用)
│   └── test_new.csv
│   └── train.csv
├── baseline code.py #主要程式
├── compare.py #結果試算
├── copycode.py #程式及結果儲存
├── load_data.py #提供存取data相關函式
├── utils.py #提供共用函式
├── requirements.txt
└── README.md
```

## 評估指標

1. **Action Prediction**：預測擊球動作類別 (actionId)
2. **Point Prediction**：預測該回合得分方 (pointId)
3. **Rally Outcome Prediction**：預測發球方是否獲得該分 (serverGetPoint)

---

-  **Action Prediction** 採用 **Macro F1-score** 作為評估指標：**F1<sub>*Action*</sub>**
-  **Point Prediction** 同樣採用 **Macro F1-score** 作為評估指標：**F1<sub>*Point*</sub>**
-  **Rally Outcome Prediction** 採用 **Area Under the ROC Curve (ROC-AUC)** 作為評估指標：**AUC**

### **Final Score**
最終評分由三項指標加權組成：

$Score=0.4*F1_{Action}+0.4*F1_{Point}+0.2*AUC$

