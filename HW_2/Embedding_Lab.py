import pandas as pd
from scipy.stats import spearmanr
import os
import glob
from gensim.downloader import load
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_base_dir():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    
    return base_dir

BASE_DIR = get_base_dir()
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "my_text8_word2vec.model")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"資料夾 {DATASET_DIR} 不存在，請先下載資料集")
WORDSIM353 = os.path.join(DATASET_DIR, "wordsim353", "combined.tab")
SIMLEX999 = os.path.join(DATASET_DIR, "SimLex-999", "SimLex-999.txt")

corpus = load("text8")

model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=3, workers=6)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print("(40pt) (1) Deriving a word embedding model")
print("模型訓練並儲存成功！")

def evaluate_custom_similarity(file_path, model, score_column_index, delimiter='\t'):
    """
    自訂的評估函數：因為 SimLex-999 和 WordSim353 的分數欄位位置不同，
    我們利用 Pandas 來精準抓取對應欄位，並計算 Spearman 相關係數。
    """
    df = pd.read_csv(file_path, sep=delimiter)
    
    human_scores = []
    model_scores = []
    oov_count = 0
    
    for index, row in df.iterrows():
        word1 = str(row.iloc[0]).strip()
        word2 = str(row.iloc[1]).strip()
        
        # 動態抓取人類評分欄位 (WordSim353 在 index 2, SimLex-999 在 index 3)
        try:
            human_score = float(row.iloc[score_column_index])
        except ValueError:
            continue # 跳過表頭或無法轉換數值的列
            
        if word1 in model.wv and word2 in model.wv:
            m_score = model.wv.similarity(word1, word2)
            model_scores.append(m_score)
            human_scores.append(human_score)
        else:
            oov_count += 1
            
    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation, oov_count, len(model_scores)

print("(30pt) (2) Estimating word similarity of WordSim-353 and SimLex-999 corpora based on your word embedding model")
print("估算詞彙相似度：")

# 評估 WordSim-353 (使用 tab 檔案，分數在第 3 欄，也就是 index 2)
ws_corr, ws_oov, ws_total = evaluate_custom_similarity(
    WORDSIM353, model, score_column_index=2
)
print(f"WordSim-353 Spearman 相關係數: {ws_corr:.4f} (跳過 {ws_oov}/{ws_total} 筆 OOV 未知字)")

# 評估 SimLex-999 (分數在第 4 欄，也就是 index 3)
sim_corr, sim_oov, sim_total = evaluate_custom_similarity(
    SIMLEX999, model, score_column_index=3
)
print(f"SimLex-999 Spearman 相關係數: {sim_corr:.4f} (跳過 {sim_oov}/{sim_total} 筆 OOV 未知字)\n")


print("--------------------------------------------------")

print("(30pt) (3) Conduct analogy prediction of BATS dataset based on your word embedding model")
print("開始進行 BATS 詞彙類比預測 (這部分會進行大量的向量運算，請耐心等候幾分鐘)...\n")


def evaluate_bats_analogy(file_path, model):
    """讀取單一 BATS 檔案並計算類比準確率"""
    pairs = []
    # 讀取檔案
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                word1 = parts[0]
                # BATS 裡面的答案可能有多個，用 '/' 分隔
                valid_answers = parts[1].split('/')
                pairs.append((word1, valid_answers))
                
    # 過濾掉模型字典裡沒有的 OOV 單字
    valid_pairs = []
    for w1, ans_list in pairs:
        if w1 in model.wv:
            valid_ans = [a for a in ans_list if a in model.wv]
            if valid_ans:
                valid_pairs.append((w1, valid_ans))
                
    if len(valid_pairs) < 2:
        return 0.0, 0 # 剩下的單字太少，無法出題

    correct = 0
    total = 0
    
    # 兩兩互相配對出題 (A:B :: C:D)
    for w1_A, ans_A in valid_pairs:
        for w1_B, ans_B in valid_pairs:
            if w1_A == w1_B: 
                continue # 不能自己跟自己比
                
            a = w1_A
            b = ans_A[0] # 取 A 的第一個答案作為關係基準
            c = w1_B
            
            try:
                # 類比核心運算：預測 D = b - a + c (也就是 positive=[b, c], negative=[a])
                # topn=1 代表我們只看模型預測的第一名
                predicted = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
                pred_word = predicted[0][0]
                
                # 如果預測出來的字，命中 B 的任何一個標準答案，就算對！
                if pred_word in ans_B:
                    correct += 1
            except:
                pass
            total += 1
            
    if total == 0: 
        return 0.0, 0
    return correct / total, total

# --- 遍歷 4 大分類資料夾 ---
# 請確保你的 BATS 資料夾路徑正確，這裡假設 4 個分類資料夾都放在目前目錄下
bats_categories = [os.path.join(DATASET_DIR, "BATS_3.0", s) for s in [
    "1_Inflectional_morphology", 
    "2_Derivational_morphology", 
    "3_Encyclopedic_semantics", 
    "4_Lexicographic_semantics"
]]

for category in bats_categories:
    if not os.path.exists(category):
        print(f"找不到資料夾 {category}，請檢查路徑！")
        continue
        
    print(f"正在分析類別: {category}")
    
    # 讀取該類別下的所有 .txt 檔案
    for filename in os.listdir(category):
        if filename.endswith(".txt"):
            file_path = os.path.join(category, filename)
            
            # 呼叫我們上面寫的評估函數
            accuracy, total_questions = evaluate_bats_analogy(file_path, model)
            
            # 為了版面乾淨，我們聽從老師的建議，過濾掉 0% 的結果，或者你可以把下面這行 if 拿掉看全部
            if accuracy > 0:
                print(f"  - [{filename}]: 準確率 {accuracy*100:.2f}% (共測試 {total_questions} 題)")
    print("-" * 40)


print("(20pt) (4) Compare with other document similarity estimation methods. For example, co-occurrence matrix with TF-IDF, SVD, …")
def tfidf_document_similarity(file_path, text_corpus, score_column_index, delimiter='\t'):
    """
    計算 TF-IDF 文件相似度
    - text_corpus: list of strings (每個元素是一個文件/句子)
    """
    df = pd.read_csv(file_path, sep=delimiter)
    human_scores = []
    model_scores = []
    oov_count = 0

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_corpus)

    for index, row in df.iterrows():
        word1 = str(row.iloc[0]).strip()
        word2 = str(row.iloc[1]).strip()
        try:
            human_score = float(row.iloc[score_column_index])
        except ValueError:
            continue

        # 生成 TF-IDF 向量
        vecs = vectorizer.transform([word1, word2])
        if vecs.shape[1] == 0:
            oov_count += 1
            continue

        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        model_scores.append(sim)
        human_scores.append(human_score)

    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation, oov_count, len(model_scores)

text_corpus = [" ".join(sentence) for sentence in corpus]  # text8 的每句
tfidf_ws_corr, tfidf_ws_oov, tfidf_ws_total = tfidf_document_similarity(WORDSIM353, text_corpus, 2)
print(f"[TF-IDF] WordSim-353 Spearman: {tfidf_ws_corr:.4f}")


print("(20pt) (5) Apply word embeddings in other tasks. For example, classification, NER, …")
ACLIMDB = os.path.join(DATASET_DIR, "aclImdb")
train_texts, train_labels = [], []
test_texts, test_labels = [], []
def readfiles(texts, labels, dir):
    for label in ['pos', 'neg']:
        files = glob.glob(os.path.join(ACLIMDB, dir, label, '*.txt'))
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(1 if label=='pos' else 0)

readfiles(train_texts, train_labels, 'train')
readfiles(test_texts, test_labels, 'test')

def texts_to_vectors(texts, w2v_model):
    vectors = []
    for text in texts:
        words = text.split()  # 基本分詞，視需要可改用 nltk / spacy
        vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            # 如果整篇文章都是 OOV，給一個零向量
            vectors.append(np.zeros(w2v_model.vector_size))
    return np.array(vectors)

X_train = texts_to_vectors(train_texts, model)
X_test = texts_to_vectors(test_texts, model)
y_train = np.array(train_labels)
y_test = np.array(test_labels)


clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print(f"IMDb 評論分類準確率: {acc*100:.2f}%")