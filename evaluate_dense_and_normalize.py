import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. 斷詞器與正規化輔助函數
# ==========================================
def secure_ngram_tokenize(text):
    if not isinstance(text, str): return []
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text).replace(" ", "")
    tokens = list(text)
    for i in range(len(text) - 1): tokens.append(text[i:i+2])
    return tokens

def min_max_normalize(scores):
    """將一組分數陣列壓縮到 0 ~ 1 之間"""
    min_val = np.min(scores)
    max_val = np.max(scores)
    # 防呆：如果全部份數都一樣，避免除以零錯誤
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

# ==========================================
# 2. 載入資料庫與 Q&A 測試集
# ==========================================
print("讀取資料庫與測試集中...")
corpus_df = pd.read_csv("document_chunks_step2.csv")
queries_df = pd.read_csv("bm25_evaluation_dataset.csv")

corpus_ids = corpus_df['Chunk_ID'].tolist()
corpus_texts = corpus_df['Text'].tolist()

# ==========================================
# 3. 建立檢索索引 (BM25 & Dense E5)
# ==========================================
# 3.1 建立 BM25 索引
print("正在建立 BM25 索引...")
tokenized_corpus = [secure_ngram_tokenize(doc) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# 3.2 載入 E5 模型並建立 Dense 向量庫
print("載入微軟 E5 模型 (intfloat/multilingual-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

# ⚠️ E5 模型專屬規則：被檢索的文件(Document)前面必須加上 "passage: "
passage_texts = ["passage: " + text for text in corpus_texts]

print("正在將所有 Chunk 轉換為向量 (第一次執行需要下載模型，並花費幾分鐘運算)...")
# convert_to_tensor=True 可以讓後續計算 cosine similarity 更快
corpus_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)

# ==========================================
# 4. 執行檢索與分數正規化
# ==========================================
eval_k = 5
dense_success = 0
dense_mrr_sum = 0.0
total_queries = len(queries_df)

print(f"\n開始評估 {total_queries} 條 Queries 的 Dense 檢索效果，並進行分數正規化...")

# 創建一個列表，用來儲存正規化後的結果，為未來的第二步做準備
normalized_results_buffer = [] 

for index, row in queries_df.iterrows():
    query_text = row['Query']
    ground_truth_id = row['Ground_Truth_Chunk_ID']
    
    # --------------------------------------------------
    # 【檢索 A】：BM25 取得所有文件分數
    # --------------------------------------------------
    tokenized_query = secure_ngram_tokenize(query_text)
    bm25_scores = np.array(bm25.get_scores(tokenized_query)) 
    
    # --------------------------------------------------
    # 【檢索 B】：Dense 取得所有文件分數
    # --------------------------------------------------
    # ⚠️ E5 模型專屬規則：使用者的提問(Query)前面必須加上 "query: "
    query_embedding = model.encode("query: " + query_text, convert_to_tensor=True)
    
    # 計算 Cosine 相似度 (回傳的形狀是 1 x N，我們用 [0] 取出變成 1D 陣列)
    dense_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    
    # --------------------------------------------------
    # 【評估】：只評估 Dense 的表現 (Baseline 2)
    # --------------------------------------------------
    # argsort 會由小排到大，[::-1] 讓它反過來變成由大排到小，取前 eval_k 名
    dense_top_indices = np.argsort(dense_scores)[::-1][:eval_k]
    dense_top_ids = [corpus_ids[i] for i in dense_top_indices]
    
    if ground_truth_id in dense_top_ids:
        dense_success += 1
        rank = dense_top_ids.index(ground_truth_id) + 1
        dense_mrr_sum += (1.0 / rank)
        
    # --------------------------------------------------
    # 【正規化】：實作論文公式 (Per-Query Min-Max)
    # --------------------------------------------------
    # 針對「這一個 Query」在所有文件中的分數，進行 0~1 壓縮
    norm_bm25 = min_max_normalize(bm25_scores)
    norm_dense = min_max_normalize(dense_scores)
    
    # 將正規化後的分數先暫存起來 (這正是我們要傳遞給「混合策略模組」的資料)
    normalized_results_buffer.append({
        "query_text": query_text,
        "ground_truth_id": ground_truth_id,
        "norm_bm25_scores": norm_bm25,
        "norm_dense_scores": norm_dense
    })

# ==========================================
# 5. 產出 Baseline 2 報告
# ==========================================
dense_recall = (dense_success / total_queries) * 100
dense_mrr = dense_mrr_sum / total_queries

print("\n" + "=" * 45)
print("📊 Baseline 2: Dense Retrieval (E5) 評估報告")
print("=" * 45)
print(f"指標 K 值：Top-{eval_k}")
print(f"🎯 命中次數：{dense_success} / {total_queries} 次")
print(f"📈 Recall@{eval_k} (召回率)：{dense_recall:.2f}%")
print(f"🥇 MRR@{eval_k} (平均倒數排名)：{dense_mrr:.4f}")
print("=" * 45)
print("✅ 第一步完成！分數皆已成功透過 Min-Max 壓縮至 0~1。")
print("下一步你可以直接將 norm_bm25 與 norm_dense 相加來做 Hybrid Search！")