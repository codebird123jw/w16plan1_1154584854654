import pandas as pd
from rank_bm25 import BM25Okapi
import re

# ==========================================
# 1. 斷詞器定義 (採用字元級別 N-gram)
# ==========================================
def secure_ngram_tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    text = text.replace(" ", "")
    tokens = list(text) # 單字
    for i in range(len(text) - 1):
        tokens.append(text[i:i+2]) # 雙字
    return tokens

# ==========================================
# 2. 載入資料並建立 BM25 索引
# ==========================================
print("讀取資料庫與測試集中...")
corpus_df = pd.read_csv("document_chunks_step2.csv")
queries_df = pd.read_csv("bm25_evaluation_dataset.csv")

print("正在建立 BM25 索引 (Indexing)...")
corpus_ids = corpus_df['Chunk_ID'].tolist()
corpus_texts = corpus_df['Text'].tolist()
tokenized_corpus = [secure_ngram_tokenize(doc) for doc in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)
print("✅ 索引建立完成！\n")

# ==========================================
# 3. 執行檢索測試與計算 Recall@K & MRR@K
# ==========================================
k_value = 5  # 評估前 5 名
success_count = 0
mrr_sum = 0.0  # 用來加總 MRR 分數
total_queries = len(queries_df)

print(f"開始評估 {total_queries} 條 Queries 的檢索效果...")

for index, row in queries_df.iterrows():
    query_text = row['Query']
    ground_truth_id = row['Ground_Truth_Chunk_ID']
    
    tokenized_query = secure_ngram_tokenize(query_text)
    
    # 回傳前 K 高分的 Chunk_ID
    top_n_scores = bm25.get_top_n(tokenized_query, corpus_ids, n=k_value)
    
    # 檢查標準答案是否有出現在這 Top-K 名單中，並計算排名
    if ground_truth_id in top_n_scores:
        success_count += 1
        # .index() 找出在陣列中的位置(從 0 開始)，所以排名要 + 1
        rank = top_n_scores.index(ground_truth_id) + 1
        mrr_sum += (1.0 / rank)

# ==========================================
# 4. 產出最終評估報告
# ==========================================
recall_at_k = (success_count / total_queries) * 100
mrr_at_k = mrr_sum / total_queries  # MRR 通常不用百分比，而是 0~1 之間的小數

print("-" * 40)
print("📊 傳統稀疏檢索 (BM25) 系統評估報告")
print("-" * 40)
print(f"測試題數：{total_queries} 題")
print(f"chunk size: 200")
print(f"指標 K 值：Top-{k_value}")
print(f"🎯 命中次數：{success_count} 次")
print(f"📈 Recall@{k_value} (召回率)：{recall_at_k:.2f}%")
print(f"🥇 MRR@{k_value} (平均倒數排名)：{mrr_at_k:.4f}")
print("-" * 40)

if mrr_at_k > 0.7:
    print("💡 結論：極優！代表不僅能找到答案，且大多數答案都排在第 1 或第 2 名。")
elif recall_at_k > 80:
    print("💡 結論：召回率不錯，但 MRR 偏低，代表答案常墊底。未來可引入 Reranker (重排器) 提升排名。")