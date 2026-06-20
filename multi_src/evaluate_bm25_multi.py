import pandas as pd
from rank_bm25 import BM25Okapi
import re
import json # 🌟 新增 json 解析

def secure_ngram_tokenize(text):
    if not isinstance(text, str): return []
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text).replace(" ", "")
    tokens = list(text)
    for i in range(len(text) - 1): tokens.append(text[i:i+2])
    return tokens

# ==========================================
# 1. 載入資料並建立 BM25 索引
# ==========================================
print("讀取資料庫與測試集中...")
corpus_df = pd.read_csv("document_chunks_step2.csv")
# 🌟 改為讀取上一步生成的 Multi-hop 測試集
queries_df = pd.read_csv("multi_hop_evaluation_dataset.csv") 

print("正在建立 BM25 索引 (Indexing)...")
corpus_ids = corpus_df['Chunk_ID'].tolist()
corpus_texts = corpus_df['Text'].tolist()
tokenized_corpus = [secure_ngram_tokenize(doc) for doc in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)
print("✅ 索引建立完成！\n")

# ==========================================
# 2. 執行多目標 (Multi-hop) 檢索測試
# ==========================================
# 🌟 因為要跨段落找多個答案，K 值通常會設稍微大一點，比如 Top-5 或 Top-10
k_value = 10  
total_queries = len(queries_df)

partial_hit_count = 0  # 只要中一個就算
strict_hit_count = 0   # 全部都中才算
proportional_recall_sum = 0.0 # 比例加總
mrr_sum = 0.0

print(f"開始評估 {total_queries} 條【跨段落 Queries】的檢索效果 (Top-{k_value})...")

for index, row in queries_df.iterrows():
    query_text = row['Query']
    # 🌟 核心：將 CSV 裡的字串還原回 Python 的 List
    gt_chunks = json.loads(row['Ground_Truth_Chunks']) 
    required_count = len(gt_chunks)
    
    tokenized_query = secure_ngram_tokenize(query_text)
    # 取得前 K 名的 Chunk_ID 清單
    top_n_scores = bm25.get_top_n(tokenized_query, corpus_ids, n=k_value)
    
    # === 計算多目標分數邏輯 ===
    
    # 1. 找出我們在 Top-K 中命中了哪些 Standard Answer
    hits_in_top_k = [gt for gt in gt_chunks if gt in top_n_scores]
    hit_count = len(hits_in_top_k)
    
    # 2. 計算各項指標
    if hit_count > 0:
        partial_hit_count += 1
        proportional_recall_sum += (hit_count / required_count) # 例如中 2 個，總共要 3 個，+0.66
        
        # 計算 MRR (找命中的 Chunk 中，排名最高的那一個)
        # ranks 是一個 List，裝著所有命中的名次 (從 1 開始)
        ranks = [top_n_scores.index(gt) + 1 for gt in hits_in_top_k]
        best_rank = min(ranks)
        mrr_sum += (1.0 / best_rank)
        
    if hit_count == required_count:
        strict_hit_count += 1

# ==========================================
# 3. 產出多目標評估報告
# ==========================================
partial_rate = (partial_hit_count / total_queries) * 100
strict_rate = (strict_hit_count / total_queries) * 100
avg_proportional_recall = (proportional_recall_sum / total_queries) * 100
mrr_at_k = mrr_sum / total_queries

print("-" * 50)
print("📊 跨段落 (Multi-hop) 傳統檢索 (BM25) 評估報告")
print("-" * 50)
print(f"測試題數：{total_queries} 題")
print(f"指標 K 值：Top-{k_value}")
print(f"🎯 Partial Hit Rate (至少中一)：{partial_rate:.2f}% (適合寬鬆評估)")
print(f"🎯 Strict Hit Rate  (全數找齊)：{strict_rate:.2f}% (非常困難)")
print(f"📈 Proportional Recall (比例召回)：{avg_proportional_recall:.2f}% (最核心指標)")
print(f"🥇 First-Hit MRR@{k_value} (首次命中排名)：{mrr_at_k:.4f}")
print("-" * 50)