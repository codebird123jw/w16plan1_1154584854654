import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer, util
import ollama
import json

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
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val: return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

# ==========================================
# 2. 載入資料庫與跨段落測試集
# ==========================================
print("讀取資料庫與【跨段落 (Multi-hop)】測試集中...")
corpus_df = pd.read_csv("document_chunks_step2.csv")
# 🌟 改讀多目標資料集
queries_df = pd.read_csv("multi_hop_evaluation_dataset.csv")

corpus_ids = corpus_df['Chunk_ID'].tolist()
corpus_texts = corpus_df['Text'].tolist()

print("建立 BM25 索引中...")
tokenized_corpus = [secure_ngram_tokenize(doc) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("載入 Dense 模型 (E5)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
passage_texts = ["passage: " + text for text in corpus_texts]
corpus_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True)

# ==========================================
# 3. 第一步：Dense 檢索評估與分數正規化 (包含詳細 CSV 匯出)
# ==========================================
eval_k = 10 
total_queries = len(queries_df)

dense_partial_hits = 0
dense_strict_hits = 0
dense_prop_recall_sum = 0.0
dense_mrr_sum = 0.0

normalized_results_buffer = [] 

# 🌟 新增：準備用來寫入詳細 CSV 的資料容器
bm25_export_data = []
e5_export_data = []

print(f"\n開始評估 {total_queries} 條 Queries，進行分數正規化與詳細紀錄匯出...")

for index, row in queries_df.iterrows():
    query_text = row['Query']
    gt_chunks = json.loads(row['Ground_Truth_Chunks']) # 這是一個 List，例如 ['chunk_01', 'chunk_02']
    required_count = len(gt_chunks)
    
    # --- 取得 BM25 正規化分數 ---
    tokenized_query = secure_ngram_tokenize(query_text)
    bm25_scores = np.array(bm25.get_scores(tokenized_query)) 
    norm_bm25 = min_max_normalize(bm25_scores)
    
    # --- 取得 Dense 正規化分數 ---
    query_embedding = model.encode("query: " + query_text, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    norm_dense = min_max_normalize(dense_scores)
    
    # ==================================================
    # 🌟 新增：紀錄前 K 名的結果供 CSV 匯出 (追蹤肉搏戰細節)
    # ==================================================
    
    # 紀錄 BM25 的前 K 名
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:eval_k]
    for rank_idx, chunk_idx in enumerate(bm25_top_indices):
        retrieved_id = corpus_ids[chunk_idx]
        bm25_export_data.append({
            "Query": query_text,
            "N_Gram_Tokens": str(tokenized_query), # 紀錄 N-gram 具體長相
            "Ground_Truth_IDs": str(gt_chunks),    # 跨段落的所有標準答案
            "Rank": rank_idx + 1,
            "Retrieved_Chunk_ID": retrieved_id,
            # 判斷這個撈出來的 ID，是否是跨段落答案的其中一塊拼圖
            "Is_Correct_Part": "Yes" if retrieved_id in gt_chunks else "No",
            "BM25_Raw_Score": round(bm25_scores[chunk_idx], 4),
            "Normalized_Score": round(norm_bm25[chunk_idx], 4),
            "Retrieved_Text": corpus_texts[chunk_idx]
        })
        
    # 紀錄 E5 的前 K 名
    dense_top_indices = np.argsort(dense_scores)[::-1][:eval_k]
    for rank_idx, chunk_idx in enumerate(dense_top_indices):
        retrieved_id = corpus_ids[chunk_idx]
        e5_export_data.append({
            "Query": query_text,
            "Ground_Truth_IDs": str(gt_chunks),
            "Rank": rank_idx + 1,
            "Retrieved_Chunk_ID": retrieved_id,
            "Is_Correct_Part": "Yes" if retrieved_id in gt_chunks else "No",
            "Cosine_Similarity": round(float(dense_scores[chunk_idx]), 4),
            "Normalized_Score": round(norm_dense[chunk_idx], 4),
            "Retrieved_Text": corpus_texts[chunk_idx]
        })

    # ==================================================
    # --- 原本的 Dense 效能計算 ---
    dense_top_ids = [corpus_ids[i] for i in dense_top_indices]
    hits_in_top = [gt for gt in gt_chunks if gt in dense_top_ids]
    hit_count = len(hits_in_top)
    
    if hit_count > 0:
        dense_partial_hits += 1
        dense_prop_recall_sum += (hit_count / required_count)
        ranks = [dense_top_ids.index(gt) + 1 for gt in hits_in_top]
        dense_mrr_sum += (1.0 / min(ranks))
        
    if hit_count == required_count:
        dense_strict_hits += 1

    # 寫入暫存區傳給後面的步驟
    normalized_results_buffer.append({
        "query_text": query_text,
        "ground_truth_chunks": gt_chunks,
        "required_count": required_count,
        "norm_bm25_scores": norm_bm25,
        "norm_dense_scores": norm_dense
    })

# 🌟 新增：將收集到的詳細資料匯出成 CSV
print("\n正在匯出檢索底層詳細報告...")
pd.DataFrame(bm25_export_data).to_csv("bm25_detailed_results_multi.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(e5_export_data).to_csv("e5_detailed_results_multi.csv", index=False, encoding="utf-8-sig")
print("✅ 成功產生 bm25_detailed_results_multi.csv (包含 N-gram 切詞與分數)")
print("✅ 成功產生 e5_detailed_results_multi.csv (包含 E5 相似度與分數)")

# --- 印出 Baseline 2 報告 ---
print("\n" + "=" * 50)
print("📊 Baseline 2: Dense Retrieval (E5) 跨段落評估報告")
print("=" * 50)
print(f"🎯 Partial Hit Rate : {(dense_partial_hits/total_queries)*100:.2f}%")
print(f"🎯 Strict Hit Rate  : {(dense_strict_hits/total_queries)*100:.2f}%")
print(f"📈 Proportional Recall: {(dense_prop_recall_sum/total_queries)*100:.2f}%")
print(f"🥇 First-Hit MRR@{eval_k}  : {dense_mrr_sum/total_queries:.4f}")

# ==========================================
# 4. 第二步：靜態混合 (Fixed Hybrid) 網格搜索
# ==========================================
print("\n" + "=" * 50)
print("🚀 開始執行 Step 2: 靜態混合網格搜索 (Grid Search)")
print("優化目標：尋找能最大化『比例召回率 (Proportional Recall)』的 Alpha")
print("=" * 50)

alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_alpha = 0.0
best_prop_recall = 0.0
best_mrr = 0.0

print(f"{'Alpha':<8} | {'Prop. Recall (%)':<18} | {'Strict Rate (%)':<18} | {'MRR':<10}")
print("-" * 60)

for alpha in alpha_candidates:
    current_prop_sum = 0.0
    current_strict = 0
    current_mrr_sum = 0.0
    
    for item in normalized_results_buffer:
        gt_chunks = item['ground_truth_chunks']
        req_count = item['required_count']
        norm_bm25 = item['norm_bm25_scores']
        norm_dense = item['norm_dense_scores']
        
        hybrid_scores = (alpha * norm_dense) + ((1.0 - alpha) * norm_bm25)
        top_indices = np.argsort(hybrid_scores)[::-1][:eval_k]
        top_ids = [corpus_ids[i] for i in top_indices]
        
        hits = [gt for gt in gt_chunks if gt in top_ids]
        hit_c = len(hits)
        
        if hit_c > 0:
            current_prop_sum += (hit_c / req_count)
            ranks = [top_ids.index(gt) + 1 for gt in hits]
            current_mrr_sum += (1.0 / min(ranks))
        if hit_c == req_count:
            current_strict += 1
            
    avg_prop = (current_prop_sum / total_queries) * 100
    avg_strict = (current_strict / total_queries) * 100
    avg_mrr = current_mrr_sum / total_queries
    
    print(f"{alpha:<8.1f} | {avg_prop:<18.2f} | {avg_strict:<18.2f} | {avg_mrr:<10.4f}")
    
    if avg_prop > best_prop_recall or (avg_prop == best_prop_recall and avg_mrr > best_mrr):
        best_prop_recall = avg_prop
        best_mrr = avg_mrr
        best_alpha = alpha

print("-" * 60)
print(f"🏆 最佳固定權重 Alpha: {best_alpha}")

# ==========================================
# 5. 第三步：實作 DAT 動態權重分配 (LLM 評估器)
# ==========================================
print("\n" + "=" * 50)
print("🧠 開始執行 Step 3: DAT 動態權重引擎 (Local LLM Evaluator)")
print("=" * 50)

def get_llm_score_ollama(query, document, model_name="gemma3:12b"):
    """
    🌟 修改版 Prompt：加入「部分解答」的概念，適應跨段落情境
    """
    prompt = f"""你是一個專業的文件檢索評估員。這是一個【跨段落綜合問題】。
請評估以下單一段落是否包含了回答此問題的「關鍵拼圖或線索」。
請嚴格根據以下標準給分，並且「只能輸出一個介於 0 到 5 的單一數字」。

評分標準:
5 分: 完全命中，單一此段落就幾乎能完整回答該問題。
3-4 分: 部分命中。段落無法完整回答問題，但提供了回答問題所必須的【關鍵線索或部分事實】。
1-2 分: 稍微沾邊。提到相似詞彙，但對回答該問題的核心無太大幫助。
0 分: 完全無關。

【使用者問題】: {query}
【檢索到的段落】: {document}
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0} 
        )
        score_str = response['message']['content'].strip()
        match = re.search(r'\d', score_str)
        if match:
            return min(max(int(match.group()), 0), 5)
        return 0 
    except Exception as e:
        return 0 

# ... (前面保留：準備啟動 DAT 推論的 print 與 LLM 函數等) ...

dat_prop_sum = 0.0
dat_strict_hits = 0
dat_mrr_sum = 0.0

# 🌟 1. 新增：準備用來裝對決資料的容器
dat_case_study_data = [] 

print(f"啟動 DAT 推論，預計推論 {total_queries * 2} 次...")

for index, item in enumerate(normalized_results_buffer):
    query_text = item['query_text']
    gt_chunks = item['ground_truth_chunks'] # 這是一個 List
    req_count = item['required_count']
    norm_bm25 = item['norm_bm25_scores']
    norm_dense = item['norm_dense_scores']
    
    # 🌟 2. 修改：找出跨段落 Ground Truth 的所有實際文本，並把它們串接起來
    gt_texts = []
    for gt_id in gt_chunks:
        if gt_id in corpus_ids:
            gt_index = corpus_ids.index(gt_id)
            # 在文本前面加上 [chunk_ID] 標籤，方便你在 Excel 裡閱讀
            gt_texts.append(f"[{gt_id}]\n{corpus_texts[gt_index]}")
    # 用兩個換行符號把多個段落隔開
    ground_truth_text = "\n\n".join(gt_texts) 
    
    # 3-1. 抽取各自的 Top-1 文件
    top1_bm25_idx = np.argmax(norm_bm25)
    top1_dense_idx = np.argmax(norm_dense)
    doc_b1 = corpus_texts[top1_bm25_idx]
    doc_v1 = corpus_texts[top1_dense_idx]
    
    # 3-2. Local LLM 評分器 (取得 Sb 與 Sv)
    score_b = get_llm_score_ollama(query_text, doc_b1) 
    score_v = get_llm_score_ollama(query_text, doc_v1) 
    
    # 3-3. 計算動態 Alpha
    if score_v == 0 and score_b == 0: alpha_q = 0.5
    elif score_v == 5 and score_b != 5: alpha_q = 1.0
    elif score_b == 5 and score_v != 5: alpha_q = 0.0
    else: alpha_q = score_v / (score_v + score_b) if (score_v + score_b) > 0 else 0.5
    alpha_q = round(alpha_q, 1)
    
    # 3-4. 進行最終檢索與排序
    dat_scores = (alpha_q * norm_dense) + ((1.0 - alpha_q) * norm_bm25)
    top_indices = np.argsort(dat_scores)[::-1][:eval_k]
    top_ids = [corpus_ids[i] for i in top_indices]
    
    # 計算 DAT 多目標成績
    hits = [gt for gt in gt_chunks if gt in top_ids]
    hit_c = len(hits)
    
    # 🌟 3. 修改：跨段落排名的紀錄方式
    final_rank = "Out of Top 10" # 假設 eval_k 是 10
    if hit_c > 0:
        dat_prop_sum += (hit_c / req_count)
        # 找出命中名單中，排名最高的那一個 (First-Hit Rank)
        ranks = [top_ids.index(gt) + 1 for gt in hits]
        best_rank = min(ranks)
        dat_mrr_sum += (1.0 / best_rank)
        # 紀錄：顯示最好名次，以及總共撈到幾塊拼圖
        final_rank = f"First Hit Rank {best_rank} (Found {hit_c}/{req_count})"
        
    if hit_c == req_count:
        dat_strict_hits += 1

    # 🌟 4. 新增：將這一回合的完整戰況存入 List
    dat_case_study_data.append({
        "Query": query_text,
        "Ground_Truth_Text": ground_truth_text,
        "BM25_Top1_Text": doc_b1,
        "E5_Top1_Text": doc_v1,
        "Score_B_BM25": score_b,
        "Score_V_E5": score_v,
        "Calculated_Alpha": alpha_q,
        "DAT_Final_Rank": final_rank
    })

    if (index + 1) % 10 == 0:
        print(f"進度 [{index+1}/{total_queries}] | Sb={score_b}, Sv={score_v} -> Alpha={alpha_q}")

# 🌟 5. 新增：迴圈結束後，匯出 DAT 案例分析報告
print("\n正在匯出 DAT 跨段落案例分析報告 (DAT_case_study_multi.csv)...")
pd.DataFrame(dat_case_study_data).to_csv("DAT_case_study_multi.csv", index=False, encoding="utf-8-sig")
print("✅ 成功產生 DAT_case_study_multi.csv！")

# ... (後面保留計算平均分數與印出 Baseline 3 報告的程式碼) ...

dat_avg_prop = (dat_prop_sum / total_queries) * 100
dat_avg_strict = (dat_strict_hits / total_queries) * 100
dat_avg_mrr = dat_mrr_sum / total_queries

print("\n" + "=" * 50)
print("🌟 Baseline 3: DAT 動態混合策略 (Dynamic Hybrid) 跨段落評估報告")
print("=" * 50)
print(f"📈 Proportional Recall: {dat_avg_prop:.2f}% (對比 Fixed 最佳: {best_prop_recall:.2f}%)")
print(f"🎯 Strict Hit Rate  : {dat_avg_strict:.2f}%")
print(f"🥇 First-Hit MRR@{eval_k}  : {dat_avg_mrr:.4f} (對比 Fixed 最佳: {best_mrr:.4f})")
print("=" * 50)
