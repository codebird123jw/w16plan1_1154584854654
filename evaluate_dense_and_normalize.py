import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer, util
import ollama
import time

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
# 4. 執行檢索與分數正規化 (並加入 CSV 匯出紀錄)
# ==========================================
eval_k = 5
dense_success = 0
dense_mrr_sum = 0.0
total_queries = len(queries_df)

print(f"\n開始評估 {total_queries} 條 Queries 的檢索效果，並記錄詳細結果至 CSV...")

normalized_results_buffer = [] 

# === 準備用來寫入 CSV 的資料容器 ===
bm25_export_data = []
e5_export_data = []

for index, row in queries_df.iterrows():
    query_text = row['Query']
    ground_truth_id = row['Ground_Truth_Chunk_ID']
    
    # --------------------------------------------------
    # 【檢索 A】：BM25 取得所有文件分數與正規化
    # --------------------------------------------------
    tokenized_query = secure_ngram_tokenize(query_text)
    bm25_scores = np.array(bm25.get_scores(tokenized_query)) 
    norm_bm25 = min_max_normalize(bm25_scores)
    
    # --------------------------------------------------
    # 【檢索 B】：Dense 取得所有文件分數與正規化
    # --------------------------------------------------
    query_embedding = model.encode("query: " + query_text, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    norm_dense = min_max_normalize(dense_scores)
    
    # --------------------------------------------------
    # 【評估】：Dense 效能計算
    # --------------------------------------------------
    dense_top_indices = np.argsort(dense_scores)[::-1][:eval_k]
    dense_top_ids = [corpus_ids[i] for i in dense_top_indices]
    
    if ground_truth_id in dense_top_ids:
        dense_success += 1
        rank = dense_top_ids.index(ground_truth_id) + 1
        dense_mrr_sum += (1.0 / rank)
        
    # --------------------------------------------------
    # 🌟 新增：紀錄前 5 名的結果供 CSV 匯出 (追蹤肉搏戰細節)
    # --------------------------------------------------
    # BM25 的前 5 名
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:eval_k]
    for rank_idx, chunk_idx in enumerate(bm25_top_indices):
        bm25_export_data.append({
            "Query": query_text,
            "N_Gram_Tokens": str(tokenized_query), # 紀錄 N-gram 切出來的具體長相
            "Ground_Truth_ID": ground_truth_id,
            "Rank": rank_idx + 1,
            "Retrieved_Chunk_ID": corpus_ids[chunk_idx],
            "Is_Correct": "Yes" if corpus_ids[chunk_idx] == ground_truth_id else "No",
            "BM25_Raw_Score": round(bm25_scores[chunk_idx], 4),
            "Normalized_Score": round(norm_bm25[chunk_idx], 4),
            "Retrieved_Text": corpus_texts[chunk_idx]
        })
        
    # E5 的前 5 名
    for rank_idx, chunk_idx in enumerate(dense_top_indices):
        e5_export_data.append({
            "Query": query_text,
            "Ground_Truth_ID": ground_truth_id,
            "Rank": rank_idx + 1,
            "Retrieved_Chunk_ID": corpus_ids[chunk_idx],
            "Is_Correct": "Yes" if corpus_ids[chunk_idx] == ground_truth_id else "No",
            "Cosine_Similarity": round(dense_scores[chunk_idx], 4),
            "Normalized_Score": round(norm_dense[chunk_idx], 4),
            "Retrieved_Text": corpus_texts[chunk_idx]
        })

    # 將正規化後的分數先暫存起來 (傳遞給後面的 DAT 模組)
    normalized_results_buffer.append({
        "query_text": query_text,
        "ground_truth_id": ground_truth_id,
        "norm_bm25_scores": norm_bm25,
        "norm_dense_scores": norm_dense
    })

# === 將收集到的詳細資料匯出成 CSV ===
print("正在匯出詳細檢索報告...")
pd.DataFrame(bm25_export_data).to_csv("bm25_detailed_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(e5_export_data).to_csv("e5_detailed_results.csv", index=False, encoding="utf-8-sig")
print("✅ 成功產生 bm25_detailed_results.csv (包含 N-gram 切詞與分數)")
print("✅ 成功產生 e5_detailed_results.csv (包含 E5 相似度與分數)")

# ==========================================
# 5. 產出 Baseline 2 報告
# ==========================================
# ...(保留原本的第 5 步印出報表的程式碼)...
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

# ==========================================
# 6. 第二步：靜態混合 (Fixed Hybrid) 網格搜索 Grid Search
# ==========================================
print("\n" + "=" * 45)
print("🚀 開始執行 Step 2: 靜態混合網格搜索 (Grid Search)")
print("尋找最佳的固定權重 Alpha (Dense 的佔比)")
print("公式: Score = Alpha * Norm_Dense + (1 - Alpha) * Norm_BM25")
print("=" * 45)

# 定義要測試的 alpha 值 (0.0 到 1.0，間隔 0.1)
alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_alpha = 0.0
best_recall = 0.0
best_mrr = 0.0
grid_search_results = []

print(f"{'Alpha':<8} | {'Recall@5 (%)':<15} | {'MRR@5':<10}")
print("-" * 40)

# 針對每一個 alpha 值進行測試
for alpha in alpha_candidates:
    success_count = 0
    mrr_sum = 0.0
    
    # 遍歷剛剛第一步存在記憶體裡的所有 Q&A 正規化分數
    for item in normalized_results_buffer:
        ground_truth_id = item['ground_truth_id']
        norm_bm25 = item['norm_bm25_scores']
        norm_dense = item['norm_dense_scores']
        
        # 核心公式：混合分數計算 (論文公式 5)
        hybrid_scores = (alpha * norm_dense) + ((1.0 - alpha) * norm_bm25)
        
        # 找出最高分的前 5 名
        top_indices = np.argsort(hybrid_scores)[::-1][:eval_k]
        top_ids = [corpus_ids[i] for i in top_indices]
        
        # 計算命中與排名
        if ground_truth_id in top_ids:
            success_count += 1
            rank = top_ids.index(ground_truth_id) + 1
            mrr_sum += (1.0 / rank)
            
    # 計算這個 alpha 下的總成績
    current_recall = (success_count / total_queries) * 100
    current_mrr = mrr_sum / total_queries
    
    grid_search_results.append({
        "alpha": alpha,
        "recall": current_recall,
        "mrr": current_mrr
    })
    
    # 印出當下結果 (格式化排版)
    print(f"{alpha:<8.1f} | {current_recall:<15.2f} | {current_mrr:<10.4f}")
    
    # 紀錄最佳表現 (優先看 Recall，若 Recall 一樣看 MRR)
    if current_recall > best_recall or (current_recall == best_recall and current_mrr > best_mrr):
        best_recall = current_recall
        best_mrr = current_mrr
        best_alpha = alpha

# ==========================================
# 7. 輸出最佳結論
# ==========================================
print("-" * 40)
print(f"🏆 最佳固定權重 (Optimal Fixed Alpha): {best_alpha}")
print(f"✅ 最佳 Recall@5 : {best_recall:.2f}%")
print(f"✅ 最佳 MRR@5    : {best_mrr:.4f}")
print("=" * 45)
print("恭喜！你已經完成了論文中的 Fixed Hybrid Baseline！")


import ollama
import time
import re

# ==========================================
# 8. 第三步：實作 DAT 動態權重分配 (Local LLM: Gemma3:12b)
# ==========================================
print("\n" + "=" * 45)
print("🧠 開始執行 Step 3: DAT 動態權重引擎 (Local LLM Evaluator)")
print("使用本地端 gemma3:12b 評估 Top-1 文件，動態計算 Alpha")
print("=" * 45)

def get_llm_score_ollama(query, document, model_name="gemma3:12b"):
    """
    呼叫本地 Ollama 進行檢索品質評分 (0~5分)
    """
    prompt = f"""你是一個專業的文件檢索評估員。請評估以下【檢索到的段落】是否能回答【使用者問題】。
請嚴格根據以下標準給分，並且「只能輸出一個介於 0 到 5 的單一數字」，絕對不要輸出任何解釋、標點符號或其他文字。

評分標準 (Scoring Rubric):
5 分 (Direct hit): 完全命中。段落直接且完整地回答了問題。
3-4 分 (Good wrong result): 概念相近。段落雖然沒有直接給出答案，但概念非常接近，有很高的機率答案就在這份上下文中。
1-2 分 (Bad wrong result): 稍微相關但會誤導。段落只提到一點點相關詞彙，但內容完全偏離問題，無參考價值。
0 分 (Completely off-track): 完全無關。段落內容與問題毫無關聯。

【使用者問題】: {query}
【檢索到的段落】: {document}
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0} # 溫度設為 0，確保它不亂發揮
        )
        
        score_str = response['message']['content'].strip()
        
        # 【防呆機制】：抓取字串中的第一個出現的數字
        match = re.search(r'\d', score_str)
        if match:
            score = int(match.group())
            return min(max(score, 0), 5) # 確保極端情況下不會超出 0~5
        else:
            print(f"⚠️ 找不到數字，模型回傳了: {score_str}")
            return 0 
            
    except Exception as e:
        print(f"🚨 LLM 評分發生錯誤: {e}")
        return 0 

# ... (前面保留原來的程式碼) ...

# 紀錄 DAT 的結果與匯出資料
dat_success_count = 0
dat_mrr_sum = 0.0
dat_case_study_data = [] # 🌟 新增：用來裝對決資料的容器

print(f"準備對 {total_queries} 條 Query 進行動態分配...\n")
print("⏳ 注意：因為使用本地端 LLM 進行 600 次推論，需要一段時間運算，請耐心等候...")

for index, item in enumerate(normalized_results_buffer):
    query_text = item['query_text']
    ground_truth_id = item['ground_truth_id']
    norm_bm25 = item['norm_bm25_scores']
    norm_dense = item['norm_dense_scores']
    
    # 🌟 新增：找出 Ground Truth 的實際文本 (為了簡報對比用)
    gt_index = corpus_ids.index(ground_truth_id)
    ground_truth_text = corpus_texts[gt_index]
    
    # 3-1. 抽取各自的 Top-1 文件
    top1_bm25_idx = np.argmax(norm_bm25)
    top1_dense_idx = np.argmax(norm_dense)
    
    doc_b1 = corpus_texts[top1_bm25_idx]
    doc_v1 = corpus_texts[top1_dense_idx]
    
    # 3-2. Local LLM 評分器 (取得 Sb 與 Sv)
    score_b = get_llm_score_ollama(query_text, doc_b1) 
    score_v = get_llm_score_ollama(query_text, doc_v1) 
    
    # 3-3. 計算動態 Alpha
    if score_v == 0 and score_b == 0:
        alpha_q = 0.5
    elif score_v == 5 and score_b != 5:
        alpha_q = 1.0
    elif score_b == 5 and score_v != 5:
        alpha_q = 0.0
    else:
        if (score_v + score_b) > 0:
            alpha_q = score_v / (score_v + score_b)
        else:
            alpha_q = 0.5
    alpha_q = round(alpha_q, 1)
    
    # 3-4. 最終融合與排序
    dat_scores = (alpha_q * norm_dense) + ((1.0 - alpha_q) * norm_bm25)
    top_indices = np.argsort(dat_scores)[::-1][:eval_k]
    top_ids = [corpus_ids[i] for i in top_indices]
    
    # 🌟 新增：紀錄 DAT 融合後，正確答案到底排第幾名
    final_rank = "Out of Top 5"
    if ground_truth_id in top_ids:
        dat_success_count += 1
        rank_num = top_ids.index(ground_truth_id) + 1
        dat_mrr_sum += (1.0 / rank_num)
        final_rank = f"Rank {rank_num}"

    # 🌟 新增：將這一回合的完整戰況存入 List
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
        print(f"進度 [{index+1}/{total_queries}] | 觀察: 題目 {index+1} 判定 Sb={score_b}, Sv={score_v} -> 分配 Alpha={alpha_q}")

# 🌟 新增：迴圈結束後，將對決資料匯出成 CSV
print("\n正在匯出 DAT 案例分析報告 (DAT_case_study.csv)...")
pd.DataFrame(dat_case_study_data).to_csv("DAT_case_study.csv", index=False, encoding="utf-8-sig")
print("✅ 成功產生 DAT_case_study.csv！請用 Excel 打開它來挑選簡報素材。")

# ... (後面保留產出 DAT 評估報告的 print 程式碼) ...

# ==========================================
# 9. 產出 DAT 評估報告
# ==========================================
dat_recall = (dat_success_count / total_queries) * 100
dat_mrr = dat_mrr_sum / total_queries

print("\n" + "=" * 45)
print("🌟 Baseline 3: DAT 動態混合策略 (Dynamic Hybrid) 評估報告")
print("=" * 45)
print(f"🎯 命中次數：{dat_success_count} / {total_queries} 次")
print(f"📈 Recall@{eval_k} (召回率)：{dat_recall:.2f}% (請比較 Fixed Hybrid 的 {best_recall:.2f}%)")
print(f"🥇 MRR@{eval_k} (平均倒數排名)：{dat_mrr:.4f} (請比較 Fixed Hybrid 的 {best_mrr:.4f})")
print("=" * 45)
