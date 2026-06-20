import pandas as pd
import ollama
import json
import random

# ==========================================
# 補上第一步：連續滑動視窗 (Sliding Window) 構建邏輯
# ==========================================

def create_sliding_windows(csv_path, window_size=6, stride=3):
    """
    將單一的破碎 Chunk 有邏輯地融合成物理上連續的大視窗
    - window_size: 每個視窗包含幾個 Chunk (預設 6 個，約 1200 字)
    - stride: 每次視窗滑動的步幅 (預設 3，代表前後視窗重疊 3 個 Chunk 避免邊界斷層)
    """
    print(f"正在讀取並依序組合滑動視窗 (Window Size: {window_size}, Stride: {stride})...")
    df = pd.read_csv(csv_path)
    
    # 確保按照 Chunk_ID 的物理順序排列，這樣縫合的文字才具有連續邏輯
    df = df.sort_values(by='Chunk_ID').reset_index(drop=True)
    
    windows = []
    total_chunks = len(df)
    
    # 利用步幅控制迴圈滑動
    for i in range(0, total_chunks, stride):
        window_df = df.iloc[i : i + window_size]
        
        # 防呆：如果尾部的段落太少（例如只剩 1-2 個 Chunk），就跳過不建立新視窗
        if len(window_df) < 3: 
            break
            
        # 提取這個視窗內含的所有 Chunk 完整資料 (包含 ID、Text、頁碼)
        chunks_list = window_df.to_dict(orient='records')
        included_chunk_ids = window_df['Chunk_ID'].tolist()
        pages = window_df['Page_Number'].unique().tolist()
        
        windows.append({
            "Window_ID": f"window_{len(windows)+1:04d}",
            "Included_Chunk_IDs": included_chunk_ids,
            "Pages": pages,
            "Chunks_Data": chunks_list  # 儲存完整的結構以便後續處理
        })
        
    windows_df = pd.DataFrame(windows)
    print(f"✅ 成功將 {total_chunks} 個 Chunks 縫合並轉換為 {len(windows_df)} 個大視窗！")
    return windows_df

def sample_windows(windows_df, sample_size=100):
    """從建構好的大視窗中隨機抽出指定數量的 Window 作為問題生成種子"""
    if len(windows_df) < sample_size:
        sample_size = len(windows_df)
        
    sampled_df = windows_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"✅ 成功隨機抽出 {sample_size} 個視窗區塊作為種子資料。")
    return sampled_df

# ==========================================
# 第三步：升級為跨段落 (Multi-hop) Prompt 工程
# ==========================================

def get_system_prompt_multi():
    """設計嚴格的跨段落 System Prompt，強迫模型尋找關聯並輸出關聯 ID 陣列"""
    return """你是一個專業的繁體中文機密文件分析師與考題設計專家。
任務：閱讀使用者提供的【多段落文本】，並根據這些內容生成 2 到 3 個「困難、具體，且必須跨段落綜合理解」的繁體中文提問。

規則：
1. 提問的答案【必須】涵蓋 2 個或以上的段落內容才能完整回答。絕對不要問只需看單一段落就能解答的簡單事實題。
2. 仔細閱讀各段落的內容，並精確找出回答該問題需要用到哪幾個段落的資訊，將對應的段落 ID 記錄下來。
3. 詢問方式要自然、直接，不要包含「根據這段文本」、「請問」等廢話。
4. 絕對不能輸出任何多餘的解釋、思考過程或對話。
5. 必須嚴格按照以下的 JSON 陣列格式輸出，屬性名稱必須完全一致：
[
  {
    "query": "你的第一個跨段落綜合問題",
    "ground_truth_chunks": ["chunk_00001", "chunk_00003"] 
  },
  {
    "query": "你的第二個跨段落綜合問題",
    "ground_truth_chunks": ["chunk_00004", "chunk_00005", "chunk_00006"]
  }
]
"""

# ==========================================
# 第四步：自動化生成與【跨段落專屬】防呆校正機制
# ==========================================

def generate_multi_hop_queries(sampled_windows_df, model_name="gemma3:12b"):
    generated_data = []
    total = len(sampled_windows_df)
    
    print(f"開始啟動 {model_name} 進行跨段落自動化生成，共 {total} 筆任務...")
    
    for index, row in sampled_windows_df.iterrows():
        window_id = row['Window_ID']
        valid_chunk_ids = row['Included_Chunk_IDs']
        chunks_data = row['Chunks_Data']
        
        # 🌟 補上：建構帶有「段落 ID 標籤」的結構化文本送給模型，讓模型擁有邊界感知能力
        structured_text = ""
        for chunk in chunks_data:
            structured_text += f"【段落 ID: {chunk['Chunk_ID']}】\n{chunk['Text']}\n\n"
            
        print(f"進度 [{index+1}/{total}] - 正在處理 {window_id} (涵蓋 {valid_chunk_ids[0]} ~ {valid_chunk_ids[-1]})...")
        
        messages = [
            {"role": "system", "content": get_system_prompt_multi()},
            {"role": "user", "content": f"【多段落文本如下】：\n{structured_text}"}
        ]
        
        try:
            # 呼叫 Ollama，利用 format='json' 強制模型輸出 JSON 格式
            response = ollama.chat(
                model=model_name,
                messages=messages,
                format='json',
                options={"temperature": 0.4} # 稍微提高一點溫度(0.4)，有利於模型進行跨段落的創意聯想
            )
            
            result_str = response['message']['content'].strip()
            queries_json = json.loads(result_str)
            
            # 【終極版解析防呆】適應跨段落新格式的形狀抓取
            extracted_queries = []
            if isinstance(queries_json, list):
                extracted_queries = queries_json
            elif isinstance(queries_json, dict):
                for key, value in queries_json.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        extracted_queries = value
                        break
                if len(extracted_queries) == 0 and "query" in queries_json:
                    extracted_queries = [queries_json] 
            
            # 🌟 補上：跨段落專屬校正邏輯（驗證欄位並過濾虛幻的 Chunk_ID）
            valid_queries = []
            for q in extracted_queries:
                if isinstance(q, dict) and 'query' in q and 'ground_truth_chunks' in q:
                    query_text = q['query'].strip()
                    gt_chunks = q['ground_truth_chunks']
                    
                    # 確保 gt_chunks 是 List 且裡面的 ID 確實存在於目前這個滑動視窗中 (防幻覺)
                    if isinstance(gt_chunks, list):
                        real_gt_chunks = [cid for cid in gt_chunks if cid in valid_chunk_ids]
                        
                        # 只有當確實抓到有效的 Ground Truth 且問題非空時才保留
                        if query_text and len(real_gt_chunks) > 0:
                            valid_queries.append({
                                "query": query_text,
                                "ground_truth_chunks": real_gt_chunks
                            })
            
            # 確保有抓到有效的跨段落考題
            if len(valid_queries) > 0:
                generated_data.append({
                    "Window_ID": window_id,
                    "Included_Chunk_IDs": valid_chunk_ids,
                    "Generated_Queries": valid_queries 
                })
            else:
                print(f"⚠️ {window_id} 模型沒有產出有效的跨段落問題。內容：{result_str}")
                
        except json.JSONDecodeError:
            print(f"❌ {window_id} JSON 解析失敗，模型格式崩潰。已跳過。")
        except Exception as e:
            print(f"🚨 {window_id} 發生未知錯誤：{str(e)}")
            
    return generated_data

# === 執行主程式 ===
if __name__ == "__main__":
    # 1. 讀取你第二步切塊產出的原始 CSV
    input_csv = "document_chunks_step2.csv"
    
    # 2. 補上：呼叫滑動視窗函數，將 Chunk 重新縫合，並抽出 100 個大區塊作為測試來源
    # (註：因為合併後總區塊數變少，若要先驗證模型，sample_size 可以先設 5)
    df_windows = create_sliding_windows(input_csv, window_size=6, stride=3)
    df_sample = sample_windows(df_windows, sample_size=100) 
    
    # 3. 呼叫 Ollama 進行跨段落生成
    raw_results = generate_multi_hop_queries(df_sample, model_name="gemma3:12b") 
    
    # 4. 將帶有跨段落層級的原始 JSON 結果存檔備份 (多增設 _multi 字樣以茲區隔)
    output_json_path = "raw_generated_queries_multi.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 跨段落生成完畢！成功率：{len(raw_results)}/{len(df_sample)}。")
    print(f"帶有多重標籤的 JSON 數據已備份至：{output_json_path}")