import pandas as pd
import ollama
import json
import random

# ==========================================
# 第三步：隨機抽樣與 Prompt 工程
# ==========================================

def get_system_prompt():
    """設計嚴格的 System Prompt，逼迫模型只能輸出 JSON 陣列"""
    return """你是一個專業的繁體中文機密文件分析師。
任務：閱讀使用者提供的【文本段落】，並根據該內容生成 3 個「困難且具體」的繁體中文提問。
規則：
1. 提問的答案必須能從該文本中找到。
2. 不要包含「根據這段文本」、「請問」等廢話，直接給出問題。
3. 絕對不能輸出任何多餘的解釋或對話。
4. 必須嚴格按照以下的 JSON 陣列格式輸出：
[
  {"query": "你的第一個問題"},
  {"query": "你的第二個問題"},
  {"query": "你的第三個問題"}
]
"""

def sample_chunks(csv_path, sample_size=100):
    """讀取第二步的 CSV 並隨機抽出指定數量的 Chunk"""
    print(f"讀取資料庫：{csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # 確保資料量足夠抽樣
    if len(df) < sample_size:
        sample_size = len(df)
        
    # random_state=42 確保每次抽樣結果一致，方便你 debug
    sampled_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"✅ 成功隨機抽出 {sample_size} 筆 Chunk 作為種子資料。")
    return sampled_df

# ==========================================
# 第四步：自動化生成與防呆機制
# ==========================================

def generate_queries(sampled_df, model_name="gemma3:12b"):
    generated_data = []
    total = len(sampled_df)
    
    print(f"開始啟動 {model_name} 進行自動化生成，共 {total} 筆任務...")
    
    for index, row in sampled_df.iterrows():
        chunk_id = row['Chunk_ID']
        page_num = row['Page_Number']
        text_content = row['Text']
        
        print(f"進度 [{index+1}/{total}] - 正在處理 {chunk_id} (第 {page_num} 頁)...")
        
        # 組裝發給 Ollama 的訊息
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"【文本段落】：\n{text_content}"}
        ]
        
        try:
            # 呼叫 Ollama，利用 format='json' 強制模型輸出 JSON 格式
            response = ollama.chat(
                model=model_name,
                messages=messages,
                format='json',
                options={"temperature": 0.3} # 溫度調低，讓輸出更穩定、少幻覺
            )
            
            # 取得模型回傳的字串
            result_str = response['message']['content'].strip()
            
            # 將字串解析為 JSON
            queries_json = json.loads(result_str)
            
            # 【終極版解析防呆】不管模型給什麼形狀，硬抓出來
            extracted_queries = []
            
            # 情況 1：模型很乖，給了標準的陣列 [...]
            if isinstance(queries_json, list):
                extracted_queries = queries_json
                
            # 情況 2：模型給了字典 {...}
            elif isinstance(queries_json, dict):
                # 嘗試找字典裡有沒有陣列 (例如 "questions": [...])
                for key, value in queries_json.items():
                    if isinstance(value, list):
                        extracted_queries = value
                        break
                        
                # 情況 3 (應對你遇到的狀況)：模型只給了單一字典 {"query": "問題"}
                if len(extracted_queries) == 0 and "query" in queries_json:
                    # 我們手動把它包裝成陣列
                    extracted_queries = [queries_json] 
            
            # 確保抓出來的東西裡面，確實有包含 'query' 這個欄位
            valid_queries = [q for q in extracted_queries if isinstance(q, dict) and 'query' in q]
            
            # 確保有抓到東西
            if len(valid_queries) > 0:
                generated_data.append({
                    "Chunk_ID": chunk_id,
                    "Page_Number": page_num,
                    "Chunk_Text": text_content,
                    "Generated_Queries": valid_queries # 可能是 1~3 個問題
                })
            else:
                print(f"⚠️ {chunk_id} 模型沒有產出有效問題。內容：{result_str}")
                
        except json.JSONDecodeError:
            print(f"❌ {chunk_id} JSON 解析失敗，模型幻覺了。已跳過。")
        except Exception as e:
            print(f"🚨 {chunk_id} 發生未知錯誤：{str(e)}")
            
    return generated_data

# === 執行主程式 ===
if __name__ == "__main__":
    # 1. 讀取你上一步產出的 CSV 檔案
    input_csv = "document_chunks_step2.csv"
    
    # 2. 隨機抽出 100 筆 (如果想先測試，可以把數字改成 5)
    df_sample = sample_chunks(input_csv, sample_size=300) # 建議先用 5 筆測試模型反應時間
    
    # 3. 呼叫 Ollama 生成
    # 注意：請確認你在終端機跑的 Gemma 版本名稱，如果是 2b 請改成 gemma:2b
    raw_results = generate_queries(df_sample, model_name="gemma3:12b") 
    
    # 4. 將帶有層級的原始 JSON 結果先存檔 (防呆備份)
    with open("raw_generated_queries.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 生成完畢！成功率：{len(raw_results)}/{len(df_sample)}。")
    print("原始生成的 JSON 已備份至 raw_generated_queries.json")