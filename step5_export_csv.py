import json
import pandas as pd

# 1. 讀取第四步產出的原始 JSON 檔案
with open("raw_generated_queries.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

flat_data = []

# 2. 資料攤平邏輯
for item in raw_data:
    chunk_id = item["Chunk_ID"]
    page_num = item["Page_Number"]
    chunk_text = item["Chunk_Text"]
    
    # 把裡面的 1~3 個問題拆開，每一題變成獨立的一行
    for q_obj in item["Generated_Queries"]:
        # 確保有抓到 'query' 這個 key
        query_text = q_obj.get("query", "").strip()
        
        if query_text:  # 避免空字串
            flat_data.append({
                "Query": query_text,
                "Ground_Truth_Chunk_ID": chunk_id,
                "Page_Number": page_num,
                "Chunk_Text": chunk_text
            })

# 3. 建構 DataFrame 並匯出
df_final = pd.DataFrame(flat_data)
output_csv = "bm25_evaluation_dataset.csv"
df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 第五步完成！成功將 {len(raw_data)} 個 Chunks 攤平成 {len(df_final)} 條獨立的測試 Q&A。")
print(f"檔案已儲存為：{output_csv}")