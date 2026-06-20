import json
import pandas as pd

# 1. 讀取跨段落生成的原始 JSON 檔案
input_json = "raw_generated_queries_multi.json"
print(f"正在讀取跨段落 JSON 資料庫：{input_json} ...")

with open(input_json, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

flat_data = []

# 2. 資料攤平與序列化邏輯
for item in raw_data:
    # 抓取視窗層級的資訊
    window_id = item.get("Window_ID", "")
    included_chunks = item.get("Included_Chunk_IDs", [])
    
    # 遍歷該視窗產出的所有問題
    for q_obj in item.get("Generated_Queries", []):
        query_text = q_obj.get("query", "").strip()
        gt_chunks = q_obj.get("ground_truth_chunks", [])
        
        # 防呆機制：確保問題不為空，且確實有對應的答案陣列
        if query_text and isinstance(gt_chunks, list) and len(gt_chunks) > 0:
            flat_data.append({
                "Query": query_text,
                # 💡 核心邏輯：將 List 轉成 JSON 字串，安全存入 CSV 單一儲存格
                "Ground_Truth_Chunks": json.dumps(gt_chunks, ensure_ascii=False),
                "Window_ID": window_id,
                # 將這題所在的視窗範圍也存下來備查
                "Window_Included_Chunks": json.dumps(included_chunks, ensure_ascii=False)
            })

# 3. 建構 DataFrame 並匯出
df_final = pd.DataFrame(flat_data)
output_csv = "multi_hop_evaluation_dataset.csv"
df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n" + "="*50)
print(f"✅ 跨段落資料攤平完成！")
print(f"成功將 {len(raw_data)} 個大視窗，拆解攤平成 {len(df_final)} 條獨立的多目標測試 Q&A。")
print(f"檔案已儲存為：{output_csv}")
print("="*50)

# 4. 印出前兩筆資料確認格式
print("\n【資料預覽】：")
print(df_final[['Query', 'Ground_Truth_Chunks']].head(2))