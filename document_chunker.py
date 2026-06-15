import fitz  # PyMuPDF
import pandas as pd
import re


#機密CODE
def clean_text(text):
    # 1. 先把「換行」和「多餘空白」全部壓平！
    # 這樣如果原本是 "機密，\n到XXX年"，就會變成 "機密， 到XXX年"
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 開始進行暴力清洗
    # 情況 A：如果你的 PDF 裡面真的印著英文字母 "XXX"
    text = text.replace("機密,,", "")#機密CODE
    #text = text.replace("，）", "")
    
    # 情況 B
    # \d+ 代表一個或多個數字，\s* 代表可能存在的空白
    #text = re.sub(r'機密，\s*到\s*\d+\s*年', '', text)
    
    # 情況 C
    #text = text.replace("機密，到ＸＸＸ年", "")
    
    return text.strip()

def process_pdf_to_chunks(pdf_path, chunk_size=200, overlap=50):
    """
    讀取 PDF 並進行滑動視窗切塊
    - chunk_size: 每個 chunk 大約的字元數
    - overlap: 與前一個 chunk 重疊的字元數
    """
    print(f"開始解析文件：{pdf_path}")
    doc = fitz.open(pdf_path)
    chunks_data = []
    chunk_counter = 1
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        if not text:
            continue
            
        cleaned_text = clean_text(text)
        
        # 滑動視窗切塊邏輯
        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            chunk_text = cleaned_text[start:end]
            
            # 如果這段文字太短（例如頁面最後幾個字），可以考慮跳過或合併
            if len(chunk_text) > 50: 
                chunks_data.append({
                    "Chunk_ID": f"chunk_{chunk_counter:05d}",
                    "Page_Number": page_num + 1, # 讓頁碼從 1 開始
                    "Text": chunk_text
                })
                chunk_counter += 1
            
            # 移動視窗，減去 overlap 以產生重疊
            start += (chunk_size - overlap)
            
    print(f"解析完成！總共生成了 {len(chunks_data)} 個 Chunks。")
    return chunks_data

# === 執行主程式 ===
if __name__ == "__main__":
    # 替換成你的機密 PDF 檔案路徑
    pdf_file_path = "..).pdf" #機密CODE
    
    # 執行切塊 (設定每段 400 字，重疊 50 字)
    all_chunks = process_pdf_to_chunks(pdf_file_path, chunk_size=400, overlap=50)
    
    # 轉成 DataFrame 並輸出檢查
    df_chunks = pd.DataFrame(all_chunks)
    
    # 將切塊結果先存下來，方便後續檢查與呼叫 LLM
    output_csv = "document_chunks_step2.csv"
    df_chunks.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 切塊資料已儲存至：{output_csv}")
    
    # 印出前兩筆資料看看長相
    print("\n資料預覽：")
    print(df_chunks.head(2))
