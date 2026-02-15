import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

# --- Config ---
PERSIST_DIRECTORY = "./chroma_db_health"
FILES_TO_READ = [
    "diabetes_knowledge.md", 
    "kidney_knowledge.md"
] 

def main():
    print("=== เริ่มต้นกระบวนการสร้าง Knowledge Base (รวม 2 โรค) ===")

    # 1. ล้าง Database เก่าทิ้งก่อน (สำคัญมาก! ไม่งั้นข้อมูลเก่าจะค้าง)
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"🧹 กำลังล้างข้อมูลเก่าใน {PERSIST_DIRECTORY}...")
        shutil.rmtree(PERSIST_DIRECTORY)
    
    # 2. อ่านไฟล์ Markdown ทุกไฟล์ที่ระบุไว้
    all_text_data = ""
    for file_path in FILES_TO_READ:
        if os.path.exists(file_path):
            print(f"📖 อ่านไฟล์: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                all_text_data += f.read() + "\n\n" # เอาเนื้อหามาต่อกัน
        else:
            print(f"⚠️ หาไฟล์ {file_path} ไม่เจอ (ข้าม)")

    if not all_text_data:
        print("❌ Error: ไม่ได้ข้อมูลอะไรมาเลย เช็คชื่อไฟล์ด่วน")
        return

    # 3. ตัดแบ่งข้อมูล (Chunking)
    # ตัดที่ # (ชื่อโรค) และ ## (หัวข้อ)
    headers_to_split_on = [
        ("#", "Disease"),
        ("##", "Topic"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_chunks = splitter.split_text(all_text_data)
    
    print(f"✅ ตัดแบ่งข้อมูลได้ทั้งหมด: {len(all_chunks)} ชิ้น (ควรจะมากกว่า 3 ชิ้น)")

    # 4. โหลด AI Model (Force CPU แก้บั๊ก Mac)
    print("🤖 กำลังโหลด AI Model...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. สร้าง Vector Database
    print("💾 กำลังบันทึกข้อมูลลง Database...")
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"🎉 เสร็จสิ้น! ข้อมูลใน Database ตอนนี้มี: {db._collection.count()} รายการ")
    
    # 6. Test เลยว่าเจอโรคไตไหม
    print("\n--- Test ค้นหาโรคไต ---")
    results = db.similarity_search("ค่า eGFR 40 อยู่ระยะไหน", k=2)
    for doc in results:
        print(f"Found: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()