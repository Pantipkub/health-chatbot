# rag_utils.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Config ต้องตรงกับตอน create_db.py เป๊ะๆ!
PERSIST_DIRECTORY = "./chroma_db_health"

# โหลด Embedding Model (บังคับ CPU เหมือนเดิมเพื่อแก้บั๊ก Mac)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# โหลด Database เตรียมไว้เลย (จะได้ไม่ต้องโหลดใหม่ทุกครั้งที่เรียก)
vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function
)

def retrieve_context(query: str, k: int = 3) -> str:
    """
    รับคำถาม -> ค้นหา Vector DB -> คืนค่าเป็น Text (Context)
    """
    try:
        results = vector_db.similarity_search(query, k=k)
        
        if not results:
            return ""

        # เอาเนื้อหามาต่อๆ กัน พร้อมบอกที่มา
        context_str = ""
        for i, doc in enumerate(results):
            source = doc.metadata.get("Disease", "Unknown Source")
            topic = doc.metadata.get("Topic", "")
            content = doc.page_content.replace("\n", " ") # ลบ newline ให้ต่อกันสวยๆ
            
            context_str += f"[ข้อมูลที่ {i+1} จาก: {source} - {topic}]:\n{content}\n\n"
            
        return context_str

    except Exception as e:
        print(f"Error retrieval: {e}")
        return ""