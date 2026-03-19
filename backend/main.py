from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ลบ chat_memory_store ออกไปเลย! เราจะใช้ความจำจาก Open WebUI แทน

app = FastAPI()

# ----- เพิ่ม CORS Middleware เพื่อให้ Open WebUI ยิง API เข้ามาได้ -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

# ---------- OpenAI-compatible schema ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    
    # 1. แปลงประวัติการคุยที่ Open WebUI ส่งมา ให้เป็นรูปแบบที่ LangGraph เข้าใจ
    langchain_messages = []
    for msg in req.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))

    # 2. โยนประวัติทั้งหมดเข้า LangGraph
    result = graph.invoke({
        "messages": langchain_messages,
        "steps": [],
        "current_node": "",
        "intent": None
    })

    # 3. ดึงคำตอบสุดท้ายของ AI
    assistant_msg = result["messages"][-1]

    # ==========================================
    # 🌟 ส่วนที่เพิ่มเข้ามา: ดึงข้อมูล Token มาจัดเรียงใหม่
    # ==========================================
    usage_data = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    # ตรวจสอบว่ามี usage_metadata ส่งมาจาก Vertex AI หรือไม่
    if hasattr(assistant_msg, "usage_metadata") and assistant_msg.usage_metadata:
        meta = assistant_msg.usage_metadata
        # แปลงชื่อคีย์ของ LangChain ให้กลายเป็นภาษาที่ OpenWebUI เข้าใจ
        usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
        usage_data["completion_tokens"] = meta.get("output_tokens", 0)
        usage_data["total_tokens"] = meta.get("total_tokens", 0)
    # ==========================================

    # 4. ส่งกลับไปให้ Open WebUI โชว์บนหน้าจอ
    return {
        "id": "chatcmpl-health",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_msg.content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": usage_data  # <--- แปะบิลค่า Token ส่งไปด้วยตรงนี้!
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "health-model",  # ชื่อที่คุณต้องการให้โผล่ใน Open WebUI
                "object": "model",
                "created": 1667925528,
                "owned_by": "health-chatbot"
            }
        ]
    }