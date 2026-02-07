from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

chat_memory_store: dict[str, list[BaseMessage]] = {}

app = FastAPI()

graph = build_graph()

# ---------- OpenAI-compatible schema ----------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

def get_memory(session_id: str) -> list[BaseMessage]:
    return chat_memory_store.setdefault(session_id, [])

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    session_id = "demo-user"  # เดี๋ยวค่อยเปลี่ยนเป็นของจริง

    user_message = req.messages[-1].content
    
    memory = get_memory(session_id)

    # ---- inject memory + user msg เข้า graph ----
    result = graph.invoke({
        "messages": memory + [HumanMessage(content=user_message)],
        "steps": [],
        "current_node": "",
        "intent": None
    })

    assistant_msg = result["messages"][-1]

    # ---- update chat memory (อยู่นอกกราฟ) ----
    memory.append(HumanMessage(content=user_message))
    memory.append(assistant_msg)

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
        ]
    }
