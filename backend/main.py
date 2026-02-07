from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph

app = FastAPI()

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
    user_message = req.messages[-1].content

    result = graph.invoke({
        "messages": [
            ("user", user_message)
        ],
        "steps": [],
        "current_node": "",
        "intent": ""
    })

    final_answer = result["messages"][-1].content

    return {
        "id": "chatcmpl-health",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_answer
                },
                "finish_reason": "stop"
            }
        ]
    }
