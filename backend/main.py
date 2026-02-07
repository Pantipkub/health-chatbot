import time
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from fastapi.responses import StreamingResponse
import json

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

def openai_chunk(text: str):
    return {
        "id": "chatcmpl-agent",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": text
                }
            }
        ]
    }
    


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    session_id = "demo-user"
    user_message = req.messages[-1].content
    memory = get_memory(session_id)

    def stream():
        input_state = {
            "messages": memory + [HumanMessage(content=user_message)],
            "steps": [],
            "current_node": "",
            "intent": None
        }

        # ---- stream agent steps ----
        for event in graph.stream(input_state):
            for node, state in event.items():

                if node != "__end__" and "steps" in state and state["steps"]:
                    text = f"🧠 {state['steps'][-1]}\n"
                    yield f"data: {json.dumps(openai_chunk(text))}\n\n"

                else:
                    assistant_msg = state["messages"][-1]
                    memory.append(HumanMessage(content=user_message))
                    memory.append(assistant_msg)

                    yield f"data: {json.dumps(openai_chunk(assistant_msg.content))}\n\n"
                    yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

