import hashlib
import json
import uuid
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.graph import build_graph, chat_model
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

SLOT_FIELD_NAMES = (
    "age",
    "gender",
    "underlying_disease",
    "current_medications",
    "current_symptoms",
    "fasting_status",
    "extracted_lab_values",
    "pending_slot",
)

slot_memory_store: dict[str, dict[str, Any]] = {}

# ---------- OpenAI-compatible schema ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    user: Optional[str] = None
    conversation_id: Optional[str] = None
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"


def _conversation_key(req: ChatRequest) -> str:
    """
    Prefer explicit IDs from the caller. If none are available, fall back to a
    stable fingerprint of the first user message plus the optional user field.
    """

    direct_key = req.conversation_id or req.chat_id or req.session_id
    if direct_key:
        return direct_key

    metadata = req.metadata or {}
    metadata_key = (
        metadata.get("conversation_id")
        or metadata.get("chat_id")
        or metadata.get("session_id")
    )
    if metadata_key:
        return str(metadata_key)

    first_user_message = next(
        (message.content for message in req.messages if message.role == "user"),
        "",
    )
    fingerprint_source = f"{req.user or 'anonymous'}::{first_user_message}"
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


def _request_extra_fields(req: ChatRequest) -> dict[str, Any]:
    known_fields = set(getattr(req, "__fields__", {}).keys())
    extra_fields = getattr(req, "model_extra", None)
    if isinstance(extra_fields, dict):
        return extra_fields
    return {
        key: value
        for key, value in req.__dict__.items()
        if key not in known_fields
    }


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_openwebui_request(req: ChatRequest, *, is_webui_task: bool) -> None:
    # TEMP_OPENWEBUI_LOG: remove this function and its call after confirming
    # which OpenWebUI field should be used as the per-chat memory key.
    metadata = req.metadata or {}
    extra_fields = _request_extra_fields(req)
    role_counts: dict[str, int] = {}
    for message in req.messages:
        role_counts[message.role] = role_counts.get(message.role, 0) + 1

    first_user_message = next(
        (message.content for message in req.messages if message.role == "user"),
        "",
    )
    last_message = req.messages[-1] if req.messages else None

    print("\n[TEMP_OPENWEBUI_LOG] Incoming /v1/chat/completions request")
    print(f"[TEMP_OPENWEBUI_LOG] is_webui_task={is_webui_task}")
    print(f"[TEMP_OPENWEBUI_LOG] model={req.model!r} user={req.user!r}")
    print(
        "[TEMP_OPENWEBUI_LOG] explicit_ids="
        f"conversation_id={req.conversation_id!r}, "
        f"chat_id={req.chat_id!r}, session_id={req.session_id!r}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] metadata="
        f"{json.dumps(metadata, ensure_ascii=False, default=str)}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] extra_fields="
        f"{json.dumps(extra_fields, ensure_ascii=False, default=str)}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] messages="
        f"count={len(req.messages)}, roles={role_counts}"
    )
    print(
        "[TEMP_OPENWEBUI_LOG] first_user_sha256="
        f"{hashlib.sha256(first_user_message.encode('utf-8')).hexdigest() if first_user_message else None}"
    )
    if last_message:
        print(
            "[TEMP_OPENWEBUI_LOG] last_message="
            f"role={last_message.role!r}, preview={_preview_text(last_message.content)!r}"
        )
    print(f"[TEMP_OPENWEBUI_LOG] selected_memory_key={_conversation_key(req)!r}")


def _initial_slot_state(conversation_key: str) -> dict[str, Any]:
    saved_slots = slot_memory_store.get(conversation_key, {})
    return {field_name: saved_slots.get(field_name) for field_name in SLOT_FIELD_NAMES}


def _save_slot_state(conversation_key: str, graph_result: dict[str, Any]) -> None:
    current_slots = slot_memory_store.setdefault(conversation_key, {})
    for field_name in SLOT_FIELD_NAMES:
        if field_name in graph_result:
            current_slots[field_name] = graph_result.get(field_name)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    
    # 1. Convert Open WebUI messages to LangGraph message format
    langchain_messages = []
    for msg in req.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))

    # Intercept background tasks generated by Open WebUI to prevent them from entering the graph
    last_message_content = req.messages[-1].content if req.messages else ""
    
    is_webui_task = any(keyword in last_message_content for keyword in [
        "Generate a concise, 3-5 word title",
        "Generate 1-3 broad tags",
        "Suggest 3-5 relevant follow-up questions"
    ])

    _log_openwebui_request(req, is_webui_task=is_webui_task)

    usage_data = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    if is_webui_task:
        print("\n[Interceptor] Open WebUI automated task detected. Bypassing LangGraph.")
        # Call the model directly without saving to memory
        response = chat_model.invoke(langchain_messages)
        assistant_content = response.content
        
        # Extract token usage for the automated task
        meta = getattr(response, "usage_metadata", {}) or {}
        usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
        usage_data["completion_tokens"] = meta.get("output_tokens", 0)
        usage_data["total_tokens"] = meta.get("total_tokens", 0)
        
    else:
        print("\n[Interceptor] Normal user message detected. Routing to LangGraph.")
        conversation_key = _conversation_key(req)
        # 2. Pass the entire history into LangGraph
        result = graph.invoke({
            "messages": langchain_messages,
            "steps": [],
            "current_node": "",
            "intent": None,
            **_initial_slot_state(conversation_key),
        })
        _save_slot_state(conversation_key, result)

        # 3. Retrieve the final response from AI
        assistant_msg = result["messages"][-1]
        assistant_content = assistant_msg.content

        # Extract token usage for the normal chat
        meta = getattr(assistant_msg, "usage_metadata", {}) or {}
        usage_data["prompt_tokens"] = meta.get("input_tokens", 0)
        usage_data["completion_tokens"] = meta.get("output_tokens", 0)
        usage_data["total_tokens"] = meta.get("total_tokens", 0)

    # 4. Return the response to Open WebUI
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": usage_data
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
