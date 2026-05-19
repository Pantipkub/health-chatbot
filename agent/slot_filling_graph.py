"""
Slot-filling / information-extraction LangGraph for a medical checkup chatbot.

This graph collects structured patient information before handing off to the
main medical analyst node. It is intentionally self-contained so it can be used
directly, or composed into a larger application graph later.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .state import AgentState

try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:  # pragma: no cover - depends on deployment extras
    ChatVertexAI = None  # type: ignore[assignment]

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - depends on deployment extras
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]


CRITICAL_LAB_ALIASES = {
    "fbs": "FBS",
    "fasting blood sugar": "FBS",
    "glucose": "Glucose",
    "hba1c": "HbA1c",
    "a1c": "HbA1c",
    "total cholesterol": "Total Cholesterol",
    "cholesterol": "Total Cholesterol",
    "hdl": "HDL",
    "ldl": "LDL",
    "triglyceride": "Triglycerides",
    "triglycerides": "Triglycerides",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "bun": "BUN",
    "ast": "AST",
    "alt": "ALT",
}


def _get_extraction_llm():
    """
    Lazily create the extraction model.

    Defaults to Vertex AI because this project already depends on
    langchain-google-vertexai. Set SLOT_FILLING_LLM_PROVIDER=google_genai to use
    ChatGoogleGenerativeAI instead.
    """

    provider = os.getenv("SLOT_FILLING_LLM_PROVIDER", "vertex").strip().lower()
    model_name = os.getenv("SLOT_FILLING_MODEL", "gemini-2.5-flash-lite")

    if provider in {"google_genai", "google-generative-ai", "genai"}:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "langchain-google-genai is not installed. Install it or use "
                "SLOT_FILLING_LLM_PROVIDER=vertex."
            )
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    if ChatVertexAI is None:
        raise RuntimeError(
            "langchain-google-vertexai is not installed. Install it or use "
            "SLOT_FILLING_LLM_PROVIDER=google_genai."
        )

    return ChatVertexAI(model=model_name, temperature=0)


def _latest_user_text(messages: List[BaseMessage]) -> str:
    """Return the latest human message content, or an empty string."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            content = message.content
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)
    return ""


def _latest_assistant_text_before_latest_user(messages: List[BaseMessage]) -> str:
    """Return the assistant message immediately before the latest human message."""

    latest_human_index: Optional[int] = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            latest_human_index = index
            break

    if latest_human_index is None:
        return ""

    for message in reversed(messages[:latest_human_index]):
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)

    return ""


def _message_content_to_text(content: Any) -> str:
    """Normalize LLM message content into text before JSON parsing."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()

    return str(content).strip()


def _parse_raw_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a raw JSON object. The fallback only extracts the first object-shaped
    span so malformed model wrappers do not crash the graph.
    """

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Extractor response must be a JSON object.")

    return parsed


def _normalize_gender(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"male", "man", "m", "ชาย", "ผู้ชาย"}:
        return "male"
    if normalized in {"female", "woman", "f", "หญิง", "ผู้หญิง"}:
        return "female"
    return None


def _normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "fasting", "fasted", "ใช่", "ใช่ครับ", "ใช่ค่ะ", "งด", "งดอาหาร"}:
        return "yes"
    if normalized in {"no", "n", "false", "not fasting", "non-fasting", "non fasting", "ไม่", "ไม่ใช่", "ไม่ใช่ครับ", "ไม่ใช่ค่ะ", "ไม่ได้งด", "ไม่ได้งดอาหาร"}:
        return "no"
    return None


def _normalize_age(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 < age < 130:
        return age
    return None


def _normalize_string_list(value: Any) -> Optional[List[str]]:
    """
    Convert extracted list-like values into a clean list.

    None means "not mentioned". An empty list means the user explicitly reported
    no known items for that slot.
    """

    if value is None:
        return None
    if value == []:
        return []
    if isinstance(value, str):
        if not value.strip():
            return None
        raw_items = re.split(r",|;|\n", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        return None

    cleaned: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            cleaned.append(text)
            seen.add(key)

    return cleaned


def _merge_lists(
    current: Optional[List[str]],
    extracted: Optional[List[str]],
) -> Optional[List[str]]:
    if extracted is None:
        return current
    if extracted == []:
        return []
    if not current:
        return extracted

    merged = list(current)
    seen = {item.casefold() for item in current}
    for item in extracted:
        key = item.casefold()
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _normalize_lab_values(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None

    normalized_values: Dict[str, float] = {}
    for raw_name, raw_value in value.items():
        if raw_value is None or raw_value == "":
            continue

        key = str(raw_name).strip()
        canonical_key = CRITICAL_LAB_ALIASES.get(key.casefold(), key)

        try:
            normalized_values[canonical_key] = float(raw_value)
        except (TypeError, ValueError):
            continue

    return normalized_values


def _merge_lab_values(
    current: Optional[Dict[str, float]],
    extracted: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if extracted is None:
        return current
    if not current:
        return extracted
    return {**current, **extracted}


def _current_slots_for_prompt(state: AgentState) -> Dict[str, Any]:
    return {
        "age": state.get("age"),
        "gender": state.get("gender"),
        "underlying_disease": state.get("underlying_disease"),
        "current_medications": state.get("current_medications"),
        "current_symptoms": state.get("current_symptoms"),
        "fasting_status": state.get("fasting_status"),
        "extracted_lab_values": state.get("extracted_lab_values"),
        "pending_slot": state.get("pending_slot"),
    }


def _updates_from_pending_slot(state: AgentState, latest_user_message: str) -> Dict[str, Any]:
    """Interpret short answers using the slot that the graph asked for last."""

    pending_slot = state.get("pending_slot")
    if not pending_slot:
        return {}

    updates: Dict[str, Any] = {}

    if pending_slot == "fasting_status":
        fasting_status = _normalize_yes_no(latest_user_message)
        if fasting_status is not None:
            updates["fasting_status"] = fasting_status
            updates["pending_slot"] = None

    elif pending_slot == "age":
        age = _normalize_age(latest_user_message)
        if age is not None:
            updates["age"] = age
            updates["pending_slot"] = None

    elif pending_slot == "gender":
        gender = _normalize_gender(latest_user_message)
        if gender is not None:
            updates["gender"] = gender
            updates["pending_slot"] = None

    return updates


def _merge_extracted_slots(state: AgentState, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Merge valid extracted values with existing AgentState slots."""

    updates: Dict[str, Any] = {}

    age = _normalize_age(extracted.get("age"))
    if age is not None:
        updates["age"] = age

    gender = _normalize_gender(extracted.get("gender"))
    if gender is not None:
        updates["gender"] = gender

    fasting_status = _normalize_yes_no(extracted.get("fasting_status"))
    if fasting_status is not None:
        updates["fasting_status"] = fasting_status

    for slot_name in (
        "underlying_disease",
        "current_medications",
        "current_symptoms",
    ):
        extracted_list = _normalize_string_list(extracted.get(slot_name))
        merged_list = _merge_lists(state.get(slot_name), extracted_list)
        if merged_list is not state.get(slot_name):
            updates[slot_name] = merged_list

    lab_values = _normalize_lab_values(extracted.get("extracted_lab_values"))
    merged_labs = _merge_lab_values(state.get("extracted_lab_values"), lab_values)
    if merged_labs is not state.get("extracted_lab_values"):
        updates["extracted_lab_values"] = merged_labs

    if updates:
        updates["pending_slot"] = None

    return updates


def extract_info_node(state: AgentState) -> Dict[str, Any]:
    """
    Extract patient slots from the latest user message and merge them into state.

    The LLM is intentionally instructed to produce only a raw JSON object. If
    parsing fails, the node returns no slot updates and allows the graph router
    to ask the next missing question.
    """

    messages = state.get("messages", [])
    latest_user_message = _latest_user_text(messages)
    if not latest_user_message:
        return {}

    pending_updates = _updates_from_pending_slot(state, latest_user_message)
    state_for_prompt: AgentState = {**state, **pending_updates}
    previous_assistant_message = _latest_assistant_text_before_latest_user(messages)

    current_slots = json.dumps(
        _current_slots_for_prompt(state_for_prompt),
        ensure_ascii=False,
        sort_keys=True,
    )

    SYSTEM_PROMPT = f"""
คุณคือระบบสกัดข้อมูลผู้ป่วยแบบ slot filling สำหรับแชทบอทตรวจสุขภาพ
หน้าที่ของคุณคือสกัดข้อมูลจากข้อความล่าสุดของผู้ใช้เท่านั้น และตอบกลับเป็น JSON ดิบตาม schema ที่กำหนด

ให้สกัดเฉพาะข้อเท็จจริงที่ผู้ใช้ระบุชัดเจนในข้อความล่าสุด ห้ามเดาหรือเติมข้อมูลเอง
ใช้ข้อมูลสถานะปัจจุบันด้านล่างเพื่อเข้าใจว่าผู้ใช้กำลังแก้ไขข้อมูลเดิมหรือเพิ่มข้อมูลใหม่เท่านั้น

สถานะข้อมูลที่ทราบอยู่แล้ว:
{current_slots}

คำถามล่าสุดจากผู้ช่วยก่อนข้อความนี้:
{previous_assistant_message or "ไม่มี"}

กฎการตอบ:
- ตอบกลับเป็น JSON object ดิบที่ valid เท่านั้น
- ห้ามใส่ markdown, code fence, คำอธิบาย, comment, หรือข้อความสนทนาอื่นๆ
- ใช้ double quotes กับ key และ string value ทุกตัว
- ถ้า slot ใดไม่ได้ถูกกล่าวถึงในข้อความล่าสุด ให้ใส่ null
- ถ้าผู้ใช้ระบุชัดเจนว่าไม่มีโรคประจำตัว/ไม่มียาที่กิน/ไม่มีอาการ ให้ใส่ empty list [] ใน slot นั้น
- ถ้าข้อความล่าสุดเป็นคำตอบสั้นๆ เช่น "ใช่", "ใช่ครับ", "ไม่ใช่", "ชาย", "หญิง" ให้ตีความตาม pending_slot หรือคำถามล่าสุดจากผู้ช่วย
- แปลง gender ให้เป็น "male" หรือ "female" เท่านั้น
- แปลง fasting_status ให้เป็น "yes" หรือ "no" เท่านั้น
- ค่าผลแล็บให้ดึงเฉพาะตัวเลข ไม่ต้องใส่หน่วย ตัวอย่าง:
  {{"FBS": 120.0, "HbA1c": 6.4, "LDL": 140.0}}

รูปแบบ JSON ที่ต้องตอบ:
{{
  "age": null,
  "gender": null,
  "underlying_disease": null,
  "current_medications": null,
  "current_symptoms": null,
  "fasting_status": null,
  "extracted_lab_values": null
}}
""".strip()

    try:
        llm = _get_extraction_llm()
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=latest_user_message),
            ]
        )
        raw_text = _message_content_to_text(response.content)
        extracted = _parse_raw_json_object(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[slot_filling] JSON parsing failed: {exc}")
        return {}
    except Exception as exc:
        print(f"[slot_filling] Extraction failed: {exc}")
        return {}

    extracted_updates = _merge_extracted_slots(state_for_prompt, extracted)
    return {**pending_updates, **extracted_updates}


def route_after_extraction(state: AgentState) -> str:
    """Route to the first missing critical slot, in priority order."""

    lab_values = state.get("extracted_lab_values")
    if not lab_values:
        return "ask_lab_node"

    if state.get("fasting_status") is None:
        return "ask_fasting_node"

    if state.get("age") is None:
        return "ask_age_node"

    if state.get("gender") is None:
        return "ask_gender_node"

    return "our_agent"


def ask_lab_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "extracted_lab_values",
        "messages": [
            AIMessage(
                content=(
                    "กรุณาส่งค่าผลตรวจสุขภาพที่ต้องการให้ช่วยดู เช่น FBS, "
                    "HbA1c, LDL, HDL, triglycerides, creatinine, eGFR, AST "
                    "หรือ ALT ครับ"
                )
            )
        ]
    }


def ask_fasting_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "fasting_status",
        "messages": [
            AIMessage(
                content=(
                    "ผลเลือดชุดนี้ตรวจหลังงดอาหารหรือไม่ครับ? กรุณาตอบว่าใช่หรือไม่ใช่"
                )
            )
        ]
    }


def ask_age_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "age",
        "messages": [
            AIMessage(
                content="ผู้ที่เป็นเจ้าของผลตรวจอายุเท่าไรครับ?"
            )
        ]
    }


def ask_gender_node(state: AgentState) -> Dict[str, Any]:
    return {
        "pending_slot": "gender",
        "messages": [
            AIMessage(
                content=(
                    "เพศของผู้ที่เป็นเจ้าของผลตรวจคือชายหรือหญิงครับ? "
                    "ข้อมูลนี้ช่วยให้เทียบช่วงอ้างอิงได้เหมาะสมขึ้น"
                )
            )
        ]
    }


def our_agent(state: AgentState) -> Dict[str, Any]:
    """
    Placeholder for the main medical analyst agent.

    Replace this node with the real RAG/analysis agent once slot collection is
    complete.
    """

    labs = state.get("extracted_lab_values") or {}
    lab_summary = ", ".join(f"{name}: {value:g}" for name, value in labs.items())

    content = (
        "ขอบคุณครับ ตอนนี้มีข้อมูลสำคัญพอสำหรับส่งต่อให้ agent วิเคราะห์หลักแล้ว: "
        f"อายุ {state.get('age')}, เพศ {state.get('gender')}, "
        f"สถานะการงดอาหาร {state.get('fasting_status')}, ค่าผลตรวจ [{lab_summary}] "
        "ขั้นถัดไปจะเป็นการทำงานของ medical analyst agent หลัก โดย placeholder นี้ยังไม่วินิจฉัยโรคหรือให้แผนการรักษาครับ"
    )

    return {"messages": [AIMessage(content=content)], "pending_slot": None}


def build_slot_filling_graph():
    """Build and compile the slot-filling LangGraph."""

    graph = StateGraph(AgentState)

    graph.add_node("extract_info_node", extract_info_node)
    graph.add_node("ask_lab_node", ask_lab_node)
    graph.add_node("ask_fasting_node", ask_fasting_node)
    graph.add_node("ask_age_node", ask_age_node)
    graph.add_node("ask_gender_node", ask_gender_node)
    graph.add_node("our_agent", our_agent)

    graph.set_entry_point("extract_info_node")

    graph.add_conditional_edges(
        "extract_info_node",
        route_after_extraction,
        {
            "ask_lab_node": "ask_lab_node",
            "ask_fasting_node": "ask_fasting_node",
            "ask_age_node": "ask_age_node",
            "ask_gender_node": "ask_gender_node",
            "our_agent": "our_agent",
        },
    )

    # These nodes intentionally end the turn. On the next user response, invoke
    # the graph again with the accumulated state and the new HumanMessage.
    graph.add_edge("ask_lab_node", END)
    graph.add_edge("ask_fasting_node", END)
    graph.add_edge("ask_age_node", END)
    graph.add_edge("ask_gender_node", END)
    graph.add_edge("our_agent", END)

    return graph.compile()


app = build_slot_filling_graph()


if __name__ == "__main__":
    example_state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "I am a 45 year old male. My FBS is 126 and LDL is 154. "
                    "I fasted before the test."
                )
            )
        ],
        "age": None,
        "gender": None,
        "underlying_disease": None,
        "current_medications": None,
        "current_symptoms": None,
        "fasting_status": None,
        "extracted_lab_values": None,
    }

    result = app.invoke(example_state)
    print(result)
