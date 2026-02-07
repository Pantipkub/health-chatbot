from typing import Annotated, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # debug
    steps: list[str]                    # log ว่า agent ทำอะไรไปบ้าง
    current_node: Optional[str]         # ตอนนี้อยู่ node ไหน

    # medical-specific
    intent: Optional[str]               # symptom, general_health
