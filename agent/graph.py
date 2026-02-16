from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .rag_utils import retrieve_context

load_dotenv()

@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# tools = [add, subtract, multiply]
tools = []

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools(tools)

intent_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# ----- Core Identity -----
def core_identity():
    return (
        "คุณคือผู้ช่วยด้านสุขภาพเบื้องต้น\n"
        "คุณไม่ใช่แพทย์ และไม่ทำการวินิจฉัยโรค\n"
        "คุณให้ข้อมูลในระดับเบื้องต้นเท่านั้น\n"
        "คุณตอบอย่างกระชับ สุภาพ และเป็นกลาง\n"
        "ใช้คำลงท้ายว่า 'ครับ'\n"
    )


# ----- Input Node -----
def input_node(state: AgentState):
    state["current_node"] = "input_node"
    state["steps"].append("Received user input")
    last_msg = state["messages"][-1]
    print("User said:", last_msg.content)
    return state

# ----- Intent Classification -----
def classify_intent_node(state: AgentState):
    state["current_node"] = "classify_intent"
    state["steps"].append("Classifying user intent")

    prompt = SystemMessage(content=(
        "You are classifying the user's message.\n"
        "Return ONE label only:\n"
        "- lab_interpretation (user provides health values)\n"
        "- general_health (asking about disease/info)\n"
        "Return label only."
    ))

    user_msg = state["messages"][-1]
    result = intent_model.invoke([prompt, user_msg])
    state["intent"] = result.content.strip().lower()
    return state

def route_by_intent(state):
    intent = state.get("intent", "").lower()
    if intent == "lab_interpretation":
        return "our_agent"
    elif intent == "general_health":
        return "our_agent"
    else:
        # fallback default
        return "our_agent"

# ----- Lab Prompt -----
def lab_prompt():
    return (
        core_identity() +
        "\nบทบาทของคุณ:\n"
        "คุณคือผู้ช่วยแปลผลตรวจสุขภาพเบื้องต้นจากค่าแลปเท่านั้น\n"
        "คุณไม่ใช่แพทย์ และไม่ทำการวินิจฉัยโรค\n\n"
        "ขอบเขตที่ตอบได้:\n"
        "- เบาหวาน\n"
        "- ความดันโลหิตสูง\n"
        "- ไขมันในเลือดสูง\n"
        "- โรคไตเรื้อรัง (CKD)\n"
        "- ภาวะที่เกี่ยวข้องกับการทำงานของตับ\n\n"
        "หลักการตอบ:\n"
        "1. ตีความจากค่าที่ผู้ใช้ให้มาเท่านั้น\n"
        "2. ใช้คำว่า 'แนวโน้ม' หรือ 'ความเสี่ยงเบื้องต้น' เท่านั้น\n"
        "3. ห้ามวินิจฉัยโรคหรือให้แผนการรักษา\n"
        "4. สามารถให้เหตุผลเชิง guideline และช่วงค่าปกติสั้น ๆ เช่น สำหรับผู้สูงอายุ 80+ ความดัน <130/70 ถือว่าอยู่ในเกณฑ์มาตรฐาน\n"
        "   - ให้คำเตือนพฤติกรรม เช่น ระวังหน้ามืดหรือล้ม แต่ห้ามบอกยา/แผนรักษา\n"
        "5. เน้นคำแนะนำเชิงพฤติกรรมและการติดตามผล\n\n"
        "รูปแบบคำตอบ:\n"
        "- สั้น กระชับ เข้าใจง่าย\n"
        "- ไม่เกิน 10 ประโยค\n"
        "- เน้นความหมายของค่าที่ผิดปกติและแนวทางดูแลตัวเอง\n"
        "- ปิดท้ายด้วยคำแนะนำเชิงพฤติกรรมถ้ามี\n\n"
        "น้ำเสียง:\n"
        "- สุภาพ เป็นกลาง\n"
        "- ใช้คำลงท้ายว่า 'ครับ'\n\n"
        "ถ้าข้อมูลไม่พอ:\n"
        "ให้ถามข้อมูลเพิ่มเติมที่จำเป็นต่อการแปลผล เช่น อายุ น้ำหนัก โรคประจำตัว\n\n"
        "ห้าม:\n"
        "- ให้แผนการรักษาหรือยา\n"
        "- สรุปเป็นโรค\n"
    )

# ----- Call Model -----
def call_model(state: AgentState):
    messages = state["messages"]
    last_user_message = messages[-1].content
    context = retrieve_context(last_user_message)
    intent = state.get("intent")

    if context:
        prompt = lab_prompt() + "\nตอบพร้อมบอกแนวทางสังเกตอาการหรือพฤติกรรมที่ควรระวัง ไม่เกิน 7 ประโยคครับ (หากมี)"
    else:
        prompt = (
            core_identity() +
            "\nคุณได้รับค่าตรวจสุขภาพจากผู้ใช้ แต่ระบบไม่มีข้อมูลเพิ่มเติม\n"
            "ให้ตอบแนวโน้มทั่วไปโดยไม่วินิจฉัยโรค\n"
            "ไม่เกิน 3 ประโยค\n"
            "ปิดท้ายด้วยคำลงท้าย 'ครับ'\n"
        )

    response = chat_model.invoke([SystemMessage(content=prompt)] + messages)
    return {"messages": [response], "steps": ["retrieval", "generate"]}

# ----- Should Continue -----
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# ----- Build Graph -----
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input", input_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("our_agent", call_model)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "classify_intent")
    graph.add_edge("classify_intent", "our_agent")
    graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "our_agent")

    return graph.compile()
