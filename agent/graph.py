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


# ----- Function -----

def input_node(state: AgentState):
    """
    Entry node that receives user input and initializes debugging metadata.
    """
    
    state["current_node"] = "input_node"
    state["steps"].append("Received user input")

    last_msg = state["messages"][-1]
    print("User said:", last_msg.content)

    return state

def classify_intent_node(state: AgentState):
    """
    Analyze user message and classify medical intent using LLM.
    """

    state["current_node"] = "classify_intent"
    state["steps"].append("Classifying user intent with LLM")

    prompt = SystemMessage(content=
        "You are a medical triage assistant.\n"
        "Classify the user's intent into ONE of the following:\n"
        "- symptom\n"
        "- general_health\n"
        "- administrative\n\n"
        "Respond with only the label."
    )

    user_msg = state["messages"][-1]
    result = intent_model.invoke([prompt, user_msg])

    state["intent"] = result.content.strip().lower()
    return state

def route_by_intent(state):
    intent = state.get("intent")
    if intent in ("symptom", "general_health"):
        return intent
    return "general_health"  # fallback

# ----- Conditional -----
def should_continue(state: AgentState):
    """
    Decide whether to continue calling the tools.
    """

    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: # ไม่มี tool_calls จาก LLM แล้ว = ต้องการจะตอบ User แล้ว

        return "end"
    else:
        return "continue"   # LLM ขอเรียกใช้ tool อยู่
    

def call_model(state: AgentState):
    """
    Node นี้ทำหน้าที่ RAG: ค้นหาข้อมูล -> สร้าง Prompt -> ให้ Gemini ตอบ
    """
    messages = state["messages"]
    last_user_message = messages[-1].content # เอาคำถามล่าสุดมา
    
    # 1. ไปค้นข้อมูลใน DB (Retrieval)
    print(f"--- Debug: กำลังค้นข้อมูลเรื่อง '{last_user_message}' ---")
    context = retrieve_context(last_user_message)
    
    # 2. สร้าง Prompt ที่มี Context (Augmented Generation)
    if context:
        rag_prompt = (
            "คุณคือ AI ผู้ช่วยด้านสุขภาพอัจฉริยะ (Health Assistant) "
            "หน้าที่ของคุณคือให้ข้อมูลสุขภาพโดยอ้างอิงจาก 'ข้อมูลที่ค้นพบ' ด้านล่างนี้เท่านั้น\n\n"
            f"--- ข้อมูลที่ค้นพบ (Context) ---\n{context}\n"
            "------------------------------\n\n"
            "ข้อปฏิบัติ:\n"
            "1. ตอบคำถามโดยใช้ข้อมูลจาก Context เป็นหลัก\n"
            "2. ถ้าข้อมูลใน Context มีตัวเลขหรือเกณฑ์ (เช่น ค่าเบาหวาน, ค่าไต) ให้ระบุตัวเลขนั้นให้ชัดเจน\n"
            "3. ถ้าใน Context ไม่มีข้อมูลที่ผู้ใช้ถาม ให้ตอบว่า 'ขออภัย ข้อมูลในคลังความรู้ปัจจุบันยังมีไม่เพียงพอสำหรับเรื่องนี้'\n"
            "4. ห้ามวินิจฉัยโรคฟันธง (เช่น 'คุณเป็นโรคไตแน่ๆ') แต่ให้ใช้คำว่า 'มีความเสี่ยง' หรือ 'ตามเกณฑ์ระบุว่า...'\n"
            "5. ใช้ภาษาที่เป็นกันเอง สุภาพ และเข้าใจง่าย\n"
        )
    else:
        # กรณีค้นไม่เจออะไรเลย
        rag_prompt = (
            "คุณคือผู้ช่วยด้านสุขภาพ แต่ตอนนี้คุณไม่พบข้อมูลอ้างอิงในฐานข้อมูล "
            "ให้ตอบผู้ใช้ว่าคุณสามารถให้คำแนะนำได้เฉพาะเรื่อง เบาหวาน และ โรคไต ตาม Guideline ของไทยเท่านั้น "
            "และแนะนำให้ผู้ใช้ปรึกษาแพทย์หากมีอาการรุนแรง"
        )

    # 3. ส่งให้ Gemini (System Prompt + History)
    # เราจะแทรก System Prompt ใหม่เข้าไปในการเรียกครั้งนี้โดยเฉพาะ
    response = chat_model.invoke([SystemMessage(content=rag_prompt)] + messages)
    
    return {"messages": [response], "steps": ["retrieval", "generate"]}


# ----- Generate graph -----
def build_graph():
    graph = StateGraph(AgentState)

    # ----- add nodes -----
    graph.add_node("input", input_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("our_agent", call_model)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    # ----- entry point -----
    graph.set_entry_point("input")

    # ----- normal edges -----
    graph.add_edge("input", "classify_intent")

    # ----- routing by intent -----
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "symptom": "our_agent",
            "general_health": "our_agent"
        }
    )

    # ----- tool loop (optional ตอนนี้ tools = []) -----
    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )

    graph.add_edge("tools", "our_agent")

    app = graph.compile()

    return app
