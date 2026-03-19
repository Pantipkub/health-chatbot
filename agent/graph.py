from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
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

intent_model = ChatVertexAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

chat_model = ChatVertexAI(
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
    Node สำหรับตอบคำถาม: ดึง Context มาใส่ใน Prompt จริงๆ
    """
    messages = state["messages"]
    last_user_message = messages[-1].content 
    
    # 1. ดึงข้อมูลจาก Vector DB (Markdown)
    context = retrieve_context(last_user_message)
    
    if context:
        rag_prompt = (
            "คุณคือผู้ช่วยประเมินความเสี่ยงสุขภาพอัจฉริยะ (Health Risk Screener)\n"
            "เป้าหมาย: วิเคราะห์ผลตรวจเลือดเบื้องต้นและให้คำแนะนำในการปฏิบัติตัว\n\n"
            f"### ข้อมูลอ้างอิงจากคู่มือสุขภาพ:\n{context}\n\n"  
            "**กฎสำคัญในการตอบ:**\n"
            "1. 💭 คิดอะไรอยู่: วิเคราะห์ค่าแลปเทียบกับ 'ข้อมูลอ้างอิงจากคู่มือสุขภาพ' ที่ให้มาด้านบนเท่านั้น ห้ามใช้เกณฑ์อื่น\n"
            "2. 📋 สรุปผล: ใช้คำว่า 'ผลการประเมินความเสี่ยง' แทนการวินิจฉัยโรค\n"
            "3. 👨‍⚕️ คำแนะนำ: หากข้อมูลในคู่มือระบุเป้าหมายตามช่วงอายุ ให้ยึดตามนั้นเป็นหลัก\n"
        )
    else:
        rag_prompt = (
            "ตอนนี้ผมมีข้อมูลจำกัดแค่เรื่อง 'เบาหวาน โรคไต ความดัน และไขมัน' "
            "ดูเหมือนคำถามของคุณจะไม่อยู่ในฐานข้อมูลของผมครับ..."
        )

    # ส่งคำสั่งที่มี "ข้อมูลอ้างอิง (Context)" ไปให้ Gemini
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
