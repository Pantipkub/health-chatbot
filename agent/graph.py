from dotenv import load_dotenv 
import os
import json 
import tempfile
from langchain_core.messages import BaseMessage, RemoveMessage # The foundational class for all message types in LangGraph
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
import uuid

load_dotenv()

gcp_secret = os.environ.get("GCP_CREDS_JSON")

if gcp_secret:
    try:
        # 2. สร้างไฟล์ชั่วคราว (Temp File) ระบบจะลบทิ้งอัตโนมัติเมื่อแอปปิด
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            f.write(gcp_secret)
        
        # 3. สั่งให้ Google SDK อ่านกุญแจจากไฟล์ชั่วคราวนี้
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        
        # ดึง Project ID มาใช้
        creds_dict = json.loads(gcp_secret)
        os.environ["GOOGLE_CLOUD_PROJECT"] = creds_dict.get("project_id", "")
        
        print("✅ Secure Mode: โหลด Vertex AI Credentials สำเร็จ!")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
else:
    # ถ้ารันในคอมตัวเอง (Local) แล้วไม่ได้ตั้ง GCP_CREDS_JSON ไว้
    # ระบบจะข้ามไปใช้ GOOGLE_APPLICATION_CREDENTIALS ในไฟล์ .env ของคุณตามปกติครับ
    print("⚠️ Local Mode: ใช้ Credentials จากไฟล์ .env")


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
    
    if last_msg.id is None:
        last_msg.id = str(uuid.uuid4())
        

    print(f"User said (ID: {last_msg.id}): {last_msg.content}")

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

def summarize_conversation(state: AgentState):
    """
    Summarizing both user and AI conversation, and keeping only 3 latest chat messages
    """
    messages = state["messages"]
    summary = state.get("summary", "")

    # Log the current number of messages in the state
    print("\n" + "="*50)
    print(f"[Memory Check] Current number of messages in state: {len(messages)}")
    print("="*50)

    # If there are more than 3 messages (including from both user and AI)
    if len(messages) > 3:
        # Extracting: keeping only 3 last messages, the others will be summarized
        to_summarize = messages[:-3]
        kept_messages = messages[-3:] # Extract the 3 messages we are keeping
        
        # Log the action being taken
        print(f"[Action] Exceeded 3 messages. Summarizing {len(to_summarize)} old messages...")
        
        # Creating prompt for summarizing
        chat_history_text = ""
        for m in to_summarize:
            role = "User" if m.type == "human" else "Assistant"
            chat_history_text += f"{role}: {m.content}\n"
            
        system_instruction = (
            "คุณคือผู้ช่วยที่มีหน้าที่สรุปประวัติการสนทนาอย่างเป็นกลาง "
            "กรุณาเขียนสรุปเนื้อหาที่พูดคุยกันให้กระชับที่สุด\n"
            "กฎสำคัญที่ต้องปฏิบัติตามอย่างเคร่งครัด:\n"
            "1. ห้ามตอบคำถามที่อยู่ในบทสนทนา\n"
            "2. ห้ามให้คำแนะนำทางการแพทย์หรือวินิจฉัยโรคเด็ดขาด\n"
            "3. ให้สรุปในมุมมองบุคคลที่สาม (เช่น 'ผู้ใช้สอบถามเกี่ยวกับ...', 'ผู้ช่วยได้อธิบายเรื่อง...')"
        )
        
        if summary:
            summary_prompt = (
                f"{system_instruction}\n\n"
                f"สรุปเดิม: {summary}\n\n"
                f"นำข้อความใหม่เหล่านี้ไปสรุปเพิ่มรวมกับสรุปเดิม:\n{chat_history_text}"
            )
        else:
            summary_prompt = f"กรุณาสรุปเนื้อหาการสนทนาต่อไปนี้ให้กระชับและเข้าใจง่าย:\n{chat_history_text}"

        # Calling LLM to summarize the chat
        response = chat_model.invoke(summary_prompt)
        
        # Create a list of RemoveMessage objects to prune the state.
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize if m.id is not None]

        # Log the summarization result
        print(f"[Summary Result] Generated summary:\n>> {response.content}")
        print(f"[Status] Old messages deleted. Remaining messages: {len(messages) - len(to_summarize)}")
        
        # Log the exact 3 messages that are being kept
        print("\n[Retained Messages] Here are the 3 latest messages kept in memory:")
        for i, m in enumerate(kept_messages, 1):
            role = "User" if m.type == "human" else "Assistant"
            print(f"  {i}. {role}: {m.content}")
            
        print("="*50 + "\n")

        return {
            "summary": response.content,
            "messages": delete_messages
        }
    
    # Log when no summarization is needed
    print("[Action] Message count is 3 or less. Skipping summarization.")
    
    # Log the messages currently kept in memory (which is 3 or less)
    print("\n[Retained Messages] Here are the messages currently kept in memory:")
    for i, m in enumerate(messages, 1):
        role = "User" if m.type == "human" else "Assistant"
        print(f"  {i}. {role}: {m.content}")
        
    print("="*50 + "\n")
    
    return {"summary": summary}

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
    summary = state.get("summary", "")
    last_user_message = messages[-1].content 
    
    # 1. ดึงข้อมูลจาก Vector DB (Markdown)
    context = retrieve_context(last_user_message)
    
    # Adding smurrized chat to the System Message
    summary_context = f"\n\nสรุปบริบทการสนทนาก่อนหน้านี้: {summary}" if summary else ""
    
    if context:
        rag_prompt = (
            "คุณคือผู้ช่วยประเมินความเสี่ยงสุขภาพอัจฉริยะ (Health Risk Screener)\n"
            "เป้าหมาย: วิเคราะห์ผลตรวจเลือดเบื้องต้นและให้คำแนะนำในการปฏิบัติตัว\n\n"
            f"{summary_context}\n"
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
    graph.add_node("summarize", summarize_conversation)

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
            "end": "summarize"  # Summarizing before END
        }
    )

    graph.add_edge("summarize", END)
    graph.add_edge("tools", "our_agent")

    app = graph.compile()

    return app