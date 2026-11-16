import os
import sqlite3
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langfuse.langchain import CallbackHandler

from rag_pipeline import rag_system, search_knowledge_base

langfuse_handler = CallbackHandler()

load_dotenv()

# Bind RAG tools to model
tools = [search_knowledge_base]

model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    request_timeout=60.0,
    api_key=os.environ.get("OPENR_API"),
    temperature=0.1,
    streaming=True,
)
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a knowledge base containing "
    "documents (PDFs, CSVs, Markdown, and text files). When the user asks factual questions "
    "about specific documents, data, or technical details, always use the search_knowledge_base "
    "tool to retrieve accurate context before answering. Synthesize clear, well-referenced answers."
)

class convo(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def Chatnode(state: convo, config: RunnableConfig = None):
    msgs = list(state["messages"])
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    attached_files = rag_system.get_thread_files(thread_id) if thread_id else []

    if attached_files:
        files_str = ", ".join(f"'{f}'" for f in attached_files)
        prompt = (
            f"You are a helpful AI assistant. The user has uploaded the following document(s) "
            f"to this conversation: [{files_str}].\n"
            f"- You ALREADY have full awareness of these files. NEVER ask the user to provide, name, "
            f"or re-upload these files.\n"
            f"- When the user asks any question related to them or general queries about the data, "
            f"immediately call the `search_knowledge_base` tool to retrieve accurate passages before answering.\n"
            f"- Synthesize clear, well-referenced answers using the retrieved information."
        )
    else:
        prompt = (
            "You are a helpful AI assistant. Answer user queries accurately and concisely. "
            "If the user asks about documents that are not uploaded, politely explain that no documents "
            "are attached to this chat yet and they can attach files using the attachment button."
        )

    # Filter out previous system messages so prompt dynamically updates when files are added
    non_system_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
    active_msgs = [SystemMessage(content=prompt)] + non_system_msgs

    callbacks = [langfuse_handler] if "langfuse_handler" in globals() and langfuse_handler else []
    response = model_with_tools.invoke(active_msgs, config={"callbacks": callbacks})
    return {"messages": [response]}

os.makedirs("Database", exist_ok=True)
conn = sqlite3.connect(database="Database/Chat_database.db", check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# Build Agentic RAG graph with ToolNode loop
tool_node = ToolNode(tools)

graph = StateGraph(convo)
graph.add_node("Chatnode", Chatnode)
graph.add_node("tools", tool_node)

graph.add_edge(START, "Chatnode")
graph.add_conditional_edges("Chatnode", tools_condition)
graph.add_edge("tools", "Chatnode")

chatbot = graph.compile(checkpointer=checkpointer)

def get_threads():
    all_threads = []
    seen = set()
    for checkpoint in checkpointer.list(config=None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if tid not in seen:
            seen.add(tid)
            all_threads.append(tid)
    return all_threads

def get_thread_messages(thread_id: str) -> list[dict]:
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    raw_messages = state.values.get("messages", []) if state and state.values else []
    formatted = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            formatted.append({"role": "assistant", "content": msg.content})
    return formatted

# To check in terminal api response
if __name__ == "__main__":
    usr_inp = input("what is your question: ")
    res = chatbot.invoke(
        {"messages": [HumanMessage(content=usr_inp)]},
        config={"configurable": {"thread_id": "cli-test"}},
    )
    last_msg = res["messages"][-1]
    print("Response:", last_msg.content)
