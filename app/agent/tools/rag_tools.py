from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.rag.pipeline import rag_system

@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """Search uploaded documents (PDF, CSV, Markdown, text, resumes) in this chat's knowledge base."""
    thread_id = None
    if isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    return rag_system.query(query, thread_id=thread_id)
