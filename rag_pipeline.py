"""HirePulse RAG Pipeline Module.

Exposes document loading, ChromaDB hybrid search, and knowledge base retrieval tools.
"""

from app.config import DOCS_DIR, PERSIST_DIR
from app.rag.embeddings import embeddings
from app.rag.loaders import load_file
from app.rag.pipeline import HybridRAG, rag_system
from app.agent.tools.rag_tools import search_knowledge_base

__all__ = [
    "DOCS_DIR",
    "PERSIST_DIR",
    "embeddings",
    "load_file",
    "HybridRAG",
    "rag_system",
    "search_knowledge_base",
]
