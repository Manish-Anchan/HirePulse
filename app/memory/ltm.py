import uuid
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import PERSIST_DIR
from app.rag.embeddings import embeddings


class LongTermMemory:
    """Manages cross-thread persistent user facts, preferences, and context using ChromaDB."""

    def __init__(self, collection_name: str = "user_long_term_memory"):
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    def add_memory(
        self,
        text: str,
        user_id: str = "default_user",
        category: str = "preference",
        similarity_threshold: float = 0.85,
    ) -> Dict[str, str]:
        """Store or update a fact in long-term memory with semantic deduplication."""
        text = text.strip()
        if not text:
            return {"id": "", "action": "empty"}

        # Check for existing memory on the same topic for this user
        try:
            existing = self.vectorstore.similarity_search_with_relevance_scores(
                text, k=1, filter={"user_id": str(user_id)}
            )
            if existing:
                doc, score = existing[0]
                existing_id = doc.metadata.get("id")

                # 1. Exact duplicate check (normalized text)
                if doc.page_content.strip().lower() == text.lower():
                    return {"id": existing_id, "action": "exact_duplicate"}

                # 2. Semantic near-duplicate / update check
                if score >= similarity_threshold:
                    doc.page_content = text
                    doc.metadata["category"] = category
                    self.vectorstore.add_documents([doc], ids=[existing_id])
                    return {"id": existing_id, "action": "updated"}
        except Exception:
            pass

        # 3. New distinct memory
        memory_id = str(uuid.uuid4())
        doc = Document(
            page_content=text,
            metadata={
                "id": memory_id,
                "user_id": str(user_id),
                "category": category,
            },
        )
        self.vectorstore.add_documents([doc], ids=[memory_id])
        return {"id": memory_id, "action": "created"}

    def search_memories(self, query: str, user_id: str = "default_user", k: int = 4) -> List[str]:
        """Retrieve the most relevant long-term memories for a given query."""
        if not query or not query.strip():
            return []

        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"user_id": str(user_id)},
            )
            return [doc.page_content for doc in results if doc.page_content]
        except Exception:
            return []

    def get_all_memories(self, user_id: str = "default_user") -> List[Dict[str, str]]:
        """List all memories for a user."""
        try:
            data = self.vectorstore.get(where={"user_id": str(user_id)})
            memories = []
            if data and data.get("documents"):
                documents = data.get("documents") or []
                metadatas = data.get("metadatas") or []
                for text, meta in zip(documents, metadatas):
                    meta = meta or {}
                    memories.append({
                        "id": meta.get("id", ""),
                        "memory": text,
                        "category": meta.get("category", "preference"),
                    })
            return memories
        except Exception:
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Remove a memory by its unique ID."""
        try:
            self.vectorstore.delete(ids=[memory_id])
            return True
        except Exception:
            return False


# Global singleton instance used across the app
ltm_store = LongTermMemory()
