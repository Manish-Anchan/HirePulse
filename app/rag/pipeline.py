from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

from app.config import DOCS_DIR, PERSIST_DIR
from app.rag.embeddings import embeddings
from app.rag.loaders import load_file

# ChromaDB vector store
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
)

# Text splitter & cross-encoder reranker
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
reranker = FlashrankRerank(top_n=4)


class HybridRAG:
    """Hybrid RAG (BM25 keyword search + Chroma dense search + FlashRank reranker)

    Each conversation thread maintains its own isolated pool of documents so
    files uploaded in one chat never leak into another.
    """

    def __init__(self):
        self.vectorstore = vectorstore
        self.thread_chunks: Dict[str, List[Document]] = {}
        self.thread_files: Dict[str, List[str]] = {}
        self._load_existing_threads_from_db()

    def _load_existing_threads_from_db(self):
        """Restore in-memory thread files and chunks from ChromaDB on startup."""
        stored = self.vectorstore.get()
        if not stored or not stored.get("documents"):
            return

        documents = stored["documents"]
        metadatas = stored.get("metadatas") or []

        for text, meta in zip(documents, metadatas):
            meta = meta or {}
            thread_id = meta.get("thread_id")
            if not thread_id:
                continue

            doc = Document(page_content=text, metadata=meta)
            self.thread_chunks.setdefault(thread_id, []).append(doc)

            filename = meta.get("filename")
            files_list = self.thread_files.setdefault(thread_id, [])
            if filename and filename not in files_list:
                files_list.append(filename)

    def ingest_documents(self, file_paths: List[str], thread_id: Optional[str] = None) -> Dict[str, any]:
        """Read files, split them into chunks, and store them in ChromaDB and thread memory."""
        raw_documents = []
        for file_path in file_paths:
            raw_documents.extend(load_file(file_path, thread_id=thread_id))

        if not raw_documents:
            return {"chunks": 0, "files": []}

        chunks = text_splitter.split_documents(raw_documents)
        self.vectorstore.add_documents(chunks)

        saved_filenames = [Path(p).name for p in file_paths]
        if thread_id:
            self.thread_chunks.setdefault(thread_id, []).extend(chunks)
            thread_file_list = self.thread_files.setdefault(thread_id, [])
            for name in saved_filenames:
                if name not in thread_file_list:
                    thread_file_list.append(name)

        return {"chunks": len(chunks), "files": saved_filenames}

    def get_thread_files(self, thread_id: str) -> List[str]:
        """Return the names of files uploaded to a specific chat thread."""
        return list(self.thread_files.get(thread_id, []))

    def _create_hybrid_retriever(self, chunks: List[Document], thread_id: str):
        """Combine BM25 keyword matching with Chroma vector search, then rerank."""
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = 10

        dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10, "filter": {"thread_id": str(thread_id)}}
        )

        hybrid_ensemble = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=[0.5, 0.5],
        )

        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_ensemble,
        )

    def query(self, query_text: str, thread_id: Optional[str] = None) -> str:
        """Search documents strictly scoped to this thread's uploaded files."""
        chunks = self.thread_chunks.get(thread_id, []) if thread_id else []
        if not chunks:
            return "No documents available in knowledge base for this chat yet. Please upload files first."

        retriever = self._create_hybrid_retriever(chunks, thread_id)
        try:
            matched_docs = retriever.invoke(query_text)
        except Exception:
            matched_docs = self.vectorstore.similarity_search(
                query_text, k=4, filter={"thread_id": str(thread_id)}
            )

        if not matched_docs:
            return "No relevant information found in the attached documents for this chat."

        formatted_passages = [
            f"--- Document Source {i}: {doc.metadata.get('filename', 'Unknown')} ---\n{doc.page_content}"
            for i, doc in enumerate(matched_docs, start=1)
        ]
        return "\n\n".join(formatted_passages)


# Global singleton instance used across the app
rag_system = HybridRAG()
