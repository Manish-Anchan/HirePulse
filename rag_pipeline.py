import os
import glob
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "Database", "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "Database", "uploads")
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# 1. Google Gemini Embeddings
gemini_key = os.environ.get("GEMINI_API") or os.environ.get("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=gemini_key,
)

# 2. Chroma Vectorstore
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

# FlashRank compressor
flashrank_compressor = FlashrankRerank(top_n=4)


def load_file(file_path: str, thread_id: Optional[str] = None) -> List[Document]:
    """Parse PDF, CSV, MD, or TXT documents, attaching metadata."""
    p = Path(file_path)
    ext = p.suffix.lower()
    filename = p.name
    docs: List[Document] = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == ".csv":
            # Autodetect or default utf-8 with fallback
            try:
                loader = CSVLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                loader = CSVLoader(file_path, encoding="latin-1")
                docs = loader.load()
        elif ext in [".md", ".markdown", ".txt", ".json", ".py", ".log"]:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="latin-1")
                docs = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Attach standard metadata
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["filename"] = filename
            if thread_id:
                d.metadata["thread_id"] = str(thread_id)
        return docs
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise e


class HybridRAG:
    def __init__(self):
        self.vectorstore = vectorstore
        self.global_chunks: List[Document] = []
        self.thread_chunks: Dict[str, List[Document]] = {}
        self.thread_files: Dict[str, List[str]] = {}
        self._initialize_from_store()

    def _initialize_from_store(self):
        """Rehydrate in-memory chunks from ChromaDB for BM25 indexing."""
        try:
            data = self.vectorstore.get()
            if data and data.get("documents"):
                for i, text in enumerate(data["documents"]):
                    meta = data["metadatas"][i] if data.get("metadatas") else {}
                    doc = Document(page_content=text, metadata=meta)
                    t_id = meta.get("thread_id")
                    if t_id:
                        if t_id not in self.thread_chunks:
                            self.thread_chunks[t_id] = []
                            self.thread_files[t_id] = []
                        self.thread_chunks[t_id].append(doc)
                        fn = meta.get("filename")
                        if fn and fn not in self.thread_files[t_id]:
                            self.thread_files[t_id].append(fn)
                    else:
                        self.global_chunks.append(doc)
        except Exception as e:
            print("Note: Could not read existing documents from ChromaDB:", e)

    def ingest_documents(self, file_paths: List[str], thread_id: Optional[str] = None) -> Dict[str, any]:
        """Parse files, chunk, write to ChromaDB, and index for thread memory."""
        all_docs: List[Document] = []
        successful_files: List[str] = []

        for path in file_paths:
            docs = load_file(path, thread_id=thread_id)
            if docs:
                all_docs.extend(docs)
                successful_files.append(Path(path).name)

        if not all_docs:
            return {"chunks": 0, "files": []}

        chunks = text_splitter.split_documents(all_docs)
        self.vectorstore.add_documents(chunks)

        if thread_id:
            if thread_id not in self.thread_chunks:
                self.thread_chunks[thread_id] = []
                self.thread_files[thread_id] = []
            self.thread_chunks[thread_id].extend(chunks)
            for fn in successful_files:
                if fn not in self.thread_files[thread_id]:
                    self.thread_files[thread_id].append(fn)
        else:
            self.global_chunks.extend(chunks)

        return {"chunks": len(chunks), "files": successful_files}

    def get_thread_files(self, thread_id: str) -> List[str]:
        return list(self.thread_files.get(thread_id, []))

    def _build_retriever_for_pool(self, chunks: List[Document], filter_kwargs: Optional[dict] = None):
        """Construct hybrid BM25 + Chroma retriever with FlashRank reranker for a document pool."""
        if not chunks:
            return None

        # Dense retriever (with optional Chroma metadata filter)
        search_kw = {"k": 10}
        if filter_kwargs:
            search_kw["filter"] = filter_kwargs
        dense_retriever = self.vectorstore.as_retriever(search_kwargs=search_kw)

        # Sparse BM25 retriever
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 10

        # RRF Hybrid Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense_retriever],
            weights=[0.5, 0.5],
        )

        # FlashRank Cross-Encoder Reranker
        return ContextualCompressionRetriever(
            base_compressor=flashrank_compressor,
            base_retriever=ensemble,
        )

    def query(self, query_text: str, thread_id: Optional[str] = None) -> str:
        """Query documents strictly scoped to this thread if files are attached."""
        retriever = None

        # 1. Thread isolation: If thread has uploaded documents, ONLY search this thread's pool
        if thread_id and thread_id in self.thread_chunks and self.thread_chunks[thread_id]:
            retriever = self._build_retriever_for_pool(
                self.thread_chunks[thread_id],
                filter_kwargs={"thread_id": str(thread_id)},
            )
            if not retriever:
                files = self.get_thread_files(thread_id)
                return f"Documents attached to this chat ({', '.join(files)}) have no indexed chunks."
        elif self.global_chunks:
            # Fallback only if no thread-specific documents exist
            retriever = self._build_retriever_for_pool(self.global_chunks)
        else:
            return "No documents available in knowledge base for this chat yet. Please upload files first."

        try:
            results = retriever.invoke(query_text)
        except Exception as e:
            print("Retriever error, falling back to direct similarity search:", e)
            search_kwargs = {"k": 4}
            if thread_id and thread_id in self.thread_chunks:
                search_kwargs["filter"] = {"thread_id": str(thread_id)}
            results = self.vectorstore.similarity_search(query_text, **search_kwargs)

        if not results:
            return "No relevant information found in the attached documents for this chat."

        formatted = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", doc.metadata.get("filename", "Unknown"))
            formatted.append(f"--- Document Source {i}: {source} ---\n{doc.page_content}")

        return "\n\n".join(formatted)


# Global instance
rag_system = HybridRAG()


@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """Search uploaded documents (PDF, CSV, Markdown, text) in this chat's knowledge base using hybrid keyword + dense vector search and reranking."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    return rag_system.query(query, thread_id=thread_id)
