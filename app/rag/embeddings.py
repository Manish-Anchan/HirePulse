from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import EMBEDDING_MODEL, GEMINI_API

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return shared Google Generative AI embeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API,
    )

# Shared singleton instance
embeddings = get_embeddings()
