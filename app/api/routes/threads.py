from fastapi import APIRouter

from app.agent.graph import get_thread_messages
from app.memory.stm import checkpointer, get_threads
from app.rag.pipeline import rag_system

router = APIRouter()


def get_thread_title(thread_id: str) -> str:
    """Generate a clean title for a thread based on user input or uploaded docs."""
    messages = get_thread_messages(thread_id)
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip().replace("\n", " ")
            return text[:42] + ("..." if len(text) > 42 else "")
    files = rag_system.get_thread_files(thread_id)
    if files:
        return f"Doc: {files[0]}"
    return "New Chat"


@router.get("/threads")
def list_threads():
    thread_ids = get_threads()
    # Also include any active threads that have uploaded files even before the first message
    for tid in rag_system.thread_files.keys():
        if tid not in thread_ids:
            thread_ids.append(tid)

    return {
        "threads": [
            {
                "id": tid,
                "title": get_thread_title(tid),
                "files": rag_system.get_thread_files(tid),
            }
            for tid in thread_ids
        ]
    }


@router.get("/threads/{thread_id}/messages")
def thread_messages(thread_id: str):
    return {
        "thread_id": thread_id,
        "messages": get_thread_messages(thread_id),
        "files": rag_system.get_thread_files(thread_id),
    }


@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)
    rag_system.thread_chunks.pop(thread_id, None)
    rag_system.thread_files.pop(thread_id, None)
    return {"deleted": thread_id}


@router.get("/threads/{thread_id}/files")
def list_thread_files(thread_id: str):
    return {
        "thread_id": thread_id,
        "files": rag_system.get_thread_files(thread_id),
    }
