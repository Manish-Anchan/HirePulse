import json
import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from chat_app import chatbot, checkpointer, get_thread_messages, get_threads
from rag_pipeline import DOCS_DIR, rag_system

app = FastAPI(title="Chatbot API with Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".md", ".markdown", ".txt", ".json", ".py", ".log"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


def _thread_title(thread_id: str) -> str:
    messages = get_thread_messages(thread_id)
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip().replace("\n", " ")
            return text[:42] + ("..." if len(text) > 42 else "")
    files = rag_system.get_thread_files(thread_id)
    if files:
        return f"Doc: {files[0]}"
    return "New Chat"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/threads")
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
                "title": _thread_title(tid),
                "files": rag_system.get_thread_files(tid),
            }
            for tid in thread_ids
        ]
    }


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str):
    return {
        "thread_id": thread_id,
        "messages": get_thread_messages(thread_id),
        "files": rag_system.get_thread_files(thread_id),
    }


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)
    rag_system.thread_chunks.pop(thread_id, None)
    rag_system.thread_files.pop(thread_id, None)
    return {"deleted": thread_id}


@app.get("/api/threads/{thread_id}/files")
def list_thread_files(thread_id: str):
    return {
        "thread_id": thread_id,
        "files": rag_system.get_thread_files(thread_id),
    }


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    thread_id: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    active_tid = thread_id or str(uuid.uuid4())
    thread_upload_dir = os.path.join(DOCS_DIR, active_tid)
    os.makedirs(thread_upload_dir, exist_ok=True)

    saved_paths = []
    skipped_files = []

    for file in files:
        filename = os.path.basename(file.filename or "")
        ext = os.path.splitext(filename)[1].lower()

        if not filename or ext not in ALLOWED_EXTENSIONS:
            skipped_files.append({"name": filename, "reason": f"Unsupported format ({ext or 'none'})"})
            continue

        dest_path = os.path.join(thread_upload_dir, filename)
        # Check size & stream write
        size = 0
        try:
            with open(dest_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB buffer
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        buffer.close()
                        os.remove(dest_path)
                        raise HTTPException(
                            status_code=400,
                            detail=f"File {filename} exceeds 25MB limit",
                        )
                    buffer.write(chunk)
            saved_paths.append(dest_path)
        except HTTPException:
            raise
        except Exception as e:
            skipped_files.append({"name": filename, "reason": str(e)})

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No valid files were uploaded. Skipped: {skipped_files}",
        )

    try:
        ingest_result = rag_system.ingest_documents(saved_paths, thread_id=active_tid)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index documents: {str(e)}",
        )

    return {
        "success": True,
        "thread_id": active_tid,
        "indexed_chunks": ingest_result.get("chunks", 0),
        "uploaded_files": ingest_result.get("files", []),
        "all_thread_files": rag_system.get_thread_files(active_tid),
        "skipped": skipped_files,
    }


@app.post("/api/chat")
def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'meta', 'thread_id': thread_id})}\n\n"
            for chunk, _metadata in chatbot.stream(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
