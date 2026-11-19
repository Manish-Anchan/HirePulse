import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import ALLOWED_EXTENSIONS, DOCS_DIR, MAX_FILE_SIZE
from app.rag.pipeline import rag_system

router = APIRouter()


@router.post("/documents/upload")
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
