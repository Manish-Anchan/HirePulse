from fastapi import APIRouter

from app.api.schemas.memory import MemoryCreateRequest
from app.memory.ltm import ltm_store

router = APIRouter()


@router.get("/memories")
def list_memories(user_id: str = "default_user"):
    return {"memories": ltm_store.get_all_memories(user_id=user_id)}


@router.post("/memories")
def create_memory(req: MemoryCreateRequest):
    return ltm_store.add_memory(req.text, user_id=req.user_id, category=req.category)


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    success = ltm_store.delete_memory(memory_id)
    return {"deleted": memory_id, "success": success}
