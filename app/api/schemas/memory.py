from typing import Optional
from pydantic import BaseModel

class MemoryCreateRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default_user"
    category: Optional[str] = "preference"
