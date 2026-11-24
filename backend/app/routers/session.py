from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.services.gemini_service import clear_history

router = APIRouter(prefix="/api/session", tags=["Session"])

class ClearRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)

@router.delete("/history")
def delete_history(payload: ClearRequest):
    clear_history(payload.session_id)
    return {"message": "Conversation history cleared.", "session_id": payload.session_id}
