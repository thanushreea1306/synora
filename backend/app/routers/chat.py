from fastapi import APIRouter, HTTPException
from app.schemas import TextChatRequest, SignChatRequest
from app.services.gemini_service import chat
from app.services.avatar import map_text_to_animations

router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("/text")
def text_chat(payload: TextChatRequest):
    try:
        reply = chat(payload.session_id, payload.message)
        animations, text_only = map_text_to_animations(reply)
        return {
            "session_id": payload.session_id,
            "user_text": payload.message,
            "reply": reply,
            "animations": animations,
            "text_only": text_only,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat service error: {exc}")

@router.post("/sign")
def sign_chat(payload: SignChatRequest):
    try:
        reply = chat(payload.session_id, f"The user signed the ISL sign: {payload.sign}. Respond conversationally.")
        animations, text_only = map_text_to_animations(reply)
        return {
            "session_id": payload.session_id,
            "user_text": payload.sign,
            "reply": reply,
            "animations": animations,
            "text_only": text_only,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat service error: {exc}")
