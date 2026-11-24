from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.config import MAX_IMAGE_MB
from app.services.gemini_service import chat, understand_image
from app.services.avatar import map_text_to_animations

router = APIRouter(prefix="/api/image", tags=["Image Understanding"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@router.post("/understand")
async def understand(
    session_id: str = Form(...),
    question: str = Form("Please explain this image."),
    image: UploadFile = File(...),
):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")
    data = await image.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image must be {MAX_IMAGE_MB} MB or smaller.")
    try:
        analysis = understand_image(data, image.content_type, question)
        # Send the image interpretation into the same conversation memory.
        reply = chat(session_id, "Image understanding result: " + analysis)
        animations, text_only = map_text_to_animations(reply)
        return {
            "session_id": session_id,
            "image_analysis": analysis,
            "reply": reply,
            "animations": animations,
            "text_only": text_only,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image service error: {exc}")
