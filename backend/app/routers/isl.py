from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.isl_service import recognizer

router = APIRouter(prefix="/api/isl", tags=["ISL Recognition"])

@router.post("/predict")
async def predict_sign(frame: UploadFile = File(...)):
    if frame.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP frame.")
    data = await frame.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Frame must be 5 MB or smaller.")
    try:
        return recognizer.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ISL prediction error: {exc}")
