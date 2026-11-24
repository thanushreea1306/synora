from pydantic import BaseModel, Field
from typing import Optional, List


# =========================
# TEXT CHAT REQUEST
# =========================
class TextChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=5000)


# =========================
# SIGN CHAT REQUEST
# =========================
class SignChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    sign: str = Field(min_length=1, max_length=300)


# =========================
# CHAT RESPONSE
# =========================
class ChatResponse(BaseModel):
    session_id: str
    user_text: str
    reply: str
    animations: List[str] = Field(default_factory=list)
    text_only: List[str] = Field(default_factory=list)


# =========================
# ISL PREDICTION RESPONSE
# =========================
class PredictionResponse(BaseModel):
    recognized_sign: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted: bool = False
    message: str
    landmarks: Optional[List[List[float]]] = None