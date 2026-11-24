from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
# Demo memory: sessions are held in this process and disappear on restart.
# Use Redis or a database for deployment with multiple workers.
_histories: dict[str, list] = {}

SYSTEM_INSTRUCTION = (
    "You are Synora, a helpful, patient conversational assistant for a Deaf and "
    "Hard-of-Hearing user community. Reply in clear, respectful, natural language. "
    "Do not claim that text output is a complete Indian Sign Language translation. "
    "If the user asks about a sign, explain uncertainty where appropriate."
)

def _require_client():
    if _client is None:
        raise RuntimeError("Gemini API key is missing. Add GEMINI_API_KEY to .env.")

def chat(session_id: str, message: str) -> str:
    _require_client()
    history = _histories.setdefault(session_id, [])
    contents = [
        types.Content(role=item["role"], parts=[types.Part.from_text(text=item["text"])])
        for item in history[-20:]
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty response.")
    history.extend([
        {"role": "user", "text": message},
        {"role": "model", "text": answer},
    ])
    # Keep the last 20 turns (40 messages).
    _histories[session_id] = history[-40:]
    return answer

def understand_image(image_bytes: bytes, mime_type: str, question: str) -> str:
    _require_client()
    prompt = (
        "Describe and interpret the uploaded image accurately. If it contains a "
        "math problem, solve it step by step. If details are unclear, say so. "
        "Use accessible, plain language. User's question: " + question
    )
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty image response.")
    return answer

def clear_history(session_id: str):
    _histories.pop(session_id, None)
