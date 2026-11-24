# Synora Backend (FastAPI)

This is a runnable backend starter based on the Synora Phase-1 project document:
- Webcam frame -> MediaPipe Hands -> Random Forest classifier -> confidence threshold (0.80)
- Uploaded image -> Gemini multimodal image understanding -> conversational Gemini
- Typed text / recognized sign -> Gemini conversational response with session history
- Response text -> supported avatar animation IDs + text-only fallback

## 1. Requirements
- Python 3.10 or 3.11 recommended
- A Google Gemini API key from Google AI Studio
- Webcam for collecting training data
- React frontend (separate project)

## 2. Install and configure
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
Copy `.env.example` to `.env`, then add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
CONFIDENCE_THRESHOLD=0.80
```
Never commit `.env` or expose the API key in React/browser code.

## 3. Run backend
From the `synora_backend` directory:
```bash
uvicorn app.main:app --reload
```
Open:
- API: http://127.0.0.1:8000
- Swagger testing page: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 4. Train the ISL classifier (required for actual sign prediction)
The project document proposes 25-30 isolated signs. The code does NOT include a trained model or ISL dataset because those must be collected and validated by your team.

Collect samples for each sign:
```bash
python scripts/collect_dataset.py
```
Run the collector again for every label. Capture many examples per label from multiple signers, distances, backgrounds and lighting. Aim for a balanced dataset. The collector saves 63 MediaPipe landmark features per sample to `data/isl_landmarks.csv`.

Train:
```bash
python scripts/train_model.py
```
This creates `models/isl_classifier.joblib`. Restart the API and check `/health`.

Important: the sample collector records one hand and isolated static frames. Dynamic signs, two-hand signs, facial expressions, body pose and signer-independent performance require additional capture/model design. Validate with signers excluded from training before presenting accuracy.

## 5. API endpoints

### GET /
Backend status.

### GET /health
Shows whether a trained ISL model is loaded.

### POST /api/chat/text
JSON:
```json
{"session_id":"demo-user-1","message":"Hello, how are you?"}
```

### POST /api/chat/sign
JSON:
```json
{"session_id":"demo-user-1","sign":"HELP"}
```

### POST /api/isl/predict
Multipart form-data: `frame` = one JPEG/PNG/WebP webcam frame.
Returns sign, confidence and whether the 0.80 threshold was met. If accepted, call `/api/chat/sign` with the returned sign.

### POST /api/image/understand
Multipart form-data:
- `session_id`: unique conversation ID
- `question`: optional question about the image
- `image`: JPEG/PNG/WebP file

### DELETE /api/session/history
JSON:
```json
{"session_id":"demo-user-1"}
```

## 6. React integration notes
- Create one UUID per chat and reuse it as `session_id` for all messages.
- For webcam recognition, capture a frame from the video element and POST it as multipart `frame` to `/api/isl/predict`.
- Only call `/api/chat/sign` when `accepted` is true. If false, show the backend `message` ("Please repeat the sign").
- For image uploads, send multipart `image`, `session_id`, and `question`.
- Read `animations` and `text_only` from the response. Load the matching GLB animation clips in Three.js. The backend returns animation identifiers, not the GLB files themselves.

## 7. Configure your real avatar clips
Edit `avatar_mapping.json` so each animation ID exactly matches the clip names or IDs your Three.js frontend expects. The supplied names are placeholders, not actual GLB assets. The initial document describes about 15 predefined animations; add your team's verified ISL animation mappings here.

## 8. Current prototype limitations
- Chat history is in process memory: it is lost on restart and not shared across multiple server workers. Use Redis/PostgreSQL for deployment.
- No login/authentication, rate limiting, persistent database, or production deployment hardening is included.
- Webcam endpoint processes one uploaded frame per request, not a continuous video stream.
- Recognition only works after you collect a dataset and train the model.
- The 0.80 confidence value is the experimental starting threshold from the project document, not a validated accuracy guarantee.
- Avatar mapping is deliberately limited. Unsupported response words remain text-only; it must not be described as full ISL translation.
- The Phase-1 document does not provide the actual 25-30 sign labels, training dataset, Gemini key, or the 15 animation files. Your team must supply these.
