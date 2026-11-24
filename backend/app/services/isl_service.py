from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

_mp_hands = mp.solutions.hands

class ISLRecognizer:
    def __init__(self):
        self.model = None
        self.load_error = None
        path = Path(MODEL_PATH)
        if path.exists():
            try:
                self.model = joblib.load(path)
            except Exception as exc:
                self.load_error = str(exc)
        else:
            self.load_error = f"Model file not found: {path}. Train it using scripts/train_model.py."

    def extract_landmarks(self, image_bytes: bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image frame. Send a JPEG or PNG frame.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        # 21 landmarks × (x, y, z) = 63 numeric features.
        return np.array(
            [[point.x, point.y, point.z] for point in hand.landmark],
            dtype=np.float32,
        ).flatten()

    def predict(self, image_bytes: bytes):
        features = self.extract_landmarks(image_bytes)
        if features is None:
            return {
                "recognized_sign": None, "confidence": 0.0, "accepted": False,
                "message": "No hand detected. Please position your hand clearly and try again.",
                "landmarks": None,
            }
        if self.model is None:
            return {
                "recognized_sign": None, "confidence": 0.0, "accepted": False,
                "message": self.load_error or "ISL model is not trained yet.",
                "landmarks": features.tolist(),
            }

        probabilities = self.model.predict_proba([features])[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = str(self.model.classes_[best_index])
        accepted = confidence >= CONFIDENCE_THRESHOLD
        return {
            "recognized_sign": label if accepted else None,
            "confidence": round(confidence, 4),
            "accepted": accepted,
            "message": (
                f"Recognized: {label}" if accepted
                else "Please repeat the sign. Recognition confidence is below the threshold."
            ),
            "landmarks": features.tolist(),
        }

recognizer = ISLRecognizer()
