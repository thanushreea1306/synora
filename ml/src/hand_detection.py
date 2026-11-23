from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CAMERA_INDEX = 0
MAX_HANDS = 2
WINDOW_NAME = "SYNORA - Hand Detection"
MIRROR_PREVIEW = True


def handedness_label(categories):
    if not categories:
        return "Unknown"

    category = categories[0]
    label = getattr(category, "display_name", "") or getattr(category, "category_name", "")
    if label.lower() in ("left", "right"):
        return f"{label} hand"
    return label or "Unknown"


def draw_hands(image, result):
    height, width = image.shape[:2]
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    handedness = result.handedness or []

    for hand_index, landmarks in enumerate(result.hand_landmarks or []):
        points = []
        for landmark in landmarks:
            x = int(round(landmark.x * width))
            y = int(round(landmark.y * height))
            if MIRROR_PREVIEW:
                x = width - 1 - x
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            points.append((x, y))

        for connection in connections:
            if connection.start < len(points) and connection.end < len(points):
                cv2.line(
                    image,
                    points[connection.start],
                    points[connection.end],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        for point in points:
            cv2.circle(image, point, 3, (0, 0, 255), -1, cv2.LINE_AA)

        categories = handedness[hand_index] if hand_index < len(handedness) else []
        label = handedness_label(categories)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        if points:
            text_x = min(point[0] for point in points)
            text_y = min(point[1] for point in points) - 10
        else:
            text_x = 0
            text_y = text_height + 4

        text_x = max(0, min(text_x, width - text_width - 1))
        text_y = max(text_height + 4, min(text_y, height - 8))
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main():
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        print(f"Error: Hand Landmarker model not found or empty: {MODEL_PATH}", file=sys.stderr)
        print(f"Download the official model from: {MODEL_URL}", file=sys.stderr)
        return 1

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )

    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"Error: could not load Hand Landmarker model {MODEL_PATH}: {exc}", file=sys.stderr)
        return 1

    cap = None
    try:
        with landmarker:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print(
                    f"Error: could not open webcam index {CAMERA_INDEX}. "
                    "Check that a camera is connected and not in use by another application.",
                    file=sys.stderr,
                )
                return 1

            print("SYNORA hand detection running. Press Q to quit.")
            last_timestamp_ms = -1

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: could not read a frame from the webcam.", file=sys.stderr)
                    return 1

                timestamp_ms = time.monotonic_ns() // 1_000_000
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                display_frame = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame.copy()
                draw_hands(display_frame, result)
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print("\nHand detection stopped.")
    except Exception as exc:
        print(f"Error: hand detection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
