"""Simple local data collector: press L to label a sign, then enter the label.
Press SPACE to save one sample, Q to quit. Creates data/isl_landmarks.csv.
"""
import csv
from pathlib import Path
import cv2
import mediapipe as mp

OUT = Path("data/isl_landmarks.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
hands_module = mp.solutions.hands

def main():
    label = input("Enter sign label (example HELLO): ").strip().upper()
    if not label:
        raise SystemExit("A label is required.")
    new_file = not OUT.exists()
    cap = cv2.VideoCapture(0)
    saved = 0
    with OUT.open("a", newline="", encoding="utf-8") as f, hands_module.Hands(
        max_num_hands=1, min_detection_confidence=0.6
    ) as hands:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            features = None
            if result.multi_hand_landmarks:
                points = result.multi_hand_landmarks[0].landmark
                features = [v for p in points for v in (p.x, p.y, p.z)]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.multi_hand_landmarks[0], hands_module.HAND_CONNECTIONS
                )
            cv2.putText(frame, f"Label: {label} | SPACE save | Q quit | Saved: {saved}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Synora dataset collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and features is not None:
                writer.writerow([label] + features)
                f.flush()
                saved += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples for {label} to {OUT}")

if __name__ == "__main__":
    main()
