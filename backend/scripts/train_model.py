"""Train a RandomForest from a CSV of MediaPipe hand landmarks.

CSV format: label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
Each row is one labeled hand sample (63 numeric landmark features).
Collect samples from multiple people, lighting conditions and hand positions.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA = Path("data/isl_landmarks.csv")
OUT = Path("models/isl_classifier.joblib")

def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column.")
    X = df.drop(columns=["label"]).astype(float)
    y = df["label"].astype(str)
    if X.shape[1] != 63:
        raise SystemExit(f"Expected 63 landmark columns, got {X.shape[1]}.")
    if y.nunique() < 2:
        raise SystemExit("At least two sign classes are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions, zero_division=0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved model to {OUT}")

if __name__ == "__main__":
    main()
