from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\thanu\synora")
sys.path.insert(0, str(ROOT))

from ml.src import prepare_video_features as feature_source
from ml.src.experiments.rf_v4_live_alignment import preprocessing, v4_service
from ml.src.experiments.rf_v2_720p30.capture import load_v2_landmarker_settings
from ml.src.inference import rf_feature_builder
from ml.src.inference.rf_service import RFInferenceService

V1_ROOT = ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
MANIFEST_PATH = V1_ROOT / "video_manifest.csv"
RF_FEATURES_PATH = V1_ROOT / "rf_features.npy"
SELECTED = [
    ("0f7d0db66e2443788b125b60354f43e23b02dec63ad5657cd8ba23a0b033fad2", "loud"),
    ("8f74b3199c7316dcae786977c69bcdef0bdefc18552c7cb22802845fb9961b2b", "quiet"),
    ("07b0469065f97b55f6f5863b9f7a1253122ad44ee3374a84df45776db90cd39e", "happy"),
]


def read_manifest():
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def extract_live_records(video_path: Path, row: dict[str, str]):
    settings, model_path = load_v2_landmarker_settings()
    session = preprocessing.LandmarkSession(model_path, settings)
    session.reset()
    records = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    try:
        while len(records) < preprocessing.MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            processed = session.detect(frame, resize=True)
            record = preprocessing.make_landmark_record(
                processed=processed,
                split="test",
                video_id=row["video_id"],
                official_rel_path=row["official_rel_path_posix"],
                source_video_abs=str(video_path),
                class_folder=row["class_folder"],
                class_label=row["class_label"],
                label_id=int(row["label_id"]),
                source_timestamp_ms=processed.frame_index * 1000.0 / 25.0,
                source_frame_index=processed.frame_index,
                source_fps=25.0,
                source_width=1920,
                source_height=1080,
            )
            records.append(record)
    finally:
        capture.release()
        session.close()
    if len(records) < preprocessing.MIN_FRAMES:
        raise RuntimeError(f"live extraction produced only {len(records)} frames")
    metadata = {
        "fps": preprocessing.TARGET_FPS,
        "width": preprocessing.TARGET_WIDTH,
        "height": preprocessing.TARGET_HEIGHT,
        "duration_s": len(records) / preprocessing.TARGET_FPS,
    }
    return records, metadata


def compare_vectors(offline: np.ndarray, live: np.ndarray, names: list[str]):
    diff = np.abs(offline.astype(np.float64) - live.astype(np.float64))
    first = int(np.argmax(diff > 0)) if bool((diff > 0).any()) else None
    largest = int(np.argmax(diff))
    return {
        "max_abs_difference": float(diff.max()),
        "differing_feature_count": int((diff > 0).sum()),
        "first_differing_index": first,
        "first_differing_feature": names[first] if first is not None else None,
        "largest_difference_index": largest,
        "largest_difference_feature": names[largest],
        "largest_difference_offline": float(offline[largest]),
        "largest_difference_live": float(live[largest]),
    }


def prediction_summary(result: dict):
    return {
        "status": result.get("status"),
        "accepted": result.get("accepted"),
        "predicted_sign": result.get("predicted_sign"),
        "raw_confidence": result.get("raw_confidence"),
        "calibrated_confidence": result.get("calibrated_confidence"),
        "errors": result.get("errors"),
    }


def main() -> int:
    rows = read_manifest()
    by_id = {row["video_id"]: row for row in rows}
    features = np.load(RF_FEATURES_PATH)
    schema = json.loads((V1_ROOT / "rf_feature_schema.json").read_text(encoding="utf-8"))
    names = list(schema["feature_columns"])
    rf_service = RFInferenceService()
    v4 = v4_service.V4RecognitionService()
    output = []
    for video_id, expected_label in SELECTED:
        row = by_id[video_id]
        index = int(row["sample_index"])
        offline_vector = features[index]
        offline_row = rf_feature_builder.RFFeatureRow(
            feature_names=tuple(names),
            values=offline_vector.reshape(1, -1).astype(np.float32),
            diagnostics={},
        )
        offline_result = rf_service._predict_feature_row(offline_row, {})
        video_path = Path(row["source_video_abs"])
        records, metadata = extract_live_records(video_path, row)
        live_row = rf_feature_builder.build_rf_feature_row(
            records,
            metadata=metadata,
            expected_feature_names=tuple(names),
        )
        live_result = rf_service.predict(records, metadata=metadata)
        v4_result = v4.predict(records, metadata=metadata)
        output.append(
            {
                "video_id": video_id,
                "expected_label": expected_label,
                "source_video": str(video_path),
                "offline_frame_count": int(row["frame_count"]),
                "live_frame_count": len(records),
                "rf_feature_comparison": compare_vectors(offline_vector, live_row.values[0], names),
                "offline_rf_prediction": prediction_summary(offline_result),
                "live_rf_prediction": prediction_summary(live_result),
                "live_v4_prediction": prediction_summary(v4_result),
            }
        )
    print(json.dumps({"parity": output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
