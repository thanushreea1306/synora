from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(r"C:\Users\thanu\synora")
sys.path.insert(0, str(ROOT))

from ml.src.train_baselines_v1 import RF_ESTIMATOR_CONFIG, load_and_validate

FEATURES_DIR = ROOT / "ml/dataset/processed/synora_include21_video_features_v1"

NON_LANDMARK_PREFIXES = (
    "frame_count",
    "duration_s",
    "fps",
    "width",
    "height",
    "padded_frame_count",
    "padded_frame_ratio",
    "no_hand_frame_count",
    "no_hand_frame_ratio",
    "any_hand_frame_ratio",
    "both_hands_frame_count",
    "both_hands_frame_ratio",
    "duplicate_handedness_frames",
    "left_observed_frames",
    "left_presence_ratio",
    "left_missing_ratio",
    "left_conf_mean",
    "left_conf_std",
    "left_conf_min",
    "left_conf_max",
    "left_conf_last",
    "right_observed_frames",
    "right_presence_ratio",
    "right_missing_ratio",
    "right_conf_mean",
    "right_conf_std",
    "right_conf_min",
    "right_conf_max",
    "right_conf_last",
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> int:
    data = load_and_validate(FEATURES_DIR)
    names = list(data["feature_names"]) if "feature_names" in data else None
    if names is None:
        schema = json.loads((FEATURES_DIR / "rf_feature_schema.json").read_text(encoding="utf-8"))
        names = list(schema["feature_columns"])
    removed_indices = [i for i, name in enumerate(names) if name in NON_LANDMARK_PREFIXES]
    kept_indices = [i for i in range(len(names)) if i not in set(removed_indices)]
    missing = sorted(set(NON_LANDMARK_PREFIXES) - {names[i] for i in removed_indices})
    if missing:
        raise RuntimeError(f"expected non-landmark features missing: {missing}")
    y = data["y"]
    splits = data["split_indices"]
    x_all = data["X_rf"]
    result = {
        "run_mode": "in_memory_no_model_saves",
        "rf_config": RF_ESTIMATOR_CONFIG,
        "feature_count_all": int(x_all.shape[1]),
        "feature_count_ablated": len(kept_indices),
        "removed_feature_count": len(removed_indices),
        "removed_features": [names[i] for i in removed_indices],
        "splits": {key: len(value) for key, value in splits.items()},
        "reported_production": {
            "validation": {
                "macro_f1": 0.8523809523809525,
                "accuracy": 0.8571428571428571,
                "balanced_accuracy": 0.8571428571428571,
            },
            "test": {
                "macro_f1": 0.8187074829931973,
                "accuracy": 0.8214285714285714,
                "balanced_accuracy": 0.8214285714285714,
            },
        },
    }
    for label, columns in (("all_features", list(range(x_all.shape[1]))), ("landmark_motion_only", kept_indices)):
        model = RandomForestClassifier(**RF_ESTIMATOR_CONFIG)
        model.fit(x_all[splits["train"]][:, columns], y[splits["train"]])
        predictions = {
            split: model.predict(x_all[splits[split]][:, columns])
            for split in ("val", "test")
        }
        result[label] = {
            split: metrics(y[splits[split]], predictions[split])
            for split in ("val", "test")
        }
    result["deltas_ablated_minus_all"] = {
        split: {
            key: result["landmark_motion_only"][split][key] - result["all_features"][split][key]
            for key in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for split in ("val", "test")
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
