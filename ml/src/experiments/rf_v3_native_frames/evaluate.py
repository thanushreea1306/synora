from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v3_generalisation import calibration, cv as v3_cv, model as v3_model
    from ml.src.experiments.rf_v3_native_frames import config
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v3_generalisation import calibration, cv as v3_cv, model as v3_model
    from experiments.rf_v3_native_frames import config


CLASS_COUNT = v3_model.CLASS_COUNT
UNKNOWN = "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=config.DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=config.DEFAULT_REPORT_ROOT)
    parser.add_argument("--baseline-model-root", type=Path, default=config.DEFAULT_SELECTED_V3_ROOT)
    parser.add_argument("--baseline-report-root", type=Path, default=config.DEFAULT_V4_REPORT_ROOT)
    parser.add_argument("--v1-config", type=Path, default=config.DEFAULT_V1_EXPERIMENT_CONFIG)
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(weights_path: Path, config_path: Path) -> tuple[Any, dict[str, Any]]:
    model_config = v2_dataset.read_json(config_path)
    model_instance = v3_model.RegularizedTemporalMLP(dict(model_config))
    with np.load(weights_path) as weights:
        model_instance.weights = {key: weights[key].copy() for key in weights.files}
    return model_instance, model_config


def predict_logits(model: Any, tensor: np.ndarray) -> np.ndarray:
    prepared, real = model.prepare_inputs(tensor)
    return np.asarray(model.predict_logits(prepared, real), dtype=np.float64)


def calibration_values(report: Mapping[str, Any]) -> tuple[float, float]:
    temperature = float(report["temperature"]["temperature"])
    threshold = float(report["threshold"]["chosen"]["threshold"])
    if not 0.0 < temperature < np.inf or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid calibration values")
    return temperature, threshold


def fixed_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(CLASS_COUNT)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_model(
    y: np.ndarray,
    logits: np.ndarray,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    raw_probabilities = calibration.softmax(logits, 1.0)
    calibrated_probabilities = calibration.softmax(logits, temperature)
    raw_predictions = np.argmax(raw_probabilities, axis=1).astype(np.int64)
    candidates = np.argmax(calibrated_probabilities, axis=1).astype(np.int64)
    confidences = calibrated_probabilities.max(axis=1)
    accepted = confidences >= threshold
    accepted_metrics = (
        fixed_label_metrics(y[accepted], candidates[accepted]) if accepted.any() else {"accuracy": 0.0, "macro_f1": 0.0}
    )
    return {
        "raw_probabilities": raw_probabilities,
        "calibrated_probabilities": calibrated_probabilities,
        "raw_predictions": raw_predictions,
        "candidate_predictions": candidates,
        "confidences": confidences,
        "accepted": accepted,
        "raw_metrics": v3_model.metrics(y, raw_predictions),
        "rejection": {
            "temperature": temperature,
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "accepted_count": int(accepted.sum()),
            "rejected_count": int((~accepted).sum()),
            "accepted_accuracy": accepted_metrics["accuracy"],
            "accepted_macro_f1": accepted_metrics["macro_f1"],
            "predicted_unknown_count": int((~accepted).sum()),
        },
    }


def compact_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_metrics": result["raw_metrics"],
        "rejection": result["rejection"],
    }


def sample_statistics(tensor: np.ndarray, row: Mapping[str, str]) -> dict[str, Any]:
    frame_real = tensor[:, 130] > 0.5
    present_left = tensor[:, 126] > 0.5
    present_right = tensor[:, 127] > 0.5
    no_hand = tensor[:, 131] > 0.5
    confidence = np.maximum(tensor[:, 133], tensor[:, 134])
    real_confidence = confidence[frame_real]
    both_hands = present_left & present_right & frame_real
    frame_count = int(row["frame_count"])
    return {
        "video_id": row["video_id"],
        "frame_count": frame_count,
        "source_frame_count": int(row["source_frame_count"]),
        "no_hand_frames": int(np.count_nonzero(no_hand & frame_real)),
        "both_hands_ratio": float(both_hands.mean()) if frame_count else 0.0,
        "mean_confidence": float(real_confidence.mean()) if real_confidence.size else 0.0,
        "padded_ratio": float(1.0 - frame_count / 89.0),
    }


def duration_label(frame_count: int) -> str:
    if frame_count <= 49:
        return "short_43_49"
    if frame_count <= 69:
        return "mid_50_69"
    return "long_70_89"


def boundary_label(frame_count: int) -> str:
    if frame_count == 43:
        return "exact_43"
    if frame_count >= 80:
        return "near_cap_80_plus"
    return "interior"


def grouped_indices(statistics: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = {
        "missing_hands": {"any_no_hand": [], "no_missing_hand": []},
        "two_hand_dominant": {"both_hands_majority": [], "single_or_absent": []},
        "low_confidence": {"mean_confidence_below_0_8": [], "mean_confidence_at_least_0_8": []},
        "duration": {"short_43_49": [], "mid_50_69": [], "long_70_89": []},
        "boundary": {"exact_43": [], "near_cap_80_plus": [], "interior": []},
        "padded_ratio": {"below_70_percent_real": [], "at_least_70_percent_real": []},
    }
    for index, row in enumerate(statistics):
        groups["missing_hands"]["any_no_hand" if int(row["no_hand_frames"]) > 0 else "no_missing_hand"].append(index)
        groups["two_hand_dominant"]["both_hands_majority" if float(row["both_hands_ratio"]) >= 0.5 else "single_or_absent"].append(index)
        groups["low_confidence"]["mean_confidence_below_0_8" if float(row["mean_confidence"]) < 0.8 else "mean_confidence_at_least_0_8"].append(index)
        groups["duration"][duration_label(int(row["frame_count"]))].append(index)
        groups["boundary"][boundary_label(int(row["frame_count"]))].append(index)
        real_ratio = 1.0 - float(row["padded_ratio"])
        groups["padded_ratio"]["at_least_70_percent_real" if real_ratio >= 0.7 else "below_70_percent_real"].append(index)
    return groups


def group_metrics(
    indices: Sequence[int],
    y: np.ndarray,
    native_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, Any]:
    if not indices:
        return {
            "count": 0,
            "native": {"accuracy": None, "macro_f1": None},
            "baseline": {"accuracy": None, "macro_f1": None},
            "native_minus_baseline": {"accuracy": None, "macro_f1": None},
        }
    selected = np.asarray(list(indices), dtype=np.int64)
    native_metrics = fixed_label_metrics(y[selected], native_predictions[selected])
    baseline_metrics = fixed_label_metrics(y[selected], baseline_predictions[selected])
    return {
        "count": int(len(selected)),
        "native": native_metrics,
        "baseline": baseline_metrics,
        "native_minus_baseline": {
            "accuracy": native_metrics["accuracy"] - baseline_metrics["accuracy"],
            "macro_f1": native_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        },
    }


def robustness_report(
    statistics: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    native_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, Any]:
    grouped = grouped_indices(statistics)
    result: dict[str, Any] = {}
    for group_name, buckets in grouped.items():
        result[group_name] = {
            bucket: group_metrics(indices, y, native_predictions, baseline_predictions)
            for bucket, indices in buckets.items()
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    report_root = args.report_root.resolve()
    model_root = args.model_root.resolve()
    selection = v2_dataset.read_json(report_root / "selection.json")
    training_manifest = v2_dataset.read_json(report_root / "training_manifest.json")
    if selection.get("selection_uses_test") is not False or training_manifest.get("test_split_read") is not False:
        raise RuntimeError("training or selection manifest does not certify no test use")

    native_model, native_config = load_model(model_root / "selected/weights.npz", model_root / "selected/config.json")
    baseline_model, baseline_config = load_model(
        args.baseline_model_root / "weights.npz",
        args.baseline_model_root / "config.json",
    )
    native_calibration = v2_dataset.read_json(report_root / "calibration.json")
    baseline_calibration = v2_dataset.read_json(args.baseline_report_root / "calibration.json")
    native_temperature, native_threshold = calibration_values(native_calibration)
    baseline_temperature, baseline_threshold = calibration_values(baseline_calibration)

    native_features_root = data_root / "features_native"
    resampled_features_root = data_root / "features_resampled"
    native_tensor = np.load(native_features_root / "base_tensor.npy")
    native_manifest = v2_dataset.read_csv(native_features_root / "video_manifest.csv")
    native_labels = np.load(native_features_root / "labels.npy")
    resampled_tensor = np.load(resampled_features_root / "mlp_tensor.npy")
    resampled_manifest = v2_dataset.read_csv(resampled_features_root / "video_manifest.csv")
    resampled_labels = np.load(resampled_features_root / "labels.npy")
    native_by_id = {row["video_id"]: index for index, row in enumerate(native_manifest)}
    resampled_by_id = {row["video_id"]: index for index, row in enumerate(resampled_manifest)}
    common_ids = sorted(set(native_by_id) & set(resampled_by_id))
    if not common_ids:
        raise RuntimeError("no common native/resampled test samples")
    native_indices = np.asarray([native_by_id[item] for item in common_ids], dtype=np.int64)
    resampled_indices = np.asarray([resampled_by_id[item] for item in common_ids], dtype=np.int64)
    y = np.asarray([int(native_labels[index]) for index in native_indices], dtype=np.int64)
    if not np.array_equal(y, resampled_labels[resampled_indices]):
        raise RuntimeError("native and resampled test labels differ")
    if len(native_manifest) != config.EXPECTED_TEST_COUNT:
        raise RuntimeError(f"unexpected native final test count: {len(native_manifest)}")
    if len(resampled_manifest) != config.EXPECTED_RESAMPLED_TEST_COUNT:
        raise RuntimeError(f"unexpected resampled final test count: {len(resampled_manifest)}")
    if len(common_ids) != config.EXPECTED_RESAMPLED_TEST_COUNT:
        raise RuntimeError(f"unexpected pairwise final test count: {len(common_ids)}")
    excluded_from_pairwise = sorted(set(native_by_id) - set(resampled_by_id))

    native_logits = predict_logits(native_model, native_tensor[native_indices])
    baseline_logits = predict_logits(baseline_model, resampled_tensor[resampled_indices])
    native_result = evaluate_model(y, native_logits, native_temperature, native_threshold)
    baseline_result = evaluate_model(y, baseline_logits, baseline_temperature, baseline_threshold)
    native_full_indices = np.arange(len(native_manifest), dtype=np.int64)
    native_full_y = np.asarray(native_labels, dtype=np.int64)
    native_full_logits = predict_logits(native_model, native_tensor[native_full_indices])
    native_full_result = evaluate_model(
        native_full_y,
        native_full_logits,
        native_temperature,
        native_threshold,
    )
    excluded_indices = np.asarray([native_by_id[item] for item in excluded_from_pairwise], dtype=np.int64)
    native_excluded_result = evaluate_model(
        native_full_y[excluded_indices],
        native_full_logits[excluded_indices],
        native_temperature,
        native_threshold,
    )
    target_names = [
        str(record["class_label"])
        for record in sorted(
            v2_dataset.read_json(args.v1_config)["label_mapping"],
            key=lambda item: int(item["model_index"]),
        )
    ]
    delta = {
        "accuracy": native_result["raw_metrics"]["accuracy"] - baseline_result["raw_metrics"]["accuracy"],
        "macro_f1": native_result["raw_metrics"]["macro_f1"] - baseline_result["raw_metrics"]["macro_f1"],
        "balanced_accuracy": native_result["raw_metrics"]["balanced_accuracy"]
        - baseline_result["raw_metrics"]["balanced_accuracy"],
        "rejection_coverage": native_result["rejection"]["coverage"] - baseline_result["rejection"]["coverage"],
        "accepted_accuracy": native_result["rejection"]["accepted_accuracy"]
        - baseline_result["rejection"]["accepted_accuracy"],
    }
    metrics_payload = {
        "experiment_id": config.EXPERIMENT_ID,
        "test_split_use": "final_evaluation_only",
        "model_selection_used_test": False,
        "calibration_used_test": False,
        "sample_count": int(len(common_ids)),
        "native_full_test_sample_count": int(len(native_manifest)),
        "excluded_from_pairwise_count": int(len(excluded_from_pairwise)),
        "excluded_from_pairwise_video_ids": excluded_from_pairwise,
        "native": compact_evaluation(native_result),
        "baseline_v3_resampled": compact_evaluation(baseline_result),
        "native_full_test": compact_evaluation(native_full_result),
        "native_excluded_from_pairwise": compact_evaluation(native_excluded_result),
        "native_minus_baseline": delta,
        "native_classification_report": v3_cv.classification_report_dict(
            y, native_result["raw_predictions"], target_names
        ),
        "baseline_classification_report": v3_cv.classification_report_dict(
            y, baseline_result["raw_predictions"], target_names
        ),
        "native_confusion_matrix": confusion_matrix(
            y, native_result["raw_predictions"], labels=list(range(CLASS_COUNT))
        ).tolist(),
        "baseline_confusion_matrix": confusion_matrix(
            y, baseline_result["raw_predictions"], labels=list(range(CLASS_COUNT))
        ).tolist(),
        "model_identity": {
            "native_config": native_config,
            "baseline_config": baseline_config,
            "native_weights_sha256": sha256_file(model_root / "selected/weights.npz"),
            "baseline_weights_sha256": sha256_file(args.baseline_model_root / "weights.npz"),
            "native_calibration_sha256": sha256_file(report_root / "calibration.json"),
            "baseline_calibration_sha256": sha256_file(args.baseline_report_root / "calibration.json"),
        },
    }
    v2_dataset.write_json(report_root / "test_metrics.json", metrics_payload)

    statistics = [
        sample_statistics(native_tensor[index], native_manifest[native_by_id[item]])
        for item, index in zip(common_ids, native_indices)
    ]
    robustness = robustness_report(
        statistics,
        y,
        native_result["raw_predictions"],
        baseline_result["raw_predictions"],
    )
    v2_dataset.write_json(
        report_root / "robustness.json",
        {
            "experiment_id": config.EXPERIMENT_ID,
            "test_split_use": "final_evaluation_only",
            "groups": robustness,
            "native_only_excluded_from_pairwise": {
                "video_ids": excluded_from_pairwise,
                "count": int(len(excluded_from_pairwise)),
                "native": compact_evaluation(native_excluded_result),
                "reason": "V2 resampling would exceed the frozen 89-frame cap; native capture keeps every real frame",
            },
            "notes": {
                "missing_hands": "bucketed from no_hand_frame on frame_real frames in native tensors",
                "two_hand_dominant": "both hands present on at least half of real frames",
                "low_confidence": "mean max hand confidence below 0.8 on real frames",
                "duration": "source frame count buckets at 30 FPS",
                "boundary": "minimum-length and near-cap cases",
                "unknown": "rejection coverage and accepted accuracy are reported in test_metrics.json",
            },
        },
    )

    csv_path = report_root / "per_sample_predictions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_id",
                "true_label",
                "native_prediction",
                "baseline_prediction",
                "native_confidence",
                "baseline_confidence",
                "native_accepted",
                "baseline_accepted",
                "frame_count",
                "no_hand_frames",
                "both_hands_ratio",
                "mean_confidence",
            ]
        )
        for position, item in enumerate(common_ids):
            stats = statistics[position]
            writer.writerow(
                [
                    item,
                    target_names[int(y[position])],
                    target_names[int(native_result["raw_predictions"][position])],
                    target_names[int(baseline_result["raw_predictions"][position])],
                    float(native_result["confidences"][position]),
                    float(baseline_result["confidences"][position]),
                    bool(native_result["accepted"][position]),
                    bool(baseline_result["accepted"][position]),
                    stats["frame_count"],
                    stats["no_hand_frames"],
                    stats["both_hands_ratio"],
                    stats["mean_confidence"],
                ]
            )

    input_hashes = {
        "native_features/base_tensor.npy": sha256_file(native_features_root / "base_tensor.npy"),
        "native_features/video_manifest.csv": sha256_file(native_features_root / "video_manifest.csv"),
        "resampled_features/mlp_tensor.npy": sha256_file(resampled_features_root / "mlp_tensor.npy"),
        "resampled_features/video_manifest.csv": sha256_file(resampled_features_root / "video_manifest.csv"),
        "native_selected/weights.npz": sha256_file(model_root / "selected/weights.npz"),
        "native_selected/config.json": sha256_file(model_root / "selected/config.json"),
        "baseline_selected/weights.npz": sha256_file(args.baseline_model_root / "weights.npz"),
        "baseline_selected/config.json": sha256_file(args.baseline_model_root / "config.json"),
        "native_calibration.json": sha256_file(report_root / "calibration.json"),
        "baseline_calibration.json": sha256_file(args.baseline_report_root / "calibration.json"),
    }
    output_hashes = {
        "test_metrics.json": sha256_file(report_root / "test_metrics.json"),
        "robustness.json": sha256_file(report_root / "robustness.json"),
        "per_sample_predictions.csv": sha256_file(csv_path),
    }
    v2_dataset.write_json(
        report_root / "test_manifest.json",
        {
            "experiment_id": config.EXPERIMENT_ID,
            "test_split_use": "final_evaluation_only",
            "model_selection_used_test": False,
            "calibration_used_test": False,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "reproduction_command": [
                sys.executable,
                "-m",
                "ml.src.experiments.rf_v3_native_frames.evaluate",
                "--data-root",
                str(data_root),
                "--model-root",
                str(model_root),
                "--report-root",
                str(report_root),
                "--baseline-model-root",
                str(args.baseline_model_root.resolve()),
                "--baseline-report-root",
                str(args.baseline_report_root.resolve()),
                "--v1-config",
                str(args.v1_config.resolve()),
            ],
        },
    )
    print(
        json.dumps(
            {
                "sample_count": len(common_ids),
                "native_macro_f1": native_result["raw_metrics"]["macro_f1"],
                "baseline_macro_f1": baseline_result["raw_metrics"]["macro_f1"],
                "macro_f1_delta": delta["macro_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
