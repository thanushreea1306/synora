from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v3_generalisation import calibration, cv as v3_cv, model as v3_model
    from ml.src.experiments.rf_v4_live_alignment import augment, models, parity
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v3_generalisation import calibration, cv as v3_cv, model as v3_model
    from experiments.rf_v4_live_alignment import augment, models, parity


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V2_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_V2_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_V4_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment/features"
DEFAULT_V4_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v4_live_alignment"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v4_live_alignment"
DEFAULT_V1_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
EXPERIMENT_ID = "synora_rf_v4_live_alignment_20261123"
SEED = models.SEED
CLASS_COUNT = models.CLASS_COUNT
N_SPLITS = 5
MIN_CV_CLASS_SUPPORT = 2
SELECTION_REGRESSION_PENALTY = 1.5
V3_VARIANT = "v3_regularized_cv"
V4_VARIANTS = (
    "v4_base_mlp_cv",
    "v4_enhanced_tcn_cv",
    "v4_enhanced_attention_cv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-features-root", type=Path, default=DEFAULT_V2_FEATURES_ROOT)
    parser.add_argument("--v2-extraction-root", type=Path, default=DEFAULT_V2_EXTRACTION_ROOT)
    parser.add_argument("--v4-features-root", type=Path, default=DEFAULT_V4_FEATURES_ROOT)
    parser.add_argument("--v4-extraction-root", type=Path, default=DEFAULT_V4_EXTRACTION_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    return v2_dataset.sha256_file(path)


def _write_json(path: Path, payload: Any) -> None:
    v2_dataset.write_json(path, payload)


def load_data(args: argparse.Namespace) -> dict[str, Any]:
    v3_args = argparse.Namespace(
        features_root=args.v2_features_root,
        extraction_root=args.v2_extraction_root,
        v1_config=args.v1_config,
    )
    v2 = v3_cv.load_data(v3_args)
    v4_root = args.v4_features_root.resolve()
    extraction_root = args.v4_extraction_root.resolve()
    extraction_config = v2_dataset.read_json(extraction_root / "config.json")
    if extraction_config.get("split_policy", {}).get("test_split_read") is not False:
        raise ValueError("V4 extraction config does not certify no test split access")
    feature_summary = v2_dataset.read_json(v4_root / "feature_build_summary.json")
    schema = v2_dataset.read_json(v4_root / "feature_schema.json")
    manifest = v2_dataset.read_csv(v4_root / "video_manifest.csv")
    extraction_manifest = v2_dataset.read_csv(extraction_root / "manifest.csv")
    if not manifest:
        raise ValueError("V4 feature manifest is empty")
    base_tensor = np.load(v4_root / "base_tensor.npy")
    enhanced_tensor = np.load(v4_root / "enhanced_tensor.npy")
    labels = np.load(v4_root / "labels.npy")
    expected_channels = len(schema["enhanced_channel_names"])
    if schema["enhanced_channel_names"] != models.features.ENHANCED_CHANNEL_NAMES:
        raise ValueError("V4 enhanced channel order mismatch")
    if base_tensor.shape != (len(manifest), models.features.T_VALUE, models.BASE_FRAME_CHANNEL_COUNT):
        raise ValueError(f"unexpected V4 base tensor shape: {base_tensor.shape}")
    if enhanced_tensor.shape != (len(manifest), models.features.T_VALUE, expected_channels):
        raise ValueError(f"unexpected V4 enhanced tensor shape: {enhanced_tensor.shape}")
    if labels.shape != (len(manifest),) or not np.isfinite(base_tensor).all() or not np.isfinite(enhanced_tensor).all():
        raise ValueError("invalid V4 tensors or labels")
    if feature_summary["sample_count"] != len(manifest):
        raise ValueError("V4 feature summary count mismatch")
    v4_y = np.asarray([int(row["model_index"]) for row in manifest], dtype=np.int64)
    if not np.array_equal(labels, v4_y):
        raise ValueError("V4 labels.npy does not match manifest")
    v4_by_id = {row["video_id"]: index for index, row in enumerate(manifest)}
    v2_by_id = {row["video_id"]: index for index, row in enumerate(v2["manifest"])}
    common_ids = sorted(set(v4_by_id) & set(v2_by_id))
    if len(common_ids) < 2:
        raise ValueError("not enough common samples for V3/V4 comparison")
    v4_indices = np.asarray([v4_by_id[item] for item in common_ids], dtype=np.int64)
    v2_indices = np.asarray([v2_by_id[item] for item in common_ids], dtype=np.int64)
    y_v2 = v2["y"][v2_indices]
    y_v4 = v4_y[v4_indices]
    if not np.array_equal(y_v2, y_v4):
        raise ValueError("V2/V3 and V4 labels differ on common samples")
    v4_manifest = [manifest[index] for index in v4_indices]
    v2_manifest = [v2["manifest"][index] for index in v2_indices]
    split_counts = {split: sum(1 for row in v4_manifest if row["split"] == split) for split in v2_dataset.SPLITS}
    if set(split_counts) != set(v2_dataset.SPLITS):
        raise ValueError("unexpected split labels in common subset")
    quality_rows = {row["video_id"]: row for row in extraction_manifest if row["status"] == "complete"}
    quality = []
    for row in v4_manifest:
        extraction_row = quality_rows.get(row["video_id"])
        if extraction_row is None:
            raise ValueError(f"missing V4 extraction row for {row['video_id']}")
        quality.append(
            {
                "video_id": row["video_id"],
                "source_fps": float(extraction_row["source_fps"]),
                "source_frame_count": int(extraction_row["source_frame_count"]),
                "target_frame_count": int(extraction_row["target_frame_count"]),
                "duplicate_source_frames": int(extraction_row["duplicate_source_frames"]),
                "resize_interpolation": extraction_row["resize_interpolation"],
                "tracking_lifecycle": extraction_row["tracking_lifecycle"],
                "no_hand_frames": int(extraction_row["no_hand_frames"]),
                "multiple_hand_frames": int(extraction_row["multiple_hand_frames"]),
                "handedness_mismatch_frames": int(extraction_row["handedness_mismatch_frames"]),
            }
        )
    return {
        "v2": v2,
        "v4_features_root": v4_root,
        "v4_extraction_root": extraction_root,
        "common_ids": common_ids,
        "v2_indices": v2_indices,
        "v4_indices": v4_indices,
        "v3_tensor": v2["tensor"][v2_indices],
        "base_tensor": base_tensor[v4_indices],
        "enhanced_tensor": enhanced_tensor[v4_indices],
        "y": y_v4,
        "groups": np.asarray(common_ids, dtype=object),
        "v4_manifest": v4_manifest,
        "v2_manifest": v2_manifest,
        "target_names": v2["target_names"],
        "split_counts": split_counts,
        "quality": quality,
    }


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    return v3_cv.make_folds(y, groups, n_splits=n_splits)


def build_v3_config() -> dict[str, Any]:
    config = v3_model.build_config(
        V3_VARIANT,
        label_smoothing=0.10,
        class_weight_power=1.0,
    )
    config["selection_data"] = "333 common train+val samples shared with V4"
    return config


def build_v4_configs() -> dict[str, Any]:
    augmentation = augment.default_augmentation_config()
    return {
        "v4_base_mlp_cv": models.build_config(
            "v4_base_mlp_cv",
            feature_mode="base",
            architecture="framewise_shared_mlp_masked_mean_max_pooling",
            input_dim=models.BASE_FRAME_CHANNEL_COUNT,
            kernel_size=1,
            attention=False,
            augmentation=None,
        ),
        "v4_enhanced_tcn_cv": models.build_config(
            "v4_enhanced_tcn_cv",
            feature_mode="enhanced",
            architecture="compact_causal_tcn",
            input_dim=models.ENHANCED_FRAME_CHANNEL_COUNT,
            kernel_size=3,
            attention=False,
            augmentation=augmentation,
        ),
        "v4_enhanced_attention_cv": models.build_config(
            "v4_enhanced_attention_cv",
            feature_mode="enhanced",
            architecture="compact_causal_tcn_temporal_attention",
            input_dim=models.ENHANCED_FRAME_CHANNEL_COUNT,
            kernel_size=3,
            attention=True,
            augmentation=augmentation,
        ),
    }


def _fit_fold_v4(
    name: str,
    config: Mapping[str, Any],
    tensor: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    fold_number: int,
    base_preparer: Any,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    prepared, real = models.extract_inputs(tensor, config["feature_mode"], base_preparer)
    inner_train, inner_val = train_test_split(
        train_index,
        test_size=0.2,
        stratify=y[train_index],
        random_state=SEED + fold_number,
    )
    inner_config = deepcopy(dict(config))
    inner_config["seed"] = SEED + fold_number
    inner_model = models.TemporalNetwork(inner_config)
    inner_history = inner_model.fit(
        prepared[inner_train],
        y[inner_train],
        inner_config,
        real[inner_train],
        prepared[inner_val],
        real[inner_val],
        y[inner_val],
        preextracted=True,
    )
    best_epoch = max(1, int(inner_history["best_epoch"] or 1))
    outer_config = deepcopy(dict(config))
    outer_config["seed"] = SEED + fold_number
    outer_config["max_epochs"] = best_epoch
    outer_model = models.TemporalNetwork(outer_config)
    outer_model.fit(
        prepared[train_index],
        y[train_index],
        outer_config,
        real[train_index],
        preextracted=True,
    )
    logits = outer_model.predict_logits(prepared[val_index], real[val_index])
    probabilities = calibration.softmax(logits, 1.0)
    fold_info = {
        "fold": fold_number,
        "variant": name,
        "train_count": int(len(train_index)),
        "val_count": int(len(val_index)),
        "inner_best_epoch": best_epoch,
        "inner_best_macro_f1": inner_history["best_validation_macro_f1"],
        "inner_epochs_completed": inner_history["epochs_completed"],
    }
    return outer_model, fold_info, logits, probabilities


def run_v4_cv(
    name: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    base_preparer: Any,
) -> dict[str, Any]:
    feature_mode = config["feature_mode"]
    tensor = data["base_tensor"] if feature_mode == "base" else data["enhanced_tensor"]
    y = data["y"]
    oof_logits = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
    oof_probabilities = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
    fold_info: list[dict[str, Any]] = []
    for fold_number, (train_index, val_index) in enumerate(folds, start=1):
        _, info, logits, probabilities = _fit_fold_v4(
            name,
            config,
            tensor,
            y,
            train_index,
            val_index,
            fold_number,
            base_preparer,
        )
        oof_logits[val_index] = logits
        oof_probabilities[val_index] = probabilities
        fold_info.append(info)
    if not np.isfinite(oof_logits).all() or not np.isfinite(oof_probabilities).all():
        raise ValueError("V4 OOF predictions are incomplete")
    predictions = np.argmax(oof_probabilities, axis=1).astype(np.int64)
    calibrated = calibration.calibration_report(y, oof_logits, min_coverage=0.90)
    return {
        "kind": "v4",
        "variant": name,
        "config": deepcopy(dict(config)),
        "oof_logits": oof_logits,
        "oof_probabilities": oof_probabilities,
        "oof_predictions": predictions,
        "metrics": v3_model.metrics(y, predictions),
        "classification_report": v3_cv.classification_report_dict(y, predictions, data["target_names"]),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=list(range(CLASS_COUNT))
        ).tolist(),
        "calibration": calibrated,
        "folds": fold_info,
    }


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": result["kind"],
        "variant": result.get("variant", result["config"].get("variant")),
        "metrics": result["metrics"],
        "calibration": calibration.summary(result["calibration"]),
        "folds": result["folds"],
        "config": result["config"],
    }


def _worst_class_f1(report: Mapping[str, Any], target_names: Sequence[str]) -> float:
    return min(float(report[str(name)]["f1-score"]) for name in target_names)


def selection_score(
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
    target_names: Sequence[str],
) -> dict[str, float]:
    baseline_worst = _worst_class_f1(baseline["classification_report"], target_names)
    worst = _worst_class_f1(result["classification_report"], target_names)
    regression = max(0.0, baseline_worst - worst)
    macro_f1 = float(result["metrics"]["macro_f1"])
    ece_after = float(result["calibration"]["temperature"]["ece_after"])
    score = macro_f1 - SELECTION_REGRESSION_PENALTY * regression - 0.01 * ece_after
    return {
        "score": float(score),
        "macro_f1": macro_f1,
        "worst_class_f1": worst,
        "baseline_worst_class_f1": baseline_worst,
        "worst_class_regression": float(regression),
        "ece_after": ece_after,
    }


def select_variant(
    results: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    target_names: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    scored = {
        name: selection_score(result, baseline, target_names)
        for name, result in results.items()
    }
    chosen = sorted(
        scored,
        key=lambda name: (
            scored[name]["score"],
            scored[name]["macro_f1"],
            -scored[name]["ece_after"],
        ),
        reverse=True,
    )[0]
    return chosen, scored


def _train_final_v4(
    name: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    epochs: int,
    base_preparer: Any,
) -> tuple[Any, dict[str, Any]]:
    final_config = deepcopy(dict(config))
    final_config["seed"] = SEED
    final_config["max_epochs"] = max(1, int(epochs))
    tensor = data["base_tensor"] if final_config["feature_mode"] == "base" else data["enhanced_tensor"]
    prepared, real = models.extract_inputs(tensor, final_config["feature_mode"], base_preparer)
    model_instance = models.TemporalNetwork(final_config)
    history = model_instance.fit(
        prepared,
        data["y"],
        final_config,
        real,
        preextracted=True,
    )
    return model_instance, {
        "kind": "v4",
        "variant": name,
        "feature_mode": final_config["feature_mode"],
        "epochs": int(epochs),
        "config": final_config,
        "parameter_count": model_instance.parameter_count(),
        "history": history,
    }


def _train_final_v3(
    data: Mapping[str, Any],
    epochs: int,
    base_preparer: Any,
) -> tuple[Any, dict[str, Any]]:
    config = build_v3_config()
    config["seed"] = SEED
    config["max_epochs"] = max(1, int(epochs))
    prepared, real = base_preparer.prepare_inputs(data["v3_tensor"])
    model_instance = v3_model.RegularizedTemporalMLP(config)
    history = model_instance.fit_regularized(prepared, real, data["y"], config)
    return model_instance, {
        "kind": "v3",
        "variant": V3_VARIANT,
        "feature_mode": "base",
        "epochs": int(epochs),
        "config": config,
        "parameter_count": sum(int(value.size) for value in model_instance.weights.values()),
        "history": history,
    }


def _per_class_csv(
    path: Path,
    baseline: Mapping[str, Any],
    selected: Mapping[str, Any],
    target_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_index",
                "class_label",
                "v3_common_cv_f1",
                "v3_common_cv_support",
                "selected_cv_f1",
                "selected_cv_support",
                "cv_delta_f1",
            ]
        )
        for index, name in enumerate(target_names):
            before_row = baseline["classification_report"][str(name)]
            after_row = selected["classification_report"][str(name)]
            before = float(before_row["f1-score"])
            after = float(after_row["f1-score"])
            writer.writerow(
                [
                    index,
                    name,
                    before,
                    int(before_row["support"]),
                    after,
                    int(after_row["support"]),
                    after - before,
                ]
            )


def _quality_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "source_fps": sorted({float(row["source_fps"]) for row in rows}),
        "duplicate_source_frames": int(sum(int(row["duplicate_source_frames"]) for row in rows)),
        "resized_1920x1080_to_1280x720": int(
            sum(
                1
                for row in rows
                if int(row["source_frame_count"]) > 0
                and row["resize_interpolation"] == "cv2.INTER_AREA"
            )
        ),
        "tracking_lifecycle": sorted({str(row["tracking_lifecycle"]) for row in rows}),
        "no_hand_frames": int(sum(int(row["no_hand_frames"]) for row in rows)),
        "multiple_hand_frames": int(sum(int(row["multiple_hand_frames"]) for row in rows)),
        "handedness_mismatch_frames": int(sum(int(row["handedness_mismatch_frames"]) for row in rows)),
        "source_frame_range": [
            min(int(row["source_frame_count"]) for row in rows),
            max(int(row["source_frame_count"]) for row in rows),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    data = load_data(args)
    base_preparer = v3_model.RegularizedTemporalMLP(v3_model.build_config("v4_base_preparer", label_smoothing=0.0, class_weight_power=0.0))
    folds = make_folds(data["y"], data["groups"], args.folds)
    common_data = dict(data)
    common_data["base_model"] = base_preparer
    v3_result = v3_cv.run_v3_cv(build_v3_config(), common_data, folds) if hasattr(v3_cv, "run_v3_cv") else None
    if v3_result is None:
        prepared, real = base_preparer.prepare_inputs(data["v3_tensor"])
        y = data["y"]
        oof_logits = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
        oof_probabilities = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
        fold_info: list[dict[str, Any]] = []
        for fold_number, (train_index, val_index) in enumerate(folds, start=1):
            _, info, logits, probabilities = v3_cv._fit_fold(
                build_v3_config(),
                prepared,
                real,
                y,
                train_index,
                val_index,
                fold_number,
            )
            oof_logits[val_index] = logits
            oof_probabilities[val_index] = probabilities
            fold_info.append(info)
        predictions = np.argmax(oof_probabilities, axis=1).astype(np.int64)
        v3_result = {
            "kind": "v3",
            "variant": V3_VARIANT,
            "config": build_v3_config(),
            "oof_logits": oof_logits,
            "oof_probabilities": oof_probabilities,
            "oof_predictions": predictions,
            "metrics": v3_model.metrics(y, predictions),
            "classification_report": v3_cv.classification_report_dict(y, predictions, data["target_names"]),
            "confusion_matrix": __import__("sklearn.metrics", fromlist=["confusion_matrix"])
            .confusion_matrix(y, predictions, labels=list(range(CLASS_COUNT)))
            .tolist(),
            "calibration": calibration.calibration_report(y, oof_logits, min_coverage=0.90),
            "folds": fold_info,
        }
    results: dict[str, dict[str, Any]] = {V3_VARIANT: v3_result}
    v4_configs = build_v4_configs()
    for name, config in v4_configs.items():
        results[name] = run_v4_cv(name, config, data, folds, base_preparer)
    selected_name, scores = select_variant(results, v3_result, data["target_names"])
    selected = results[selected_name]
    epochs = int(np.median([int(item["inner_best_epoch"]) for item in selected["folds"]]))
    if selected["kind"] == "v3":
        final_model, final_info = _train_final_v3(data, epochs, base_preparer)
    else:
        final_model, final_info = _train_final_v4(
            selected_name,
            v4_configs[selected_name],
            data,
            epochs,
            base_preparer,
        )

    model_root = args.model_root.resolve()
    report_root = args.report_root.resolve()
    model_dir = model_root / "selected"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    weight_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in final_model.weights.items()}
    np.savez_compressed(model_dir / "weights.npz", **weight_arrays)
    _write_json(model_dir / "config.json", final_info)

    np.savez_compressed(
        report_root / "oof_predictions.npz",
        labels=data["y"],
        groups=data["groups"].astype(str),
        **{f"{name}_logits": result["oof_logits"] for name, result in results.items()},
        **{f"{name}_probabilities": result["oof_probabilities"] for name, result in results.items()},
    )
    calibration_payload = {
        key: value for key, value in selected["calibration"].items() if key != "probabilities"
    }
    _write_json(report_root / "calibration.json", calibration_payload)
    _write_json(
        report_root / "cv_results.json",
        {
            "folds": [
                {"fold": index, "train_indices": train.tolist(), "val_indices": val.tolist()}
                for index, (train, val) in enumerate(folds, start=1)
            ],
            "variants": {name: _compact_result(result) for name, result in results.items()},
            "selection_scores": scores,
        },
    )
    _write_json(
        report_root / "classification_reports.json",
        {name: result["classification_report"] for name, result in results.items()},
    )
    _write_json(
        report_root / "confusion_matrices.json",
        {name: result["confusion_matrix"] for name, result in results.items()},
    )
    _per_class_csv(
        report_root / "per_class_before_after.csv",
        v3_result,
        selected,
        data["target_names"],
    )
    parity_evidence = parity.write_parity_evidence(
        report_root / "parity_evidence.json",
        data["v4_extraction_root"],
    )
    input_hashes = {
        "v2_features/mlp_tensor.npy": sha256_file(data["v2"]["features_root"] / "mlp_tensor.npy"),
        "v2_features/mlp_preprocessing.npz": sha256_file(data["v2"]["features_root"] / "mlp_preprocessing.npz"),
        "v2_features/mlp_channel_names.json": sha256_file(data["v2"]["features_root"] / "mlp_channel_names.json"),
        "v2_features/video_manifest.csv": sha256_file(data["v2"]["features_root"] / "video_manifest.csv"),
        "v4_features/base_tensor.npy": sha256_file(data["v4_features_root"] / "base_tensor.npy"),
        "v4_features/enhanced_tensor.npy": sha256_file(data["v4_features_root"] / "enhanced_tensor.npy"),
        "v4_features/labels.npy": sha256_file(data["v4_features_root"] / "labels.npy"),
        "v4_features/feature_schema.json": sha256_file(data["v4_features_root"] / "feature_schema.json"),
        "v4_features/video_manifest.csv": sha256_file(data["v4_features_root"] / "video_manifest.csv"),
        "v4_features/feature_build_summary.json": sha256_file(data["v4_features_root"] / "feature_build_summary.json"),
        "v4_extraction/config.json": sha256_file(data["v4_extraction_root"] / "config.json"),
        "v4_extraction/manifest.csv": sha256_file(data["v4_extraction_root"] / "manifest.csv"),
        "v1_experiment_config": sha256_file(args.v1_config),
    }
    output_hashes = {
        "selected/weights.npz": sha256_file(model_dir / "weights.npz"),
        "selected/config.json": sha256_file(model_dir / "config.json"),
        "calibration.json": sha256_file(report_root / "calibration.json"),
        "selection.json": None,
    }
    selection = {
        "experiment_id": EXPERIMENT_ID,
        "selection_metric": "oof_macro_f1_minus_worst_class_regression_penalty_minus_ece",
        "selection_data": "333 common train+val samples shared by V3/V4; no test split",
        "groups": "video_id; signer identity unavailable",
        "selected_variant": selected_name,
        "selected_kind": selected["kind"],
        "selected_metrics": selected["metrics"],
        "v3_common_cv_metrics": v3_result["metrics"],
        "selection_scores": scores,
        "selection_uses_test": False,
        "final_training_samples": int(len(data["y"])),
        "final_epochs": epochs,
        "final_parameter_count": final_info["parameter_count"],
    }
    _write_json(report_root / "selection.json", selection)
    training_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_split_read": False,
        "cv_protocol": "StratifiedGroupKFold(video_id) over 333 common train+val samples; inner split for early stopping",
        "cv_folds": len(folds),
        "split_counts": data["split_counts"],
        "candidates": [V3_VARIANT, *V4_VARIANTS],
        "selected_variant": selected_name,
        "selected_kind": selected["kind"],
        "selected_config": final_info["config"],
        "final_epochs": epochs,
        "fit_seconds": round(time.perf_counter() - started, 3),
        "quality_summary": _quality_summary(data["quality"]),
        "parity_passed": bool(parity_evidence["passed"]),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "reproduction_command": [
            sys.executable,
            "-m",
            "ml.src.experiments.rf_v4_live_alignment.train_cv",
            "--v2-features-root",
            str(data["v2"]["features_root"]),
            "--v2-extraction-root",
            str(data["v2"]["extraction_root"]),
            "--v4-features-root",
            str(data["v4_features_root"]),
            "--v4-extraction-root",
            str(data["v4_extraction_root"]),
            "--model-root",
            str(model_root),
            "--report-root",
            str(report_root),
            "--v1-config",
            str(args.v1_config),
        ],
    }
    output_hashes["selection.json"] = sha256_file(report_root / "selection.json")
    training_manifest["output_hashes"] = output_hashes
    _write_json(report_root / "training_manifest.json", training_manifest)
    print(
        json.dumps(
            {
                "selected_variant": selected_name,
                "selected_kind": selected["kind"],
                "macro_f1": selected["metrics"]["macro_f1"],
                "v3_macro_f1": v3_result["metrics"]["macro_f1"],
                "parity_passed": parity_evidence["passed"],
                "fit_seconds": training_manifest["fit_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
