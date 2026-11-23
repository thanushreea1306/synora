from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from ml.src import prepare_video_features as feature_source
    from ml.src import train_baselines_v1 as v1
    from ml.src.experiments.rf_v2_720p30 import dataset
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    import train_baselines_v1 as v1
    from experiments.rf_v2_720p30 import dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v2_720p30"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v2_720p30"
DEFAULT_V1_MODEL = REPO_ROOT / "ml/models/synora_baseline_v1_20261121/random_forest/random_forest.joblib"
DEFAULT_V1_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
DEFAULT_V1_METRICS = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/validation_metrics.json"
SEED = 20261121
CLASS_COUNT = 21
MIN_THRESHOLD_COVERAGE = 0.90


class ValidatedTemporalMLP(v1.TemporalMLP):
    def fit_validated(self, features, real, y, config, val_features, val_real, y_val):
        self.initialize(int(config["seed"]))
        rng = np.random.default_rng(int(config["seed"]))
        first_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        second_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        step = 0
        best_score = -np.inf
        best_epoch = 0
        best_weights = None
        history = []
        n_samples = len(y)
        batch_size = int(config["batch_size"])
        lr = float(config["learning_rate"])
        beta1 = float(config["beta1"])
        beta2 = float(config["beta2"])
        epsilon = float(config["epsilon"])
        weight_decay = float(config["weight_decay"])
        patience = int(config["early_stopping_patience"])
        min_delta = float(config["min_delta"])
        stale = 0
        for epoch in range(1, int(config["max_epochs"]) + 1):
            order = rng.permutation(n_samples)
            total_loss = 0.0
            total_batches = 0
            for start in range(0, n_samples, batch_size):
                batch_index = order[start : start + batch_size]
                logits, cache = self.forward(features[batch_index], real[batch_index])
                shifted = logits - logits.max(axis=1, keepdims=True)
                exponent = np.exp(shifted)
                probabilities = exponent / exponent.sum(axis=1, keepdims=True)
                target = np.eye(CLASS_COUNT, dtype=np.float32)[y[batch_index]]
                data_loss = -np.log(np.clip(probabilities[target.astype(bool)], 1e-12, 1.0)).mean()
                regularization = sum(
                    np.sum(value * value) for key, value in self.weights.items() if key.startswith("W")
                )
                loss = data_loss + weight_decay * regularization
                gradients = self.backward(
                    cache,
                    ((probabilities - target) / len(batch_index)).astype(np.float32),
                    weight_decay,
                )
                step += 1
                correction1 = 1.0 - beta1**step
                correction2 = 1.0 - beta2**step
                for key in self.weights:
                    first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradients[key]
                    second_moment[key] = beta2 * second_moment[key] + (1.0 - beta2) * (
                        gradients[key] * gradients[key]
                    )
                    adjusted_first = first_moment[key] / correction1
                    adjusted_second = second_moment[key] / correction2
                    self.weights[key] -= (
                        lr * adjusted_first / (np.sqrt(adjusted_second) + epsilon)
                    ).astype(np.float32)
                total_loss += float(loss)
                total_batches += 1
            val_prediction = self.predict(val_features, val_real)
            val_metrics = compute_metrics(y_val, val_prediction)
            score = val_metrics["macro_f1"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(total_loss / max(total_batches, 1)),
                    "val_macro_f1": float(score),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                }
            )
            if score > best_score + min_delta:
                best_score = score
                best_epoch = epoch
                best_weights = deepcopy(self.weights)
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.weights = best_weights
        return {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_score),
            "epochs_completed": len(history),
            "stopped_early": len(history) < int(config["max_epochs"]),
            "early_stopping_data": "val",
            "history": history,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--extraction-root", type=Path, default=DEFAULT_EXTRACTION_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--v1-model", type=Path, default=DEFAULT_V1_MODEL)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    parser.add_argument("--v1-metrics", type=Path, default=DEFAULT_V1_METRICS)
    return parser.parse_args(argv)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def json_report(y_true, y_pred, target_names) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    converted: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            converted[key] = {subkey: float(subvalue) for subkey, subvalue in value.items()}
        else:
            converted[key] = float(value)
    return converted


def confidence_summary(y_true, proba) -> dict[str, Any]:
    max_confidence = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = predictions == y_true
    result = {
        "count": int(len(max_confidence)),
        "min": float(np.min(max_confidence)),
        "max": float(np.max(max_confidence)),
        "mean": float(np.mean(max_confidence)),
        "median": float(np.median(max_confidence)),
        "quantiles": {
            str(q): float(np.quantile(max_confidence, q))
            for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
        "correct_mean": float(np.mean(max_confidence[correct])) if correct.any() else None,
        "incorrect_mean": float(np.mean(max_confidence[~correct])) if (~correct).any() else None,
    }
    return result


def threshold_sweep(y_true, proba) -> dict[str, Any]:
    baseline = compute_metrics(y_true, np.argmax(proba, axis=1))
    predictions = np.argmax(proba, axis=1)
    max_confidence = np.max(proba, axis=1)
    rows = []
    for threshold in np.round(np.arange(0.0, 0.951, 0.01), 2):
        accepted = max_confidence >= threshold
        coverage = float(accepted.mean())
        accepted_predictions = np.where(accepted, predictions, -1)
        row = {
            "threshold": float(threshold),
            "coverage": coverage,
            "accepted_count": int(accepted.sum()),
            "rejected_count": int((~accepted).sum()),
            "accepted_accuracy": float(accuracy_score(y_true[accepted], predictions[accepted]))
            if accepted.any()
            else 0.0,
            "accepted_macro_f1": float(
                f1_score(
                    y_true[accepted],
                    predictions[accepted],
                    labels=list(range(CLASS_COUNT)),
                    average="macro",
                    zero_division=0,
                )
            )
            if accepted.any()
            else 0.0,
            "macro_f1_with_rejection": float(
                f1_score(
                    y_true,
                    accepted_predictions,
                    labels=list(range(CLASS_COUNT)),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        rows.append(row)
    eligible = [row for row in rows if row["coverage"] >= MIN_THRESHOLD_COVERAGE]
    improved = [row for row in eligible if row["accepted_accuracy"] > baseline["accuracy"]]
    if improved:
        chosen = sorted(
            improved,
            key=lambda row: (row["accepted_accuracy"], row["coverage"], -row["threshold"]),
            reverse=True,
        )[0]
    else:
        chosen = next(row for row in rows if row["threshold"] == 0.0)
    return {
        "selection_metric": "accepted_accuracy",
        "selection_rule": (
            "choose the highest accepted accuracy among thresholds with at least 90% validation "
            "coverage and accuracy above the unthresholded model; otherwise use 0.0"
        ),
        "min_coverage": MIN_THRESHOLD_COVERAGE,
        "baseline": baseline,
        "chosen": chosen,
        "candidates": rows,
    }


def load_v2_data(args: argparse.Namespace) -> dict[str, Any]:
    features_root = args.features_root.resolve()
    extraction_root = args.extraction_root.resolve()
    extraction_config = dataset.read_json(extraction_root / "config.json")
    feature_summary = dataset.read_json(features_root / "feature_build_summary.json")
    schema = dataset.read_json(features_root / "rf_feature_schema.json")
    channels = dataset.read_json(features_root / "mlp_channel_names.json")
    v1_config = dataset.read_json(args.v1_config)
    manifest = dataset.read_csv(features_root / "video_manifest.csv")
    if len(manifest) != 346:
        raise ValueError(f"unexpected v2 manifest count: {len(manifest)}")
    if feature_summary["total_samples"] != len(manifest):
        raise ValueError("feature summary count mismatch")
    if extraction_config["split_policy"]["test_split_read"] is not False:
        raise ValueError("extraction config does not record test_split_read correctly")
    if extraction_config["split_policy"]["test_split_read"]:
        raise ValueError("v2 extraction read the test split")
    if len(channels) != 135 or channels != v1.TemporalMLP.expected_channels():
        raise ValueError("MLP channel order mismatch")
    if list(schema["feature_columns"]) != list(feature_source.base_feature_names()):
        raise ValueError("RF feature schema mismatch")
    if schema["v1_schema_sha256"] != dataset.sha256_file(
        REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json"
    ):
        raise ValueError("v1 feature schema hash mismatch")

    mapping = {int(record["original_label_id"]): int(record["model_index"]) for record in v1_config["label_mapping"]}
    target_names = [str(record["class_label"]) for record in v1_config["label_mapping"]]
    split_indices = {
        split: np.asarray([index for index, row in enumerate(manifest) if row["split"] == split], dtype=np.int64)
        for split in dataset.SPLITS
    }
    expected_counts = {"train": 305, "val": 41}
    for split in dataset.SPLITS:
        if len(split_indices[split]) != expected_counts[split]:
            raise ValueError(f"unexpected {split} count: {len(split_indices[split])}")
    train_video_ids = {manifest[i]["video_id"] for i in split_indices["train"]}
    val_video_ids = {manifest[i]["video_id"] for i in split_indices["val"]}
    if train_video_ids & val_video_ids:
        raise ValueError("train/val video overlap")

    y = np.asarray([mapping[int(row["label_id"])] for row in manifest], dtype=np.int64)
    rf = np.load(features_root / "rf_features.npy")
    labels = np.load(features_root / "labels.npy")
    tensor = np.load(features_root / "mlp_tensor.npy")
    flat = np.load(features_root / "mlp_flat.npy")
    if rf.shape != (346, 831) or labels.shape != (346,) or tensor.shape != (346, 89, 135) or flat.shape != (346, 12015):
        raise ValueError("unexpected v2 matrix shapes")
    raw_label_ids = np.asarray([int(row["label_id"]) for row in manifest], dtype=np.int64)
    if not np.array_equal(labels, raw_label_ids):
        raise ValueError("labels.npy does not match manifest label IDs")
    if not np.array_equal(flat, tensor.reshape(346, -1)):
        raise ValueError("mlp_flat.npy is not a reshape of mlp_tensor.npy")
    if not np.isfinite(rf).all() or not np.isfinite(tensor).all():
        raise ValueError("non-finite v2 features")

    with np.load(features_root / "mlp_preprocessing.npz") as preprocessing:
        mean = preprocessing["mean"].copy()
        std = preprocessing["std"].copy()
    if mean.shape != (2, 21, 3) or std.shape != (2, 21, 3):
        raise ValueError("v2 preprocessing shape mismatch")
    return {
        "features_root": features_root,
        "extraction_root": extraction_root,
        "manifest": manifest,
        "split_indices": split_indices,
        "y": y,
        "rf": rf,
        "tensor": tensor,
        "mean": mean,
        "std": std,
        "channel_names": channels,
        "feature_names": list(schema["feature_columns"]),
        "label_mapping": v1_config["label_mapping"],
        "target_names": target_names,
        "v1_config": v1_config,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_v2_data(args)
    train_idx = data["split_indices"]["train"]
    val_idx = data["split_indices"]["val"]
    y = data["y"]
    rf = data["rf"]
    target_names = data["target_names"]

    model_root = args.model_root.resolve()
    report_root = args.report_root.resolve()
    rf_model_dir = model_root / "random_forest"
    mlp_model_dir = model_root / "temporal_mlp"
    for path in (rf_model_dir, mlp_model_dir, report_root):
        path.mkdir(parents=True, exist_ok=True)

    rf_config = deepcopy(v1.RF_CONFIG)
    rf_estimator_config = {key: value for key, value in rf_config.items() if key not in {"estimator", "scaling"}}
    started = time.perf_counter()
    rf_model = RandomForestClassifier(**rf_estimator_config)
    rf_model.fit(rf[train_idx], y[train_idx])
    rf_fit_seconds = time.perf_counter() - started
    rf_proba_val = rf_model.predict_proba(rf[val_idx])
    rf_pred_val = np.argmax(rf_proba_val, axis=1).astype(np.int64)
    joblib.dump(rf_model, rf_model_dir / "random_forest.joblib")
    dataset.write_json(rf_model_dir / "config.json", rf_config)

    v1_rf_model = joblib.load(args.v1_model)
    v1_rf_proba_val = v1_rf_model.predict_proba(rf[val_idx])
    v1_rf_pred_val = np.argmax(v1_rf_proba_val, axis=1).astype(np.int64)
    v1_rf_metrics_v2_val = compute_metrics(y[val_idx], v1_rf_pred_val)
    v1_frozen_metrics = dataset.read_json(args.v1_metrics)["metrics"]

    mlp_config = deepcopy(v1.MLP_CONFIG)
    mlp_config["input_shape"] = [len(train_idx), 89, 135]
    mlp_config["training_input_shape"] = [len(train_idx), 89, 135]
    mlp_config["frame_input_shape"] = [89, 135]
    mlp_config["preprocessing"] = {
        "source": "v2 mlp_preprocessing.npz",
        "fit_splits": ["train"],
        "refit": True,
        "scaled_channels": "observed normalized landmark coordinates only",
    }
    mlp_config["early_stopping_data"] = "val"
    temporal_model = ValidatedTemporalMLP(mlp_config)
    prepared_features, real_mask = temporal_model.prepare_inputs(data["tensor"])
    started = time.perf_counter()
    mlp_history = temporal_model.fit_validated(
        prepared_features[train_idx],
        real_mask[train_idx],
        y[train_idx],
        mlp_config,
        prepared_features[val_idx],
        real_mask[val_idx],
        y[val_idx],
    )
    mlp_fit_seconds = time.perf_counter() - started
    mlp_proba_val = temporal_model.predict_proba(prepared_features[val_idx], real_mask[val_idx])
    mlp_pred_val = np.argmax(mlp_proba_val, axis=1).astype(np.int64)
    weight_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in temporal_model.weights.items()}
    np.savez_compressed(mlp_model_dir / "weights.npz", **weight_arrays)
    dataset.write_json(mlp_model_dir / "config.json", mlp_config)
    dataset.write_json(mlp_model_dir / "training_history.json", mlp_history)
    dataset.write_json(
        mlp_model_dir / "preprocessing_reference.json",
        {
            "path": str(data["features_root"] / "mlp_preprocessing.npz"),
            "sha256": dataset.sha256_file(data["features_root"] / "mlp_preprocessing.npz"),
            "fit_splits": ["train"],
            "used_without_modification": True,
        },
    )

    predictions = {
        "frozen_v1_rf_on_v2_val": v1_rf_pred_val,
        "rf_v2": rf_pred_val,
        "temporal_mlp_v2": mlp_pred_val,
    }
    probabilities = {
        "frozen_v1_rf_on_v2_val": v1_rf_proba_val,
        "rf_v2": rf_proba_val,
        "temporal_mlp_v2": mlp_proba_val,
    }
    metrics = {name: compute_metrics(y[val_idx], pred) for name, pred in predictions.items()}
    candidates = ("rf_v2", "temporal_mlp_v2")
    selected_model = max(candidates, key=lambda name: metrics[name]["macro_f1"])
    selected_proba = probabilities[selected_model]
    threshold_policy = threshold_sweep(y[val_idx], selected_proba)

    classification_reports = {
        name: json_report(y[val_idx], pred, target_names) for name, pred in predictions.items()
    }
    confusion_matrices = {
        name: confusion_matrix(y[val_idx], pred, labels=list(range(CLASS_COUNT))).tolist()
        for name, pred in predictions.items()
    }
    confidence = {name: confidence_summary(y[val_idx], proba) for name, proba in probabilities.items()}

    input_hashes = {
        "features/rf_features.npy": dataset.sha256_file(data["features_root"] / "rf_features.npy"),
        "features/labels.npy": dataset.sha256_file(data["features_root"] / "labels.npy"),
        "features/mlp_tensor.npy": dataset.sha256_file(data["features_root"] / "mlp_tensor.npy"),
        "features/mlp_preprocessing.npz": dataset.sha256_file(data["features_root"] / "mlp_preprocessing.npz"),
        "features/rf_feature_schema.json": dataset.sha256_file(data["features_root"] / "rf_feature_schema.json"),
        "features/video_manifest.csv": dataset.sha256_file(data["features_root"] / "video_manifest.csv"),
        "extraction/config.json": dataset.sha256_file(data["extraction_root"] / "config.json"),
        "v1_rf_model": dataset.sha256_file(args.v1_model),
        "v1_experiment_config": dataset.sha256_file(args.v1_config),
        "v1_validation_metrics": dataset.sha256_file(args.v1_metrics),
    }
    output_hashes = {
        "random_forest/random_forest.joblib": dataset.sha256_file(rf_model_dir / "random_forest.joblib"),
        "random_forest/config.json": dataset.sha256_file(rf_model_dir / "config.json"),
        "temporal_mlp/weights.npz": dataset.sha256_file(mlp_model_dir / "weights.npz"),
        "temporal_mlp/config.json": dataset.sha256_file(mlp_model_dir / "config.json"),
        "temporal_mlp/preprocessing_reference.json": dataset.sha256_file(
            mlp_model_dir / "preprocessing_reference.json"
        ),
    }
    training_manifest = {
        "experiment_id": "synora_rf_v2_720p30_20261122",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_split_read": False,
        "selection_split": "val",
        "split_counts": {"train": len(train_idx), "val": len(val_idx)},
        "rf_config": rf_config,
        "mlp_config": mlp_config,
        "fit_seconds": {"rf_v2": rf_fit_seconds, "temporal_mlp_v2": mlp_fit_seconds},
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numeric_mode": "float32 NumPy linear algebra; deterministic fixed-seed initialization; no dropout",
        "reproduction_command": [
            sys.executable,
            "-m",
            "ml.src.experiments.rf_v2_720p30.train_v2",
            "--features-root",
            str(data["features_root"]),
            "--extraction-root",
            str(data["extraction_root"]),
            "--model-root",
            str(model_root),
            "--report-root",
            str(report_root),
        ],
    }

    comparison = {
        "frozen_v1_on_original_v1_val": v1_frozen_metrics,
        "frozen_v1_on_v2_val": v1_rf_metrics_v2_val,
        "v2_models": {name: metrics[name] for name in candidates},
        "notes": [
            "v1 artifacts and test split were not modified or used.",
            "The 13 over-cap clips were excluded before v2 feature building and recorded in excluded.jsonl.",
            "All v2 model selection and confidence calibration used validation only.",
        ],
    }
    dataset.write_json(report_root / "validation_metrics.json", metrics)
    dataset.write_json(report_root / "comparison.json", comparison)
    dataset.write_json(report_root / "classification_reports.json", classification_reports)
    dataset.write_json(report_root / "confusion_matrices.json", confusion_matrices)
    dataset.write_json(report_root / "confidence_calibration.json", confidence)
    dataset.write_json(report_root / "confidence_threshold.json", threshold_policy)
    dataset.write_json(
        report_root / "model_selection.json",
        {
            "selection_metric": "macro_f1",
            "selection_split": "val",
            "selected_model": selected_model,
            "metrics": metrics,
            "selection_uses_test": False,
            "rejection_threshold": threshold_policy["chosen"]["threshold"],
        },
    )
    dataset.write_json(report_root / "training_manifest.json", training_manifest)

    lines = [
        "# SYNORA RF v2 720p30 Training Report",
        "",
        "- Experiment: `synora_rf_v2_720p30_20261122`",
        "- Test split read: no",
        "- Train samples: 305; validation samples: 41",
        "- Excluded over-cap clips: 13",
        "",
        "## Validation metrics",
        "",
        "| Model | Macro-F1 | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['macro_f1']:.6f} | {values['accuracy']:.6f} | {values['balanced_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen v1 reference",
            "",
            f"- Original v1 validation macro-F1: {v1_frozen_metrics['random_forest']['macro_f1']:.6f}",
            f"- Frozen v1 RF evaluated on v2 validation features: {v1_rf_metrics_v2_val['macro_f1']:.6f}",
            "",
            "## Selection",
            "",
            f"- Selected model: `{selected_model}`",
            f"- Rejection threshold: {threshold_policy['chosen']['threshold']:.2f}",
            f"- Threshold coverage: {threshold_policy['chosen']['coverage']:.6f}",
            f"- Threshold accepted accuracy: {threshold_policy['chosen']['accepted_accuracy']:.6f}",
            "",
            "## Weaknesses",
            "",
            "- Validation-set size is 41 after excluding one over-cap clip; class-wise estimates are unstable.",
            "- A single signer recorded at a close-range desk setup is not a signer-independent evaluation.",
            "- The frozen v1 model was exposed to v2 metadata values (`30 FPS`, `1280x720`) outside its original acquisition schema; its v2-feature score is diagnostic only.",
        ]
    )
    (report_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
