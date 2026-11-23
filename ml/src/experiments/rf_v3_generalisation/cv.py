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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v2_720p30 import train_v2
    from ml.src.experiments.rf_v3_generalisation import calibration, model
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v2_720p30 import train_v2
    from experiments.rf_v3_generalisation import calibration, model


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v3_generalisation"
DEFAULT_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v3_generalisation"
DEFAULT_V1_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
DEFAULT_V2_REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v2_720p30"
EXPERIMENT_ID = "synora_rf_v3_generalisation_20261122"
SEED = model.SEED
CLASS_COUNT = model.CLASS_COUNT
N_SPLITS = 5
MIN_CV_CLASS_SUPPORT = 2
VARIANTS = {
    "v2_baseline_cv": {"label_smoothing": 0.0, "class_weight_power": 0.0},
    "regularized_cv": {"label_smoothing": 0.05, "class_weight_power": 0.5},
    "strong_regularized_cv": {"label_smoothing": 0.10, "class_weight_power": 1.0},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--extraction-root", type=Path, default=DEFAULT_EXTRACTION_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    parser.add_argument("--v2-report-root", type=Path, default=DEFAULT_V2_REPORT_ROOT)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(args: argparse.Namespace) -> dict[str, Any]:
    loader_args = argparse.Namespace(
        features_root=args.features_root,
        extraction_root=args.extraction_root,
        v1_config=args.v1_config,
    )
    data = train_v2.load_v2_data(loader_args)
    manifest = data["manifest"]
    splits = {row["split"] for row in manifest}
    if splits != {"train", "val"}:
        raise ValueError(f"unexpected splits in train/val feature manifest: {sorted(splits)}")
    groups = np.asarray([row["video_id"] for row in manifest], dtype=object)
    if len(set(groups)) != len(groups):
        raise ValueError("video_id groups are not unique")
    data["groups"] = groups
    return data


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(y, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_index, val_index in splitter.split(np.zeros(len(labels)), labels, groups):
        train_groups = set(groups[train_index])
        val_groups = set(groups[val_index])
        if train_groups & val_groups:
            raise ValueError("CV fold groups overlap")
        support = np.bincount(labels[val_index], minlength=CLASS_COUNT)
        if support.min() < MIN_CV_CLASS_SUPPORT:
            raise ValueError(f"CV fold class support too small: {support.min()}")
        folds.append((np.asarray(train_index, dtype=np.int64), np.asarray(val_index, dtype=np.int64)))
    return folds


def _fit_fold(
    config: Mapping[str, Any],
    prepared: np.ndarray,
    real: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    fold_number: int,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    inner_train, inner_val = train_test_split(
        train_index,
        test_size=0.2,
        stratify=y[train_index],
        random_state=SEED + fold_number,
    )
    inner_config = deepcopy(dict(config))
    inner_config["seed"] = SEED + fold_number
    inner_model = model.RegularizedTemporalMLP(inner_config)
    inner_history = inner_model.fit_regularized(
        prepared[inner_train],
        real[inner_train],
        y[inner_train],
        inner_config,
        prepared[inner_val],
        real[inner_val],
        y[inner_val],
    )
    best_epoch = max(1, int(inner_history["best_epoch"] or 1))
    outer_config = deepcopy(dict(config))
    outer_config["seed"] = SEED + fold_number
    outer_config["max_epochs"] = best_epoch
    outer_model = model.RegularizedTemporalMLP(outer_config)
    outer_model.fit_regularized(prepared[train_index], real[train_index], y[train_index], outer_config)
    logits = outer_model.predict_logits(prepared[val_index], real[val_index])
    probabilities = calibration.softmax(logits, 1.0)
    fold_info = {
        "fold": fold_number,
        "train_count": int(len(train_index)),
        "val_count": int(len(val_index)),
        "inner_best_epoch": best_epoch,
        "inner_best_macro_f1": inner_history["best_validation_macro_f1"],
        "inner_epochs_completed": inner_history["epochs_completed"],
    }
    return outer_model, fold_info, logits, probabilities


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(CLASS_COUNT)),
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )
    return {key: {name: float(value) for name, value in row.items()} if isinstance(row, dict) else float(row) for key, row in report.items()}


def run_cv(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    prepared, real = data["base_model"].prepare_inputs(data["tensor"])
    y = data["y"]
    oof_logits = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
    oof_proba = np.full((len(y), CLASS_COUNT), np.nan, dtype=np.float64)
    fold_info: list[dict[str, Any]] = []
    for fold_number, (train_index, val_index) in enumerate(folds, start=1):
        _, info, logits, probabilities = _fit_fold(config, prepared, real, y, train_index, val_index, fold_number)
        oof_logits[val_index] = logits
        oof_proba[val_index] = probabilities
        fold_info.append(info)
    if not np.isfinite(oof_logits).all() or not np.isfinite(oof_proba).all():
        raise ValueError("OOF predictions are incomplete")
    predictions = np.argmax(oof_proba, axis=1).astype(np.int64)
    calibrated = calibration.calibration_report(y, oof_logits, min_coverage=0.90)
    return {
        "config": deepcopy(dict(config)),
        "oof_logits": oof_logits,
        "oof_probabilities": oof_proba,
        "oof_predictions": predictions,
        "metrics": model.metrics(y, predictions),
        "classification_report": classification_report_dict(y, predictions, data["target_names"]),
        "confusion_matrix": confusion_matrix(y, predictions, labels=list(range(CLASS_COUNT))).tolist(),
        "calibration": calibrated,
        "folds": fold_info,
    }


def compact_results(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": result["metrics"],
        "calibration": calibration.summary(result["calibration"]),
        "folds": result["folds"],
        "config": result["config"],
    }


def select_variant(results: Mapping[str, Mapping[str, Any]]) -> str:
    return sorted(
        results,
        key=lambda name: (
            results[name]["metrics"]["macro_f1"],
            results[name]["metrics"]["accuracy"],
            -results[name]["calibration"]["temperature"]["ece_after"],
        ),
        reverse=True,
    )[0]


def train_final_model(config: Mapping[str, Any], data: Mapping[str, Any], epochs: int) -> tuple[Any, dict[str, Any]]:
    final_config = deepcopy(dict(config))
    final_config["seed"] = SEED
    final_config["max_epochs"] = max(1, int(epochs))
    prepared, real = data["base_model"].prepare_inputs(data["tensor"])
    final_model = model.RegularizedTemporalMLP(final_config)
    history = final_model.fit_regularized(prepared, real, data["y"], final_config)
    return final_model, {"epochs": int(epochs), "history": history, "config": final_config}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_per_class_csv(path: Path, baseline: Mapping[str, Any], selected: Mapping[str, Any], original: Mapping[str, Any], target_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_index",
                "class_label",
                "v2_original_val_f1",
                "v2_original_val_support",
                "baseline_cv_f1",
                "baseline_cv_support",
                "improved_cv_f1",
                "improved_cv_support",
                "cv_delta_f1",
            ]
        )
        for index, name in enumerate(target_names):
            key = str(name)
            original_row = original.get(key, {})
            baseline_row = baseline.get(key, {})
            selected_row = selected.get(key, {})
            before = float(baseline_row.get("f1-score", 0.0))
            after = float(selected_row.get("f1-score", 0.0))
            writer.writerow(
                [
                    index,
                    name,
                    float(original_row.get("f1-score", 0.0)),
                    int(original_row.get("support", 0)),
                    before,
                    int(baseline_row.get("support", 0)),
                    after,
                    int(selected_row.get("support", 0)),
                    after - before,
                ]
            )


def _metrics_table(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| Variant | Macro-F1 | Accuracy | Balanced accuracy | ECE before | ECE after | Threshold |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        metrics = result["metrics"]
        calibration_summary = calibration.summary(result["calibration"])
        lines.append(
            f"| {name} | {metrics['macro_f1']:.6f} | {metrics['accuracy']:.6f} | "
            f"{metrics['balanced_accuracy']:.6f} | {calibration_summary['ece_before']:.6f} | "
            f"{calibration_summary['ece_after']:.6f} | {calibration_summary['threshold']:.2f} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    data = load_data(args)
    data["base_model"] = model.RegularizedTemporalMLP(build_config_for_base())
    folds = make_folds(data["y"], data["groups"], args.folds)
    results: dict[str, dict[str, Any]] = {}
    for name, settings in VARIANTS.items():
        config = model.build_config(name, **settings)
        results[name] = run_cv(config, data, folds)
    selected_name = select_variant(results)
    selected = results[selected_name]
    baseline = results["v2_baseline_cv"]
    epochs = int(np.median([int(item["inner_best_epoch"]) for item in selected["folds"]]))
    final_model, final_info = train_final_model(selected["config"], data, epochs)

    model_dir = args.model_root.resolve() / "temporal_mlp"
    report_dir = args.report_root.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    weight_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in final_model.weights.items()}
    np.savez_compressed(model_dir / "weights.npz", **weight_arrays)
    _write_json(model_dir / "config.json", final_info["config"])
    np.savez_compressed(
        report_dir / "oof_predictions.npz",
        labels=data["y"],
        groups=data["groups"].astype(str),
        **{f"{name}_logits": result["oof_logits"] for name, result in results.items()},
        **{f"{name}_probabilities": result["oof_probabilities"] for name, result in results.items()},
    )
    calibration_payload = {key: value for key, value in selected["calibration"].items() if key != "probabilities"}
    _write_json(report_dir / "calibration.json", calibration_payload)
    fold_payload = [
        {"fold": index, "train_indices": train.tolist(), "val_indices": val.tolist()}
        for index, (train, val) in enumerate(folds, start=1)
    ]
    _write_json(
        report_dir / "cv_results.json",
        {"folds": fold_payload, "variants": {name: compact_results(result) for name, result in results.items()}},
    )
    _write_json(
        report_dir / "classification_reports.json",
        {name: result["classification_report"] for name, result in results.items()},
    )
    _write_json(
        report_dir / "confusion_matrices.json",
        {name: result["confusion_matrix"] for name, result in results.items()},
    )
    original_classification = v2_dataset.read_json(args.v2_report_root / "classification_reports.json")
    original_validation = v2_dataset.read_json(args.v2_report_root / "validation_metrics.json")
    _write_per_class_csv(
        report_dir / "per_class_before_after.csv",
        baseline["classification_report"],
        selected["classification_report"],
        original_classification["temporal_mlp_v2"],
        data["target_names"],
    )
    selection = {
        "experiment_id": EXPERIMENT_ID,
        "selection_metric": "out_of_fold_macro_f1",
        "selection_data": "346 train+val samples; no test split",
        "groups": "video_id; signer identity unavailable",
        "selected_variant": selected_name,
        "selected_metrics": selected["metrics"],
        "baseline_metrics": baseline["metrics"],
        "v2_original_val_metrics": original_validation["temporal_mlp_v2"],
        "selection_uses_test": False,
        "final_training_samples": int(len(data["y"])),
        "final_epochs": epochs,
    }
    _write_json(report_dir / "selection.json", selection)
    input_hashes = {
        "features/rf_features.npy": sha256_file(data["features_root"] / "rf_features.npy"),
        "features/labels.npy": sha256_file(data["features_root"] / "labels.npy"),
        "features/mlp_tensor.npy": sha256_file(data["features_root"] / "mlp_tensor.npy"),
        "features/mlp_preprocessing.npz": sha256_file(data["features_root"] / "mlp_preprocessing.npz"),
        "features/rf_feature_schema.json": sha256_file(data["features_root"] / "rf_feature_schema.json"),
        "features/video_manifest.csv": sha256_file(data["features_root"] / "video_manifest.csv"),
        "extraction/config.json": sha256_file(data["extraction_root"] / "config.json"),
        "v1_experiment_config": sha256_file(args.v1_config),
    }
    output_hashes = {
        "temporal_mlp/weights.npz": sha256_file(model_dir / "weights.npz"),
        "temporal_mlp/config.json": sha256_file(model_dir / "config.json"),
        "calibration.json": sha256_file(report_dir / "calibration.json"),
        "selection.json": sha256_file(report_dir / "selection.json"),
    }
    training_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_split_read": False,
        "cv_protocol": "StratifiedGroupKFold(video_id) over train+val only",
        "cv_folds": len(folds),
        "split_counts": {"train": 305, "val": 41, "cv_total": 346},
        "candidates": list(VARIANTS),
        "selected_variant": selected_name,
        "selected_config": selected["config"],
        "final_epochs": epochs,
        "fit_seconds": round(time.perf_counter() - started, 3),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "reproduction_command": [
            sys.executable,
            "-m",
            "ml.src.experiments.rf_v3_generalisation.cv",
            "--features-root",
            str(data["features_root"]),
            "--extraction-root",
            str(data["extraction_root"]),
            "--model-root",
            str(args.model_root.resolve()),
            "--report-root",
            str(report_dir),
        ],
    }
    _write_json(report_dir / "training_manifest.json", training_manifest)
    calibration_summary = calibration.summary(selected["calibration"])
    lines = [
        "# SYNORA v3 Generalisation Report",
        "",
        "- Experiment: `synora_rf_v3_generalisation_20261122`",
        "- Test split read: no",
        "- Validation protocol: 5-fold stratified group CV over all 346 train+val clips, grouped by `video_id`",
        "- Signer identity: unavailable; this is not a signer-independent evaluation",
        "",
        "## Model comparison on the same CV protocol",
        "",
    ]
    lines.extend(_metrics_table(results))
    lines.extend(
        [
            "",
            "## Selected model",
            "",
            f"- Variant: `{selected_name}`",
            f"- OOF macro-F1: {selected['metrics']['macro_f1']:.6f}",
            f"- OOF accuracy: {selected['metrics']['accuracy']:.6f}",
            f"- Final training epochs: {epochs}",
            f"- Temperature: {calibration_summary['temperature']:.6f}",
            f"- Rejection threshold: {calibration_summary['threshold']:.2f}",
            f"- Accepted accuracy at threshold: {calibration_summary['accepted_accuracy']:.6f}",
            f"- Coverage at threshold: {calibration_summary['coverage']:.6f}",
            "",
            "## Limitations",
            "",
            "- The original v2 validation set has only 41 clips, 20 classes with 2 clips and narrow with 1.",
            "- CV improves sample support per class but cannot remove shared-signer or shared-setup bias because no signer IDs exist.",
            "- Offline extraction resamples 25 FPS sources to 30 FPS with duplicated source frames; live capture does not duplicate frames.",
            "- Raw confidence is not a calibrated probability; temperature scaling is fitted on out-of-fold train/val predictions only.",
            "- No webcam predictions, test-split metrics, or real-world accuracy claims are made.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selected={selected_name} macro_f1={selected['metrics']['macro_f1']:.6f}")
    return 0


def build_config_for_base() -> dict[str, Any]:
    return model.build_config("feature_loader_base", label_smoothing=0.0, class_weight_power=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
