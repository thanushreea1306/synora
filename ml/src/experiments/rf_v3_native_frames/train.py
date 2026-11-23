from __future__ import annotations

import argparse
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
N_SPLITS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-features-root", type=Path, default=config.DEFAULT_V2_FEATURES_ROOT)
    parser.add_argument("--v2-extraction-root", type=Path, default=config.DEFAULT_V2_EXTRACTION_ROOT)
    parser.add_argument("--v4-features-root", type=Path, default=config.DEFAULT_V4_FEATURES_ROOT)
    parser.add_argument("--v4-extraction-root", type=Path, default=config.DEFAULT_V4_EXTRACTION_ROOT)
    parser.add_argument("--v1-config", type=Path, default=config.DEFAULT_V1_EXPERIMENT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=config.DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=config.DEFAULT_REPORT_ROOT)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equivalence(data_root: Path) -> dict[str, Any]:
    report = v2_dataset.read_json(data_root / "equivalence.json")
    if report.get("passed") is not True:
        raise RuntimeError("exact tensor equivalence check did not pass")
    if report.get("existing_v4_trainval", {}).get("all_exact") is not True:
        raise RuntimeError("existing V4 train/val tensor is not exactly reproducible")
    return report


def build_config() -> dict[str, Any]:
    result = v3_model.build_config(
        config.VARIANT,
        label_smoothing=0.10,
        class_weight_power=1.0,
    )
    result["seed"] = config.SEED
    result["selection_data"] = "333 common train+val samples shared with V4; no test split"
    result["training_input"] = "V4 native base tensor; one detection per real source frame"
    result["preprocessing"] = {
        "source": "existing mlp_preprocessing.npz",
        "fit_splits": ["train"],
        "refit": False,
        "scaled_channels": "observed normalized landmark coordinates only",
    }
    return result


def load_training_data(args: argparse.Namespace) -> dict[str, Any]:
    _require_equivalence(args.data_root.resolve())
    loader_args = argparse.Namespace(
        features_root=args.v2_features_root,
        extraction_root=args.v2_extraction_root,
        v1_config=args.v1_config,
    )
    v2 = v3_cv.load_data(loader_args)
    v4_root = args.v4_features_root.resolve()
    v4_extraction_root = args.v4_extraction_root.resolve()
    extraction_config = v2_dataset.read_json(v4_extraction_root / "config.json")
    if extraction_config.get("split_policy", {}).get("test_split_read") is not False:
        raise ValueError("V4 extraction config does not certify no test split access")
    manifest = v2_dataset.read_csv(v4_root / "video_manifest.csv")
    base_tensor = np.load(v4_root / "base_tensor.npy")
    labels = np.load(v4_root / "labels.npy")
    if base_tensor.shape != (len(manifest), 89, 135) or labels.shape != (len(manifest),):
        raise ValueError("unexpected V4 feature shapes")
    if not np.isfinite(base_tensor).all():
        raise ValueError("non-finite V4 base tensor")
    v4_by_id = {row["video_id"]: index for index, row in enumerate(manifest)}
    v2_by_id = {row["video_id"]: index for index, row in enumerate(v2["manifest"])}
    common_ids = sorted(set(v4_by_id) & set(v2_by_id))
    if len(common_ids) != 333:
        raise ValueError(f"unexpected common train/val count: {len(common_ids)}")
    v4_indices = np.asarray([v4_by_id[item] for item in common_ids], dtype=np.int64)
    v2_indices = np.asarray([v2_by_id[item] for item in common_ids], dtype=np.int64)
    y = np.asarray([int(manifest[index]["model_index"]) for index in v4_indices], dtype=np.int64)
    expected_y = v2["y"][v2_indices]
    if not np.array_equal(y, expected_y) or not np.array_equal(labels[v4_indices], y):
        raise ValueError("V2/V4 labels differ on common train/val samples")
    native_tensor = base_tensor[v4_indices]
    resampled_tensor = v2["tensor"][v2_indices]
    if np.array_equal(native_tensor, resampled_tensor):
        raise ValueError("native and resampled training tensors are identical")
    split_counts = {
        split: sum(1 for index in v4_indices if manifest[index]["split"] == split)
        for split in v2_dataset.SPLITS
    }
    if split_counts != {"train": 293, "val": 40}:
        raise ValueError(f"unexpected common split counts: {split_counts}")
    return {
        "v2": v2,
        "native_tensor": native_tensor,
        "resampled_tensor": resampled_tensor,
        "y": y,
        "groups": np.asarray(common_ids, dtype=object),
        "manifest": [manifest[index] for index in v4_indices],
        "target_names": v2["target_names"],
        "split_counts": split_counts,
        "v4_root": v4_root,
        "v4_extraction_root": v4_extraction_root,
    }


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config": result["config"],
        "metrics": result["metrics"],
        "calibration": calibration.summary(result["calibration"]),
        "folds": result["folds"],
        "classification_report": result["classification_report"],
        "confusion_matrix": result["confusion_matrix"],
    }


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    data = load_training_data(args)
    base_preparer = v3_model.RegularizedTemporalMLP(
        v3_model.build_config("native_frames_preparer", label_smoothing=0.0, class_weight_power=0.0)
    )
    folds = v3_cv.make_folds(data["y"], data["groups"], args.folds)
    config_value = build_config()
    cv_data = {
        "tensor": data["native_tensor"],
        "base_model": base_preparer,
        "y": data["y"],
        "target_names": data["target_names"],
    }
    cv_result = v3_cv.run_cv(config_value, cv_data, folds)
    calibrated = calibration.summary(cv_result["calibration"])
    epochs = int(np.median([int(item["inner_best_epoch"]) for item in cv_result["folds"]]))
    final_config = deepcopy(config_value)
    final_config["seed"] = config.SEED
    final_config["max_epochs"] = max(1, epochs)
    prepared, real = base_preparer.prepare_inputs(data["native_tensor"])
    final_model = v3_model.RegularizedTemporalMLP(final_config)
    history = final_model.fit_regularized(prepared, real, data["y"], final_config)
    final_info = {
        "kind": "v3",
        "variant": config.VARIANT,
        "feature_mode": "base",
        "epochs": epochs,
        "parameter_count": sum(int(value.size) for value in final_model.weights.values()),
        "history": history,
        "config": final_config,
    }

    model_root = args.model_root.resolve()
    report_root = args.report_root.resolve()
    selected_dir = model_root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    weight_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in final_model.weights.items()}
    np.savez_compressed(selected_dir / "weights.npz", **weight_arrays)
    v2_dataset.write_json(selected_dir / "config.json", final_config)
    np.savez_compressed(
        report_root / "oof_predictions.npz",
        labels=data["y"],
        groups=data["groups"].astype(str),
        logits=cv_result["oof_logits"],
        probabilities=cv_result["oof_probabilities"],
    )
    calibration_payload = {
        key: value for key, value in cv_result["calibration"].items() if key != "probabilities"
    }
    v2_dataset.write_json(report_root / "calibration.json", calibration_payload)
    v2_dataset.write_json(report_root / "cv_results.json", _compact_result(cv_result))
    v2_dataset.write_json(
        report_root / "selection.json",
        {
            "experiment_id": config.EXPERIMENT_ID,
            "variant": config.VARIANT,
            "selection_metric": "out_of_fold_macro_f1",
            "selection_data": "333 common train+val samples shared with V4; no test split",
            "selection_uses_test": False,
            "calibration_data": "out_of_fold_train_val_predictions",
            "calibration_uses_test": False,
            "metrics": cv_result["metrics"],
            "calibration": calibrated,
            "final_epochs": epochs,
        },
    )
    v2_dataset.write_json(
        report_root / "training.json",
        {
            "experiment_id": config.EXPERIMENT_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.SEED,
            "cv_protocol": "StratifiedGroupKFold(video_id) over 333 common train+val samples; inner split for early stopping",
            "cv_folds": len(folds),
            "split_counts": data["split_counts"],
            "group_count": int(len(set(data["groups"]))),
            "final_info": final_info,
            "oof_metrics": cv_result["metrics"],
            "calibration": calibrated,
            "fit_seconds": round(time.perf_counter() - started, 3),
        },
    )
    input_hashes = {
        "v2_features/mlp_tensor.npy": sha256_file(args.v2_features_root / "mlp_tensor.npy"),
        "v2_features/mlp_preprocessing.npz": sha256_file(args.v2_features_root / "mlp_preprocessing.npz"),
        "v2_features/mlp_channel_names.json": sha256_file(args.v2_features_root / "mlp_channel_names.json"),
        "v2_features/video_manifest.csv": sha256_file(args.v2_features_root / "video_manifest.csv"),
        "v4_features/base_tensor.npy": sha256_file(data["v4_root"] / "base_tensor.npy"),
        "v4_features/video_manifest.csv": sha256_file(data["v4_root"] / "video_manifest.csv"),
        "v4_extraction/config.json": sha256_file(data["v4_extraction_root"] / "config.json"),
        "v4_extraction/manifest.csv": sha256_file(data["v4_extraction_root"] / "manifest.csv"),
        "v1_experiment_config": sha256_file(args.v1_config),
        "equivalence.json": sha256_file(args.data_root.resolve() / "equivalence.json"),
    }
    output_hashes = {
        "selected/weights.npz": sha256_file(selected_dir / "weights.npz"),
        "selected/config.json": sha256_file(selected_dir / "config.json"),
        "calibration.json": sha256_file(report_root / "calibration.json"),
        "cv_results.json": sha256_file(report_root / "cv_results.json"),
        "training.json": sha256_file(report_root / "training.json"),
    }
    v2_dataset.write_json(
        report_root / "training_manifest.json",
        {
            "experiment_id": config.EXPERIMENT_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test_split_read": False,
            "selection_uses_test": False,
            "calibration_uses_test": False,
            "final_model_training_samples": int(len(data["y"])),
            "final_epochs": epochs,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "reproduction_command": [
                sys.executable,
                "-m",
                "ml.src.experiments.rf_v3_native_frames.train",
                "--v2-features-root",
                str(args.v2_features_root.resolve()),
                "--v2-extraction-root",
                str(args.v2_extraction_root.resolve()),
                "--v4-features-root",
                str(data["v4_root"]),
                "--v4-extraction-root",
                str(data["v4_extraction_root"]),
                "--v1-config",
                str(args.v1_config.resolve()),
                "--data-root",
                str(args.data_root.resolve()),
                "--model-root",
                str(model_root),
                "--report-root",
                str(report_root),
            ],
        },
    )
    print(
        json.dumps(
            {
                "variant": config.VARIANT,
                "oof_macro_f1": cv_result["metrics"]["macro_f1"],
                "final_epochs": epochs,
                "parameter_count": final_info["parameter_count"],
                "fit_seconds": v2_dataset.read_json(report_root / "training.json")["fit_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
