from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import mediapipe as mp
import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v2_720p30 import build_features_v2, dataset as v2_dataset, extract_v2
    from ml.src.experiments.rf_v3_native_frames import config
    from ml.src.experiments.rf_v4_live_alignment import extract_v4, features as v4_features, preprocessing
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    from experiments.rf_v2_720p30 import build_features_v2, dataset as v2_dataset, extract_v2
    from experiments.rf_v3_native_frames import config
    from experiments.rf_v4_live_alignment import extract_v4, features as v4_features, preprocessing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "features", "verify", "all"), default="all")
    parser.add_argument("--v1-features-dir", type=Path, default=config.DEFAULT_V1_FEATURES)
    parser.add_argument("--v1-landmark-config", type=Path, default=config.DEFAULT_V1_LANDMARK_CONFIG)
    parser.add_argument("--v1-experiment-config", type=Path, default=config.DEFAULT_V1_EXPERIMENT_CONFIG)
    parser.add_argument("--v2-features-root", type=Path, default=config.DEFAULT_V2_FEATURES_ROOT)
    parser.add_argument("--v4-features-root", type=Path, default=config.DEFAULT_V4_FEATURES_ROOT)
    parser.add_argument("--v4-extraction-root", type=Path, default=config.DEFAULT_V4_EXTRACTION_ROOT)
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    return parser.parse_args(argv)


def official_test_metadata(v1_landmark_config: Mapping[str, Any]) -> dict[str, Any]:
    test_config = v1_landmark_config["official_splits"][config.SPLIT]
    path = Path(test_config["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = v2_dataset.sha256_file(path)
    if actual != test_config["sha256"]:
        raise RuntimeError("official test split hash mismatch")
    return {"path": path, "sha256": actual}


def load_test_entries(
    v1_manifest_path: str | Path,
    v1_landmark_config: Mapping[str, Any],
    v1_experiment_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = official_test_metadata(v1_landmark_config)
    official_paths = v2_dataset.official_split_lines(metadata["path"])
    mapping = {record["class_folder"]: record for record in v1_experiment_config["label_mapping"]}
    entries: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in v2_dataset.read_csv(v1_manifest_path):
        if row["split"] != config.SPLIT:
            continue
        rel_path = row["official_rel_path_posix"]
        if rel_path not in official_paths:
            raise ValueError(f"official test split mismatch for {row['video_id']}")
        if row["video_id"] in seen_video_ids or rel_path in seen_paths:
            raise ValueError(f"duplicate test entry: {row['video_id']}")
        if row["class_folder"] not in mapping:
            raise ValueError(f"unmapped class folder: {row['class_folder']}")
        mapped = mapping[row["class_folder"]]
        if int(mapped["original_label_id"]) != int(row["label_id"]):
            raise ValueError(f"label id mismatch for {row['video_id']}")
        source_path = Path(row["source_video_abs"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        seen_video_ids.add(row["video_id"])
        seen_paths.add(rel_path)
        entries.append(
            {
                "video_id": row["video_id"],
                "split": config.SPLIT,
                "official_rel_path_posix": rel_path,
                "source_video_abs": str(source_path.resolve()),
                "class_folder": row["class_folder"],
                "class_label": row["class_label"],
                "label_id": int(row["label_id"]),
                "model_index": int(mapped["model_index"]),
                "source_frame_count": int(row["frame_count"]),
                "source_fps": float(row["fps"]),
                "source_width": int(float(row["width"])),
                "source_height": int(float(row["height"])),
                "source_duration_s": float(row["duration_s"]),
            }
        )
    entries.sort(key=lambda item: item["official_rel_path_posix"])
    if len(entries) != config.EXPECTED_TEST_COUNT:
        raise ValueError(f"unexpected test count: {len(entries)}")
    return entries


def build_extraction_payload(
    method: str,
    v1_manifest_path: Path,
    v1_landmark_config_path: Path,
    v1_experiment_config_path: Path,
    v1_landmark_config: Mapping[str, Any],
    v1_experiment_config: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    model_path: Path,
    code_hash: str,
) -> dict[str, Any]:
    if method not in ("native", "resampled"):
        raise ValueError(f"unsupported extraction method: {method}")
    metadata = official_test_metadata(v1_landmark_config)
    if method == "native":
        temporal = "one detection per real source frame; no frame duplication and no frame resampling"
        lifecycle = "landmarker reset at recording start; offline extraction resets once per clip"
    else:
        temporal = "nearest real source frame at each 30 FPS target timestamp; duplicate source frames are real-frame repeats"
        lifecycle = "landmarker created once per clip with monotonically increasing timestamps"
    return {
        "schema_version": "synora_rf_v3_native_frames_test_extraction_config_v1",
        "experiment_id": config.EXPERIMENT_ID,
        "method": method,
        "split_policy": {
            "allowed_splits": [config.SPLIT],
            "test_split_read": True,
            "test_use": "final_evaluation_only",
            "training_split_access": False,
            "model_selection_access": False,
            "calibration_access": False,
            "selected_test_count": len(entries),
        },
        "acquisition": {
            "model_fps": 30.0,
            "width": 1280,
            "height": 720,
            "min_frames": 43,
            "max_frames": 89,
            "camera_mode": "V4 native 1280x720 at 30 FPS",
        },
        "preprocessing": {
            "spatial": "resize every decoded source frame to 1280x720 with cv2.INTER_AREA",
            "temporal": temporal,
            "resize_interpolation": "cv2.INTER_AREA",
            "resize_call": "cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)",
            "color_conversion": "cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)",
            "timestamp_policy": "next_timestamp_ms(reset_frame_index, 30.0, last_timestamp_ms)",
            "tracking_lifecycle": lifecycle,
            "no_crop": True,
            "no_mirroring": True,
            "no_augmentation": True,
            "no_smoothing": True,
            "no_synthesis": True,
            "no_silent_frame_skip": True,
            "no_silent_truncation": True,
        },
        "extraction": v1_landmark_config["extraction"],
        "frame_schema_version": feature_source.FRAME_SCHEMA_VERSION,
        "label_mapping": v1_experiment_config["label_mapping"],
        "official_splits": {
            config.SPLIT: {"path": str(metadata["path"]), "sha256": metadata["sha256"]}
        },
        "v1_inputs": {
            "features_dir": str(v1_manifest_path.parent.resolve()),
            "landmark_config_sha256": v2_dataset.sha256_file(v1_landmark_config_path),
            "experiment_config_sha256": v2_dataset.sha256_file(v1_experiment_config_path),
            "v1_manifest_sha256": v2_dataset.sha256_file(v1_manifest_path),
        },
        "model": {
            "path": str(model_path),
            "sha256": v2_dataset.sha256_file(model_path),
        },
        "code_sha256": code_hash,
        "runtime": {
            "python": platform.python_version(),
            "opencv-python": cv2.__version__,
            "mediapipe": mp.__version__ if hasattr(mp, "__version__") else "unknown",
            "platform": platform.platform(),
        },
    }


def write_extraction_config(root: Path, payload: Mapping[str, Any]) -> Path:
    payload = dict(payload)
    payload["config_hash"] = extract_v4.sha256_json(payload)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != payload["config_hash"]:
            raise RuntimeError(f"existing extraction config mismatch: {path}")
        return path
    v2_dataset.write_json(path, payload)
    return path


def extract_test(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    v1_manifest_path = args.v1_features_dir / "video_manifest.csv"
    v1_landmark_config = v2_dataset.read_json(args.v1_landmark_config)
    v1_experiment_config = v2_dataset.read_json(args.v1_experiment_config)
    entries = load_test_entries(v1_manifest_path, v1_landmark_config, v1_experiment_config)
    landmarker_payload, model_path = preprocessing.load_landmarker_settings(args.v1_landmark_config)
    code_hash = v2_dataset.sha256_file(Path(__file__).resolve())
    native_root = data_root / "extraction_native"
    resampled_root = data_root / "extraction_resampled"
    native_payload = build_extraction_payload(
        "native",
        v1_manifest_path,
        args.v1_landmark_config,
        args.v1_experiment_config,
        v1_landmark_config,
        v1_experiment_config,
        entries,
        model_path,
        code_hash,
    )
    resampled_payload = build_extraction_payload(
        "resampled",
        v1_manifest_path,
        args.v1_landmark_config,
        args.v1_experiment_config,
        v1_landmark_config,
        v1_experiment_config,
        entries,
        model_path,
        code_hash,
    )
    write_extraction_config(native_root, native_payload)
    write_extraction_config(resampled_root, resampled_payload)
    v2_dataset.write_jsonl(data_root / "selected_test.jsonl", entries)
    v2_dataset.write_json(data_root / "label_mapping.json", v1_experiment_config["label_mapping"])

    native_rows: list[dict[str, Any]] = []
    resampled_rows: list[dict[str, Any]] = []
    warning_counts: dict[str, int] = {}
    for number, entry in enumerate(entries, start=1):
        native_row, native_warnings = extract_v4.process_video(
            entry, landmarker_payload, model_path, native_root
        )
        resampled_row, resampled_warnings = extract_v2.process_video(
            entry, landmarker_payload, model_path, resampled_root
        )
        native_rows.append(native_row)
        resampled_rows.append(resampled_row)
        for message in [*native_warnings, *resampled_warnings]:
            key = message.strip()
            warning_counts[key] = warning_counts.get(key, 0) + 1
        print(
            f"[{number}/{len(entries)}] test {entry['video_id'][:12]} "
            f"native={native_row['status']} resampled={resampled_row['status']}"
        )

    v2_dataset.write_csv(native_root / "manifest.csv", native_rows, extract_v4.MANIFEST_FIELDS)
    v2_dataset.write_csv(resampled_root / "manifest.csv", resampled_rows, extract_v2.MANIFEST_FIELDS)
    summary = {
        "experiment_id": config.EXPERIMENT_ID,
        "test_split_use": "final_evaluation_only",
        "selected_count": len(entries),
        "native_complete": sum(1 for row in native_rows if row["status"] == "complete"),
        "resampled_complete": sum(1 for row in resampled_rows if row["status"] == "complete"),
        "native_excluded": [row["video_id"] for row in native_rows if row["status"] != "complete"],
        "resampled_excluded": [row["video_id"] for row in resampled_rows if row["status"] != "complete"],
        "warning_counts": warning_counts,
    }
    v2_dataset.write_json(data_root / "extraction_summary.json", summary)
    return summary


def _load_scaler(v2_features_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(v2_features_root / "mlp_preprocessing.npz") as payload:
        mean = payload["mean"].copy()
        std = payload["std"].copy()
    channel_names = v2_dataset.read_json(v2_features_root / "mlp_channel_names.json")
    if len(channel_names) != 135:
        raise ValueError("base channel count mismatch")
    return mean, std, channel_names


def _label_mapping(args: argparse.Namespace) -> dict[int, int]:
    experiment_config = v2_dataset.read_json(args.v1_experiment_config)
    return {
        int(record["original_label_id"]): int(record["model_index"])
        for record in experiment_config["label_mapping"]
    }


def build_test_features(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    native_root = data_root / "extraction_native"
    resampled_root = data_root / "extraction_resampled"
    native_output = data_root / "features_native"
    resampled_output = data_root / "features_resampled"
    native_output.mkdir(parents=True, exist_ok=True)
    resampled_output.mkdir(parents=True, exist_ok=True)
    mean, std, channel_names = _load_scaler(args.v2_features_root)
    mapping = _label_mapping(args)

    native_manifest = [
        row for row in v2_dataset.read_csv(native_root / "manifest.csv") if row["status"] == "complete"
    ]
    native_samples: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    for index, row in enumerate(native_manifest):
        landmark_path = Path(row["landmark_file"])
        sample_row = {
            "video_id": row["video_id"],
            "split": row["split"],
            "official_rel_path_posix": row["official_rel_path_posix"],
            "source_video_abs": row["source_video_abs"],
            "output_file": str(landmark_path),
            "class_folder": row["class_folder"],
            "class_label": row["class_label"],
            "label_id": int(row["label_id"]),
            "duration_s": float(row["source_duration_s"]),
            "source_duration_s": float(row["source_duration_s"]),
        }
        sample = build_features_v2.build_sample_from_records(
            sample_row,
            v2_dataset.read_jsonl(landmark_path),
            row["landmark_sha256"],
        )
        if sample["frame_count"] != int(row["target_frame_count"]):
            raise ValueError(f"native frame count mismatch for {row['video_id']}")
        native_samples.append(sample)
        expected = mapping[int(row["label_id"])]
        if expected != int(row["model_index"]):
            raise ValueError(f"native label mismatch for {row['video_id']}")
        native_rows.append(
            {
                "sample_index": index,
                "video_id": row["video_id"],
                "split": row["split"],
                "class_folder": row["class_folder"],
                "class_label": row["class_label"],
                "label_id": int(row["label_id"]),
                "model_index": expected,
                "source_frame_count": int(row["source_frame_count"]),
                "source_fps": float(row["source_fps"]),
                "source_width": int(row["source_width"]),
                "source_height": int(row["source_height"]),
                "frame_count": sample["frame_count"],
                "duration_s": sample["frame_count"] / 30.0,
                "landmark_sha256": row["landmark_sha256"],
            }
        )
    native_tensor = np.asarray(
        [v4_features.build_base_tensor(sample, mean, std, channel_names) for sample in native_samples],
        dtype=np.float32,
    )
    native_labels = np.asarray([row["model_index"] for row in native_rows], dtype=np.int64)
    if native_tensor.shape != (len(native_rows), 89, 135):
        raise ValueError(f"unexpected native test tensor shape: {native_tensor.shape}")
    np.save(native_output / "base_tensor.npy", native_tensor)
    np.save(native_output / "labels.npy", native_labels)
    v2_dataset.write_csv(
        native_output / "video_manifest.csv",
        native_rows,
        [*native_rows[0].keys()],
    )
    v2_dataset.write_json(
        native_output / "feature_schema.json",
        {
            "schema_version": "synora_rf_v3_native_frames_native_feature_schema_v1",
            "channel_names": channel_names,
            "preprocessing": "existing v2 coordinate mean/std; fit on v2 train only and not refit",
            "representation": "V4 native one-detection-per-real-source-frame",
            "test_split_use": "final_evaluation_only",
            "t_value": 89,
        },
    )

    resampled_manifest = [
        row for row in v2_dataset.read_csv(resampled_root / "manifest.csv") if row["status"] == "complete"
    ]
    resampled_samples = [build_features_v2.build_sample(row, resampled_root) for row in resampled_manifest]
    resampled_tensor = np.asarray(
        [
            feature_source.build_mlp_tensor(sample, mean, std, channel_names)
            for sample in resampled_samples
        ],
        dtype=np.float32,
    )
    resampled_labels: list[int] = []
    resampled_rows: list[dict[str, Any]] = []
    for index, (row, sample) in enumerate(zip(resampled_manifest, resampled_samples)):
        expected = mapping[int(row["label_id"])]
        if expected != int(row["model_index"]):
            raise ValueError(f"resampled label mismatch for {row['video_id']}")
        if sample["frame_count"] != int(row["target_frame_count"]):
            raise ValueError(f"resampled frame count mismatch for {row['video_id']}")
        resampled_labels.append(expected)
        resampled_rows.append(
            {
                "sample_index": index,
                "video_id": row["video_id"],
                "split": row["split"],
                "class_folder": row["class_folder"],
                "class_label": row["class_label"],
                "label_id": int(row["label_id"]),
                "model_index": expected,
                "source_frame_count": int(row["source_frame_count"]),
                "source_fps": float(row["source_fps"]),
                "source_width": int(row["source_width"]),
                "source_height": int(row["source_height"]),
                "frame_count": sample["frame_count"],
                "duration_s": sample["frame_count"] / 30.0,
                "landmark_sha256": row["landmark_sha256"],
            }
        )
    if resampled_tensor.shape != (len(resampled_rows), 89, 135):
        raise ValueError(f"unexpected resampled test tensor shape: {resampled_tensor.shape}")
    np.save(resampled_output / "mlp_tensor.npy", resampled_tensor)
    np.save(resampled_output / "labels.npy", np.asarray(resampled_labels, dtype=np.int64))
    v2_dataset.write_csv(
        resampled_output / "video_manifest.csv",
        resampled_rows,
        [*resampled_rows[0].keys()],
    )
    v2_dataset.write_json(
        resampled_output / "feature_schema.json",
        {
            "schema_version": "synora_rf_v3_native_frames_resampled_feature_schema_v1",
            "channel_names": channel_names,
            "preprocessing": "existing v2 coordinate mean/std; fit on v2 train only and not refit",
            "representation": "V2 nearest-real-frame resampling to 30 FPS",
            "test_split_use": "final_evaluation_only",
            "t_value": 89,
        },
    )
    summary = {
        "native_test_samples": len(native_rows),
        "resampled_test_samples": len(resampled_rows),
        "native_tensor_shape": list(native_tensor.shape),
        "resampled_tensor_shape": list(resampled_tensor.shape),
        "test_split_use": "final_evaluation_only",
    }
    v2_dataset.write_json(data_root / "test_feature_summary.json", summary)
    return summary


def _rebuild_native_tensor(row: Mapping[str, str], extraction_root: Path, mean: np.ndarray, std: np.ndarray, names: Sequence[str]) -> np.ndarray:
    landmark_path = Path(row["landmark_file"])
    sample_row = {
        "video_id": row["video_id"],
        "split": row["split"],
        "official_rel_path_posix": row["official_rel_path_posix"],
        "source_video_abs": row["source_video_abs"],
        "output_file": str(landmark_path),
        "class_folder": row["class_folder"],
        "class_label": row["class_label"],
        "label_id": int(row["label_id"]),
        "duration_s": float(row["source_duration_s"]),
        "source_duration_s": float(row["source_duration_s"]),
    }
    sample = build_features_v2.build_sample_from_records(
        sample_row,
        v2_dataset.read_jsonl(landmark_path),
        row["landmark_sha256"],
    )
    return v4_features.build_base_tensor(sample, mean, std, names)


def _exact_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(f"tensor shape mismatch: {actual.shape} != {expected.shape}")
    unequal = int(np.count_nonzero(np.any(actual != expected, axis=tuple(range(1, actual.ndim)))))
    return {
        "rows": int(len(actual)),
        "exact_rows": int(len(actual) - unequal),
        "unequal_rows": unequal,
        "max_abs_diff": float(np.abs(actual - expected).max()) if actual.size else 0.0,
        "all_exact": unequal == 0,
    }


def verify_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    mean, std, channel_names = _load_scaler(args.v2_features_root)
    data_root = args.data_root.resolve()

    v4_feature_root = args.v4_features_root.resolve()
    v4_base = np.load(v4_feature_root / "base_tensor.npy")
    v4_feature_manifest = v2_dataset.read_csv(v4_feature_root / "video_manifest.csv")
    v4_extraction_rows = {
        row["video_id"]: row
        for row in v2_dataset.read_csv(args.v4_extraction_root / "manifest.csv")
    }
    v4_rebuilt: list[np.ndarray] = []
    for index, row in enumerate(v4_feature_manifest):
        extraction_row = v4_extraction_rows[row["video_id"]]
        landmark_path = args.v4_extraction_root / extraction_row["landmark_file"]
        sample_row = {
            "video_id": row["video_id"],
            "split": row["split"],
            "official_rel_path_posix": extraction_row["official_rel_path_posix"],
            "source_video_abs": extraction_row["source_video_abs"],
            "output_file": str(landmark_path),
            "class_folder": row["class_folder"],
            "class_label": row["class_label"],
            "label_id": int(row["label_id"]),
            "duration_s": float(row["duration_s"]),
            "source_duration_s": float(extraction_row["source_duration_s"]),
        }
        sample = build_features_v2.build_sample_from_records(
            sample_row,
            v2_dataset.read_jsonl(landmark_path),
            extraction_row["landmark_sha256"],
        )
        v4_rebuilt.append(v4_features.build_base_tensor(sample, mean, std, channel_names))
    v4_summary = _exact_summary(np.asarray(v4_rebuilt, dtype=np.float32), v4_base)

    native_extraction_root = data_root / "extraction_native"
    native_feature_root = data_root / "features_native"
    native_base = np.load(native_feature_root / "base_tensor.npy")
    native_rows = [
        row for row in v2_dataset.read_csv(native_extraction_root / "manifest.csv") if row["status"] == "complete"
    ]
    native_rebuilt = np.asarray(
        [
            _rebuild_native_tensor(row, native_extraction_root, mean, std, channel_names)
            for row in native_rows
        ],
        dtype=np.float32,
    )
    native_summary = _exact_summary(native_rebuilt, native_base)

    resampled_extraction_root = data_root / "extraction_resampled"
    resampled_feature_root = data_root / "features_resampled"
    resampled_tensor = np.load(resampled_feature_root / "mlp_tensor.npy")
    resampled_rows = [
        row for row in v2_dataset.read_csv(resampled_extraction_root / "manifest.csv") if row["status"] == "complete"
    ]
    resampled_rebuilt = np.asarray(
        [
            feature_source.build_mlp_tensor(
                build_features_v2.build_sample(row, resampled_extraction_root),
                mean,
                std,
                channel_names,
            )
            for row in resampled_rows
        ],
        dtype=np.float32,
    )
    resampled_summary = _exact_summary(resampled_rebuilt, resampled_tensor)

    report = {
        "experiment_id": config.EXPERIMENT_ID,
        "native_live_path": "build_features_v2.build_sample_from_records -> features.build_base_tensor -> RegularizedTemporalMLP.prepare_inputs",
        "existing_v4_trainval": v4_summary,
        "native_test": native_summary,
        "resampled_test": resampled_summary,
        "passed": all(
            item["all_exact"] for item in (v4_summary, native_summary, resampled_summary)
        ),
    }
    v2_dataset.write_json(data_root / "equivalence.json", report)
    if not report["passed"]:
        raise RuntimeError("exact live/offline tensor equivalence failed")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("extract", "all"):
        extract_test(args)
    if args.stage in ("features", "all"):
        build_test_features(args)
    if args.stage in ("verify", "all"):
        verify_equivalence(args)
    print(json.dumps({"stage": args.stage, "data_root": str(args.data_root.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
