from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ml.src.experiments.rf_v2_720p30 import build_features_v2
    from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
    from ml.src.experiments.rf_v4_live_alignment import features
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import build_features_v2
    from experiments.rf_v2_720p30 import dataset as v2_dataset
    from experiments.rf_v4_live_alignment import features


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "features"
DEFAULT_V2_FEATURES = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
DEFAULT_V1_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
ID_COLUMNS = ["sample_index", "video_id", "split", "class_folder", "class_label", "label_id"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v2-features-root", type=Path, default=DEFAULT_V2_FEATURES)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = v2_dataset.read_csv(data_root / "manifest.csv")
    complete = [row for row in manifest if row["status"] == "complete"]
    if not complete:
        raise ValueError("no complete v4 extraction rows")
    v1_config = v2_dataset.read_json(args.v1_config)
    mapping = {int(record["original_label_id"]): int(record["model_index"]) for record in v1_config["label_mapping"]}
    with np.load(args.v2_features_root / "mlp_preprocessing.npz") as payload:
        mean = payload["mean"].copy()
        std = payload["std"].copy()
    channel_names = v2_dataset.read_json(args.v2_features_root / "mlp_channel_names.json")
    if len(channel_names) != 135:
        raise ValueError("v2 channel count mismatch")

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(complete):
        landmark_path = data_root / row["landmark_file"]
        records = v2_dataset.read_jsonl(landmark_path)
        frame_count = len(records)
        sample_row = {
            "video_id": row["video_id"],
            "split": row["split"],
            "official_rel_path_posix": row["official_rel_path_posix"],
            "source_video_abs": row["source_video_abs"],
            "output_file": str(landmark_path),
            "class_folder": row["class_folder"],
            "class_label": row["class_label"],
            "label_id": int(row["label_id"]),
            "duration_s": frame_count / 30.0,
            "source_duration_s": float(row["source_duration_s"]),
        }
        sample = build_features_v2.build_sample_from_records(sample_row, records, v2_dataset.sha256_file(landmark_path))
        if sample["frame_count"] != int(row["target_frame_count"]):
            raise ValueError(f"v4 frame count mismatch for {row['video_id']}")
        samples.append(sample)

    base_tensors = np.asarray(
        [features.build_base_tensor(sample, mean, std, channel_names) for sample in samples],
        dtype=np.float32,
    )
    enhanced_tensors = np.asarray(
        [features.build_enhanced_tensor(sample) for sample in samples],
        dtype=np.float32,
    )
    labels = np.asarray([mapping[int(sample["meta"]["label_id"])] for sample in samples], dtype=np.int64)
    if base_tensors.shape != (len(samples), 89, 135):
        raise ValueError(f"unexpected base tensor shape: {base_tensors.shape}")
    if enhanced_tensors.shape != (len(samples), 89, len(features.ENHANCED_CHANNEL_NAMES)):
        raise ValueError(f"unexpected enhanced tensor shape: {enhanced_tensors.shape}")
    if not np.isfinite(base_tensors).all() or not np.isfinite(enhanced_tensors).all():
        raise ValueError("non-finite v4 feature tensor")

    manifest_rows: list[dict[str, Any]] = []
    for index, (row, sample) in enumerate(zip(complete, samples)):
        manifest_rows.append(
            {
                "sample_index": index,
                "video_id": row["video_id"],
                "split": row["split"],
                "class_folder": row["class_folder"],
                "class_label": row["class_label"],
                "label_id": int(row["label_id"]),
                "model_index": mapping[int(row["label_id"])],
                "source_frame_count": int(row["source_frame_count"]),
                "source_fps": float(row["source_fps"]),
                "source_width": int(row["source_width"]),
                "source_height": int(row["source_height"]),
                "frame_count": sample["frame_count"],
                "duration_s": sample["frame_count"] / 30.0,
                "landmark_sha256": sample["landmark_sha256"],
            }
        )
    fieldnames = ID_COLUMNS + [
        "model_index",
        "source_frame_count",
        "source_fps",
        "source_width",
        "source_height",
        "frame_count",
        "duration_s",
        "landmark_sha256",
    ]
    v2_dataset.write_csv(output_root / "video_manifest.csv", manifest_rows, fieldnames)
    np.save(output_root / "base_tensor.npy", base_tensors)
    np.save(output_root / "enhanced_tensor.npy", enhanced_tensors)
    np.save(output_root / "labels.npy", labels)
    v2_dataset.write_json(
        output_root / "feature_schema.json",
        {
            "schema_version": "synora_rf_v4_feature_schema_v1",
            "base_channel_names": channel_names,
            "enhanced_channel_names": features.ENHANCED_CHANNEL_NAMES,
            "base_preprocessing": "existing v2 coordinate mean/std; fit on v2 train only and not refit",
            "enhanced_preprocessing": "deterministic bounded scaling; no fitted statistics",
            "absolute_coordinates_preserved": True,
            "missing_landmarks_masked": True,
            "t_value": 89,
        },
    )
    v2_dataset.write_json(
        output_root / "feature_build_summary.json",
        {
            "sample_count": len(samples),
            "split_counts": {
                split: sum(1 for row in manifest_rows if row["split"] == split)
                for split in v2_dataset.SPLITS
            },
            "base_tensor_shape": list(base_tensors.shape),
            "enhanced_tensor_shape": list(enhanced_tensors.shape),
            "test_split_read": False,
        },
    )
    print(f"built_v4_features samples={len(samples)} enhanced_channels={len(features.ENHANCED_CHANNEL_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
