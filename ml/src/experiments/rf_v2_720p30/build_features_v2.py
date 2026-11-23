from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v2_720p30 import dataset
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:  # type: ignore[name-defined]
        import sys

        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    from experiments.rf_v2_720p30 import dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "features"
ID_COLUMNS = ["sample_index", "video_id", "split", "class_folder", "class_label", "label_id"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def build_sample_from_records(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    landmark_sha256: str | None = None,
) -> dict[str, Any]:
    records = sorted(records, key=lambda item: int(item["frame_index"]))
    frame_count = len(records)
    if frame_count <= 0 or frame_count > feature_source.T_VALUE:
        raise ValueError(f"invalid frame_count {frame_count} for {row.get('video_id')}")
    coords = np.zeros((feature_source.T_VALUE, 2, feature_source.LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros((feature_source.T_VALUE, 2), dtype=bool)
    confidence = np.zeros((feature_source.T_VALUE, 2), dtype=np.float32)
    frame_real = np.zeros(feature_source.T_VALUE, dtype=bool)
    no_hand = np.zeros(feature_source.T_VALUE, dtype=bool)
    padded = np.zeros(feature_source.T_VALUE, dtype=bool)
    duplicate_count = 0
    for position, record in enumerate(records):
        if int(record["frame_index"]) != position:
            raise ValueError("non-sequential frame_index")
        if record.get("schema_version") != feature_source.FRAME_SCHEMA_VERSION:
            raise ValueError("schema mismatch")
        if record.get("decode_ok") is not True or record.get("infer_ok") is not True:
            raise ValueError("unsuccessful frame")
        if float(record["fps"]) != 30.0 or int(record["width"]) != 1280 or int(record["height"]) != 720:
            raise ValueError("v2 acquisition metadata mismatch")
        frame_coords, frame_presence, frame_confidence, frame_no_hand, frame_duplicates = (
            feature_source.canonical_frame(record)
        )
        coords[position] = frame_coords
        presence[position] = frame_presence
        confidence[position] = frame_confidence
        no_hand[position] = frame_no_hand
        frame_real[position] = True
        duplicate_count += int(frame_duplicates)
    for position in range(frame_count, feature_source.T_VALUE):
        padded[position] = True
    meta = {
        "video_id": row["video_id"],
        "split": row["split"],
        "official_rel_path_posix": row["official_rel_path_posix"],
        "source_video_abs": row["source_video_abs"],
        "output_file": row["output_file"],
        "class_folder": row["class_folder"],
        "class_label": row["class_label"],
        "label_id": int(row["label_id"]),
        "frames_written": int(frame_count),
        "duration_s": float(row["duration_s"]),
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "source_duration_s": float(row.get("source_duration_s", row["duration_s"])),
    }
    return {
        "meta": meta,
        "coords": coords,
        "presence": presence,
        "confidence": confidence,
        "frame_real": frame_real,
        "no_hand": no_hand,
        "padded": padded,
        "frame_count": frame_count,
        "duplicate_count": duplicate_count,
        "landmark_sha256": landmark_sha256 or "",
    }


def build_sample(row: dict[str, str], data_root: Path) -> dict[str, Any]:
    landmark_path = (
        data_root / row["landmark_file"]
        if not Path(row["landmark_file"]).is_absolute()
        else Path(row["landmark_file"])
    )
    records = dataset.read_jsonl(landmark_path)
    row_for_sample = {
        "video_id": row["video_id"],
        "split": row["split"],
        "official_rel_path_posix": row["official_rel_path_posix"],
        "source_video_abs": row["source_video_abs"],
        "output_file": str(landmark_path),
        "class_folder": row["class_folder"],
        "class_label": row["class_label"],
        "label_id": int(row["label_id"]),
        "duration_s": float(row["transformed_duration_s"]),
        "source_duration_s": float(row["source_duration_s"]),
    }
    return build_sample_from_records(row_for_sample, records, dataset.sha256_file(landmark_path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = dataset.read_csv(data_root / "manifest.csv")
    complete = [row for row in manifest if row["status"] == "complete"]
    for split in dataset.SPLITS:
        if not any(row["split"] == split for row in complete):
            raise ValueError(f"no complete {split} rows")
    samples = [build_sample(row, data_root) for row in complete]

    feature_names = list(feature_source.base_feature_names())
    if len(feature_names) != 831:
        raise ValueError(f"unexpected RF feature count: {len(feature_names)}")
    v1_schema = json.loads(
        (REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if list(v1_schema["feature_columns"]) != feature_names:
        raise ValueError("v2 feature names do not match frozen v1 schema")
    rf_values = np.asarray(
        [list(feature_source.compute_rf_features(sample).values()) for sample in samples],
        dtype=feature_source.RF_DTYPE,
    )
    labels = np.asarray([int(sample["meta"]["label_id"]) for sample in samples], dtype=np.int64)
    if rf_values.shape != (len(samples), 831):
        raise ValueError(f"unexpected RF matrix shape: {rf_values.shape}")

    mean, std = feature_source.fit_coordinate_scaler(samples)
    channel_names = feature_source.mlp_channel_names()
    tensors = np.asarray(
        [feature_source.build_mlp_tensor(sample, mean, std, channel_names) for sample in samples],
        dtype=feature_source.MLP_DTYPE,
    )
    flat = np.asarray([tensor.reshape(-1) for tensor in tensors], dtype=feature_source.MLP_DTYPE)
    if tensors.shape != (len(samples), feature_source.T_VALUE, len(channel_names)):
        raise ValueError(f"unexpected MLP tensor shape: {tensors.shape}")

    np.save(output_root / "rf_features.npy", rf_values)
    np.save(output_root / "labels.npy", labels)
    np.save(output_root / "mlp_tensor.npy", tensors)
    np.save(output_root / "mlp_flat.npy", flat)
    np.savez_compressed(output_root / "mlp_preprocessing.npz", mean=mean, std=std)

    dataset.write_json(
        output_root / "rf_feature_schema.json",
        {
            "schema_version": "synora_rf_v2_feature_schema_v1",
            "feature_columns": feature_names,
            "scaling": "none",
            "dtype": "float32",
            "feature_count": 831,
            "v1_schema_sha256": dataset.sha256_file(
                REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1/rf_feature_schema.json"
            ),
        },
    )
    dataset.write_json(output_root / "mlp_channel_names.json", channel_names)
    dataset.write_json(
        output_root / "mlp_preprocessing_reference.json",
        {
            "path": str(output_root / "mlp_preprocessing.npz"),
            "sha256": dataset.sha256_file(output_root / "mlp_preprocessing.npz"),
            "fit_splits": ["train"],
        },
    )

    manifest_rows = [feature_source.metadata_row(sample, index) for index, sample in enumerate(samples)]
    dataset.write_csv(
        output_root / "video_manifest.csv",
        manifest_rows,
        feature_source.VIDEO_METADATA_COLUMNS,
    )
    csv_rows = []
    for index, sample in enumerate(samples):
        features = feature_source.compute_rf_features(sample)
        row = {name: sample["meta"][name] for name in ID_COLUMNS if name in sample["meta"]}
        row["sample_index"] = index
        for name in feature_names:
            row[name] = float(features[name])
        csv_rows.append(row)
    dataset.write_csv(output_root / "rf_features.csv", csv_rows, ID_COLUMNS + feature_names)

    dataset.write_json(
        output_root / "feature_build_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(data_root / "manifest.csv"),
            "total_samples": len(samples),
            "split_counts": {
                split: sum(1 for sample in samples if sample["meta"]["split"] == split)
                for split in dataset.SPLITS
            },
            "rf_shape": list(rf_values.shape),
            "mlp_tensor_shape": list(tensors.shape),
            "mlp_flat_shape": list(flat.shape),
            "feature_count": len(feature_names),
            "dtype": str(rf_values.dtype),
            "scaling": "none",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
