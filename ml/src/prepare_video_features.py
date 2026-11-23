import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "synora_video_features_v1"
FRAME_SCHEMA_VERSION = "synora_frame_v1"
SPLITS = ("train", "val", "test")
HAND_LABELS = ("Left", "Right")
COORDINATE_AXES = ("x", "y", "z")
T_VALUE = 89
LANDMARK_COUNT = 21
RF_DTYPE = np.float32
MLP_DTYPE = np.float32
VIDEO_METADATA_COLUMNS = [
    "sample_index",
    "video_id",
    "official_rel_path_posix",
    "source_video_abs",
    "landmark_file",
    "landmark_sha256",
    "class_folder",
    "folder_id",
    "class_label",
    "label_id",
    "split",
    "frame_count",
    "frames_written_manifest",
    "duration_s",
    "fps",
    "width",
    "height",
    "left_observed_frames",
    "right_observed_frames",
    "no_hand_frames",
    "padded_frames",
    "duplicate_handedness_frames",
]
RF_CSV_ID_COLUMNS = ["sample_index", "video_id", "class_folder", "class_label", "label_id", "split"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDMARK_ROOT = REPO_ROOT / "ml/dataset/processed/synora_include30_landmarks_v1"
DEFAULT_DESIGN_JSON = REPO_ROOT / "ml/reports/dataset_adequacy/recommended_experiment_design.json"
DEFAULT_ADEQUACY_REPORT = REPO_ROOT / "ml/reports/dataset_adequacy/dataset_adequacy_report.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-root", type=Path, default=DEFAULT_LANDMARK_ROOT)
    parser.add_argument("--design-json", type=Path, default=DEFAULT_DESIGN_JSON)
    parser.add_argument("--adequacy-report", type=Path, default=DEFAULT_ADEQUACY_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def coordinate_name(side, stat, landmark_index, axis):
    return f"{side}_coord_{stat}_lm{landmark_index:02d}_{axis}"


def delta_name(side, stat, landmark_index, axis):
    return f"{side}_coord_{stat}_lm{landmark_index:02d}_{axis}"


def base_feature_names():
    names = [
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
    ]
    for side in ("left", "right"):
        names.extend(
            [
                f"{side}_observed_frames",
                f"{side}_presence_ratio",
                f"{side}_missing_ratio",
                f"{side}_conf_mean",
                f"{side}_conf_std",
                f"{side}_conf_min",
                f"{side}_conf_max",
                f"{side}_conf_last",
            ]
        )
        for stat in ("mean", "std", "min", "max"):
            for landmark_index in range(LANDMARK_COUNT):
                for axis in COORDINATE_AXES:
                    names.append(coordinate_name(side, stat, landmark_index, axis))
        for stat in ("mean_abs_delta", "std_abs_delta"):
            for landmark_index in range(LANDMARK_COUNT):
                for axis in COORDINATE_AXES:
                    names.append(delta_name(side, stat, landmark_index, axis))
    for stat in ("mean", "std", "min", "max"):
        names.append(f"interhand_distance_{stat}")
    for landmark_index in range(LANDMARK_COUNT):
        for stat in ("mean", "std"):
            names.append(f"interhand_lm{landmark_index:02d}_distance_{stat}")
    return names


def mlp_channel_names():
    names = []
    for side in ("left", "right"):
        for landmark_index in range(LANDMARK_COUNT):
            for axis in COORDINATE_AXES:
                names.append(f"{side}_lm{landmark_index:02d}_{axis}")
    names.extend(
        [
            "left_hand_present",
            "right_hand_present",
            "left_hand_missing",
            "right_hand_missing",
            "frame_real",
            "no_hand_frame",
            "frame_padded",
            "left_confidence",
            "right_confidence",
        ]
    )
    return names


def canonical_frame(record):
    coords = np.zeros((2, LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros(2, dtype=bool)
    confidence = np.zeros(2, dtype=np.float32)
    selected = {}
    duplicate_count = 0
    for hand in record.get("hands", []):
        label = hand.get("handedness_label")
        if label not in HAND_LABELS:
            continue
        score = float(hand.get("handedness_score", 0.0))
        if label in selected:
            duplicate_count += 1
            if score <= float(selected[label].get("handedness_score", 0.0)):
                continue
        selected[label] = hand
    for side, label in enumerate(HAND_LABELS):
        hand = selected.get(label)
        if hand is None:
            continue
        landmarks = hand.get("normalized_landmarks")
        world_landmarks = hand.get("world_landmarks")
        if not isinstance(landmarks, list) or len(landmarks) != LANDMARK_COUNT:
            raise ValueError(f"invalid normalized_landmarks for {label}")
        if not isinstance(world_landmarks, list) or len(world_landmarks) != LANDMARK_COUNT:
            raise ValueError(f"invalid world_landmarks for {label}")
        for landmark_index, landmark in enumerate(landmarks):
            values = []
            for axis in COORDINATE_AXES:
                value = float(landmark[axis])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {label} landmark value")
                values.append(value)
            coords[side, landmark_index] = values
        presence[side] = True
        confidence[side] = float(hand.get("handedness_score", 0.0))
    no_hand = not any(presence)
    return coords, presence, confidence, no_hand, duplicate_count


def build_sample(meta, class_lookup, design_classes, official_paths, class_by_folder):
    landmark_path = Path(meta["output_file"])
    if not landmark_path.exists():
        raise FileNotFoundError(landmark_path)
    records = read_jsonl(landmark_path)
    records.sort(key=lambda item: int(item["frame_index"]))
    frame_count = len(records)
    if frame_count > T_VALUE:
        raise ValueError(f"video {meta['video_id']} has {frame_count} frames exceeding T={T_VALUE}")
    if frame_count != int(meta["frames_written"]):
        raise ValueError(f"video {meta['video_id']} frame count mismatch")
    coords = np.zeros((T_VALUE, 2, LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros((T_VALUE, 2), dtype=bool)
    confidence = np.zeros((T_VALUE, 2), dtype=np.float32)
    frame_real = np.zeros(T_VALUE, dtype=bool)
    no_hand = np.zeros(T_VALUE, dtype=bool)
    padded = np.zeros(T_VALUE, dtype=bool)
    duplicate_count = 0
    for frame_index, record in enumerate(records):
        if int(record["frame_index"]) != frame_index:
            raise ValueError(f"non-sequential frame index in {meta['video_id']}")
        if record.get("video_id") != meta["video_id"]:
            raise ValueError(f"video_id mismatch in {landmark_path}")
        if record.get("split") != meta["split"]:
            raise ValueError(f"split mismatch in {landmark_path}")
        if record.get("class_folder") != meta["class_folder"]:
            raise ValueError(f"class mismatch in {landmark_path}")
        if record.get("schema_version") != FRAME_SCHEMA_VERSION:
            raise ValueError(f"schema mismatch in {landmark_path}")
        if record.get("decode_ok") is not True or record.get("infer_ok") is not True:
            raise ValueError(f"unsuccessful frame in {landmark_path}")
        frame_real[frame_index] = True
        frame_coords, frame_presence, frame_confidence, frame_no_hand, frame_duplicates = canonical_frame(record)
        coords[frame_index] = frame_coords
        presence[frame_index] = frame_presence
        confidence[frame_index] = frame_confidence
        no_hand[frame_index] = frame_no_hand
        duplicate_count += frame_duplicates
    for frame_index in range(frame_count, T_VALUE):
        padded[frame_index] = True
    if meta["official_rel_path_posix"] not in official_paths[meta["split"]]:
        raise ValueError(f"official split mismatch for {meta['video_id']}")
    if meta["class_folder"] not in design_classes:
        raise ValueError(f"non-primary class in manifest: {meta['class_folder']}")
    expected_label_id = class_lookup[meta["class_label"]]
    if int(meta["label_id"]) != expected_label_id:
        raise ValueError(f"label id mismatch for {meta['video_id']}")
    if class_by_folder[meta["class_folder"]] != meta["class_label"]:
        raise ValueError(f"class label mismatch for {meta['video_id']}")
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
        "landmark_sha256": sha256_file(landmark_path),
    }


def add_zero_stat_block(features, side, names):
    for name in names:
        features[name] = 0.0


def compute_rf_features(sample):
    features = collections.OrderedDict()
    meta = sample["meta"]
    frame_count = sample["frame_count"]
    real_indices = np.where(sample["frame_real"])[0]
    real_count = len(real_indices)
    both = sample["presence"][:, 0] & sample["presence"][:, 1] & sample["frame_real"]
    neither = sample["no_hand"] & sample["frame_real"]
    features["frame_count"] = frame_count
    features["duration_s"] = float(meta["duration_s"])
    features["fps"] = float(meta["fps"])
    features["width"] = float(meta["width"])
    features["height"] = float(meta["height"])
    features["padded_frame_count"] = int(sample["padded"].sum())
    features["padded_frame_ratio"] = float(sample["padded"].sum() / T_VALUE)
    features["no_hand_frame_count"] = int(neither.sum())
    features["no_hand_frame_ratio"] = float(neither.sum() / real_count) if real_count else 0.0
    features["any_hand_frame_ratio"] = float((real_count - neither.sum()) / real_count) if real_count else 0.0
    features["both_hands_frame_count"] = int(both.sum())
    features["both_hands_frame_ratio"] = float(both.sum() / real_count) if real_count else 0.0
    features["duplicate_handedness_frames"] = int(sample["duplicate_count"])
    for side_index, side in enumerate(("left", "right")):
        present = sample["presence"][:, side_index] & sample["frame_real"]
        present_indices = np.where(present)[0]
        observed = int(present.sum())
        features[f"{side}_observed_frames"] = observed
        features[f"{side}_presence_ratio"] = float(observed / real_count) if real_count else 0.0
        features[f"{side}_missing_ratio"] = 1.0 - (float(observed / real_count) if real_count else 0.0)
        conf_values = sample["confidence"][present, side_index]
        for stat in ("mean", "std", "min", "max"):
            features[f"{side}_conf_{stat}"] = float(getattr(np, stat)(conf_values)) if observed else 0.0
        features[f"{side}_conf_last"] = float(sample["confidence"][real_indices[-1], side_index]) if real_count else 0.0
        stat_names = []
        for stat in ("mean", "std", "min", "max"):
            for landmark_index in range(LANDMARK_COUNT):
                for axis in COORDINATE_AXES:
                    stat_names.append(coordinate_name(side, stat, landmark_index, axis))
        delta_names = []
        for stat in ("mean_abs_delta", "std_abs_delta"):
            for landmark_index in range(LANDMARK_COUNT):
                for axis in COORDINATE_AXES:
                    delta_names.append(delta_name(side, stat, landmark_index, axis))
        add_zero_stat_block(features, side, stat_names)
        add_zero_stat_block(features, side, delta_names)
        if observed:
            values = sample["coords"][present, side_index].reshape(observed, 63)
            stat_arrays = {
                "mean": np.mean(values, axis=0),
                "std": np.std(values, axis=0),
                "min": np.min(values, axis=0),
                "max": np.max(values, axis=0),
            }
            position = 0
            for landmark_index in range(LANDMARK_COUNT):
                for axis in COORDINATE_AXES:
                    for stat in ("mean", "std", "min", "max"):
                        features[coordinate_name(side, stat, landmark_index, axis)] = float(stat_arrays[stat][position])
                    position += 1
            delta_values = []
            real_frames = sample["frame_real"]
            for frame_index in range(1, T_VALUE):
                if real_frames[frame_index - 1] and real_frames[frame_index] and present[frame_index - 1] and present[frame_index]:
                    delta_values.append(
                        np.abs(
                            sample["coords"][frame_index, side_index]
                            - sample["coords"][frame_index - 1, side_index]
                        ).reshape(63)
                    )
            if delta_values:
                delta_array = np.asarray(delta_values, dtype=np.float32)
                delta_arrays = {
                    "mean_abs_delta": np.mean(delta_array, axis=0),
                    "std_abs_delta": np.std(delta_array, axis=0),
                }
                for stat, array in delta_arrays.items():
                    for landmark_index in range(LANDMARK_COUNT):
                        for axis_index, axis in enumerate(COORDINATE_AXES):
                            offset = landmark_index * 3 + axis_index
                            features[delta_name(side, stat, landmark_index, axis)] = float(array[offset])
    names = ["interhand_distance_mean", "interhand_distance_std", "interhand_distance_min", "interhand_distance_max"]
    for landmark_index in range(LANDMARK_COUNT):
        names.append(f"interhand_lm{landmark_index:02d}_distance_mean")
        names.append(f"interhand_lm{landmark_index:02d}_distance_std")
    for name in names:
        features[name] = 0.0
    if both.any():
        distances = np.linalg.norm(sample["coords"][both, 0] - sample["coords"][both, 1], axis=2)
        flat_distances = distances.reshape(distances.shape[0], LANDMARK_COUNT)
        for stat in ("mean", "std", "min", "max"):
            features[f"interhand_distance_{stat}"] = float(getattr(np, stat)(flat_distances))
        for landmark_index in range(LANDMARK_COUNT):
            features[f"interhand_lm{landmark_index:02d}_distance_mean"] = float(np.mean(flat_distances[:, landmark_index]))
            features[f"interhand_lm{landmark_index:02d}_distance_std"] = float(np.std(flat_distances[:, landmark_index]))
    return features


def fit_coordinate_scaler(samples):
    per_side = []
    for side_index in range(2):
        chunks = []
        for sample in samples:
            if sample["meta"]["split"] != "train":
                continue
            mask = sample["presence"][:, side_index] & sample["frame_real"]
            if mask.any():
                chunks.append(sample["coords"][mask, side_index])
        if not chunks:
            raise ValueError("no train observations for hand scaling")
        per_side.append(np.concatenate(chunks, axis=0))
    mean = np.stack([np.mean(values, axis=0) for values in per_side]).astype(np.float32)
    std = np.stack([np.std(values, axis=0) for values in per_side]).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite coordinate scaler statistics")
    return mean, std


def build_mlp_tensor(sample, mean, std, channel_names):
    coords = sample["coords"].copy()
    for side_index in range(2):
        mask = sample["presence"][:, side_index] & sample["frame_real"]
        if mask.any():
            coords[mask, side_index] = (coords[mask, side_index] - mean[side_index]) / std[side_index]
    tensor = np.zeros((T_VALUE, len(channel_names)), dtype=MLP_DTYPE)
    tensor[:, 0:63] = coords[:, 0].reshape(T_VALUE, 63)
    tensor[:, 63:126] = coords[:, 1].reshape(T_VALUE, 63)
    tensor[:, 126] = sample["presence"][:, 0].astype(np.float32)
    tensor[:, 127] = sample["presence"][:, 1].astype(np.float32)
    tensor[:, 128] = (~sample["presence"][:, 0] & sample["frame_real"]).astype(np.float32)
    tensor[:, 129] = (~sample["presence"][:, 1] & sample["frame_real"]).astype(np.float32)
    tensor[:, 130] = sample["frame_real"].astype(np.float32)
    tensor[:, 131] = sample["no_hand"].astype(np.float32)
    tensor[:, 132] = sample["padded"].astype(np.float32)
    tensor[:, 133] = sample["confidence"][:, 0]
    tensor[:, 134] = sample["confidence"][:, 1]
    if not np.isfinite(tensor).all():
        raise ValueError(f"non-finite MLP tensor for {sample['meta']['video_id']}")
    return tensor


def metadata_row(sample, index):
    meta = sample["meta"]
    return {
        "sample_index": index,
        "video_id": meta["video_id"],
        "official_rel_path_posix": meta["official_rel_path_posix"],
        "source_video_abs": meta["source_video_abs"],
        "landmark_file": meta["output_file"],
        "landmark_sha256": sample["landmark_sha256"],
        "class_folder": meta["class_folder"],
        "folder_id": int(meta["class_folder"].split(".", 1)[0]),
        "class_label": meta["class_label"],
        "label_id": int(meta["label_id"]),
        "split": meta["split"],
        "frame_count": sample["frame_count"],
        "frames_written_manifest": int(meta["frames_written"]),
        "duration_s": float(meta["duration_s"]),
        "fps": float(meta["fps"]),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "left_observed_frames": int((sample["presence"][:, 0] & sample["frame_real"]).sum()),
        "right_observed_frames": int((sample["presence"][:, 1] & sample["frame_real"]).sum()),
        "no_hand_frames": int((sample["no_hand"] & sample["frame_real"]).sum()),
        "padded_frames": int(sample["padded"].sum()),
        "duplicate_handedness_frames": int(sample["duplicate_count"]),
    }


def split_inventory(samples):
    result = {}
    for split in SPLITS:
        split_samples = [sample for sample in samples if sample["meta"]["split"] == split]
        class_counts = collections.Counter(sample["meta"]["class_folder"] for sample in split_samples)
        result[split] = {
            "video_count": len(split_samples),
            "frame_count": sum(sample["frame_count"] for sample in split_samples),
            "class_counts": {key: class_counts[key] for key in sorted(class_counts, key=lambda value: int(value.split(".", 1)[0]))},
            "video_ids": [sample["meta"]["video_id"] for sample in split_samples],
        }
    return result


def mask_summary(samples):
    summary = {}
    for split in SPLITS:
        split_samples = [sample for sample in samples if sample["meta"]["split"] == split]
        frame_count = sum(sample["frame_count"] for sample in split_samples)
        real_frames = T_VALUE * len(split_samples)
        summary[split] = {
            "real_frames": frame_count,
            "padded_frames": real_frames - frame_count,
            "no_hand_frames": int(sum((sample["no_hand"] & sample["frame_real"]).sum() for sample in split_samples)),
            "left_missing_frames": int(sum((~sample["presence"][:, 0] & sample["frame_real"]).sum() for sample in split_samples)),
            "right_missing_frames": int(sum((~sample["presence"][:, 1] & sample["frame_real"]).sum() for sample in split_samples)),
            "both_hands_frames": int(sum((sample["presence"][:, 0] & sample["presence"][:, 1] & sample["frame_real"]).sum() for sample in split_samples)),
            "duplicate_handedness_frames": int(sum(sample["duplicate_count"] for sample in split_samples)),
        }
    return summary


def validation_checks(samples, expected_ids, official_paths, rf_matrix, mlp_tensor, mlp_flat, feature_names, mean, std, design):
    checks = []
    sample_ids = [sample["meta"]["video_id"] for sample in samples]
    ids_unique = len(set(sample_ids)) == len(sample_ids) == 443
    checks.append({"name": "duplicate_video_ids", "passed": ids_unique, "details": {"rows": len(sample_ids), "unique_ids": len(set(sample_ids))}})
    missing_ids = sorted(expected_ids - set(sample_ids))
    extra_ids = sorted(set(sample_ids) - expected_ids)
    checks.append({"name": "missing_videos", "passed": not missing_ids and not extra_ids, "details": {"missing": missing_ids, "extra": extra_ids}})
    label_ok = all(
        int(sample["meta"]["label_id"]) == design["class_lookup"][sample["meta"]["class_label"]]
        for sample in samples
    )
    checks.append({"name": "label_consistency", "passed": label_ok, "details": {"classes": len(design["classes"])}})
    bad_splits = [
        sample["meta"]["video_id"]
        for sample in samples
        if sample["meta"]["official_rel_path_posix"] not in official_paths[sample["meta"]["split"]]
    ]
    train_ids = set(sample["meta"]["video_id"] for sample in samples if sample["meta"]["split"] == "train")
    val_ids = set(sample["meta"]["video_id"] for sample in samples if sample["meta"]["split"] == "val")
    test_ids = set(sample["meta"]["video_id"] for sample in samples if sample["meta"]["split"] == "test")
    disjoint = not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    checks.append({"name": "split_leakage", "passed": not bad_splits and disjoint, "details": {"bad_split_assignments": bad_splits, "train_val_overlap": len(train_ids & val_ids), "train_test_overlap": len(train_ids & test_ids), "val_test_overlap": len(val_ids & test_ids)}})
    nonfinite = {
        "rf_matrix": int((~np.isfinite(rf_matrix)).sum()),
        "mlp_tensor": int((~np.isfinite(mlp_tensor)).sum()),
        "mlp_flat": int((~np.isfinite(mlp_flat)).sum()),
        "coordinate_mean": int((~np.isfinite(mean)).sum()),
        "coordinate_std": int((~np.isfinite(std)).sum()),
    }
    checks.append({"name": "non_finite_values", "passed": all(value == 0 for value in nonfinite.values()), "details": nonfinite})
    shape_ok = (
        rf_matrix.shape == (443, len(feature_names))
        and mlp_tensor.shape == (443, T_VALUE, 135)
        and mlp_flat.shape == (443, T_VALUE * 135)
    )
    checks.append({"name": "tensor_shapes", "passed": shape_ok, "details": {"rf_matrix": list(rf_matrix.shape), "mlp_tensor": list(mlp_tensor.shape), "mlp_flat": list(mlp_flat.shape)}})
    truncation = [sample["meta"]["video_id"] for sample in samples if sample["frame_count"] > T_VALUE]
    checks.append({"name": "no_truncation", "passed": not truncation, "details": {"truncated_video_ids": truncation}})
    split_counts = {split: sum(1 for sample in samples if sample["meta"]["split"] == split) for split in SPLITS}
    expected_counts = {split: design["expected_counts"][split] for split in SPLITS}
    checks.append({"name": "expected_split_counts", "passed": split_counts == expected_counts, "details": {"actual": split_counts, "expected": expected_counts}})
    frames = sum(sample["frame_count"] for sample in samples)
    checks.append({"name": "expected_frame_total", "passed": frames == design["expected_frames"], "details": {"actual": frames, "expected": design["expected_frames"]}})
    return checks


def main():
    args = parse_args()
    landmark_root = args.landmark_root.resolve()
    design_path = args.design_json.resolve()
    adequacy_path = args.adequacy_report.resolve()
    output_root = args.output_root.resolve()
    extraction_config_path = landmark_root / "config.json"
    extraction_manifest_path = landmark_root / "extraction_manifest.jsonl"
    selected_videos_path = landmark_root / "selected_videos.jsonl"
    extraction_config = json.loads(extraction_config_path.read_text(encoding="utf-8"))
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    classes = design_payload["primary_design"]["classes"]
    label_map_path = Path(extraction_config["label_map"]["path"])
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    manifest_rows = read_jsonl(extraction_manifest_path)
    class_lookup = {name: int(label_map[name]) for name in {row["class_label"] for row in manifest_rows if row["class_folder"] in classes}}
    class_by_folder = {}
    for row in manifest_rows:
        if row["class_folder"] in classes:
            class_by_folder[row["class_folder"]] = row["class_label"]
    official_paths = {}
    for split in SPLITS:
        path = Path(extraction_config["official_splits"][split]["path"])
        official_paths[split] = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    eligible = [
        row
        for row in manifest_rows
        if row["class_folder"] in classes
        and row["status"] == "complete"
        and row["included_in_run"] is True
        and row["source_video_exists"] is True
    ]
    eligible.sort(key=lambda row: (SPLITS.index(row["split"]), int(row["class_folder"].split(".", 1)[0]), row["video_id"]))
    expected_ids = {row["video_id"] for row in eligible}
    samples = [build_sample(row, class_lookup, set(classes), official_paths, class_by_folder) for row in eligible]
    mean, std = fit_coordinate_scaler(samples)
    channel_names = mlp_channel_names()
    feature_names = base_feature_names()
    rf_rows = []
    tensors = []
    for index, sample in enumerate(samples):
        features = compute_rf_features(sample)
        if list(features.keys()) != feature_names:
            raise ValueError("RF feature name mismatch")
        row = metadata_row(sample, index)
        row.update(features)
        rf_rows.append(row)
        tensors.append(build_mlp_tensor(sample, mean, std, channel_names))
    rf_matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rf_rows], dtype=RF_DTYPE)
    mlp_tensor = np.stack(tensors).astype(MLP_DTYPE)
    mlp_flat = mlp_tensor.reshape(len(samples), T_VALUE * len(channel_names)).astype(MLP_DTYPE)
    design_meta = {
        "classes": classes,
        "class_lookup": class_lookup,
        "expected_counts": design_payload["primary_design"]["videos"],
        "expected_frames": design_payload["primary_design"]["frames"]["total"],
    }
    checks = validation_checks(samples, expected_ids, official_paths, rf_matrix, mlp_tensor, mlp_flat, feature_names, mean, std, design_meta)
    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "actual_dimensions": {
            "rf_matrix": list(rf_matrix.shape),
            "mlp_tensor": list(mlp_tensor.shape),
            "mlp_flat": list(mlp_flat.shape),
        },
        "split_counts": {split: sum(1 for sample in samples if sample["meta"]["split"] == split) for split in SPLITS},
        "class_counts": {class_name: sum(1 for sample in samples if sample["meta"]["class_folder"] == class_name) for class_name in classes},
        "frame_total": sum(sample["frame_count"] for sample in samples),
        "mask_summary": mask_summary(samples),
        "rf_non_finite_values": int((~np.isfinite(rf_matrix)).sum()),
        "issues": [check for check in checks if not check["passed"]],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "validation_report.json", validation_report)
    if not validation_report["passed"]:
        raise SystemExit("validation failed; see validation_report.json")
    metadata_columns = VIDEO_METADATA_COLUMNS
    rf_csv_columns = RF_CSV_ID_COLUMNS + feature_names
    write_csv(output_root / "rf_features.csv", rf_csv_columns, [{key: row[key] for key in rf_csv_columns} for row in rf_rows])
    write_csv(output_root / "video_manifest.csv", metadata_columns, [{key: row[key] for key in metadata_columns} for row in rf_rows])
    write_csv(output_root / "mlp_sample_manifest.csv", metadata_columns, [{key: row[key] for key in metadata_columns} for row in rf_rows])
    np.save(output_root / "rf_features.npy", rf_matrix)
    np.save(output_root / "mlp_tensor.npy", mlp_tensor)
    np.save(output_root / "mlp_flat.npy", mlp_flat)
    np.savez_compressed(output_root / "mlp_preprocessing.npz", coordinate_mean=mean, coordinate_std=std)
    inventories = split_inventory(samples)
    write_json(output_root / "split_inventories.json", inventories)
    source_hashes = {
        "design_json": {"path": str(design_path), "sha256": sha256_file(design_path)},
        "adequacy_report": {"path": str(adequacy_path), "sha256": sha256_file(adequacy_path)},
        "extraction_config": {"path": str(extraction_config_path), "sha256": sha256_file(extraction_config_path)},
        "extraction_manifest": {"path": str(extraction_manifest_path), "sha256": sha256_file(extraction_manifest_path)},
        "selected_videos_audit_only": {"path": str(selected_videos_path), "sha256": sha256_file(selected_videos_path)},
        "label_map": {"path": str(label_map_path), "sha256": sha256_file(label_map_path)},
        "official_splits": {
            split: {
                "path": extraction_config["official_splits"][split]["path"],
                "sha256": sha256_file(Path(extraction_config["official_splits"][split]["path"])),
            }
            for split in SPLITS
        },
        "landmark_files": {"count": len(samples), "per_video_sha256": "video_manifest.csv"},
        "preparation_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_name": "synora_include21_video_features_v1",
        "source_of_truth": {
            "completed_and_included_videos": "extraction_manifest.jsonl",
            "ignored_stale_field": "selected_videos.jsonl.included_in_run",
            "landmark_root": str(landmark_root),
            "frame_schema_version": FRAME_SCHEMA_VERSION,
        },
        "canonicalization": {
            "slot_rule": "hands mapped by handedness_label into fixed Left and Right slots; hand_index ignored",
            "duplicate_handedness_policy": "when multiple hands share a label, keep the highest-handedness_score hand and count extras",
            "missing_policy": "absent slots remain zero with explicit presence, no-hand and padding masks",
            "duplicate_handedness_frames": int(sum(sample["duplicate_count"] for sample in samples)),
        },
        "primary_classes": [
            {
                "class_folder": class_name,
                "folder_id": int(class_name.split(".", 1)[0]),
                "class_label": class_by_folder[class_name],
                "label_id": class_lookup[class_by_folder[class_name]],
            }
            for class_name in classes
        ],
        "counts": {
            "videos": len(samples),
            "frames": sum(sample["frame_count"] for sample in samples),
            "by_split": validation_report["split_counts"],
        },
        "rf_feature_preparation": {
            "rows": len(samples),
            "feature_count": len(feature_names),
            "scaling": "none",
            "coordinate_system": "normalized_landmarks",
            "world_landmarks_used": False,
            "missing_value_policy": "zeros with explicit presence, no-hand and padding masks",
        },
        "mlp_feature_preparation": {
            "samples": len(samples),
            "temporal_length": T_VALUE,
            "channels": len(channel_names),
            "tensor_shape": list(mlp_tensor.shape),
            "flat_matrix_shape": list(mlp_flat.shape),
            "padding": "post-pad only; no truncation",
            "coordinate_system": "normalized_landmarks",
            "world_landmarks_used": False,
            "scaling": {
                "fit_splits": ["train"],
                "applied_to": "observed normalized landmark coordinates only",
                "masks_and_confidences_scaled": False,
                "missing_and_padded_coordinate_slots": "remain zero with masks",
                "statistics_file": "mlp_preprocessing.npz",
            },
            "channel_order": channel_names,
            "design_limitations": [
                "A plain frame-based MLP requires fixed-length 1-D inputs; mlp_flat.npy is the model-ready flattening of the audited tensor.",
                "Flattening treats temporal position as feature position and does not model temporal structure directly; a sequence model would require mlp_tensor.npy.",
                "T=89 preserves all existing frames but adds padded positions for shorter videos; padded coordinates are zero with frame_padded=1.",
                "Only normalized hand landmarks are used; world landmarks and body landmarks are excluded.",
                "Signer identity is unavailable; this is not a signer-independent representation.",
                f"{int(sum(sample['duplicate_count'] for sample in samples))} real frames contained duplicate handedness labels; the highest-handedness_score hand was kept and extras were counted."
            ],
        },
        "validation": validation_report,
        "source_hashes": source_hashes,
        "artifacts": {
            "video_manifest": "video_manifest.csv",
            "rf_features_csv": "rf_features.csv",
            "rf_features_npy": "rf_features.npy",
            "mlp_sample_manifest": "mlp_sample_manifest.csv",
            "mlp_tensor": "mlp_tensor.npy",
            "mlp_flat_matrix": "mlp_flat.npy",
            "mlp_preprocessing": "mlp_preprocessing.npz",
            "split_inventories": "split_inventories.json",
            "rf_feature_schema": "rf_feature_schema.json",
            "mlp_feature_schema": "mlp_feature_schema.json",
            "validation_report": "validation_report.json",
        },
    }
    write_json(output_root / "config.json", config)
    write_json(
        output_root / "rf_feature_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": len(samples),
            "metadata_columns": metadata_columns,
            "csv_id_columns": RF_CSV_ID_COLUMNS,
            "csv_columns": rf_csv_columns,
            "feature_columns": feature_names,
            "feature_dtype": str(RF_DTYPE),
            "scaling": "none",
            "missing_value_policy": "zero placeholders disambiguated by presence, no-hand, ratio and count masks",
            "canonicalization": config["canonicalization"],
        },
    )
    write_json(
        output_root / "mlp_feature_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "samples": len(samples),
            "tensor_shape": list(mlp_tensor.shape),
            "flat_matrix_shape": list(mlp_flat.shape),
            "temporal_length": T_VALUE,
            "channel_count": len(channel_names),
            "channel_order": channel_names,
            "coordinate_channel_indices": [0, 125],
            "mask_channel_indices": [126, 132],
            "confidence_channel_indices": [133, 134],
            "dtype": str(MLP_DTYPE),
            "coordinate_system": "normalized_landmarks",
            "world_landmarks_used": False,
            "scaling": config["mlp_feature_preparation"]["scaling"],
            "canonicalization": config["canonicalization"],
            "design_limitations": config["mlp_feature_preparation"]["design_limitations"],
        },
    )
    print(json.dumps({"output_root": str(output_root), "dimensions": validation_report["actual_dimensions"], "split_counts": validation_report["split_counts"], "frame_total": validation_report["frame_total"], "validation_passed": validation_report["passed"], "mask_summary": validation_report["mask_summary"]}, indent=2))


if __name__ == "__main__":
    main()
