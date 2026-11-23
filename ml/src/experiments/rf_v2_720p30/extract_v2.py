from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

try:
    from ml.src import extract_landmarks as extraction
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v2_720p30 import dataset
    from ml.src.experiments.rf_v2_720p30.capture import load_v2_landmarker_settings
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import extract_landmarks as extraction
    import prepare_video_features as feature_source
    from experiments.rf_v2_720p30 import dataset
    from experiments.rf_v2_720p30.capture import load_v2_landmarker_settings


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V1_FEATURES = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
DEFAULT_V1_LANDMARK_CONFIG = REPO_ROOT / "ml/dataset/processed/synora_include30_landmarks_v1/config.json"
DEFAULT_V1_EXPERIMENT_CONFIG = REPO_ROOT / "ml/reports/synora_baseline_v1_20261121/experiment_config.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"

MANIFEST_FIELDS = [
    "video_id",
    "split",
    "official_rel_path_posix",
    "class_folder",
    "class_label",
    "label_id",
    "model_index",
    "landmark_file",
    "landmark_sha256",
    "status",
    "reason",
    "source_video_abs",
    "source_video_sha256",
    "source_frame_count",
    "source_fps",
    "source_width",
    "source_height",
    "source_duration_s",
    "target_fps",
    "target_width",
    "target_height",
    "target_frame_count",
    "transformed_duration_s",
    "source_indices",
    "repeated_source_frames",
    "unused_source_frames",
    "no_hand_frames",
    "multiple_hand_frames",
    "handedness_mismatch_frames",
    "decode_errors",
    "inference_errors",
    "elapsed_s",
]


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-features-dir", type=Path, default=DEFAULT_V1_FEATURES)
    parser.add_argument("--v1-landmark-config", type=Path, default=DEFAULT_V1_LANDMARK_CONFIG)
    parser.add_argument("--v1-experiment-config", type=Path, default=DEFAULT_V1_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def build_config_payload(
    args: argparse.Namespace,
    v1_landmark_config: dict[str, Any],
    v1_experiment_config: dict[str, Any],
    entries: list[dict[str, Any]],
    model_path: Path,
    code_hash: str,
) -> dict[str, Any]:
    official_dir = Path(v1_landmark_config["official_splits"]["train"]["path"]).parent
    official_hashes = {
        split: {
            "path": v1_landmark_config["official_splits"][split]["path"],
            "sha256": v1_landmark_config["official_splits"][split]["sha256"],
        }
        for split in dataset.SPLITS
    }
    for split in dataset.SPLITS:
        current = Path(official_hashes[split]["path"])
        if not current.is_file():
            raise FileNotFoundError(current)
        actual = dataset.sha256_file(current)
        if actual != official_hashes[split]["sha256"]:
            raise RuntimeError(f"official {split} split hash mismatch")
    if not official_dir.is_dir():
        raise FileNotFoundError(official_dir)
    return {
        "schema_version": "synora_rf_v2_extraction_config_v1",
        "experiment_id": "synora_rf_v2_720p30_20261122",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "AI4Bharat INCLUDE official 21-class subset",
        "split_policy": {
            "allowed_splits": list(dataset.SPLITS),
            "test_split_read": False,
            "official_train_val_only": True,
            "selected_train_count": sum(1 for entry in entries if entry["split"] == "train"),
            "selected_val_count": sum(1 for entry in entries if entry["split"] == "val"),
        },
        "acquisition": {
            "target_fps": dataset.TARGET_FPS,
            "target_width": dataset.TARGET_WIDTH,
            "target_height": dataset.TARGET_HEIGHT,
            "max_frames": dataset.MAX_FRAMES,
            "camera_mode": "officially enumerated BisonCam NB Pro 1280x720 @ 30 FPS",
        },
        "preprocessing": {
            "spatial": "resize every decoded source frame from its native size to 1280x720 with cv2.INTER_AREA",
            "temporal": "nearest real source frame at each 30 FPS target timestamp; duplicate source frames are real-frame repeats, not synthesized landmarks",
            "resize_interpolation": "cv2.INTER_AREA",
            "resampling": "round(index * source_fps / 30.0), clamped to last real source frame",
            "no_crop": True,
            "no_mirroring": True,
            "no_augmentation": True,
            "no_smoothing": True,
            "no_synthesis": True,
            "no_silent_frame_skip": True,
            "no_silent_truncation": True,
            "over_cap_policy": "exclude clip from v2 extraction when round(source_duration_s * 30) exceeds 89; record in excluded.jsonl",
        },
        "extraction": v1_landmark_config["extraction"],
        "frame_schema_version": feature_source.FRAME_SCHEMA_VERSION,
        "label_mapping": v1_experiment_config["label_mapping"],
        "official_splits": official_hashes,
        "v1_inputs": {
            "features_dir": str(args.v1_features_dir.resolve()),
            "landmark_config": str(args.v1_landmark_config.resolve()),
            "experiment_config": str(args.v1_experiment_config.resolve()),
            "v1_manifest_sha256": dataset.sha256_file(args.v1_features_dir / "video_manifest.csv"),
            "v1_landmark_config_sha256": dataset.sha256_file(args.v1_landmark_config),
        },
        "model": {
            "path": str(model_path),
            "sha256": dataset.sha256_file(model_path),
        },
        "code_sha256": code_hash,
        "runtime": {
            "python": platform.python_version(),
            "opencv-python": cv2.__version__,
            "mediapipe": mp.__version__ if hasattr(mp, "__version__") else "unknown",
            "platform": platform.platform(),
        },
        "norm_rect_warning": {
            "observed_in_stage1": True,
            "v1_pipeline_unchanged": True,
            "v2_policy": "record warnings; no landmark alteration unless outputs fail schema/parity checks",
        },
    }


def write_config(output_root: Path, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload["config_hash"] = sha256_json(payload)
    path = output_root / "config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != payload["config_hash"]:
            raise RuntimeError("existing v2 config.json does not match current configuration")
        return path
    dataset.write_json(path, payload)
    return path


def process_video(
    entry: dict[str, Any],
    payload: dict[str, Any],
    model_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    started = time.perf_counter()
    row = dict.fromkeys(MANIFEST_FIELDS, "")
    row.update(
        {
            "video_id": entry["video_id"],
            "split": entry["split"],
            "official_rel_path_posix": entry["official_rel_path_posix"],
            "class_folder": entry["class_folder"],
            "class_label": entry["class_label"],
            "label_id": entry["label_id"],
            "model_index": entry["model_index"],
            "source_video_abs": entry["source_video_abs"],
        }
    )
    warning_messages: list[str] = []
    cap = cv2.VideoCapture(entry["source_video_abs"])
    if not cap.isOpened():
        row.update(status="failed", reason="SOURCE_UNREADABLE", elapsed_s=time.perf_counter() - started)
        return row, warning_messages
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not source_fps > 0.0 or source_frame_count <= 0 or source_width <= 0 or source_height <= 0:
            row.update(status="failed", reason="INVALID_SOURCE_METADATA", elapsed_s=time.perf_counter() - started)
            return row, warning_messages
        plan = dataset.resample_plan(source_frame_count, source_fps)
        row.update(
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            source_width=source_width,
            source_height=source_height,
            source_duration_s=plan["source_duration_s"],
            target_fps=dataset.TARGET_FPS,
            target_width=dataset.TARGET_WIDTH,
            target_height=dataset.TARGET_HEIGHT,
            target_frame_count=plan["target_frame_count"],
            transformed_duration_s=plan["transformed_duration_s"] or "",
            source_indices=json.dumps(plan["source_indices"]),
            repeated_source_frames=plan["repeated_source_frames"],
            unused_source_frames=plan["unused_source_frames"],
        )
        if plan["status"] == "excluded":
            row.update(status="excluded", reason=plan["reason"], elapsed_s=time.perf_counter() - started)
            return row, warning_messages

        needed = set(plan["source_indices"])
        decoded: dict[int, Any] = {}
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index in needed:
                decoded[index] = frame.copy()
            index += 1
        if index != source_frame_count or len(decoded) != len(needed):
            row.update(
                status="failed",
                reason="PARTIAL_SOURCE_DECODE",
                decode_errors=1,
                elapsed_s=time.perf_counter() - started,
            )
            return row, warning_messages

        output_path = output_root / "landmarks" / entry["split"] / f"{entry['video_id']}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        no_hand_frames = 0
        multiple_hand_frames = 0
        mismatch_frames = 0
        inference_errors = 0
        last_timestamp_ms = -1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            landmarker = extraction.make_landmarker(model_path, payload)
            try:
                with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
                    for target_index, source_index in enumerate(plan["source_indices"]):
                        timestamp_ms = extraction.next_timestamp_ms(
                            target_index,
                            dataset.TARGET_FPS,
                            last_timestamp_ms,
                        )
                        last_timestamp_ms = timestamp_ms
                        resized = dataset.resize_frame(decoded[source_index])
                        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        try:
                            detection = landmarker.detect_for_video(mp_image, timestamp_ms)
                        except Exception as exc:
                            inference_errors += 1
                            warning_messages.append(f"inference error: {type(exc).__name__}: {exc}")
                            continue
                        hands, mismatch = extraction.serialize_hands(detection)
                        if not hands:
                            no_hand_frames += 1
                        if len(hands) > 1:
                            multiple_hand_frames += 1
                        if mismatch:
                            mismatch_frames += 1
                        record = {
                            "schema_version": feature_source.FRAME_SCHEMA_VERSION,
                            "split": entry["split"],
                            "video_id": entry["video_id"],
                            "official_rel_path": entry["official_rel_path_posix"],
                            "source_video_abs": entry["source_video_abs"],
                            "archive": None,
                            "class_folder": entry["class_folder"],
                            "class_label": entry["class_label"],
                            "label_id": entry["label_id"],
                            "frame_index": target_index,
                            "timestamp_ms": timestamp_ms,
                            "source_timestamp_ms": float(source_index * 1000.0 / source_fps),
                            "fps": dataset.TARGET_FPS,
                            "width": dataset.TARGET_WIDTH,
                            "height": dataset.TARGET_HEIGHT,
                            "source_frame_index": source_index,
                            "source_fps": source_fps,
                            "source_width": source_width,
                            "source_height": source_height,
                            "num_hands": len(hands),
                            "hands": hands,
                            "handedness_world_landmark_mismatch": mismatch,
                            "decode_ok": True,
                            "infer_ok": True,
                            "error": None,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
            finally:
                landmarker.close()
        warning_messages.extend(str(item.message) for item in caught)
    finally:
        cap.release()

    written = 0
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as handle:
            written = sum(1 for line in handle if line.strip())
    row.update(
        landmark_file=str(output_path),
        landmark_sha256=dataset.sha256_file(output_path) if written else "",
        source_video_sha256=dataset.sha256_file(entry["source_video_abs"]),
        status="complete" if written == plan["target_frame_count"] and inference_errors == 0 else "failed",
        reason=None if written == plan["target_frame_count"] and inference_errors == 0 else "INFERENCE_OR_WRITE_MISMATCH",
        no_hand_frames=no_hand_frames,
        multiple_hand_frames=multiple_hand_frames,
        handedness_mismatch_frames=mismatch_frames,
        inference_errors=inference_errors,
        elapsed_s=time.perf_counter() - started,
    )
    return row, warning_messages


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    v1_experiment_config = json.loads(args.v1_experiment_config.read_text(encoding="utf-8"))
    v1_landmark_config = json.loads(args.v1_landmark_config.read_text(encoding="utf-8"))
    official_split_dir = Path(
        v1_landmark_config["official_splits"]["train"]["path"]
    ).parent
    entries = dataset.load_train_val_entries(
        args.v1_features_dir / "video_manifest.csv",
        official_split_dir,
        v1_experiment_config["label_mapping"],
    )
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        entries = entries[: args.limit]
    payload, model_path = load_v2_landmarker_settings(args.v1_landmark_config)
    code_hash = dataset.sha256_file(Path(__file__).resolve())
    config_payload = build_config_payload(
        args,
        v1_landmark_config,
        v1_experiment_config,
        entries,
        model_path,
        code_hash,
    )
    write_config(output_root, config_payload)

    dataset.write_jsonl(output_root / "selected_trainval.jsonl", entries)
    dataset.write_json(output_root / "label_mapping.json", v1_experiment_config["label_mapping"])

    manifest_rows: list[dict[str, Any]] = []
    warning_messages: list[str] = []
    for number, entry in enumerate(entries, start=1):
        row, warnings_for_video = process_video(entry, config_payload, model_path, output_root)
        manifest_rows.append(row)
        warning_messages.extend(warnings_for_video)
        print(
            f"[{number}/{len(entries)}] {row['split']} {row['video_id'][:12]} "
            f"{row['status']} {row['reason'] or ''}"
        )

    dataset.write_csv(output_root / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    excluded_rows = [row for row in manifest_rows if row["status"] in ("excluded", "failed")]
    dataset.write_jsonl(output_root / "excluded.jsonl", excluded_rows)
    warning_counts: dict[str, int] = {}
    for message in warning_messages:
        key = message.strip()
        warning_counts[key] = warning_counts.get(key, 0) + 1
    dataset.write_json(output_root / "warnings.json", warning_counts)

    complete = [row for row in manifest_rows if row["status"] == "complete"]
    counts = {
        split: sum(1 for row in complete if row["split"] == split)
        for split in dataset.SPLITS
    }
    dataset.write_json(
        output_root / "extraction_summary.json",
        {
            "selected_counts": {
                split: sum(1 for entry in entries if entry["split"] == split)
                for split in dataset.SPLITS
            },
            "complete_counts": counts,
            "excluded_counts": {
                split: sum(1 for row in excluded_rows if row["split"] == split)
                for split in dataset.SPLITS
            },
            "failed_video_ids": [row["video_id"] for row in manifest_rows if row["status"] == "failed"],
            "excluded_video_ids": [row["video_id"] for row in manifest_rows if row["status"] == "excluded"],
            "warning_counts": warning_counts,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
