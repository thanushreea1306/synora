import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCHEMA_VERSION = "synora_frame_v1"
CONFIG_SCHEMA_VERSION = "synora_extraction_config_v1"
DEFAULT_DATASET_ROOT = Path(r"D:\SYNORA_DATA\INCLUDE")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "processed"
    / "synora_include30_landmarks_v1"
)
OFFICIAL_SPLITS = {
    "train": "include_train.txt",
    "val": "include_val.txt",
    "test": "include_test.txt",
}
SELECTED_CLASSES = (
    "1. loud",
    "2. quiet",
    "3. happy",
    "4. sad",
    "5. Beautiful",
    "17. flat",
    "18. curved",
    "21. tight",
    "22. loose",
    "29. clean",
    "30. dirty",
    "78. long",
    "79. short",
    "80. tall",
    "81. wide",
    "82. narrow",
    "83. big large",
    "84. small little",
    "85. slow",
    "86. fast",
    "87. hot",
    "88. cold",
    "89. warm",
    "90. cool",
    "91. new",
    "92. old",
    "93. young",
    "94. good",
    "95. bad",
    "96. wet",
)
LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence", "name")
CATEGORY_FIELDS = ("index", "score", "display_name", "category_name")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    ensure_parent(path)
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def write_jsonl(path, rows):
    ensure_parent(path)
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temp_path, path)


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    return value


def class_folder_to_label(class_folder):
    if ". " in class_folder:
        class_folder = class_folder.split(". ", 1)[1]
    return "".join(class_folder.lower().split())


def build_class_archive_map(dataset_root):
    extract_root = dataset_root / "_extract_test"
    class_to_archive = {}
    for archive_dir in sorted(extract_root.glob("Adjectives_*of8")):
        adjective_root = archive_dir / "Adjectives"
        if not adjective_root.is_dir():
            continue
        for class_dir in sorted(adjective_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.lower() == "extra":
                continue
            existing = class_to_archive.get(class_dir.name)
            if existing is not None and existing != archive_dir.name:
                raise RuntimeError(
                    f"class {class_dir.name!r} found in both {existing} and {archive_dir.name}"
                )
            class_to_archive[class_dir.name] = archive_dir.name
    return class_to_archive


def load_label_map(dataset_root):
    path = (
        dataset_root
        / "_official_metadata"
        / "AI4Bharat_INCLUDE"
        / "label_maps"
        / "label_map_include.json"
    )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def build_selected_videos(dataset_root, class_to_archive, label_map):
    selected_set = set(SELECTED_CLASSES)
    entries = []
    split_root = dataset_root / "_official_metadata" / "AI4Bharat_INCLUDE" / "train_test_paths"
    for split, filename in OFFICIAL_SPLITS.items():
        split_path = split_root / filename
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in handle:
                official_rel_path = line.rstrip("\r\n")
                if not official_rel_path:
                    continue
                rel_posix = official_rel_path.replace("\\", "/")
                parts = rel_posix.split("/")
                if len(parts) < 3 or parts[0] != "Adjectives" or parts[1] not in selected_set:
                    continue
                class_folder = parts[1]
                archive = class_to_archive.get(class_folder)
                if archive is None:
                    raise RuntimeError(f"no extracted archive found for class {class_folder!r}")
                source_path = dataset_root / "_extract_test" / archive / Path(*parts)
                class_label = class_folder_to_label(class_folder)
                if class_label not in label_map:
                    raise RuntimeError(f"class {class_folder!r} has no include label-map entry")
                entries.append(
                    {
                        "video_id": sha256_text(rel_posix),
                        "split": split,
                        "official_rel_path": official_rel_path,
                        "official_rel_path_posix": rel_posix,
                        "source_video_abs": str(source_path),
                        "source_video_exists": source_path.is_file(),
                        "archive": archive,
                        "class_folder": class_folder,
                        "class_label": class_label,
                        "label_id": int(label_map[class_label]),
                        "included_in_run": False,
                    }
                )
    entries.sort(key=lambda item: (item["split"], item["official_rel_path_posix"]))
    return entries


def select_run_entries(entries, scope, pilot_count):
    if scope == "full":
        return {entry["video_id"] for entry in entries}
    train_entries = sorted(
        (entry for entry in entries if entry["split"] == "train"),
        key=lambda item: item["official_rel_path_posix"],
    )
    if pilot_count > len(train_entries):
        raise RuntimeError(
            f"requested {pilot_count} pilot videos but only {len(train_entries)} train videos exist"
        )
    return {entry["video_id"] for entry in train_entries[:pilot_count]}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_config_payload(
    args,
    dataset_root,
    output_root,
    model_path,
    split_paths,
    label_map_path,
    entries,
    run_video_ids,
):
    split_hashes = {}
    for split, path in split_paths.items():
        split_hashes[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    selected_videos_path = output_root / "selected_videos.jsonl"
    selected_videos_hash = sha256_file(selected_videos_path)
    pilot_video_ids = None
    pilot_selection_rule = None
    if args.scope == "pilot":
        pilot_video_ids = sorted(run_video_ids)
        pilot_selection_rule = (
            "first N train videos sorted by official_rel_path_posix; N="
            + str(args.pilot_train_videos)
        )
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scope": args.scope,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "selected_classes": list(SELECTED_CLASSES),
        "selected_videos_sha256": selected_videos_hash,
        "selected_video_count": len(entries),
        "official_splits": split_hashes,
        "label_map": {
            "path": str(label_map_path),
            "sha256": sha256_file(label_map_path),
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "mediapipe": package_version("mediapipe"),
            "opencv-python": package_version("opencv-python"),
            "numpy": package_version("numpy"),
            "platform": platform.platform(),
        },
        "extraction": {
            "running_mode": "VIDEO",
            "delegate": "CPU",
            "num_hands": args.num_hands,
            "min_hand_detection_confidence": args.min_hand_detection_confidence,
            "min_hand_presence_confidence": args.min_hand_presence_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "all_decoded_frames": True,
            "mirroring": False,
            "resize": False,
            "crop": False,
            "augmentation": False,
            "smoothing": False,
            "synthesis": False,
            "skip_frames": False,
        },
        "pilot": {
            "video_count": len(run_video_ids) if args.scope == "pilot" else None,
            "selection_rule": pilot_selection_rule,
            "video_ids": pilot_video_ids,
        },
    }


def validate_or_write_config(output_root, config_payload):
    config_path = output_root / "config.json"
    expected_hash = sha256_json(config_payload)
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_hash") != expected_hash:
            raise RuntimeError(
                "existing config.json does not match current configuration; "
                "use a separate output root or archive the previous run"
            )
        return existing, expected_hash
    config = dict(config_payload)
    config["config_hash"] = expected_hash
    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, config)
    return config, expected_hash


def load_existing_run(output_root, model_path, split_paths):
    config_path = output_root / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config_hash = config.get("config_hash")
    payload = {
        key: value
        for key, value in config.items()
        if key not in ("config_hash", "created_utc")
    }
    if not config_hash or config_hash != sha256_json(payload):
        raise RuntimeError("config.json hash does not match its payload")

    selected_path = output_root / "selected_videos.jsonl"
    if not selected_path.is_file():
        raise RuntimeError(f"selected_videos.jsonl not found: {selected_path}")
    if sha256_file(selected_path) != config.get("selected_videos_sha256"):
        raise RuntimeError("selected_videos.jsonl hash does not match config.json")
    if sha256_file(model_path) != config["model"]["sha256"]:
        raise RuntimeError("model hash does not match config.json")
    for split, path in split_paths.items():
        expected = config["official_splits"].get(split, {})
        if sha256_file(path) != expected.get("sha256"):
            raise RuntimeError(f"official split hash does not match config.json for {split}")

    entries = read_jsonl(selected_path)
    if len(entries) != config.get("selected_video_count"):
        raise RuntimeError("selected_videos.jsonl count does not match config.json")
    pilot_ids = set(config.get("pilot", {}).get("video_ids") or [])
    manifest_rows = read_jsonl(output_root / "extraction_manifest.jsonl")
    complete_ids = set()
    for row in manifest_rows:
        if row.get("status") != "complete":
            continue
        if row.get("config_hash") != config_hash:
            continue
        if row.get("model_sha256") != config["model"]["sha256"]:
            continue
        if not Path(row.get("output_file", "")).is_file():
            continue
        complete_ids.add(row.get("video_id"))
    if not pilot_ids.issubset(complete_ids):
        raise RuntimeError("one or more pilot videos are not complete")
    remaining_ids = {entry["video_id"] for entry in entries} - complete_ids
    return config, config_hash, entries, remaining_ids


def pending_manifest_row(entry, config_hash, config, output_root):
    output_path = output_root / "landmarks" / entry["split"] / f"{entry['video_id']}.jsonl"
    return {
        "video_id": entry["video_id"],
        "included_in_run": entry["included_in_run"],
        "status": "pending",
        "split": entry["split"],
        "official_rel_path": entry["official_rel_path"],
        "official_rel_path_posix": entry["official_rel_path_posix"],
        "source_video_abs": entry["source_video_abs"],
        "source_video_exists": entry["source_video_exists"],
        "archive": entry["archive"],
        "class_folder": entry["class_folder"],
        "class_label": entry["class_label"],
        "label_id": entry["label_id"],
        "output_file": str(output_path),
        "config_hash": config_hash,
        "model_sha256": config["model"]["sha256"],
        "metadata_frame_count": None,
        "frames_read": 0,
        "frames_written": 0,
        "no_hand_frames": 0,
        "multiple_hand_frames": 0,
        "handedness_mismatch_frames": 0,
        "decode_errors": 0,
        "inference_errors": 0,
        "elapsed_s": 0.0,
        "ms_per_frame": None,
        "error": None,
    }


def load_manifest(output_root, entries, config_hash, config):
    manifest_path = output_root / "extraction_manifest.jsonl"
    existing_rows = read_jsonl(manifest_path)
    existing_by_id = {row.get("video_id"): row for row in existing_rows}
    rows = []
    for entry in entries:
        previous = existing_by_id.get(entry["video_id"])
        output_path = output_root / "landmarks" / entry["split"] / f"{entry['video_id']}.jsonl"
        reusable = (
            previous is not None
            and previous.get("status") == "complete"
            and previous.get("config_hash") == config_hash
            and previous.get("model_sha256") == config["model"]["sha256"]
            and previous.get("output_file") == str(output_path)
            and Path(previous.get("output_file", "")).is_file()
        )
        if reusable:
            row = dict(previous)
            row["included_in_run"] = entry["included_in_run"]
        else:
            row = pending_manifest_row(entry, config_hash, config, output_root)
        rows.append(row)
    return rows


def serialize_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def serialize_landmark(landmark):
    return {field: serialize_value(getattr(landmark, field, None)) for field in LANDMARK_FIELDS}


def serialize_category(category):
    return {field: serialize_value(getattr(category, field, None)) for field in CATEGORY_FIELDS}


def handedness_label(categories):
    for category in categories:
        label = getattr(category, "display_name", None) or getattr(category, "category_name", None)
        if label:
            return str(label)
    return None


def handedness_score(categories):
    if not categories:
        return None
    return serialize_value(getattr(categories[0], "score", None))


def serialize_hands(result):
    hand_landmarks = list(result.hand_landmarks or [])
    world_landmarks = list(result.hand_world_landmarks or [])
    handedness = list(result.handedness or [])
    hands = []
    for hand_index, landmarks in enumerate(hand_landmarks):
        categories = handedness[hand_index] if hand_index < len(handedness) else []
        categories = list(categories or [])
        world = world_landmarks[hand_index] if hand_index < len(world_landmarks) else None
        hands.append(
            {
                "hand_index": hand_index,
                "handedness_label": handedness_label(categories),
                "handedness_score": handedness_score(categories),
                "handedness": [serialize_category(category) for category in categories],
                "landmark_count": len(landmarks),
                "normalized_landmarks": [serialize_landmark(item) for item in landmarks],
                "world_landmarks": (
                    [serialize_landmark(item) for item in world] if world is not None else None
                ),
            }
        )
    mismatch = len(hand_landmarks) != len(handedness) or len(hand_landmarks) != len(world_landmarks)
    return hands, mismatch


def next_timestamp_ms(frame_index, fps, last_timestamp_ms):
    if fps is not None and math.isfinite(fps) and fps > 0:
        timestamp_ms = int(round(frame_index * 1000.0 / fps))
    else:
        timestamp_ms = int(frame_index)
    return max(timestamp_ms, last_timestamp_ms + 1)


def make_landmarker(model_path, config):
    extraction = config["extraction"]
    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(extraction["num_hands"]),
        min_hand_detection_confidence=float(extraction["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(extraction["min_hand_presence_confidence"]),
        min_tracking_confidence=float(extraction["min_tracking_confidence"]),
    )
    return vision.HandLandmarker.create_from_options(options)


def process_video(row, entry, config, config_hash, model_path, output_root):
    started = time.perf_counter()
    output_path = Path(row["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(row)
    result["config_hash"] = config_hash
    result["model_sha256"] = config["model"]["sha256"]
    result["metadata_frame_count"] = None
    result["frames_read"] = 0
    result["frames_written"] = 0
    result["no_hand_frames"] = 0
    result["multiple_hand_frames"] = 0
    result["handedness_mismatch_frames"] = 0
    result["decode_errors"] = 0
    result["inference_errors"] = 0
    result["error"] = None
    result["status"] = "failed"

    cap = cv2.VideoCapture(str(entry["source_video_abs"]))
    if not cap.isOpened():
        output_path.write_text("", encoding="utf-8")
        result["status"] = "unreadable"
        result["error"] = "could not open source video"
        result["elapsed_s"] = time.perf_counter() - started
        result["ms_per_frame"] = None
        return result

    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    result["metadata_frame_count"] = metadata_frame_count
    result["fps"] = fps if math.isfinite(fps) and fps > 0 else None
    result["width"] = width
    result["height"] = height
    result["duration_s"] = (metadata_frame_count / fps) if fps > 0 else None

    last_timestamp_ms = -1
    frames_read = 0
    frames_written = 0
    no_hand_frames = 0
    multiple_hand_frames = 0
    mismatch_frames = 0
    inference_errors = 0
    errors = []
    handle = open(output_path, "w", encoding="utf-8", newline="\n")

    try:
        landmarker = make_landmarker(model_path, config)
    except Exception as exc:
        handle.close()
        cap.release()
        result["status"] = "failed"
        result["error"] = f"could not create HandLandmarker: {exc}"
        result["elapsed_s"] = time.perf_counter() - started
        result["ms_per_frame"] = None
        return result

    with landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if frames_read == 0:
                    errors.append("could not decode first frame")
                    result["status"] = "unreadable"
                    result["decode_errors"] += 1
                elif metadata_frame_count > 0 and frames_read < metadata_frame_count:
                    errors.append(
                        f"decode stopped after {frames_read} of {metadata_frame_count} frames"
                    )
                    result["status"] = "partial"
                    result["decode_errors"] += 1
                break

            frame_index = frames_read
            frames_read += 1
            source_timestamp = finite_float(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_ms = next_timestamp_ms(frame_index, fps, last_timestamp_ms)
            last_timestamp_ms = timestamp_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                detection = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as exc:
                inference_errors += 1
                errors.append(f"inference failed at frame_index={frame_index}: {exc}")
                result["status"] = "partial"
                frame_record = {
                    "schema_version": SCHEMA_VERSION,
                    "split": entry["split"],
                    "video_id": entry["video_id"],
                    "official_rel_path": entry["official_rel_path"],
                    "source_video_abs": entry["source_video_abs"],
                    "archive": entry["archive"],
                    "class_folder": entry["class_folder"],
                    "class_label": entry["class_label"],
                    "label_id": entry["label_id"],
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "source_timestamp_ms": source_timestamp,
                    "fps": fps if fps > 0 else None,
                    "width": width,
                    "height": height,
                    "num_hands": 0,
                    "hands": [],
                    "decode_ok": True,
                    "infer_ok": False,
                    "error": str(exc),
                }
                handle.write(json.dumps(frame_record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                frames_written += 1
                break

            hands, mismatch = serialize_hands(detection)
            num_hands = len(hands)
            if num_hands == 0:
                no_hand_frames += 1
            if num_hands > 1:
                multiple_hand_frames += 1
            if mismatch:
                mismatch_frames += 1

            frame_record = {
                "schema_version": SCHEMA_VERSION,
                "split": entry["split"],
                "video_id": entry["video_id"],
                "official_rel_path": entry["official_rel_path"],
                "source_video_abs": entry["source_video_abs"],
                "archive": entry["archive"],
                "class_folder": entry["class_folder"],
                "class_label": entry["class_label"],
                "label_id": entry["label_id"],
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "source_timestamp_ms": source_timestamp,
                "fps": fps if fps > 0 else None,
                "width": width,
                "height": height,
                "num_hands": num_hands,
                "hands": hands,
                "handedness_world_landmark_mismatch": mismatch,
                "decode_ok": True,
                "infer_ok": True,
                "error": None,
            }
            handle.write(json.dumps(frame_record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            frames_written += 1

    handle.close()
    cap.release()

    if metadata_frame_count > 0 and frames_read != metadata_frame_count:
        errors.append(
            f"frame count mismatch: metadata={metadata_frame_count} decoded={frames_read}"
        )
    if frames_read != frames_written:
        errors.append(f"frame write mismatch: decoded={frames_read} written={frames_written}")
    if errors and result["status"] in ("failed", "complete"):
        result["status"] = "partial"

    result["frames_read"] = frames_read
    result["frames_written"] = frames_written
    result["no_hand_frames"] = no_hand_frames
    result["multiple_hand_frames"] = multiple_hand_frames
    result["handedness_mismatch_frames"] = mismatch_frames
    result["inference_errors"] = inference_errors
    result["error"] = "; ".join(errors) if errors else None
    if not errors and frames_read > 0:
        result["status"] = "complete"
    result["elapsed_s"] = time.perf_counter() - started
    result["ms_per_frame"] = (
        result["elapsed_s"] * 1000.0 / frames_written if frames_written > 0 else None
    )
    return result


def build_leakage_audit(entries, manifest_rows, config_hash, config, elapsed_s, frames_written):
    selected_by_id = {entry["video_id"]: entry for entry in entries}
    ids_by_split = collections.defaultdict(set)
    splits_by_id = collections.defaultdict(set)
    for entry in entries:
        ids_by_split[entry["split"]].add(entry["video_id"])
        splits_by_id[entry["video_id"]].add(entry["split"])
    duplicate_ids = sorted(video_id for video_id, splits in splits_by_id.items() if len(splits) > 1)
    split_conflicts = []
    config_conflicts = []
    model_conflicts = []
    path_conflicts = []
    for row in manifest_rows:
        video_id = row["video_id"]
        selected = selected_by_id.get(video_id)
        if selected is None:
            split_conflicts.append({"video_id": video_id, "reason": "not_selected"})
            continue
        if row.get("split") != selected["split"]:
            split_conflicts.append(
                {
                    "video_id": video_id,
                    "reason": "manifest_split_mismatch",
                    "manifest_split": row.get("split"),
                    "selected_split": selected["split"],
                }
            )
        if row.get("config_hash") != config_hash:
            config_conflicts.append(video_id)
        if row.get("model_sha256") != config["model"]["sha256"]:
            model_conflicts.append(video_id)
        expected_path = str(
            Path(row.get("output_file", "")).parent.parent
            / selected["split"]
            / f"{video_id}.jsonl"
        )
        if row.get("output_file") != expected_path:
            path_conflicts.append(video_id)

    status_counts = dict(collections.Counter(row["status"] for row in manifest_rows))
    attempted_rows = [row for row in manifest_rows if row["included_in_run"]]
    attempted_frames = sum(int(row.get("frames_written") or 0) for row in attempted_rows)
    attempted_ms = sum(float(row.get("elapsed_s") or 0.0) for row in attempted_rows) * 1000.0
    measured_ms_per_frame = attempted_ms / attempted_frames if attempted_frames else None
    passed = not (
        duplicate_ids
        or split_conflicts
        or config_conflicts
        or model_conflicts
        or path_conflicts
    )
    return {
        "schema_version": "synora_leakage_audit_v1",
        "passed": passed,
        "selected_video_count": len(entries),
        "selected_by_split": {split: len(ids) for split, ids in ids_by_split.items()},
        "duplicate_video_ids_across_splits": duplicate_ids,
        "split_conflicts": split_conflicts,
        "config_hash_conflicts": config_conflicts,
        "model_hash_conflicts": model_conflicts,
        "output_path_conflicts": path_conflicts,
        "manifest_status_counts": status_counts,
        "attempted_video_count": len(attempted_rows),
        "attempted_frames_written": attempted_frames,
        "all_frames_written": frames_written,
        "run_elapsed_s": elapsed_s,
        "measured_ms_per_frame": measured_ms_per_frame,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scope", choices=("pilot", "remaining", "full"), default="pilot")
    parser.add_argument("--pilot-train-videos", type=int, default=5)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise RuntimeError(f"dataset root does not exist: {dataset_root}")
    if not model_path.is_file():
        raise RuntimeError(f"model does not exist: {model_path}")

    split_paths = {
        split: dataset_root
        / "_official_metadata"
        / "AI4Bharat_INCLUDE"
        / "train_test_paths"
        / filename
        for split, filename in OFFICIAL_SPLITS.items()
    }
    for split, path in split_paths.items():
        if not path.is_file():
            raise RuntimeError(f"official split file missing for {split}: {path}")

    if args.scope == "remaining":
        config, config_hash, entries, run_video_ids = load_existing_run(
            output_root,
            model_path,
            split_paths,
        )
        for entry in entries:
            if entry["video_id"] in run_video_ids:
                entry["included_in_run"] = True
    else:
        class_to_archive = build_class_archive_map(dataset_root)
        label_map, label_map_path = load_label_map(dataset_root)
        entries = build_selected_videos(dataset_root, class_to_archive, label_map)
        run_video_ids = select_run_entries(entries, args.scope, args.pilot_train_videos)
        for entry in entries:
            entry["included_in_run"] = entry["video_id"] in run_video_ids

        write_jsonl(output_root / "selected_videos.jsonl", entries)
        config_payload = build_config_payload(
            args,
            dataset_root,
            output_root,
            model_path,
            split_paths,
            label_map_path,
            entries,
            run_video_ids,
        )
        config, config_hash = validate_or_write_config(output_root, config_payload)

    manifest_rows = load_manifest(output_root, entries, config_hash, config)
    manifest_by_id = {row["video_id"]: row for row in manifest_rows}
    for video_id in run_video_ids:
        manifest_by_id[video_id]["included_in_run"] = True
    failure_rows = [row for row in manifest_rows if row["status"] in ("unreadable", "partial", "failed")]
    run_started = time.perf_counter()
    processed = 0

    for entry in entries:
        if not entry["included_in_run"]:
            continue
        row = manifest_by_id[entry["video_id"]]
        if row["status"] == "complete":
            continue
        updated = process_video(row, entry, config, config_hash, model_path, output_root)
        manifest_by_id[entry["video_id"]] = updated
        failure_rows = [failure for failure in failure_rows if failure["video_id"] != entry["video_id"]]
        if updated["status"] != "complete":
            failure_rows.append(updated)
        processed += 1

    manifest_rows = [manifest_by_id[entry["video_id"]] for entry in entries]
    write_jsonl(output_root / "extraction_manifest.jsonl", manifest_rows)
    write_jsonl(output_root / "failures.jsonl", failure_rows)
    elapsed_s = time.perf_counter() - run_started
    attempted_rows = [manifest_by_id[video_id] for video_id in run_video_ids]
    attempted_frames = sum(int(row.get("frames_written") or 0) for row in attempted_rows)
    measured_ms_per_frame = (
        elapsed_s * 1000.0 / attempted_frames if attempted_frames > 0 else None
    )
    audit = build_leakage_audit(
        entries,
        manifest_rows,
        config_hash,
        config,
        elapsed_s,
        attempted_frames,
    )
    write_json(output_root / "leakage_audit.json", audit)

    summary = {
        "scope": args.scope,
        "selected_videos": len(entries),
        "run_video_ids": sorted(run_video_ids),
        "processed_videos": processed,
        "attempted_videos": len(attempted_rows),
        "frames_written": attempted_frames,
        "elapsed_s": elapsed_s,
        "ms_per_frame": measured_ms_per_frame,
        "status_counts": audit["manifest_status_counts"],
        "leakage_audit_passed": audit["passed"],
        "output_root": str(output_root),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not audit["passed"]:
        return 2
    if any(row["status"] != "complete" for row in attempted_rows):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
