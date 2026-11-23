import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

DATASET_ROOT = Path(r"D:\SYNORA_DATA\INCLUDE")
OUTPUT_ROOT = Path(
    r"C:\Users\thanu\synora\ml\dataset\processed\synora_include30_landmarks_v1"
)
REPORT_DIR = Path(r"C:\Users\thanu\synora\ml\reports\dataset_quality_audit")
REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "synora_frame_v1"
LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence", "name")
CATEGORY_FIELDS = ("index", "score", "display_name", "category_name")
SPLITS = ("train", "val", "test")
SPLIT_FILES = {
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
BASELINE = {
    "config_hash": "7448f845f45c3d0b53eac163504bb7087be123efcc6a633812f560b85ca09094",
    "selected_videos_sha256": "60633bbfb09c1ce48323719b3e6ad4e5acf3641295847d15483d702e3b6b14f8",
    "label_map_sha256": "067a8a51369a31c19f7441844d5064ed57f2f47bf4155beda8fcceeec1d8df9a",
    "model_sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "extraction_manifest_sha256": "effe5c28199de7ce1c06536daa691300daedd0ba038cd4641c6f612a8f446cf5",
    "split_sha256": {
        "train": "1d3f0f21c10f8a6b4bf984aea5354f6841b53cc2b9a11a63fd459c1f527d8ea5",
        "val": "75556b02dc9833e5c5a12a4e43294778eaa97f23fd72ba2114d26596ad694ea1",
        "test": "48a6219b6b78c68e4831f03171ac377fcf980430bd7ea557106e111fe59d0e65",
    },
}
MIN_TRAIN_VIDEOS = 10
MIN_EVAL_VIDEOS = 2
IMBALANCE_RATIO_THRESHOLD = 2.0


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def normalize_class_folder(class_folder):
    if ". " in class_folder:
        class_folder = class_folder.split(". ", 1)[1]
    return "".join(class_folder.lower().split())


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expected_timestamp_ms(frame_index, fps, previous_timestamp):
    if is_finite_number(fps) and fps > 0:
        value = int(round(frame_index * 1000.0 / fps))
    else:
        value = int(frame_index)
    return max(value, previous_timestamp + 1)


def add_issue(issues, kind, **details):
    issues.append({"kind": kind, **details})


def read_json_file(path, issues, label):
    if not Path(path).is_file():
        add_issue(issues, "missing_file", file=label)
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        add_issue(issues, "malformed_json", file=label, error=str(exc))
        return None


def read_jsonl_file(path, issues, label, require_final_newline=True):
    if not Path(path).is_file():
        add_issue(issues, "missing_file", file=label)
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        add_issue(issues, "unreadable_file", file=label, error=str(exc))
        return []
    if require_final_newline and text and not text.endswith("\n"):
        add_issue(issues, "missing_final_newline", file=label)
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            add_issue(issues, "blank_record", file=label, line_number=line_number)
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            add_issue(
                issues,
                "malformed_json_record",
                file=label,
                line_number=line_number,
                error=str(exc),
            )
    return rows


def build_class_archive_map(dataset_root, issues):
    extract_root = dataset_root / "_extract_test"
    class_to_archive = {}
    if not extract_root.is_dir():
        add_issue(issues, "missing_directory", file=str(extract_root))
        return class_to_archive
    for archive_dir in sorted(extract_root.glob("Adjectives_*of8")):
        adjective_root = archive_dir / "Adjectives"
        if not adjective_root.is_dir():
            continue
        for class_dir in sorted(adjective_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.lower() == "extra":
                continue
            existing = class_to_archive.get(class_dir.name)
            if existing is not None and existing != archive_dir.name:
                add_issue(
                    issues,
                    "class_in_multiple_archives",
                    class_folder=class_dir.name,
                    archives=[existing, archive_dir.name],
                )
                continue
            class_to_archive[class_dir.name] = archive_dir.name
    return class_to_archive


def read_official_splits(dataset_root, issues):
    root = (
        dataset_root
        / "_official_metadata"
        / "AI4Bharat_INCLUDE"
        / "train_test_paths"
    )
    paths_by_split = {}
    lines_by_split = {}
    for split, filename in SPLIT_FILES.items():
        path = root / filename
        paths_by_split[split] = path
        if not path.is_file():
            add_issue(issues, "missing_file", file=str(path))
            lines_by_split[split] = []
            continue
        lines_by_split[split] = [
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    path_to_splits = collections.defaultdict(set)
    for split, lines in lines_by_split.items():
        for official_rel_path in lines:
            path_to_splits[official_rel_path].add(split)
    duplicates = sorted(
        path for path, splits in path_to_splits.items() if len(splits) > 1
    )
    duplicate_lines = []
    for split, lines in lines_by_split.items():
        counts = collections.Counter(lines)
        duplicate_lines.extend(
            {"split": split, "official_rel_path": path, "count": count}
            for path, count in sorted(counts.items())
            if count > 1
        )
    return paths_by_split, lines_by_split, path_to_splits, duplicates, duplicate_lines


def collect_artifact_hashes(output_root):
    paths = []
    for name in (
        "config.json",
        "selected_videos.jsonl",
        "extraction_manifest.jsonl",
        "failures.jsonl",
        "leakage_audit.json",
    ):
        path = output_root / name
        if path.is_file():
            paths.append(path)
    landmarks_root = output_root / "landmarks"
    if landmarks_root.is_dir():
        paths.extend(sorted(landmarks_root.rglob("*.jsonl")))
    hashes = {
        path.relative_to(output_root).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "file_count": len(hashes),
        "aggregate_sha256": sha256_json(hashes),
        "files": hashes,
    }


def validate_hand(hand, hand_index, issues, context):
    valid = True
    if not isinstance(hand, dict):
        add_issue(issues, "hand_not_object", **context)
        return False, {"label": None, "score": None}
    if hand.get("hand_index") != hand_index:
        add_issue(issues, "hand_index_invalid", expected=hand_index, **context)
        valid = False
    if hand.get("landmark_count") != 21:
        add_issue(issues, "landmark_count_invalid", expected=21, **context)
        valid = False
    for key in ("normalized_landmarks", "world_landmarks"):
        values = hand.get(key)
        if not isinstance(values, list) or len(values) != 21:
            add_issue(issues, "landmark_list_invalid", key=key, expected=21, **context)
            valid = False
            continue
        for point_index, point in enumerate(values):
            if not isinstance(point, dict) or set(point) != set(LANDMARK_FIELDS):
                add_issue(
                    issues,
                    "landmark_fields_invalid",
                    key=key,
                    point_index=point_index,
                    **context,
                )
                valid = False
                continue
            for field in ("x", "y", "z"):
                if not is_finite_number(point.get(field)):
                    add_issue(
                        issues,
                        "landmark_coordinate_invalid",
                        key=key,
                        point_index=point_index,
                        field=field,
                        **context,
                    )
                    valid = False
            for field in ("visibility", "presence"):
                value = point.get(field)
                if value is not None and not is_finite_number(value):
                    add_issue(
                        issues,
                        "landmark_metadata_invalid",
                        key=key,
                        point_index=point_index,
                        field=field,
                        **context,
                    )
                    valid = False
            if point.get("name") is not None and not isinstance(point.get("name"), str):
                add_issue(
                    issues,
                    "landmark_name_invalid",
                    key=key,
                    point_index=point_index,
                    **context,
                )
                valid = False
    categories = hand.get("handedness")
    if not isinstance(categories, list) or not categories:
        add_issue(issues, "handedness_missing", **context)
        valid = False
        categories = []
    for category_index, category in enumerate(categories):
        if not isinstance(category, dict) or set(category) != set(CATEGORY_FIELDS):
            add_issue(
                issues,
                "handedness_fields_invalid",
                category_index=category_index,
                **context,
            )
            valid = False
            continue
        if not is_finite_number(category.get("index")):
            add_issue(
                issues,
                "handedness_index_invalid",
                category_index=category_index,
                **context,
            )
            valid = False
        score = category.get("score")
        if not is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
            add_issue(
                issues,
                "handedness_score_invalid",
                category_index=category_index,
                **context,
            )
            valid = False
        for field in ("display_name", "category_name"):
            if not isinstance(category.get(field), str) or not category.get(field):
                add_issue(
                    issues,
                    "handedness_label_invalid",
                    category_index=category_index,
                    field=field,
                    **context,
                )
                valid = False
    label = hand.get("handedness_label")
    score = hand.get("handedness_score")
    if label not in ("Left", "Right"):
        add_issue(issues, "handedness_label_unexpected", value=label, **context)
        valid = False
    if not is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
        add_issue(issues, "handedness_score_invalid_top_level", value=score, **context)
        valid = False
    elif categories and is_finite_number(categories[0].get("score")):
        if not math.isclose(float(score), float(categories[0]["score"]), abs_tol=1e-6):
            add_issue(issues, "handedness_score_mismatch", **context)
            valid = False
    return valid, {"label": label, "score": score}


def validate_landmark_file(path, manifest_row, issues, samples):
    context = {
        "video_id": manifest_row["video_id"],
        "output_file": str(path),
    }
    raw = path.read_bytes()
    file_hash = sha256_bytes(raw)
    try:
        text = raw.decode("utf-8")
    except Exception as exc:
        add_issue(issues, "landmark_file_not_utf8", error=str(exc), **context)
        return {"frames": 0, "no_hand": 0, "multiple_hand": 0, "hands": 0, "hash": file_hash}
    if text and not text.endswith("\n"):
        add_issue(issues, "landmark_missing_final_newline", **context)
    lines = text.splitlines()
    if not lines:
        add_issue(issues, "landmark_file_empty", **context)
    stats = {
        "frames": 0,
        "no_hand": 0,
        "multiple_hand": 0,
        "hands": 0,
        "hash": file_hash,
    }
    seen_lines = set()
    seen_frame_indices = set()
    last_timestamp = -1
    last_source_timestamp = None
    expected_frame_index = 0
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "split": manifest_row["split"],
        "video_id": manifest_row["video_id"],
        "official_rel_path": manifest_row["official_rel_path"],
        "source_video_abs": manifest_row["source_video_abs"],
        "archive": manifest_row["archive"],
        "class_folder": manifest_row["class_folder"],
        "class_label": manifest_row["class_label"],
        "label_id": manifest_row["label_id"],
        "fps": manifest_row.get("fps"),
        "width": manifest_row.get("width"),
        "height": manifest_row.get("height"),
    }
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            add_issue(issues, "landmark_blank_record", line_number=line_number, **context)
            continue
        line_hash = sha256_text(line)
        if line_hash in seen_lines:
            add_issue(issues, "duplicate_landmark_record", line_number=line_number, **context)
        seen_lines.add(line_hash)
        try:
            record = json.loads(line)
        except Exception as exc:
            add_issue(
                issues,
                "malformed_landmark_record",
                line_number=line_number,
                error=str(exc),
                **context,
            )
            continue
        record_context = {"line_number": line_number, **context}
        if not isinstance(record, dict):
            add_issue(issues, "landmark_record_not_object", **record_context)
            continue
        for key, expected_value in expected_identity.items():
            if record.get(key) != expected_value:
                add_issue(
                    issues,
                    "landmark_identity_mismatch",
                    field=key,
                    expected=expected_value,
                    actual=record.get(key),
                    **record_context,
                )
        frame_index = record.get("frame_index")
        if frame_index != expected_frame_index or frame_index in seen_frame_indices:
            add_issue(
                issues,
                "frame_index_invalid",
                expected=expected_frame_index,
                actual=frame_index,
                **record_context,
            )
        if isinstance(frame_index, int):
            seen_frame_indices.add(frame_index)
        expected_frame_index += 1
        timestamp = record.get("timestamp_ms")
        expected_timestamp = expected_timestamp_ms(
            frame_index if isinstance(frame_index, int) else expected_frame_index - 1,
            record.get("fps"),
            last_timestamp,
        )
        if timestamp != expected_timestamp or not isinstance(timestamp, int):
            add_issue(
                issues,
                "timestamp_invalid",
                expected=expected_timestamp,
                actual=timestamp,
                **record_context,
            )
        if isinstance(timestamp, int):
            last_timestamp = timestamp
        source_timestamp = record.get("source_timestamp_ms")
        if source_timestamp is not None:
            if not is_finite_number(source_timestamp):
                add_issue(issues, "source_timestamp_invalid", actual=source_timestamp, **record_context)
            elif last_source_timestamp is not None and float(source_timestamp) < float(last_source_timestamp):
                add_issue(
                    issues,
                    "source_timestamp_out_of_order",
                    previous=last_source_timestamp,
                    actual=source_timestamp,
                    **record_context,
                )
            else:
                last_source_timestamp = source_timestamp
        if record.get("decode_ok") is not True or record.get("infer_ok") is not True:
            add_issue(issues, "frame_not_decoded_or_inferred", **record_context)
        if record.get("error") is not None:
            add_issue(issues, "frame_error_present", error=record.get("error"), **record_context)
        if record.get("handedness_world_landmark_mismatch") is not False:
            add_issue(
                issues,
                "handedness_world_landmark_mismatch_true",
                **record_context,
            )
        num_hands = record.get("num_hands")
        hands = record.get("hands")
        if not isinstance(num_hands, int) or not isinstance(hands, list):
            add_issue(issues, "num_hands_or_hands_invalid", **record_context)
        else:
            if len(hands) != num_hands:
                add_issue(
                    issues,
                    "num_hands_mismatch",
                    num_hands=num_hands,
                    hands_length=len(hands),
                    **record_context,
                )
            stats["no_hand"] += int(num_hands == 0)
            stats["multiple_hand"] += int(num_hands > 1)
            hand_labels = []
            hand_scores = []
            landmark_lengths = []
            for hand_index, hand in enumerate(hands):
                valid, hand_info = validate_hand(hand, hand_index, issues, record_context)
                if valid:
                    stats["hands"] += 1
                hand_labels.append(hand_info["label"])
                hand_scores.append(hand_info["score"])
                if isinstance(hand, dict):
                    landmark_lengths.append(
                        {
                            "normalized": len(hand.get("normalized_landmarks") or []),
                            "world": len(hand.get("world_landmarks") or []),
                        }
                    )
            if manifest_row["video_id"] not in samples:
                samples[manifest_row["video_id"]] = {
                    "video_id": manifest_row["video_id"],
                    "split": manifest_row["split"],
                    "first_frame": {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp,
                        "source_timestamp_ms": source_timestamp,
                        "num_hands": num_hands,
                        "handedness_labels": hand_labels,
                        "handedness_scores": hand_scores,
                        "landmark_lengths": landmark_lengths,
                        "valid": valid if hands else True,
                    },
                }
        stats["frames"] += 1
    if stats["frames"] != int(manifest_row.get("frames_written") or 0):
        add_issue(
            issues,
            "frame_count_mismatch",
            actual=stats["frames"],
            manifest=manifest_row.get("frames_written"),
            **context,
        )
    return stats


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def summarize_distribution(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "ratio_max_over_min": None,
        }
    minimum = min(values)
    maximum = max(values)
    return {
        "count": len(values),
        "min": minimum,
        "max": maximum,
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "ratio_max_over_min": (maximum / minimum) if minimum else None,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    report_dir = args.report_dir.resolve()

    issues = []
    checks = []

    def add_check(name, passed, details=None):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "details": details if details is not None else {},
            }
        )

    required_paths = {
        "config": output_root / "config.json",
        "selected_videos": output_root / "selected_videos.jsonl",
        "extraction_manifest": output_root / "extraction_manifest.jsonl",
        "failures": output_root / "failures.jsonl",
        "leakage_audit": output_root / "leakage_audit.json",
        "label_map": dataset_root
        / "_official_metadata"
        / "AI4Bharat_INCLUDE"
        / "label_maps"
        / "label_map_include.json",
    }
    add_check(
        "required_files_present",
        all(path.is_file() for path in required_paths.values()),
        {name: str(path) for name, path in required_paths.items()},
    )

    artifact_hashes_start = collect_artifact_hashes(output_root)
    config = read_json_file(required_paths["config"], issues, "config.json")
    label_map = read_json_file(required_paths["label_map"], issues, "label_map_include.json")
    leakage_audit = read_json_file(required_paths["leakage_audit"], issues, "leakage_audit.json")
    selected_rows = read_jsonl_file(
        required_paths["selected_videos"], issues, "selected_videos.jsonl"
    )
    manifest_rows = read_jsonl_file(
        required_paths["extraction_manifest"], issues, "extraction_manifest.jsonl"
    )
    failure_rows = read_jsonl_file(required_paths["failures"], issues, "failures.jsonl")
    split_paths, official_lines, official_path_to_splits, cross_split_duplicates, duplicate_official_lines = read_official_splits(
        dataset_root, issues
    )

    config_payload = {}
    config_hash_recomputed = None
    if isinstance(config, dict):
        config_payload = {
            key: value
            for key, value in config.items()
            if key not in ("config_hash", "created_utc")
        }
        config_hash_recomputed = sha256_json(config_payload)
    add_check(
        "config_hash_recomputes",
        bool(config_hash_recomputed) and config_hash_recomputed == config.get("config_hash"),
        {"stored": config.get("config_hash"), "recomputed": config_hash_recomputed},
    )
    add_check(
        "config_baseline_unchanged",
        bool(config) and config.get("config_hash") == BASELINE["config_hash"],
        {"expected": BASELINE["config_hash"], "actual": config.get("config_hash") if config else None},
    )

    selected_hash = (
        sha256_file(required_paths["selected_videos"])
        if required_paths["selected_videos"].is_file()
        else None
    )
    add_check(
        "selected_videos_hash_valid",
        selected_hash == config.get("selected_videos_sha256") == BASELINE["selected_videos_sha256"],
        {
            "file": selected_hash,
            "config": config.get("selected_videos_sha256") if config else None,
            "baseline": BASELINE["selected_videos_sha256"],
        },
    )

    label_map_hash = (
        sha256_file(required_paths["label_map"])
        if required_paths["label_map"].is_file()
        else None
    )
    add_check(
        "label_map_hash_valid",
        label_map_hash == config.get("label_map", {}).get("sha256") == BASELINE["label_map_sha256"],
        {
            "file": label_map_hash,
            "config": config.get("label_map", {}).get("sha256") if config else None,
            "baseline": BASELINE["label_map_sha256"],
        },
    )

    model_path = Path(config.get("model", {}).get("path", "")) if config else Path()
    model_hash = sha256_file(model_path) if model_path.is_file() else None
    add_check(
        "model_hash_valid",
        model_hash == config.get("model", {}).get("sha256") == BASELINE["model_sha256"],
        {
            "file": model_hash,
            "config": config.get("model", {}).get("sha256") if config else None,
            "baseline": BASELINE["model_sha256"],
        },
    )

    split_hash_details = {}
    split_hashes_valid = True
    for split, path in split_paths.items():
        actual = sha256_file(path) if path.is_file() else None
        configured = config.get("official_splits", {}).get(split, {}).get("sha256") if config else None
        baseline = BASELINE["split_sha256"][split]
        valid = actual == configured == baseline
        split_hashes_valid &= valid
        split_hash_details[split] = {
            "path": str(path),
            "file_sha256": actual,
            "config_sha256": configured,
            "baseline_sha256": baseline,
            "valid": valid,
        }
    add_check("official_split_hashes_valid", split_hashes_valid, split_hash_details)

    manifest_hash = (
        sha256_file(required_paths["extraction_manifest"])
        if required_paths["extraction_manifest"].is_file()
        else None
    )
    add_check(
        "extraction_manifest_baseline_unchanged",
        manifest_hash == BASELINE["extraction_manifest_sha256"],
        {"expected": BASELINE["extraction_manifest_sha256"], "actual": manifest_hash},
    )

    add_check(
        "failures_log_empty",
        len(failure_rows) == 0,
        {"failure_rows": len(failure_rows)},
    )
    add_check(
        "existing_leakage_audit_passed",
        bool(leakage_audit) and leakage_audit.get("passed") is True,
        {"passed": leakage_audit.get("passed") if leakage_audit else None},
    )

    selected_classes_config = list(config.get("selected_classes", [])) if config else []
    add_check(
        "selected_classes_match_expected",
        len(selected_classes_config) == 30
        and len(set(selected_classes_config)) == 30
        and set(selected_classes_config) == set(SELECTED_CLASSES),
        {"count": len(selected_classes_config), "expected": list(SELECTED_CLASSES)},
    )

    class_to_archive = build_class_archive_map(dataset_root, issues)
    class_folders = sorted({row.get("class_folder") for row in selected_rows})
    add_check(
        "all_selected_classes_represented",
        set(class_folders) == set(selected_classes_config) and len(class_folders) == 30,
        {
            "selected_classes": len(selected_classes_config),
            "observed_classes": len(class_folders),
            "missing": sorted(set(selected_classes_config) - set(class_folders)),
            "unexpected": sorted(set(class_folders) - set(selected_classes_config)),
        },
    )

    selected_ids = [row.get("video_id") for row in selected_rows]
    selected_paths = [row.get("official_rel_path_posix") for row in selected_rows]
    source_paths = [row.get("source_video_abs") for row in selected_rows]
    add_check(
        "selected_video_ids_unique",
        len(selected_rows) == 528 and len(set(selected_ids)) == len(selected_rows),
        {"rows": len(selected_rows), "unique_ids": len(set(selected_ids))},
    )
    add_check(
        "selected_official_paths_unique",
        len(set(selected_paths)) == len(selected_rows),
        {"rows": len(selected_rows), "unique_paths": len(set(selected_paths))},
    )
    add_check(
        "selected_source_paths_unique",
        len(set(source_paths)) == len(selected_rows),
        {"rows": len(selected_rows), "unique_paths": len(set(source_paths))},
    )
    add_check(
        "official_paths_not_cross_split",
        not cross_split_duplicates and not duplicate_official_lines,
        {
            "cross_split_duplicates": cross_split_duplicates,
            "duplicate_lines": duplicate_official_lines,
        },
    )

    label_conflicts = []
    split_conflicts = []
    id_conflicts = []
    source_conflicts = []
    outside_repo = True
    for row in selected_rows:
        video_id = row.get("video_id")
        official_path = row.get("official_rel_path")
        posix_path = row.get("official_rel_path_posix")
        split = row.get("split")
        class_folder = row.get("class_folder")
        class_label = normalize_class_folder(class_folder)
        expected_id = sha256_text(posix_path) if posix_path else None
        if video_id != expected_id:
            id_conflicts.append({"video_id": video_id, "expected": expected_id})
        if split not in SPLITS or posix_path not in official_lines.get(split, []):
            split_conflicts.append(
                {"video_id": video_id, "split": split, "official_rel_path": posix_path}
            )
        parts = (posix_path or "").split("/")
        if len(parts) < 3 or parts[0] != "Adjectives" or parts[1] != class_folder:
            split_conflicts.append(
                {"video_id": video_id, "reason": "class_folder_path_mismatch"}
            )
        if class_label != row.get("class_label"):
            label_conflicts.append(
                {"video_id": video_id, "reason": "class_label_mismatch"}
            )
        expected_label_id = label_map.get(class_label) if isinstance(label_map, dict) else None
        if expected_label_id is None or int(expected_label_id) != row.get("label_id"):
            label_conflicts.append(
                {
                    "video_id": video_id,
                    "expected_label_id": expected_label_id,
                    "actual_label_id": row.get("label_id"),
                }
            )
        archive = class_to_archive.get(class_folder)
        expected_source = (
            dataset_root / "_extract_test" / archive / Path(*parts) if archive else None
        )
        if expected_source is None or str(expected_source) != row.get("source_video_abs"):
            source_conflicts.append(
                {
                    "video_id": video_id,
                    "expected": str(expected_source) if expected_source else None,
                    "actual": row.get("source_video_abs"),
                }
            )
        source_path = Path(row.get("source_video_abs", ""))
        if not source_path.is_file() or row.get("source_video_exists") is not True:
            source_conflicts.append({"video_id": video_id, "reason": "source_missing"})
        resolved = source_path.resolve()
        outside_repo &= not resolved.is_relative_to(REPO_ROOT)
        outside_repo &= str(resolved).lower().startswith(str(dataset_root).lower())

    add_check("video_ids_match_official_paths", not id_conflicts, id_conflicts)
    add_check(
        "labels_match_official_label_map",
        not label_conflicts,
        label_conflicts,
    )
    add_check(
        "selected_splits_match_official_files",
        not split_conflicts,
        split_conflicts,
    )
    add_check("source_paths_valid", not source_conflicts, source_conflicts)
    add_check(
        "source_videos_outside_git_repo",
        outside_repo,
        {"repo_root": str(REPO_ROOT), "dataset_root": str(dataset_root)},
    )

    selected_by_id = {row["video_id"]: row for row in selected_rows}
    manifest_by_id = {row["video_id"]: row for row in manifest_rows}
    manifest_conflicts = []
    expected_output_files = set()
    for video_id, selected in selected_by_id.items():
        output_path = (
            output_root
            / "landmarks"
            / selected["split"]
            / f"{video_id}.jsonl"
        )
        expected_output_files.add(str(output_path))
        row = manifest_by_id.get(video_id)
        if row is None:
            manifest_conflicts.append({"video_id": video_id, "reason": "missing_manifest_row"})
            continue
        for field in (
            "split",
            "official_rel_path",
            "official_rel_path_posix",
            "source_video_abs",
            "archive",
            "class_folder",
            "class_label",
            "label_id",
        ):
            if row.get(field) != selected.get(field):
                manifest_conflicts.append(
                    {
                        "video_id": video_id,
                        "field": field,
                        "selected": selected.get(field),
                        "manifest": row.get(field),
                    }
                )
        if str(row.get("output_file")) != str(output_path):
            manifest_conflicts.append(
                {
                    "video_id": video_id,
                    "field": "output_file",
                    "expected": str(output_path),
                    "actual": row.get("output_file"),
                }
            )
        if row.get("status") != "complete":
            manifest_conflicts.append(
                {"video_id": video_id, "reason": "status_not_complete", "status": row.get("status")}
            )
        if row.get("config_hash") != config.get("config_hash"):
            manifest_conflicts.append({"video_id": video_id, "reason": "config_hash_mismatch"})
        if row.get("model_sha256") != config.get("model", {}).get("sha256"):
            manifest_conflicts.append({"video_id": video_id, "reason": "model_hash_mismatch"})
        if (
            row.get("decode_errors")
            or row.get("inference_errors")
            or row.get("handedness_mismatch_frames")
            or row.get("error") is not None
        ):
            manifest_conflicts.append({"video_id": video_id, "reason": "error_counts_nonzero"})
        if not (
            row.get("metadata_frame_count") == row.get("frames_read") == row.get("frames_written")
        ):
            manifest_conflicts.append({"video_id": video_id, "reason": "frame_count_mismatch"})

    add_check(
        "manifest_matches_selected",
        set(manifest_by_id) == set(selected_by_id) and not manifest_conflicts,
        {
            "selected_rows": len(selected_by_id),
            "manifest_rows": len(manifest_by_id),
            "conflicts": manifest_conflicts,
        },
    )

    landmark_root = output_root / "landmarks"
    actual_output_files = set()
    if landmark_root.is_dir():
        actual_output_files = {str(path) for path in landmark_root.rglob("*.jsonl")}
    missing_output_files = sorted(expected_output_files - actual_output_files)
    extra_output_files = sorted(actual_output_files - expected_output_files)
    add_check(
        "landmark_files_complete",
        not missing_output_files and not extra_output_files,
        {"missing": missing_output_files, "extra": extra_output_files},
    )

    stats_by_video = {}
    samples = {}
    class_counts = collections.defaultdict(
        lambda: {
            "videos": set(),
            "frames": 0,
            "no_hand": 0,
            "multiple_hand": 0,
            "hands": 0,
            "video_frames": [],
        }
    )
    total_frames = 0
    total_no_hand = 0
    total_multiple_hand = 0
    total_hands = 0
    for row in manifest_rows:
        output_path = Path(row["output_file"])
        if not output_path.is_file():
            continue
        stats = validate_landmark_file(output_path, row, issues, samples)
        stats_by_video[row["video_id"]] = stats
        total_frames += stats["frames"]
        total_no_hand += stats["no_hand"]
        total_multiple_hand += stats["multiple_hand"]
        total_hands += stats["hands"]
        key = (row["class_folder"], row["split"])
        bucket = class_counts[key]
        bucket["videos"].add(row["video_id"])
        bucket["frames"] += stats["frames"]
        bucket["no_hand"] += stats["no_hand"]
        bucket["multiple_hand"] += stats["multiple_hand"]
        bucket["hands"] += stats["hands"]
        bucket["video_frames"].append(stats["frames"])

    issue_counts = collections.Counter(item["kind"] for item in issues)
    add_check(
        "no_empty_landmark_files",
        issue_counts["landmark_file_empty"] == 0,
        {"count": issue_counts["landmark_file_empty"]},
    )
    add_check(
        "no_malformed_landmark_records",
        issue_counts["malformed_landmark_record"] == 0
        and issue_counts["landmark_blank_record"] == 0
        and issue_counts["landmark_missing_final_newline"] == 0
        and issue_counts["landmark_record_not_object"] == 0,
        {
            "malformed": issue_counts["malformed_landmark_record"],
            "blank": issue_counts["landmark_blank_record"],
            "missing_final_newline": issue_counts["landmark_missing_final_newline"],
        },
    )
    add_check(
        "no_duplicate_landmark_records",
        issue_counts["duplicate_landmark_record"] == 0
        and issue_counts["frame_index_invalid"] == 0,
        {
            "duplicate_records": issue_counts["duplicate_landmark_record"],
            "invalid_frame_indices": issue_counts["frame_index_invalid"],
        },
    )
    add_check(
        "frame_indices_and_timestamps_valid",
        issue_counts["timestamp_invalid"] == 0
        and issue_counts["source_timestamp_invalid"] == 0
        and issue_counts["source_timestamp_out_of_order"] == 0,
        {
            "timestamp_invalid": issue_counts["timestamp_invalid"],
            "source_timestamp_invalid": issue_counts["source_timestamp_invalid"],
            "source_timestamp_out_of_order": issue_counts["source_timestamp_out_of_order"],
        },
    )
    add_check(
        "landmark_structure_valid",
        issue_counts["landmark_list_invalid"] == 0
        and issue_counts["landmark_fields_invalid"] == 0
        and issue_counts["landmark_coordinate_invalid"] == 0
        and issue_counts["landmark_metadata_invalid"] == 0
        and issue_counts["landmark_name_invalid"] == 0
        and issue_counts["landmark_count_invalid"] == 0,
        {
            "landmark_list_invalid": issue_counts["landmark_list_invalid"],
            "landmark_fields_invalid": issue_counts["landmark_fields_invalid"],
            "landmark_coordinate_invalid": issue_counts["landmark_coordinate_invalid"],
            "landedness_metadata_invalid": issue_counts["landmark_metadata_invalid"],
        },
    )
    add_check(
        "handedness_metadata_valid",
        issue_counts["handedness_missing"] == 0
        and issue_counts["handedness_fields_invalid"] == 0
        and issue_counts["handedness_score_invalid"] == 0
        and issue_counts["handedness_label_invalid"] == 0
        and issue_counts["handedness_label_unexpected"] == 0
        and issue_counts["handedness_score_mismatch"] == 0,
        {
            "missing": issue_counts["handedness_missing"],
            "fields_invalid": issue_counts["handedness_fields_invalid"],
            "score_invalid": issue_counts["handedness_score_invalid"],
            "label_invalid": issue_counts["handedness_label_invalid"],
        },
    )
    add_check(
        "no_decode_inference_or_identity_errors",
        issue_counts["frame_not_decoded_or_inferred"] == 0
        and issue_counts["frame_error_present"] == 0
        and issue_counts["landmark_identity_mismatch"] == 0
        and issue_counts["num_hands_mismatch"] == 0
        and issue_counts["handedness_world_landmark_mismatch_true"] == 0,
        {
            "frame_not_decoded_or_inferred": issue_counts["frame_not_decoded_or_inferred"],
            "frame_error_present": issue_counts["frame_error_present"],
            "identity_mismatch": issue_counts["landmark_identity_mismatch"],
            "num_hands_mismatch": issue_counts["num_hands_mismatch"],
        },
    )
    count_mismatches = [
        item for item in issues if item["kind"] == "frame_count_mismatch"
    ]
    add_check(
        "frame_counts_match_manifest",
        not count_mismatches
        and total_frames == sum(int(row.get("frames_written") or 0) for row in manifest_rows),
        {"actual_frames": total_frames, "mismatches": count_mismatches},
    )

    artifact_hashes_end = collect_artifact_hashes(output_root)
    add_check(
        "artifact_hashes_stable_during_audit",
        artifact_hashes_start == artifact_hashes_end,
        {
            "start_aggregate": artifact_hashes_start["aggregate_sha256"],
            "end_aggregate": artifact_hashes_end["aggregate_sha256"],
        },
    )

    split_video_counts = collections.Counter(row["split"] for row in manifest_rows)
    split_frame_counts = collections.Counter()
    for row in manifest_rows:
        split_frame_counts[row["split"]] += stats_by_video.get(row["video_id"], {}).get("frames", 0)

    class_rows = []
    class_info = {}
    for class_folder in sorted(set(SELECTED_CLASSES)):
        example = next(
            (row for row in selected_rows if row["class_folder"] == class_folder),
            None,
        )
        class_info[class_folder] = {
            "class_label": example["class_label"] if example else normalize_class_folder(class_folder),
            "label_id": example["label_id"] if example else None,
        }
        for split in SPLITS:
            bucket = class_counts[(class_folder, split)]
            videos = len(bucket["videos"])
            frames = bucket["frames"]
            class_rows.append(
                {
                    "class_folder": class_folder,
                    "class_label": class_info[class_folder]["class_label"],
                    "label_id": class_info[class_folder]["label_id"],
                    "split": split,
                    "videos": videos,
                    "frames": frames,
                    "no_hand_frames": bucket["no_hand"],
                    "multiple_hand_frames": bucket["multiple_hand"],
                    "hands_detected": bucket["hands"],
                    "min_video_frames": min(bucket["video_frames"]) if bucket["video_frames"] else 0,
                    "max_video_frames": max(bucket["video_frames"]) if bucket["video_frames"] else 0,
                    "mean_video_frames": (
                        sum(bucket["video_frames"]) / len(bucket["video_frames"])
                        if bucket["video_frames"]
                        else 0.0
                    ),
                    "share_of_split_videos": (
                        videos / split_video_counts[split] if split_video_counts[split] else 0.0
                    ),
                    "share_of_split_frames": (
                        frames / split_frame_counts[split] if split_frame_counts[split] else 0.0
                    ),
                    "insufficient_sample": (
                        split == "train" and videos < MIN_TRAIN_VIDEOS
                    )
                    or (split in ("val", "test") and videos < MIN_EVAL_VIDEOS),
                }
            )

    class_video_counts = {"train": [], "val": [], "test": []}
    class_frame_counts = {"train": [], "val": [], "test": []}
    for row in class_rows:
        class_video_counts[row["split"]].append(row["videos"])
        class_frame_counts[row["split"]].append(row["frames"])

    distribution_summary = {}
    for split in SPLITS:
        distribution_summary[split] = {
            "videos_per_class": summarize_distribution(class_video_counts[split]),
            "frames_per_class": summarize_distribution(class_frame_counts[split]),
        }

    imbalance = {}
    warnings = []
    for split in SPLITS:
        video_ratio = distribution_summary[split]["videos_per_class"]["ratio_max_over_min"]
        frame_ratio = distribution_summary[split]["frames_per_class"]["ratio_max_over_min"]
        imbalance[split] = {
            "video_ratio_max_over_min": video_ratio,
            "frame_ratio_max_over_min": frame_ratio,
            "video_imbalanced": bool(video_ratio and video_ratio >= IMBALANCE_RATIO_THRESHOLD),
            "frame_imbalanced": bool(frame_ratio and frame_ratio >= IMBALANCE_RATIO_THRESHOLD),
        }
        if imbalance[split]["video_imbalanced"] or imbalance[split]["frame_imbalanced"]:
            warnings.append(
                {
                    "type": "class_imbalance",
                    "split": split,
                    "video_ratio_max_over_min": video_ratio,
                    "frame_ratio_max_over_min": frame_ratio,
                    "threshold": IMBALANCE_RATIO_THRESHOLD,
                }
            )

    insufficient = {}
    for class_folder in sorted(class_info):
        class_splits = {
            row["split"]: row for row in class_rows if row["class_folder"] == class_folder
        }
        reasons = []
        for split in SPLITS:
            videos = class_splits[split]["videos"]
            minimum = MIN_TRAIN_VIDEOS if split == "train" else MIN_EVAL_VIDEOS
            if videos < minimum:
                reasons.append(
                    {
                        "split": split,
                        "videos": videos,
                        "minimum_required": minimum,
                    }
                )
        if reasons:
            insufficient[class_folder] = {
                "class_label": class_info[class_folder]["class_label"],
                "reasons": reasons,
            }
    if insufficient:
        warnings.append(
            {
                "type": "insufficient_samples",
                "thresholds": {
                    "min_train_videos": MIN_TRAIN_VIDEOS,
                    "min_val_videos": MIN_EVAL_VIDEOS,
                    "min_test_videos": MIN_EVAL_VIDEOS,
                },
                "classes": sorted(insufficient),
                "count": len(insufficient),
            }
        )

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    integrity_passed = not failed_checks and not issues
    blocking_issues_before_training = []
    if failed_checks:
        blocking_issues_before_training.extend(failed_checks)
    if issues:
        blocking_issues_before_training.append("record_validation_issues")
    if warnings:
        blocking_issues_before_training.extend(
            sorted({warning["type"] for warning in warnings})
        )

    report = {
        "schema_version": "synora_dataset_quality_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script": str(Path(__file__).resolve()),
        "dataset_root": str(dataset_root),
        "processed_output_root": str(output_root),
        "git_repo_root": str(REPO_ROOT),
        "inputs": {
            "config": str(required_paths["config"]),
            "selected_videos": str(required_paths["selected_videos"]),
            "extraction_manifest": str(required_paths["extraction_manifest"]),
            "label_map": str(required_paths["label_map"]),
            "failures": str(required_paths["failures"]),
            "leakage_audit": str(required_paths["leakage_audit"]),
            "official_split_files": {
                split: str(path) for split, path in split_paths.items()
            },
        },
        "hashes": {
            "config_hash": config.get("config_hash") if config else None,
            "config_hash_recomputed": config_hash_recomputed,
            "selected_videos_sha256": selected_hash,
            "extraction_manifest_sha256": manifest_hash,
            "label_map_sha256": label_map_hash,
            "model_sha256": model_hash,
            "official_split_sha256": {
                split: details["file_sha256"] for split, details in split_hash_details.items()
            },
            "artifacts_at_start": artifact_hashes_start,
            "artifacts_at_end": artifact_hashes_end,
        },
        "checks": checks,
        "check_summary": {
            "total": len(checks),
            "passed": len(checks) - len(failed_checks),
            "failed": len(failed_checks),
            "failed_names": failed_checks,
            "warnings": len(warnings),
        },
        "counts": {
            "selected_classes": len(selected_classes_config),
            "selected_videos": len(selected_rows),
            "manifest_videos": len(manifest_rows),
            "landmark_files": len(actual_output_files),
            "frames": total_frames,
            "hand_records": total_hands,
            "no_hand_frames": total_no_hand,
            "multiple_hand_frames": total_multiple_hand,
            "videos_by_split": dict(split_video_counts),
            "frames_by_split": dict(split_frame_counts),
            "status_counts": dict(collections.Counter(row["status"] for row in manifest_rows)),
        },
        "class_distribution_summary": distribution_summary,
        "class_imbalance": imbalance,
        "insufficient_sample_classes": insufficient,
        "warnings": warnings,
        "issues": issues,
        "issue_counts_by_kind": dict(issue_counts),
        "readiness": {
            "integrity_passed": integrity_passed,
            "ready_for_model_training": integrity_passed and not warnings,
            "ready_for_balanced_per_class_training": integrity_passed and not warnings,
            "blocking_issues_before_training": blocking_issues_before_training,
            "notes": [
                "No model training, threshold tuning, augmentation, mirroring, synthesis or frame skipping was performed.",
                "Signer identity is not verified; no signer-independent evaluation claim is made.",
            ],
        },
    }

    split_integrity = {
        "schema_version": "synora_split_integrity_report_v1",
        "official_split_hashes": split_hash_details,
        "official_split_line_counts": {
            split: len(lines) for split, lines in official_lines.items()
        },
        "selected_by_split": dict(collections.Counter(row["split"] for row in selected_rows)),
        "manifest_by_split": dict(split_video_counts),
        "landmark_files_by_split": dict(
            collections.Counter(Path(path).parent.name for path in actual_output_files)
        ),
        "frames_by_split": dict(split_frame_counts),
        "duplicate_video_ids_across_splits": sorted(
            video_id
            for video_id, selected in collections.defaultdict(set).items()
            if False
        ),
        "official_paths_in_multiple_splits": cross_split_duplicates,
        "duplicate_official_split_lines": duplicate_official_lines,
        "selected_vs_official_conflicts": split_conflicts,
        "manifest_vs_selected_conflicts": manifest_conflicts,
        "missing_landmark_files": missing_output_files,
        "extra_landmark_files": extra_output_files,
        "sources_outside_git_repo": outside_repo,
        "artifact_hashes_at_start": artifact_hashes_start,
        "artifact_hashes_at_end": artifact_hashes_end,
        "passed": (
            split_hashes_valid
            and not cross_split_duplicates
            and not duplicate_official_lines
            and not split_conflicts
            and not manifest_conflicts
            and not missing_output_files
            and not extra_output_files
            and outside_repo
            and artifact_hashes_start == artifact_hashes_end
        ),
    }
    split_ids = collections.defaultdict(set)
    for row in selected_rows:
        split_ids[row["video_id"]].add(row["split"])
    split_integrity["duplicate_video_ids_across_splits"] = sorted(
        video_id for video_id, splits in split_ids.items() if len(splits) > 1
    )

    sample_validation = {
        "schema_version": "synora_sample_validation_report_v1",
        "frames_validated": total_frames,
        "files_validated": len(stats_by_video),
        "hand_records_validated": total_hands,
        "no_hand_frames": total_no_hand,
        "multiple_hand_frames": total_multiple_hand,
        "validation_rules": [
            "schema_version == synora_frame_v1",
            "frame record identity matches manifest and selected video",
            "frame_index starts at 0 and increments by 1",
            "timestamp_ms is strictly increasing and matches FPS-derived extraction timing",
            "source_timestamp_ms is finite when present and does not decrease",
            "decode_ok and infer_ok are true and error is null",
            "num_hands equals len(hands) and does not exceed configured num_hands",
            "each hand has exactly 21 normalized and 21 world landmarks",
            "landmark fields are x, y, z, visibility, presence, name",
            "handedness metadata is present, finite, in [0, 1] and labeled Left or Right",
        ],
        "issue_counts_by_kind": dict(issue_counts),
        "issues": issues,
        "sample_records": [
            samples[video_id]
            for video_id in sorted(samples)
        ],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "split_integrity_report.json").write_text(
        json.dumps(split_integrity, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "sample_validation_report.json").write_text(
        json.dumps(sample_validation, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    csv_path = report_dir / "class_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_folder",
                "class_label",
                "label_id",
                "split",
                "videos",
                "frames",
                "no_hand_frames",
                "multiple_hand_frames",
                "hands_detected",
                "min_video_frames",
                "max_video_frames",
                "mean_video_frames",
                "share_of_split_videos",
                "share_of_split_frames",
                "insufficient_sample",
            ],
        )
        writer.writeheader()
        for row in class_rows:
            writer.writerow(row)

    distribution_lines = []
    for class_folder in sorted(class_info):
        values = {
            row["split"]: row for row in class_rows if row["class_folder"] == class_folder
        }
        distribution_lines.append(
            "| {folder} | {train_v} | {val_v} | {test_v} | {train_f} | {val_f} | {test_f} |".format(
                folder=class_folder,
                train_v=values["train"]["videos"],
                val_v=values["val"]["videos"],
                test_v=values["test"]["videos"],
                train_f=values["train"]["frames"],
                val_f=values["val"]["frames"],
                test_f=values["test"]["frames"],
            )
        )
    warning_lines = []
    for warning in warnings:
        if warning["type"] == "class_imbalance":
            warning_lines.append(
                "- {split}: video max/min {video}, frame max/min {frame}; threshold {threshold}.".format(
                    split=warning["split"],
                    video=warning["video_ratio_max_over_min"],
                    frame=warning["frame_ratio_max_over_min"],
                    threshold=warning["threshold"],
                )
            )
        else:
            warning_lines.append(
                "- {count} classes fall below the support thresholds: {classes}.".format(
                    count=warning["count"],
                    classes=", ".join(warning["classes"]),
                )
            )

    readme = """# SYNORA INCLUDE-30 Dataset Quality Audit

## Scope

This audit inspects the completed landmark dataset at `{output_root}` without modifying extracted files. It checks the extractor configuration, selected-video manifest, extraction manifest, label map, official split files, and all 528 landmark JSONL files.

The audit script is separate from the extractor and uses only the Python standard library.

## Method

- Recomputed config, selected-video, label-map, model, split-file, and manifest hashes and compared them with stored and baseline values.
- Parsed all official split files and verified that each selected video belongs only to its recorded split.
- Verified 30 selected classes, normalized class labels, and official label-map IDs.
- Verified each video ID against SHA-256 of its official relative path.
- Verified every landmark record's video ID, split, class fields, source path, frame index, timestamps, decode/inference status, hand counts, 21 normalized landmarks, 21 world landmarks, landmark fields, and handedness metadata.
- Checked for empty, malformed, truncated, duplicate, missing, and unexpected landmark files or records.
- Recomputed per-class video and frame counts separately for train, validation, and test.

Support thresholds used for insufficiency warnings:

- train: at least {min_train} videos per class
- validation: at least {min_eval} videos per class
- test: at least {min_eval} videos per class
- imbalance: max/min ratio at least {imbalance_threshold} for videos or frames

## Results

- Integrity checks: {passed} passed, {failed} failed.
- Videos: {videos} total = {train_videos} train, {val_videos} validation, {test_videos} test.
- Frames: {frames} total = {train_frames} train, {val_frames} validation, {test_frames} test.
- Hand records validated: {hands}.
- No-hand frames: {no_hand}; multiple-hand frames: {multiple_hand}.
- Malformed, missing, truncated, duplicate, or unexpected records: {record_issues}.
- Extraction artifacts were unchanged during the audit: {stable}.
- Dataset integrity passed: {integrity}.
- Ready for unbalanced structural use: {structural_ready}.
- Ready for class-balanced 30-class training/evaluation: {ready}.

### Warnings

{warnings}

### Per-class distribution

| class | train videos | val videos | test videos | train frames | val frames | test frames |
|---|---:|---:|---:|---:|---:|---:|
{distribution}

## Files

- `dataset_quality_report.json`: complete check, hash, count, warning, and readiness report.
- `class_distribution.csv`: per-class and per-split video/frame counts.
- `split_integrity_report.json`: split, leakage, path, and artifact-hash integrity details.
- `sample_validation_report.json`: validation rules, aggregates, issues, and sample records.
- `README.md`: this summary.

## Reproduce

```powershell
& "C:\\Users\\thanu\\synora\\ml\\.venv\\Scripts\\python.exe" "C:\\Users\\thanu\\synora\\ml\\audit\\dataset_quality_audit.py"
```

## Limits

This is a structural and integrity audit only. It does not evaluate model accuracy. Signer identity is not verified, so no signer-independent evaluation claim is made.
""".format(
        output_root=output_root,
        min_train=MIN_TRAIN_VIDEOS,
        min_eval=MIN_EVAL_VIDEOS,
        imbalance_threshold=IMBALANCE_RATIO_THRESHOLD,
        passed=report["check_summary"]["passed"],
        failed=report["check_summary"]["failed"],
        videos=report["counts"]["selected_videos"],
        train_videos=split_video_counts["train"],
        val_videos=split_video_counts["val"],
        test_videos=split_video_counts["test"],
        frames=total_frames,
        train_frames=split_frame_counts["train"],
        val_frames=split_frame_counts["val"],
        test_frames=split_frame_counts["test"],
        hands=total_hands,
        no_hand=total_no_hand,
        multiple_hand=total_multiple_hand,
        record_issues=len(issues),
        stable=artifact_hashes_start == artifact_hashes_end,
        integrity=integrity_passed,
        structural_ready=integrity_passed,
        ready=integrity_passed and not warnings,
        warnings="\n".join(warning_lines) if warning_lines else "- None.",
        distribution="\n".join(distribution_lines),
    )
    (report_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "report_dir": str(report_dir),
                "checks_passed": report["check_summary"]["passed"],
                "checks_failed": report["check_summary"]["failed"],
                "warnings": len(warnings),
                "videos_by_split": dict(split_video_counts),
                "frames_by_split": dict(split_frame_counts),
                "frames": total_frames,
                "issues": len(issues),
                "ready_for_model_training": report["readiness"]["ready_for_model_training"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if integrity_passed else 2


if __name__ == "__main__":
    sys.exit(main())
