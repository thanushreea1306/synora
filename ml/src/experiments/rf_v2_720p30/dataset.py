from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


TARGET_FPS = 30.0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MIN_FRAMES = 43
MAX_FRAMES = 89
SPLITS = ("train", "val")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def official_split_lines(path: str | Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip().replace("\\", "/") for line in handle if line.strip()}


def load_train_val_entries(
    v1_manifest_path: str | Path,
    official_split_dir: str | Path,
    label_mapping: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapping = {record["class_folder"]: record for record in label_mapping}
    expected_split_counts = {"train": 317, "val": 42}
    official = {
        split: official_split_lines(Path(official_split_dir) / f"include_{split}.txt")
        for split in SPLITS
    }
    entries: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in read_csv(v1_manifest_path):
        split = row["split"]
        if split not in SPLITS:
            continue
        rel_path = row["official_rel_path_posix"]
        if rel_path not in official[split]:
            raise ValueError(f"official split mismatch for {row['video_id']}")
        if row["video_id"] in seen_video_ids or rel_path in seen_paths:
            raise ValueError(f"duplicate train/val entry: {row['video_id']}")
        if row["class_folder"] not in mapping:
            raise ValueError(f"unmapped class folder: {row['class_folder']}")
        mapped = mapping[row["class_folder"]]
        if int(mapped["original_label_id"]) != int(row["label_id"]):
            raise ValueError(f"label id mismatch for {row['video_id']}")
        seen_video_ids.add(row["video_id"])
        seen_paths.add(rel_path)
        entries.append(
            {
                "video_id": row["video_id"],
                "split": split,
                "official_rel_path_posix": rel_path,
                "source_video_abs": row["source_video_abs"],
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
    entries.sort(key=lambda item: (item["split"], item["official_rel_path_posix"]))
    for split in SPLITS:
        count = sum(1 for entry in entries if entry["split"] == split)
        if count != expected_split_counts[split]:
            raise ValueError(
                f"unexpected {split} count: {count}; expected {expected_split_counts[split]}"
            )
    return entries


def resample_plan(
    source_frame_count: int,
    source_fps: float,
    target_fps: float = TARGET_FPS,
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    if source_frame_count <= 0:
        raise ValueError("source_frame_count must be positive")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must be positive")
    source_duration_s = source_frame_count / source_fps
    target_frame_count = int(round(source_duration_s * target_fps))
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if target_frame_count > max_frames:
        return {
            "status": "excluded",
            "reason": "EXCEEDS_V2_FRAME_CAP",
            "source_frame_count": source_frame_count,
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "source_duration_s": float(source_duration_s),
            "target_frame_count": target_frame_count,
            "max_frames": max_frames,
            "source_indices": [],
            "transformed_duration_s": None,
            "repeated_source_frames": 0,
            "unused_source_frames": 0,
        }
    source_indices = [
        min(int(round(index * source_fps / target_fps)), source_frame_count - 1)
        for index in range(target_frame_count)
    ]
    unique = set(source_indices)
    return {
        "status": "included",
        "reason": None,
        "source_frame_count": source_frame_count,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "source_duration_s": float(source_duration_s),
        "target_frame_count": target_frame_count,
        "max_frames": max_frames,
        "source_indices": source_indices,
        "transformed_duration_s": float(target_frame_count / target_fps),
        "repeated_source_frames": int(target_frame_count - len(unique)),
        "unused_source_frames": int(source_frame_count - len(unique)),
    }


def resize_frame(frame: Any, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> Any:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
