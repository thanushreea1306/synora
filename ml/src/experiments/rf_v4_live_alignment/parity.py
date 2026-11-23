from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v4_live_alignment import preprocessing
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import prepare_video_features as feature_source
    from experiments.rf_v4_live_alignment import preprocessing


RECORD_TOLERANCE = 1e-6
FRAME_TOLERANCE = 0
COMPARED_RECORD_KEYS = (
    "schema_version",
    "fps",
    "width",
    "height",
    "frame_index",
    "timestamp_ms",
    "resize_interpolation",
    "tracking_lifecycle",
    "num_hands",
    "hands",
    "handedness_world_landmark_mismatch",
    "decode_ok",
    "infer_ok",
    "error",
)


class FakeLandmarker:
    def __init__(self, result: Any):
        self.result = result
        self.calls: list[tuple[int, Any]] = []
        self.closed = False

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        self.calls.append((timestamp_ms, np.asarray(image.numpy_view()).shape))
        return self.result

    def close(self) -> None:
        self.closed = True


def _fake_result() -> Any:
    landmarks = [
        SimpleNamespace(x=0.11 + index * 0.01, y=0.21 + index * 0.005, z=0.0 + index * 0.001, visibility=0.9, presence=0.8)
        for index in range(feature_source.LANDMARK_COUNT)
    ]
    category = SimpleNamespace(index=0, score=0.97, display_name="Left", category_name="Left")
    return SimpleNamespace(
        hand_landmarks=[landmarks],
        hand_world_landmarks=[landmarks],
        handedness=[[category]],
    )


def run_preprocessing_parity() -> dict[str, Any]:
    rng = np.random.default_rng(20261123)
    source_frame = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    live_frame = cv2.resize(
        source_frame,
        (preprocessing.TARGET_WIDTH, preprocessing.TARGET_HEIGHT),
        interpolation=preprocessing.RESIZE_INTERPOLATION,
    )
    offline_rgb = preprocessing.preprocess_frame(source_frame, resize=True)
    live_rgb = preprocessing.preprocess_frame(live_frame, resize=False)
    max_difference = int(np.abs(offline_rgb.astype(np.int16) - live_rgb.astype(np.int16)).max())
    return {
        "offline_shape": [int(value) for value in offline_rgb.shape],
        "live_shape": [int(value) for value in live_rgb.shape],
        "max_rgb_difference": max_difference,
        "tolerance": FRAME_TOLERANCE,
        "passed": max_difference <= FRAME_TOLERANCE,
        "resize_interpolation": preprocessing.RESIZE_INTERPOLATION_NAME,
        "color_conversion": "cv2.COLOR_BGR2RGB",
    }


def run_lifecycle_parity() -> dict[str, Any]:
    result = _fake_result()
    offline_landmarker = FakeLandmarker(result)
    live_landmarker = FakeLandmarker(result)
    offline_factory_count = 0
    live_factory_count = 0

    def offline_factory(path: Path, payload: Mapping[str, Any]) -> Any:
        nonlocal offline_factory_count
        offline_factory_count += 1
        return offline_landmarker

    def live_factory(path: Path, payload: Mapping[str, Any]) -> Any:
        nonlocal live_factory_count
        live_factory_count += 1
        return live_landmarker

    offline_session = preprocessing.LandmarkSession(Path("offline"), {}, offline_factory)
    live_session = preprocessing.LandmarkSession(Path("live"), {}, live_factory)
    offline_session.reset()
    live_session.reset()
    source_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    live_frame = cv2.resize(
        source_frame,
        (preprocessing.TARGET_WIDTH, preprocessing.TARGET_HEIGHT),
        interpolation=preprocessing.RESIZE_INTERPOLATION,
    )
    offline_records: list[dict[str, Any]] = []
    live_records: list[dict[str, Any]] = []
    for index in range(3):
        processed_offline = offline_session.detect(source_frame, resize=True)
        processed_live = live_session.detect(live_frame, resize=False)
        offline_records.append(
            preprocessing.make_landmark_record(
                processed=processed_offline,
                split="train",
                video_id="offline_parity",
                official_rel_path=None,
                source_video_abs=None,
                class_folder=None,
                class_label=None,
                label_id=-1,
                source_timestamp_ms=float(index * 40),
                source_frame_index=index,
                source_fps=25.0,
                source_width=1920,
                source_height=1080,
            )
        )
        live_records.append(
            preprocessing.make_landmark_record(
                processed=processed_live,
                split=None,
                video_id="live_parity",
                official_rel_path=None,
                source_video_abs=None,
                class_folder=None,
                class_label=None,
                label_id=None,
                source_timestamp_ms=None,
                source_frame_index=index,
                source_fps=preprocessing.TARGET_FPS,
                source_width=preprocessing.TARGET_WIDTH,
                source_height=preprocessing.TARGET_HEIGHT,
            )
        )
    comparison = compare_records(offline_records, live_records)
    timestamps_match = [left["timestamp_ms"] == right["timestamp_ms"] for left, right in zip(offline_records, live_records)]
    offline_session.close()
    live_session.close()
    return {
        "offline_factory_count": offline_factory_count,
        "live_factory_count": live_factory_count,
        "offline_closed": offline_landmarker.closed,
        "live_closed": live_landmarker.closed,
        "timestamp_match": all(timestamps_match),
        "timestamps": [int(item["timestamp_ms"]) for item in offline_records],
        "tracking_lifecycle": [str(item["tracking_lifecycle"]) for item in offline_records],
        "record_comparison": comparison,
        "passed": bool(
            offline_factory_count == 1
            and live_factory_count == 1
            and all(timestamps_match)
            and comparison["passed"]
        ),
    }


def compare_records(
    offline_records: Sequence[Mapping[str, Any]],
    live_records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = RECORD_TOLERANCE,
) -> dict[str, Any]:
    if len(offline_records) != len(live_records):
        return {
            "passed": False,
            "reason": "frame count mismatch",
            "offline_count": len(offline_records),
            "live_count": len(live_records),
        }
    differences: list[str] = []
    for frame_index, (offline, live) in enumerate(zip(offline_records, live_records)):
        for key in COMPARED_RECORD_KEYS:
            left = offline.get(key)
            right = live.get(key)
            if not _values_equal(left, right, tolerance):
                differences.append(f"frame {frame_index}: {key}")
    return {
        "passed": not differences,
        "tolerance": tolerance,
        "frame_count": len(offline_records),
        "differences": differences[:100],
    }


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def load_extracted_records(extraction_root: str | Path, frame_count: int = 43) -> list[dict[str, Any]]:
    extraction_path = Path(extraction_root)
    rows = list(csv.DictReader(open(extraction_path / "manifest.csv", "r", encoding="utf-8", newline="")))
    complete = [row for row in rows if row["status"] == "complete"]
    if not complete:
        raise ValueError("no complete V4 extraction records")
    landmark_path = Path(complete[0]["landmark_file"])
    records: list[dict[str, Any]] = []
    with open(landmark_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= frame_count:
                break
    if len(records) < frame_count:
        raise ValueError(f"example V4 recording has only {len(records)} frames")
    return records


def write_parity_evidence(path: str | Path, extraction_root: str | Path) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(extraction_root) / "config.json").read_text(encoding="utf-8"))
    records = load_extracted_records(extraction_root)
    summary = {
        "experiment_id": "v4_offline_live_parity",
        "test_split_read": bool(config.get("split_policy", {}).get("test_split_read")),
        "example_record_count": len(records),
        "example_tracking_lifecycle": sorted({str(row.get("tracking_lifecycle")) for row in records}),
        "example_resize_interpolation": sorted({str(row.get("resize_interpolation")) for row in records}),
        "example_fps": sorted({float(row.get("fps")) for row in records}),
        "example_dimensions": sorted({(int(row["width"]), int(row["height"])) for row in records}),
        "preprocessing": run_preprocessing_parity(),
        "lifecycle": run_lifecycle_parity(),
    }
    summary["passed"] = bool(
        summary["preprocessing"]["passed"]
        and summary["lifecycle"]["passed"]
        and summary["example_tracking_lifecycle"] == ["reset_at_recording_start"]
        and summary["example_resize_interpolation"] == [preprocessing.RESIZE_INTERPOLATION_NAME]
        and summary["example_fps"] == [preprocessing.TARGET_FPS]
        and summary["example_dimensions"] == [(preprocessing.TARGET_WIDTH, preprocessing.TARGET_HEIGHT)]
        and summary["test_split_read"] is False
    )
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
