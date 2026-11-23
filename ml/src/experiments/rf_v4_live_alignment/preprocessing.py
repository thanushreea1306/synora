from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import mediapipe as mp

try:
    from ml.src import extract_landmarks as extraction
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v2_720p30.capture import load_v2_landmarker_settings
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import extract_landmarks as extraction
    import prepare_video_features as feature_source
    from experiments.rf_v2_720p30.capture import load_v2_landmarker_settings


TARGET_FPS = 30.0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MIN_FRAMES = 43
MAX_FRAMES = 89
RESIZE_INTERPOLATION_NAME = "cv2.INTER_AREA"
RESIZE_INTERPOLATION = cv2.INTER_AREA

LandmarkerFactory = Callable[[Path, dict[str, Any]], Any]


@dataclass
class ProcessedFrame:
    result: Any
    rgb_frame: Any
    timestamp_ms: int
    frame_index: int


class LandmarkSession:
    def __init__(
        self,
        model_path: Path,
        payload: dict[str, Any],
        factory: LandmarkerFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.payload = payload
        self.factory: LandmarkerFactory = factory if factory is not None else extraction.make_landmarker
        self.landmarker: Any = None
        self.frame_index = 0
        self.last_timestamp_ms = -1

    def reset(self) -> None:
        self.close()
        self.landmarker = self.factory(self.model_path, self.payload)
        self.frame_index = 0
        self.last_timestamp_ms = -1

    def detect(self, frame: Any, *, resize: bool) -> ProcessedFrame:
        if self.landmarker is None:
            raise RuntimeError("landmark session is not started")
        rgb_frame = preprocess_frame(frame, resize=resize)
        timestamp_ms = extraction.next_timestamp_ms(
            self.frame_index,
            TARGET_FPS,
            self.last_timestamp_ms,
        )
        self.last_timestamp_ms = timestamp_ms
        self.frame_index += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return ProcessedFrame(
            result=result,
            rgb_frame=rgb_frame,
            timestamp_ms=timestamp_ms,
            frame_index=self.frame_index - 1,
        )

    def close(self) -> None:
        if self.landmarker is None:
            return
        try:
            close = getattr(self.landmarker, "close", None)
            if close is not None:
                close()
        finally:
            self.landmarker = None


def preprocess_frame(frame: Any, *, resize: bool) -> Any:
    source = frame
    if resize:
        source = cv2.resize(
            frame,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=RESIZE_INTERPOLATION,
        )
    return cv2.cvtColor(source, cv2.COLOR_BGR2RGB)


def make_landmark_record(
    *,
    processed: ProcessedFrame,
    split: str | None,
    video_id: str,
    official_rel_path: str | None,
    source_video_abs: str | None,
    class_folder: str | None,
    class_label: str | None,
    label_id: int | None,
    source_timestamp_ms: float | None,
    source_frame_index: int,
    source_fps: float | None,
    source_width: int | None,
    source_height: int | None,
) -> dict[str, Any]:
    hands, mismatch = extraction.serialize_hands(processed.result)
    record = {
        "schema_version": feature_source.FRAME_SCHEMA_VERSION,
        "split": split,
        "video_id": video_id,
        "official_rel_path": official_rel_path,
        "source_video_abs": source_video_abs,
        "archive": None,
        "class_folder": class_folder,
        "class_label": class_label,
        "label_id": label_id,
        "frame_index": processed.frame_index,
        "timestamp_ms": processed.timestamp_ms,
        "source_timestamp_ms": source_timestamp_ms,
        "fps": TARGET_FPS,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "source_frame_index": source_frame_index,
        "source_fps": source_fps,
        "source_width": source_width,
        "source_height": source_height,
        "resize_interpolation": RESIZE_INTERPOLATION_NAME,
        "tracking_lifecycle": "reset_at_recording_start",
        "num_hands": len(hands),
        "hands": hands,
        "handedness_world_landmark_mismatch": mismatch,
        "decode_ok": True,
        "infer_ok": True,
        "error": None,
    }
    feature_source.canonical_frame(record)
    return record


def frame_dimensions_detail(frame: Any) -> str | None:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        return "frame has no usable dimensions"
    if int(shape[0]) != TARGET_HEIGHT or int(shape[1]) != TARGET_WIDTH:
        return f"frame is {shape[1]}x{shape[0]}; required {TARGET_WIDTH}x{TARGET_HEIGHT}"
    return None


def load_landmarker_settings(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    return load_v2_landmarker_settings(path)
