from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

try:
    from ml.src import extract_landmarks as extraction
    from ml.src import prepare_video_features as feature_source
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import extract_landmarks as extraction
    import prepare_video_features as feature_source


DEFAULT_EXTRACTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "processed"
    / "synora_include30_landmarks_v1"
    / "config.json"
)
DEFAULT_FPS = 25.0
DEFAULT_WIDTH = 1920.0
DEFAULT_HEIGHT = 1080.0
DEFAULT_WINDOW_NAME = "SYNORA - Sign Capture"
START_KEY = "s"
END_KEY = "e"
QUIT_KEY = "q"
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATUS_OK = "ok"
STATUS_QUIT = "quit"
STATUS_ERROR = "error"

REASON_CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
REASON_CAMERA_INIT_FAILED = "CAMERA_INIT_FAILED"
REASON_CAMERA_CONFIGURATION_INVALID = "CAMERA_CONFIGURATION_INVALID"
REASON_LANDMARKER_INIT_FAILED = "LANDMARKER_INIT_FAILED"
REASON_FRAME_READ_FAILED = "FRAME_READ_FAILED"
REASON_LANDMARK_INFERENCE_FAILED = "LANDMARK_INFERENCE_FAILED"
REASON_EMPTY_RECORDING = "EMPTY_RECORDING"
REASON_RECORD_TOO_LONG = "RECORD_TOO_LONG"
REASON_USER_QUIT = "USER_QUIT"
REASON_KEYBOARD_INTERRUPT = "KEYBOARD_INTERRUPT"

REQUIRED_EXTRACTION_KEYS = (
    "running_mode",
    "delegate",
    "num_hands",
    "min_hand_detection_confidence",
    "min_hand_presence_confidence",
    "min_tracking_confidence",
    "mirroring",
    "resize",
    "crop",
    "augmentation",
    "smoothing",
    "synthesis",
    "skip_frames",
)


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    fps: float = DEFAULT_FPS
    width: float = DEFAULT_WIDTH
    height: float = DEFAULT_HEIGHT
    max_frames: int = feature_source.T_VALUE
    extraction_config_path: Path = DEFAULT_EXTRACTION_CONFIG_PATH
    model_path: Path | None = None
    show_preview: bool = True
    window_name: str = DEFAULT_WINDOW_NAME


@dataclass
class CaptureResult:
    status: str
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, float] | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    frame_count: int = 0


class OpenCVViewer:
    def __init__(self, show_preview: bool, window_name: str):
        self.show_preview = show_preview
        self.window_name = window_name

    def show(self, frame: Any, result: Any) -> None:
        if not self.show_preview:
            return
        display_frame = frame.copy()
        draw_hands(display_frame, result)
        cv2.imshow(self.window_name, display_frame)

    def poll_key(self) -> int | None:
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


def load_capture_config(path: str | Path | None = None) -> CaptureConfig:
    config_path = DEFAULT_EXTRACTION_CONFIG_PATH if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    extraction_settings = payload.get("extraction")
    if not isinstance(extraction_settings, dict):
        raise ValueError(f"extraction settings missing in {config_path}")
    missing = [key for key in REQUIRED_EXTRACTION_KEYS if key not in extraction_settings]
    if missing:
        raise ValueError(f"extraction settings missing keys: {missing}")
    model_value = payload.get("model", {}).get("path")
    if not model_value:
        raise ValueError(f"model path missing in {config_path}")
    model_path = Path(model_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"hand landmarker model not found: {model_path}")
    return CaptureConfig(
        extraction_config_path=config_path,
        model_path=model_path,
    )


def capture_recording(
    config: CaptureConfig | None = None,
    camera_factory: Any = None,
    landmarker_factory: Any = None,
    viewer: Any = None,
) -> CaptureResult:
    active_config = config if config is not None else load_capture_config()
    camera_factory = camera_factory if camera_factory is not None else cv2.VideoCapture
    landmarker_factory = landmarker_factory if landmarker_factory is not None else extraction.make_landmarker
    active_viewer = viewer if viewer is not None else OpenCVViewer(
        active_config.show_preview,
        active_config.window_name,
    )

    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    camera = None
    landmarker = None
    state = STATE_IDLE

    try:
        extraction_payload = json.loads(
            Path(active_config.extraction_config_path).read_text(encoding="utf-8")
        )
        model_path = active_config.model_path
        if model_path is None:
            model_value = extraction_payload.get("model", {}).get("path")
            if not model_value:
                return _error_result(
                    REASON_LANDMARKER_INIT_FAILED,
                    warnings,
                    "model path missing from extraction config",
                )
            model_path = Path(model_value)

        try:
            landmarker = landmarker_factory(model_path, extraction_payload)
        except Exception as exc:
            return _error_result(
                REASON_LANDMARKER_INIT_FAILED,
                warnings,
                str(exc),
            )

        try:
            camera = camera_factory(active_config.camera_index)
        except Exception as exc:
            return _error_result(
                REASON_CAMERA_INIT_FAILED,
                warnings,
                str(exc),
            )

        if not camera.isOpened():
            return _error_result(REASON_CAMERA_UNAVAILABLE, warnings)

        try:
            fps, width, height = _configure_camera(camera, active_config)
        except CameraConfigurationError as exc:
            return _error_result(
                REASON_CAMERA_CONFIGURATION_INVALID,
                warnings,
                str(exc),
            )

        global_frame_index = 0
        last_timestamp_ms = -1
        record_index = 0

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                return _error_result(REASON_FRAME_READ_FAILED, warnings)

            configuration_detail = _frame_dimensions_detail(frame, width, height, fps)
            if configuration_detail is not None:
                return _error_result(
                    REASON_CAMERA_CONFIGURATION_INVALID,
                    warnings,
                    configuration_detail,
                )

            timestamp_ms = extraction.next_timestamp_ms(
                global_frame_index,
                fps,
                last_timestamp_ms,
            )
            last_timestamp_ms = timestamp_ms
            global_frame_index += 1

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as exc:
                return _error_result(
                    REASON_LANDMARK_INFERENCE_FAILED,
                    warnings,
                    str(exc),
                )

            active_viewer.show(frame, result)
            key = _normalize_key(active_viewer.poll_key())

            if key == QUIT_KEY:
                return CaptureResult(
                    status=STATUS_QUIT,
                    records=[],
                    metadata=None,
                    reason=REASON_USER_QUIT,
                    warnings=warnings,
                    frame_count=0,
                )

            if key == START_KEY and state == STATE_IDLE:
                state = STATE_RECORDING
                records = []
                record_index = 0
                continue

            if key == END_KEY and state == STATE_RECORDING:
                if not records:
                    return _error_result(REASON_EMPTY_RECORDING, warnings)
                return CaptureResult(
                    status=STATUS_OK,
                    records=records,
                    metadata=_metadata(len(records), fps, width, height),
                    reason=None,
                    warnings=warnings,
                    frame_count=len(records),
                )

            if state != STATE_RECORDING:
                continue

            if record_index >= active_config.max_frames:
                return _error_result(REASON_RECORD_TOO_LONG, warnings)

            records.append(
                _make_record(
                    frame_index=record_index,
                    timestamp_ms=timestamp_ms,
                    source_timestamp_ms=_source_timestamp(camera),
                    fps=fps,
                    width=width,
                    height=height,
                    result=result,
                )
            )
            record_index += 1
    except KeyboardInterrupt:
        return CaptureResult(
            status=STATUS_QUIT,
            records=[],
            metadata=None,
            reason=REASON_KEYBOARD_INTERRUPT,
            warnings=warnings,
            frame_count=0,
        )
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
        try:
            active_viewer.close()
        except Exception:
            pass


def draw_hands(image: Any, result: Any) -> None:
    height, width = image.shape[:2]
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for landmarks in result.hand_landmarks or []:
        points = []
        for landmark in landmarks:
            x = int(round(float(landmark.x) * width))
            y = int(round(float(landmark.y) * height))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            points.append((x, y))
        for connection in connections:
            if connection.start < len(points) and connection.end < len(points):
                cv2.line(
                    image,
                    points[connection.start],
                    points[connection.end],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        for point in points:
            cv2.circle(image, point, 3, (0, 0, 255), -1, cv2.LINE_AA)


def _error_result(
    reason: str,
    warnings: list[str],
    detail: str | None = None,
) -> CaptureResult:
    if detail:
        warnings.append(f"{reason}: {detail}")
    return CaptureResult(
        status=STATUS_ERROR,
        records=[],
        metadata=None,
        reason=reason,
        warnings=warnings,
        frame_count=0,
    )


def _make_record(
    frame_index: int,
    timestamp_ms: int,
    source_timestamp_ms: float | None,
    fps: float,
    width: float,
    height: float,
    result: Any,
) -> dict[str, Any]:
    hands, mismatch = extraction.serialize_hands(result)
    return {
        "schema_version": feature_source.FRAME_SCHEMA_VERSION,
        "split": None,
        "video_id": "live_capture",
        "official_rel_path": None,
        "source_video_abs": None,
        "archive": None,
        "class_folder": None,
        "class_label": None,
        "label_id": None,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "source_timestamp_ms": source_timestamp_ms,
        "fps": fps,
        "width": width,
        "height": height,
        "num_hands": len(hands),
        "hands": hands,
        "handedness_world_landmark_mismatch": mismatch,
        "decode_ok": True,
        "infer_ok": True,
        "error": None,
    }


def _metadata(frame_count: int, fps: float, width: float, height: float) -> dict[str, float]:
    return {
        "duration_s": frame_count / fps,
        "fps": fps,
        "width": width,
        "height": height,
    }


class CameraConfigurationError(Exception):
    pass


def _configure_camera(camera: Any, config: CaptureConfig) -> tuple[float, int, int]:
    required_fps = float(config.fps)
    required_width = int(config.width)
    required_height = int(config.height)

    configuration_detail = (
        f"configure the camera to {required_width}x{required_height} "
        f"at {required_fps:g} FPS"
    )

    _set_camera_property(camera, cv2.CAP_PROP_FRAME_WIDTH, required_width, "width")
    _set_camera_property(camera, cv2.CAP_PROP_FRAME_HEIGHT, required_height, "height")
    _set_camera_property(camera, cv2.CAP_PROP_FPS, required_fps, "FPS")

    fps = _read_camera_property(
        camera,
        cv2.CAP_PROP_FPS,
        "FPS",
        configuration_detail,
    )
    width_value = _read_camera_property(
        camera,
        cv2.CAP_PROP_FRAME_WIDTH,
        "width",
        configuration_detail,
    )
    height_value = _read_camera_property(
        camera,
        cv2.CAP_PROP_FRAME_HEIGHT,
        "height",
        configuration_detail,
    )

    if not math.isfinite(fps) or fps <= 0.0:
        raise CameraConfigurationError(
            f"camera FPS readback is invalid: {fps!r}; required a finite positive value; "
            f"configure the camera to {required_width}x{required_height} at {required_fps:g} FPS"
        )

    actual_width = int(round(width_value))
    actual_height = int(round(height_value))
    if (
        not math.isclose(fps, required_fps, rel_tol=0.0, abs_tol=0.1)
        or actual_width != required_width
        or actual_height != required_height
    ):
        raise CameraConfigurationError(
            f"camera reported {actual_width}x{actual_height} at {fps:g} FPS; "
            f"required {required_width}x{required_height} at {required_fps:g} FPS; "
            "configure the camera to the training capture format"
        )

    return fps, actual_width, actual_height


def _set_camera_property(
    camera: Any,
    property_id: int,
    value: float,
    name: str,
) -> None:
    try:
        camera.set(property_id, value)
    except Exception as exc:
        raise CameraConfigurationError(
            f"camera {name} request failed: {type(exc).__name__}: {exc}"
        ) from exc


def _read_camera_property(
    camera: Any,
    property_id: int,
    name: str,
    help_text: str = "",
) -> float:
    try:
        value = float(camera.get(property_id))
    except Exception as exc:
        raise CameraConfigurationError(
            f"camera {name} readback failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        message = (
            f"camera {name} readback is invalid: {value!r}; "
            "required a finite positive value"
        )
        if help_text:
            message = f"{message}; {help_text}"
        raise CameraConfigurationError(message)
    return value


def _frame_dimensions_detail(
    frame: Any,
    width: int,
    height: int,
    fps: float,
) -> str | None:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        return (
            "camera returned a frame without usable dimensions; "
            f"required {width}x{height}; FPS readback {fps:g}; "
            "configure the camera to the training capture format"
        )
    try:
        actual_height = int(shape[0])
        actual_width = int(shape[1])
    except (TypeError, ValueError):
        return (
            "camera returned a frame without usable dimensions; "
            f"required {width}x{height}; FPS readback {fps:g}; "
            "configure the camera to the training capture format"
        )
    if actual_width != width or actual_height != height:
        return (
            f"camera returned {actual_width}x{actual_height} frames; "
            f"required {width}x{height}; FPS readback {fps:g}; "
            "configure the camera to the training capture format"
        )
    return None


def _source_timestamp(camera: Any) -> float | None:
    try:
        value = float(camera.get(cv2.CAP_PROP_POS_MSEC))
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def _normalize_key(key: Any) -> str | None:
    if key is None:
        return None
    try:
        code = int(key)
    except (TypeError, ValueError):
        return None
    if code in (-1, 0, 255):
        return None
    if 65 <= code <= 90:
        code += 32
    return chr(code)
