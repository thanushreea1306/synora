from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

try:
    from ml.src import extract_landmarks as extraction
    from ml.src import prepare_video_features as feature_source
    from ml.src.experiments.rf_v2_720p30 import dataset
    from ml.src.inference import landmark_capture as capture_source
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import extract_landmarks as extraction
    import prepare_video_features as feature_source
    from experiments.rf_v2_720p30 import dataset
    from inference import landmark_capture as capture_source


REQUIRED_FPS = 30.0
REQUIRED_WIDTH = 1280
REQUIRED_HEIGHT = 720
MIN_FRAMES = dataset.MIN_FRAMES
MAX_FRAMES = dataset.MAX_FRAMES
FPS_READBACK_TOLERANCE = 0.1
DEFAULT_EXTRACTION_CONFIG_PATH = capture_source.DEFAULT_EXTRACTION_CONFIG_PATH
DEFAULT_WINDOW_NAME = "SYNORA RF v2 720p30 - Capture"

STATUS_OK = "ok"
STATUS_QUIT = "quit"
STATUS_ERROR = "error"

REASON_CAMERA_UNAVAILABLE = capture_source.REASON_CAMERA_UNAVAILABLE
REASON_CAMERA_INIT_FAILED = capture_source.REASON_CAMERA_INIT_FAILED
REASON_CAMERA_CONFIGURATION_INVALID = capture_source.REASON_CAMERA_CONFIGURATION_INVALID
REASON_LANDMARKER_INIT_FAILED = capture_source.REASON_LANDMARKER_INIT_FAILED
REASON_FRAME_READ_FAILED = capture_source.REASON_FRAME_READ_FAILED
REASON_FRAME_DIMENSION_INVALID = "FRAME_DIMENSION_INVALID"
REASON_LANDMARK_INFERENCE_FAILED = capture_source.REASON_LANDMARK_INFERENCE_FAILED
REASON_INVALID_LANDMARK_RECORD = "INVALID_LANDMARK_RECORD"
REASON_EMPTY_RECORDING = capture_source.REASON_EMPTY_RECORDING
REASON_RECORD_TOO_LONG = capture_source.REASON_RECORD_TOO_LONG
REASON_USER_QUIT = capture_source.REASON_USER_QUIT
REASON_KEYBOARD_INTERRUPT = capture_source.REASON_KEYBOARD_INTERRUPT

STATE_IDLE = "idle"
STATE_RECORDING = "recording"


class CameraConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfiguration:
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    max_frames: int = MAX_FRAMES
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


class V2OpenCVViewer(capture_source.OpenCVViewer):
    def show(self, frame: Any, result: Any, status: str | None = None) -> None:
        if not self.show_preview:
            return
        display_frame = frame.copy()
        capture_source.draw_hands(display_frame, result)
        if status:
            for index, line in enumerate(str(status).splitlines()):
                cv2.putText(
                    display_frame,
                    line,
                    (12, 28 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        cv2.imshow(self.window_name, display_frame)


def configure_camera(camera: Any) -> CameraConfiguration:
    _request_camera_property(camera, cv2.CAP_PROP_FRAME_WIDTH, REQUIRED_WIDTH, "width")
    _request_camera_property(camera, cv2.CAP_PROP_FRAME_HEIGHT, REQUIRED_HEIGHT, "height")
    _request_camera_property(camera, cv2.CAP_PROP_FPS, REQUIRED_FPS, "FPS")

    fps = _read_camera_property(camera, cv2.CAP_PROP_FPS, "FPS")
    width_value = _read_camera_property(camera, cv2.CAP_PROP_FRAME_WIDTH, "width")
    height_value = _read_camera_property(camera, cv2.CAP_PROP_FRAME_HEIGHT, "height")
    width = int(round(width_value))
    height = int(round(height_value))

    if (
        not math.isclose(fps, REQUIRED_FPS, rel_tol=0.0, abs_tol=FPS_READBACK_TOLERANCE)
        or width != REQUIRED_WIDTH
        or height != REQUIRED_HEIGHT
    ):
        raise CameraConfigurationError(
            f"camera reported {width}x{height} at {fps:g} FPS; required "
            f"{REQUIRED_WIDTH}x{REQUIRED_HEIGHT} at {REQUIRED_FPS:g} FPS; no fallback attempted"
        )
    return CameraConfiguration(fps=fps, width=width, height=height)


def load_v2_landmarker_settings(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    config_path = DEFAULT_EXTRACTION_CONFIG_PATH if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    extraction_settings = payload.get("extraction")
    if not isinstance(extraction_settings, dict):
        raise ValueError(f"extraction settings missing in {config_path}")
    missing = [key for key in capture_source.REQUIRED_EXTRACTION_KEYS if key not in extraction_settings]
    if missing:
        raise ValueError(f"extraction settings missing keys in {config_path}: {missing}")
    model_value = payload.get("model", {}).get("path")
    if not model_value:
        raise ValueError(f"model path missing in {config_path}")
    model_path = Path(model_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"hand landmarker model not found: {model_path}")
    return payload, model_path


def capture_segment(
    config: CaptureConfig | None = None,
    camera_factory: Any = None,
    landmarker_factory: Any = None,
    viewer_factory: Any = None,
) -> CaptureResult:
    active_config = config if config is not None else CaptureConfig()
    active_camera_factory = camera_factory if camera_factory is not None else cv2.VideoCapture
    active_landmarker_factory = landmarker_factory if landmarker_factory is not None else extraction.make_landmarker
    active_viewer_factory = viewer_factory if viewer_factory is not None else (
        lambda: V2OpenCVViewer(active_config.show_preview, active_config.window_name)
    )

    warnings: list[str] = []
    notice: str | None = None
    camera: Any = None
    landmarker: Any = None
    viewer: Any = None
    records: list[dict[str, Any]] = []
    state = STATE_IDLE
    global_frame_index = 0
    last_timestamp_ms = -1

    try:
        try:
            payload, default_model_path = load_v2_landmarker_settings(active_config.extraction_config_path)
        except Exception as exc:
            return _error_result(
                REASON_LANDMARKER_INIT_FAILED,
                warnings,
                f"{type(exc).__name__}: {exc}",
            )

        model_path = active_config.model_path if active_config.model_path is not None else default_model_path
        try:
            landmarker = active_landmarker_factory(model_path, payload)
        except Exception as exc:
            return _error_result(
                REASON_LANDMARKER_INIT_FAILED,
                warnings,
                f"{type(exc).__name__}: {exc}",
            )

        try:
            camera = active_camera_factory(active_config.camera_index)
        except Exception as exc:
            return _error_result(
                REASON_CAMERA_INIT_FAILED,
                warnings,
                f"{type(exc).__name__}: {exc}",
            )
        if not camera.isOpened():
            return _error_result(REASON_CAMERA_UNAVAILABLE, warnings)

        try:
            camera_configuration = configure_camera(camera)
        except CameraConfigurationError as exc:
            return _error_result(REASON_CAMERA_CONFIGURATION_INVALID, warnings, str(exc))

        viewer = active_viewer_factory()

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                return _error_result(REASON_FRAME_READ_FAILED, warnings)

            dimension_detail = _frame_dimensions_detail(frame, camera_configuration)
            if dimension_detail is not None:
                return _error_result(REASON_FRAME_DIMENSION_INVALID, warnings, dimension_detail)

            timestamp_ms = extraction.next_timestamp_ms(
                global_frame_index,
                camera_configuration.fps,
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
                    f"{type(exc).__name__}: {exc}",
                )

            status = _status_text(state, len(records), active_config.max_frames, notice)
            _show_viewer(viewer, frame, result, status)
            key = _normalize_key(viewer.poll_key())

            if key == capture_source.QUIT_KEY:
                return CaptureResult(
                    status=STATUS_QUIT,
                    records=[],
                    metadata=None,
                    reason=REASON_USER_QUIT,
                    warnings=warnings,
                    frame_count=0,
                )

            if key == capture_source.START_KEY and state == STATE_IDLE:
                state = STATE_RECORDING
                records = []
                notice = None
                continue

            if key == capture_source.END_KEY and state == STATE_RECORDING:
                if len(records) < MIN_FRAMES:
                    remaining = MIN_FRAMES - len(records)
                    notice = (
                        f"Need {remaining} more frame(s) ({len(records)}/{MIN_FRAMES}); "
                        f"press E again after {MIN_FRAMES}"
                    )
                    warnings.append(notice)
                    continue
                return _success_result(records, warnings)

            if state != STATE_RECORDING:
                continue

            try:
                record = _make_record(
                    frame_index=len(records),
                    timestamp_ms=timestamp_ms,
                    source_timestamp_ms=_source_timestamp(camera),
                    result=result,
                )
            except Exception as exc:
                return _error_result(
                    REASON_INVALID_LANDMARK_RECORD,
                    warnings,
                    f"{type(exc).__name__}: {exc}",
                )
            records.append(record)
            if len(records) >= active_config.max_frames:
                warnings.append(
                    f"AUTO_STOP_AT_MAX_FRAMES: recording reached {active_config.max_frames} frames; "
                    "finalized automatically"
                )
                return _success_result(records, warnings)
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
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


def _request_camera_property(camera: Any, property_id: int, value: float, name: str) -> None:
    try:
        camera.set(property_id, value)
    except Exception as exc:
        raise CameraConfigurationError(
            f"camera {name} request failed: {type(exc).__name__}: {exc}"
        ) from exc


def _read_camera_property(camera: Any, property_id: int, name: str) -> float:
    try:
        value = float(camera.get(property_id))
    except Exception as exc:
        raise CameraConfigurationError(
            f"camera {name} readback failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise CameraConfigurationError(
            f"camera {name} readback is invalid: {value!r}; required a finite positive value"
        )
    return value


def _frame_dimensions_detail(frame: Any, configuration: CameraConfiguration) -> str | None:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        return (
            "camera returned a frame without usable dimensions; required "
            f"{REQUIRED_WIDTH}x{REQUIRED_HEIGHT}; no fallback attempted"
        )
    try:
        actual_height = int(shape[0])
        actual_width = int(shape[1])
    except (TypeError, ValueError):
        return (
            f"camera returned unusable frame dimensions {shape!r}; required "
            f"{REQUIRED_WIDTH}x{REQUIRED_HEIGHT}; no fallback attempted"
        )
    if actual_width != configuration.width or actual_height != configuration.height:
        return (
            f"camera returned a {actual_width}x{actual_height} frame; required "
            f"{configuration.width}x{configuration.height} at {configuration.fps:g} FPS; "
            "no fallback attempted"
        )
    return None


def _make_record(
    frame_index: int,
    timestamp_ms: int,
    source_timestamp_ms: float | None,
    result: Any,
) -> dict[str, Any]:
    hands, mismatch = extraction.serialize_hands(result)
    record = {
        "schema_version": feature_source.FRAME_SCHEMA_VERSION,
        "split": None,
        "video_id": "live_capture_v2_720p30",
        "official_rel_path": None,
        "source_video_abs": None,
        "archive": None,
        "class_folder": None,
        "class_label": None,
        "label_id": None,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "source_timestamp_ms": source_timestamp_ms,
        "fps": REQUIRED_FPS,
        "width": REQUIRED_WIDTH,
        "height": REQUIRED_HEIGHT,
        "num_hands": len(hands),
        "hands": hands,
        "handedness_world_landmark_mismatch": mismatch,
        "decode_ok": True,
        "infer_ok": True,
        "error": None,
    }
    feature_source.canonical_frame(record)
    return record


def _source_timestamp(camera: Any) -> float | None:
    try:
        value = float(camera.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


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


def _metadata(frame_count: int) -> dict[str, float]:
    return {
        "duration_s": frame_count / REQUIRED_FPS,
        "fps": REQUIRED_FPS,
        "width": float(REQUIRED_WIDTH),
        "height": float(REQUIRED_HEIGHT),
    }


def _status_text(
    state: str,
    frame_count: int,
    max_frames: int,
    notice: str | None = None,
) -> str:
    if state == STATE_IDLE:
        lines = [
            "Idle: press S to start | Q to quit",
            f"Limits: min {MIN_FRAMES} frames, max {max_frames} frames",
        ]
    else:
        remaining = MIN_FRAMES - frame_count
        lines = [
            f"Recording: {frame_count}/{max_frames} frames",
            f"Limits: min {MIN_FRAMES} frames, max {max_frames} frames",
        ]
        if remaining > 0:
            lines.append(f"Need {remaining} more frame(s); E is locked until {MIN_FRAMES}")
        else:
            lines.append(f"Minimum reached ({MIN_FRAMES}); press E to end")
        lines.append(f"Auto-stop at {max_frames} frames | Q to quit")
    if notice:
        lines.append(notice)
    return "\n".join(lines)


def _show_viewer(viewer: Any, frame: Any, result: Any, status: str) -> None:
    show = getattr(viewer, "show", None)
    if show is None:
        return
    try:
        show(frame, result, status)
    except TypeError:
        show(frame, result)


def _success_result(records: list[dict[str, Any]], warnings: list[str]) -> CaptureResult:
    return CaptureResult(
        status=STATUS_OK,
        records=records,
        metadata=_metadata(len(records)),
        reason=None,
        warnings=warnings,
        frame_count=len(records),
    )


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
