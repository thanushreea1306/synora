from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

try:
    from ml.src.experiments.rf_v2_720p30 import capture as v2_capture
    from ml.src.experiments.rf_v4_live_alignment import preprocessing
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    from experiments.rf_v2_720p30 import capture as v2_capture
    from experiments.rf_v4_live_alignment import preprocessing


DEFAULT_EXTRACTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[4]
    / "ml/dataset/processed/experiments/rf_v4_live_alignment/config.json"
)
DEFAULT_WINDOW_NAME = "SYNORA RF v4 720p30 - Capture"
STATUS_OK = v2_capture.STATUS_OK
STATUS_QUIT = v2_capture.STATUS_QUIT
STATUS_ERROR = v2_capture.STATUS_ERROR
REASON_CAMERA_UNAVAILABLE = v2_capture.REASON_CAMERA_UNAVAILABLE
REASON_CAMERA_INIT_FAILED = v2_capture.REASON_CAMERA_INIT_FAILED
REASON_CAMERA_CONFIGURATION_INVALID = v2_capture.REASON_CAMERA_CONFIGURATION_INVALID
REASON_FRAME_READ_FAILED = v2_capture.REASON_FRAME_READ_FAILED
REASON_FRAME_DIMENSION_INVALID = v2_capture.REASON_FRAME_DIMENSION_INVALID
REASON_LANDMARK_INFERENCE_FAILED = v2_capture.REASON_LANDMARK_INFERENCE_FAILED
REASON_INVALID_LANDMARK_RECORD = v2_capture.REASON_INVALID_LANDMARK_RECORD
REASON_USER_QUIT = v2_capture.REASON_USER_QUIT
REASON_KEYBOARD_INTERRUPT = v2_capture.REASON_KEYBOARD_INTERRUPT
STATE_IDLE = v2_capture.STATE_IDLE
STATE_RECORDING = v2_capture.STATE_RECORDING


class CameraConfigurationError(v2_capture.CameraConfigurationError):
    pass


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    max_frames: int = preprocessing.MAX_FRAMES
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


class V4OpenCVViewer(v2_capture.V2OpenCVViewer):
    pass


def load_v4_landmarker_settings(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    return preprocessing.load_landmarker_settings(path)


def capture_segment(
    config: CaptureConfig | None = None,
    camera_factory: Any = None,
    landmarker_factory: Any = None,
    viewer_factory: Any = None,
) -> CaptureResult:
    active_config = config if config is not None else CaptureConfig()
    active_camera_factory = camera_factory if camera_factory is not None else cv2.VideoCapture
    active_landmarker_factory = landmarker_factory if landmarker_factory is not None else preprocessing.extraction.make_landmarker
    active_viewer_factory = viewer_factory if viewer_factory is not None else (
        lambda: V4OpenCVViewer(active_config.show_preview, active_config.window_name)
    )
    warnings: list[str] = []
    notice: str | None = None
    camera: Any = None
    session: preprocessing.LandmarkSession | None = None
    viewer: Any = None
    records: list[dict[str, Any]] = []
    state = STATE_IDLE
    try:
        try:
            payload, default_model_path = load_v4_landmarker_settings(active_config.extraction_config_path)
        except Exception as exc:
            return _error_result(
                "LANDMARKER_INIT_FAILED",
                warnings,
                f"{type(exc).__name__}: {exc}",
            )
        model_path = active_config.model_path if active_config.model_path is not None else default_model_path
        try:
            session = preprocessing.LandmarkSession(model_path, payload, active_landmarker_factory)
            session.reset()
        except Exception as exc:
            return _error_result(
                "LANDMARKER_INIT_FAILED",
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
            camera_configuration = v2_capture.configure_camera(camera)
        except v2_capture.CameraConfigurationError as exc:
            return _error_result(REASON_CAMERA_CONFIGURATION_INVALID, warnings, str(exc))
        viewer = active_viewer_factory()
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                return _error_result(REASON_FRAME_READ_FAILED, warnings)
            dimension_detail = preprocessing.frame_dimensions_detail(frame)
            if dimension_detail is not None:
                return _error_result(REASON_FRAME_DIMENSION_INVALID, warnings, dimension_detail)
            try:
                processed = session.detect(frame, resize=False)
            except Exception as exc:
                return _error_result(
                    REASON_LANDMARK_INFERENCE_FAILED,
                    warnings,
                    f"{type(exc).__name__}: {exc}",
                )
            status = v2_capture._status_text(state, len(records), active_config.max_frames, notice)
            _show_viewer(viewer, frame, processed.result, status)
            key = v2_capture._normalize_key(viewer.poll_key())
            if key == "q":
                return CaptureResult(
                    status=STATUS_QUIT,
                    records=[],
                    metadata=None,
                    reason=REASON_USER_QUIT,
                    warnings=warnings,
                    frame_count=0,
                )
            if key == "s" and state == STATE_IDLE:
                state = STATE_RECORDING
                records = []
                notice = None
                session.reset()
                continue
            if key == "e" and state == STATE_RECORDING:
                if len(records) < preprocessing.MIN_FRAMES:
                    remaining = preprocessing.MIN_FRAMES - len(records)
                    notice = (
                        f"Need {remaining} more frame(s) ({len(records)}/{preprocessing.MIN_FRAMES}); "
                        f"press E again after {preprocessing.MIN_FRAMES}"
                    )
                    warnings.append(notice)
                    continue
                return _success_result(records, warnings, camera_configuration)
            if state != STATE_RECORDING:
                continue
            try:
                record = preprocessing.make_landmark_record(
                    processed=processed,
                    split=None,
                    video_id="live_capture_v4_720p30",
                    official_rel_path=None,
                    source_video_abs=None,
                    class_folder=None,
                    class_label=None,
                    label_id=None,
                    source_timestamp_ms=v2_capture._source_timestamp(camera),
                    source_frame_index=processed.frame_index,
                    source_fps=camera_configuration.fps,
                    source_width=camera_configuration.width,
                    source_height=camera_configuration.height,
                )
            except Exception as exc:
                return _error_result(
                    REASON_INVALID_LANDMARK_RECORD,
                    warnings,
                    f"{type(exc).__name__}: {exc}",
                )
            record["frame_index"] = len(records)
            records.append(record)
            if len(records) >= active_config.max_frames:
                warnings.append(
                    f"AUTO_STOP_AT_MAX_FRAMES: recording reached {active_config.max_frames} frames; "
                    "finalized automatically"
                )
                return _success_result(records, warnings, camera_configuration)
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
        if session is not None:
            session.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


def _show_viewer(viewer: Any, frame: Any, result: Any, status: str) -> None:
    show = getattr(viewer, "show", None)
    if show is None:
        return
    try:
        show(frame, result, status)
    except TypeError:
        show(frame, result)


def _success_result(
    records: list[dict[str, Any]],
    warnings: list[str],
    camera_configuration: Any,
) -> CaptureResult:
    return CaptureResult(
        status=STATUS_OK,
        records=records,
        metadata=v2_capture._metadata(len(records)),
        reason=None,
        warnings=warnings,
        frame_count=len(records),
    )


def _error_result(reason: str, warnings: list[str], detail: str | None = None) -> CaptureResult:
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
