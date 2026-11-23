from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

try:
    from ml.src.inference import landmark_capture as capture
    from ml.src.inference import rf_service as service_module
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import inference.landmark_capture as capture
    import inference.rf_service as service_module


INSTRUCTIONS = (
    "SYNORA isolated-sign recognition (v1)\n"
    "Press S to start one sign recording.\n"
    "Press E to end the recording and predict.\n"
    "Press Q at any time to exit cleanly.\n"
    "Maximum recording length is 89 frames. Confidence is raw and uncalibrated."
)
FATAL_CAPTURE_REASONS = {
    capture.REASON_CAMERA_UNAVAILABLE,
    capture.REASON_CAMERA_INIT_FAILED,
    capture.REASON_CAMERA_CONFIGURATION_INVALID,
    capture.REASON_LANDMARKER_INIT_FAILED,
}


@dataclass
class CLIApp:
    capture_config: Any
    service: Any
    capture_fn: Callable[..., Any] = capture.capture_recording
    camera_factory: Callable[[int], Any] | None = None
    landmarker_factory: Callable[[Any, Any], Any] | None = None
    viewer_factory: Callable[[], Any] | None = None
    printer: Callable[[str], None] = print
    max_recordings: int | None = None

    def run(self) -> int:
        self.printer(INSTRUCTIONS)
        attempts = 0
        while self.max_recordings is None or attempts < self.max_recordings:
            attempts += 1
            try:
                capture_result = self._capture_once()
            except KeyboardInterrupt:
                self.printer("Interrupted. Exiting cleanly.")
                return 130
            except Exception as exc:
                self.printer(f"Capture failed unexpectedly: {type(exc).__name__}: {exc}")
                return 1

            if capture_result.status == capture.STATUS_QUIT:
                self.printer("Q pressed. Exiting cleanly.")
                return 0

            if capture_result.status == capture.STATUS_ERROR:
                self.printer(format_capture_error(capture_result))
                if capture_result.reason in FATAL_CAPTURE_REASONS:
                    return 1
                continue

            if capture_result.status != capture.STATUS_OK:
                self.printer(f"Unexpected capture status: {capture_result.status!r}")
                return 1

            try:
                prediction = self.service.predict_capture(capture_result)
            except Exception as exc:
                self.printer(
                    "Prediction failed unexpectedly: "
                    f"{type(exc).__name__}: {exc}\n"
                    "No prediction was produced."
                )
                return 1
            self.printer(format_prediction(prediction))

        self.printer("Finished requested recordings. Exiting cleanly.")
        return 0

    def _capture_once(self) -> Any:
        kwargs: dict[str, Any] = {"config": self.capture_config}
        if self.camera_factory is not None:
            kwargs["camera_factory"] = self.camera_factory
        if self.landmarker_factory is not None:
            kwargs["landmarker_factory"] = self.landmarker_factory
        if self.viewer_factory is not None:
            kwargs["viewer"] = self.viewer_factory()
        return self.capture_fn(**kwargs)


def format_prediction(result: Any) -> str:
    lines: list[str] = []
    status = getattr(result, "get", lambda key, default=None: default)("status")
    if status != service_module.STATUS_OK:
        lines.append(f"Prediction failed: {status}")
        for error in result.get("errors", []) or []:
            code = error.get("code", "UNKNOWN_ERROR")
            message = error.get("message", "")
            lines.append(f"  {code}: {message}")
        lines.append("No prediction was produced.")
        _append_diagnostics(lines, result)
        _append_warnings(lines, result)
        lines.append(f"Confidence note: {result.get('confidence_note', '')}")
        return "\n".join(lines)

    lines.append("Prediction")
    lines.append(f"  Predicted sign: {result.get('predicted_sign')}")
    lines.append(f"  Original label ID: {result.get('original_label_id')}")
    lines.append(f"  Model class index: {result.get('model_class_index')}")
    lines.append(f"  Raw confidence: {float(result.get('raw_confidence')):.6f}")
    _append_diagnostics(lines, result)
    _append_warnings(lines, result)
    lines.append(f"Confidence note: {result.get('confidence_note', '')}")
    return "\n".join(lines)


def format_capture_error(capture_result: Any) -> str:
    reason = getattr(capture_result, "reason", None) or "UNKNOWN_ERROR"
    lines = [f"Capture error: {reason}", "No prediction was produced."]
    warnings = [str(item) for item in getattr(capture_result, "warnings", []) or []]
    if warnings:
        lines.append("Capture warnings:")
        lines.extend(f"  {item}" for item in warnings)
    return "\n".join(lines)


def _append_diagnostics(lines: list[str], result: Any) -> None:
    diagnostics = result.get("recording_diagnostics", {}) or {}
    if not diagnostics:
        return
    lines.append("Recording diagnostics:")
    for key in sorted(diagnostics):
        lines.append(f"  {key}: {diagnostics[key]}")


def _append_warnings(lines: list[str], result: Any) -> None:
    warnings = [str(item) for item in result.get("warnings", []) or []]
    if not warnings:
        return
    lines.append("Warnings:")
    lines.extend(f"  {item}" for item in warnings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synora-inference",
        description="Run one or more user-triggered isolated-sign RF predictions.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Stop after this many capture attempts; omit for continuous use.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    printer: Callable[[str], None] = print,
    service: Any = None,
    capture_config: Any = None,
    capture_fn: Callable[..., Any] | None = None,
    camera_factory: Callable[[int], Any] | None = None,
    landmarker_factory: Callable[[Any, Any], Any] | None = None,
    viewer_factory: Callable[[], Any] | None = None,
    max_recordings: int | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    limit = max_recordings if max_recordings is not None else args.max_recordings
    if limit is not None and limit < 0:
        printer("--max-recordings must be non-negative")
        return 2

    try:
        active_service = service if service is not None else service_module.RFInferenceService()
        active_config = capture_config if capture_config is not None else capture.load_capture_config()
        if capture_config is None:
            active_config = replace(
                active_config,
                camera_index=args.camera_index,
                show_preview=not args.no_preview,
            )
    except Exception as exc:
        printer(f"Initialization failed: {type(exc).__name__}: {exc}")
        return 1

    app = CLIApp(
        capture_config=active_config,
        service=active_service,
        capture_fn=capture_fn if capture_fn is not None else capture.capture_recording,
        camera_factory=camera_factory,
        landmarker_factory=landmarker_factory,
        viewer_factory=viewer_factory,
        printer=printer,
        max_recordings=limit,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
