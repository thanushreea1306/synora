from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Any, Callable

from ml.src.experiments.rf_v3_generalisation import recognize as v3_recognize
from ml.src.experiments.rf_v4_live_alignment import capture, v4_service

INSTRUCTIONS = (
    "SYNORA isolated-sign recognition (v4, 1280x720 @ 30 FPS)\n"
    "Press S to start one sign recording; tracking resets when recording starts.\n"
    "Press E to end at 43 frames or more.\n"
    "Recording stops automatically at 89 frames.\n"
    "Press Q at any time to exit cleanly.\n"
    "Limits: minimum 43 frames, maximum 89 frames. Low-confidence predictions return unknown."
)
FATAL_CAPTURE_REASONS = {
    v4_service.v4_capture.REASON_CAMERA_UNAVAILABLE,
    v4_service.v4_capture.REASON_CAMERA_INIT_FAILED,
    v4_service.v4_capture.REASON_CAMERA_CONFIGURATION_INVALID,
    "LANDMARKER_INIT_FAILED",
}


@dataclass
class CLIApp:
    capture_config: Any
    service: Any
    capture_fn: Callable[..., Any] = capture.capture_segment
    camera_factory: Callable[[int], Any] | None = None
    landmarker_factory: Callable[[Any, Any], Any] | None = None
    viewer_factory: Callable[[], Any] | None = None
    printer: Callable[[str], None] = print
    max_recordings: int | None = None

    def run(self) -> int:
        self.printer(INSTRUCTIONS)
        self.printer(f"Validation note: {v4_service.LIMITATION_NOTE}")
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
            kwargs["viewer_factory"] = self.viewer_factory
        return self.capture_fn(**kwargs)


def format_prediction(result: Any) -> str:
    lines: list[str] = []
    status = result.get("status") if hasattr(result, "get") else None
    if status != v4_service.STATUS_OK:
        lines.append(f"Prediction failed: {status}")
        for error in result.get("errors", []) or []:
            lines.append(f"  {error.get('code', 'UNKNOWN_ERROR')}: {error.get('message', '')}")
        lines.append("No prediction was produced.")
        v3_recognize._append_diagnostics(lines, result)
        v3_recognize._append_warnings(lines, result)
        return "\n".join(lines)
    if result.get("accepted"):
        lines.append("Prediction")
        lines.append(f"  Predicted sign: {result.get('predicted_sign')}")
        lines.append(f"  Original label ID: {result.get('original_label_id')}")
        lines.append(f"  Model class index: {result.get('model_class_index')}")
    else:
        lines.append("Prediction")
        lines.append(f"  Predicted sign: {v4_service.UNKNOWN_SIGN}")
        lines.append(f"  Candidate sign: {result.get('candidate_sign')}")
    lines.append(f"  Raw confidence: {float(result.get('raw_confidence')):.6f}")
    lines.append(f"  Calibrated confidence: {float(result.get('calibrated_confidence')):.6f}")
    lines.append(f"  Rejection threshold: {float(result.get('rejection_threshold')):.6f}")
    lines.append(f"  Model variant: {result.get('model', {}).get('selected_variant')}")
    v3_recognize._append_diagnostics(lines, result)
    v3_recognize._append_warnings(lines, result)
    lines.append(f"Confidence note: {result.get('confidence_note', '')}")
    lines.append(f"Validation note: {result.get('limitation_note', v4_service.LIMITATION_NOTE)}")
    return "\n".join(lines)


def format_capture_error(capture_result: Any) -> str:
    return v3_recognize.format_capture_error(capture_result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synora-v4-recognize",
        description="Run isolated-sign recognition with the SYNORA v4 live-aligned model.",
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
        active_service = service if service is not None else v4_service.V4RecognitionService()
        active_config = capture_config if capture_config is not None else capture.CaptureConfig()
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
        capture_fn=capture_fn or capture.capture_segment,
        camera_factory=camera_factory,
        landmarker_factory=landmarker_factory,
        viewer_factory=viewer_factory,
        printer=printer,
        max_recordings=limit,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
