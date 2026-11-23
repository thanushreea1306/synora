from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

try:
    from ml.src import extract_landmarks as extraction
except ModuleNotFoundError as exc:
    if exc.name not in ("ml", "ml.src"):
        raise
    SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)
    import extract_landmarks as extraction


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "processed"
    / "synora_include30_landmarks_v1"
    / "config.json"
)
DEFAULT_JUMP_THRESHOLD = 0.08
SIDES = ("Left", "Right")
PREVIEW_WINDOW_NAME = "SYNORA - Live Landmark Diagnostic"
WAIT_KEY_DELAY_MS = 1
LANDMARK_POINT_COLOR = (0, 0, 255)
LANDMARK_LINE_COLOR = (0, 255, 0)
LANDMARK_POINT_RADIUS = 3
LANDMARK_LINE_THICKNESS = 2


@dataclass
class ConfidenceStats:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float | None) -> None:
        number = _finite_float(value)
        if number is None:
            return
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class StabilityStats:
    samples: int = 0
    total: float = 0.0
    maximum: float = 0.0
    large_movement_candidates: int = 0
    last_movement: float | None = None

    def add(self, movement: float, threshold: float) -> None:
        if not math.isfinite(movement) or movement < 0.0:
            return
        self.samples += 1
        self.total += movement
        self.maximum = max(self.maximum, movement)
        self.last_movement = movement
        if movement > threshold:
            self.large_movement_candidates += 1

    @property
    def mean(self) -> float | None:
        return self.total / self.samples if self.samples else None


@dataclass
class DiagnosticSummary:
    reported_fps: float | None
    observed_fps: float | None
    width: int | None
    height: int | None
    frames_observed: int
    no_hand_frames: int
    any_hand_frames: int
    both_hands_frames: int
    left_frames: int
    right_frames: int
    left_confidence: ConfidenceStats
    right_confidence: ConfidenceStats
    left_stability: StabilityStats
    right_stability: StabilityStats
    jump_threshold: float
    elapsed_s: float
    interrupted: bool = False


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_float(value: float | None, digits: int = 6) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _is_quit_key(key: Any) -> bool:
    try:
        code = int(key)
    except (TypeError, ValueError):
        return False
    if code in (-1, 255):
        return False
    if 65 <= code <= 90:
        code += 32
    return chr(code) == "q"


def load_diagnostic_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_value = payload.get("model", {}).get("path")
    if not model_value:
        raise ValueError(f"model path missing in {config_path}")
    model_path = Path(model_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"hand landmarker model not found: {model_path}")
    return payload


def _landmark_points(landmarks: Any) -> tuple[tuple[float, float], ...] | None:
    points = []
    for landmark in landmarks or []:
        x = _finite_float(getattr(landmark, "x", None))
        y = _finite_float(getattr(landmark, "y", None))
        if x is None or y is None:
            return None
        points.append((x, y))
    return tuple(points) if points else None


def _frame_hand_infos(result: Any) -> list[tuple[str | None, float | None, tuple[tuple[float, float], ...] | None]]:
    landmarks_by_hand = list(getattr(result, "hand_landmarks", None) or [])
    handedness_by_hand = list(getattr(result, "handedness", None) or [])
    infos = []
    for index, landmarks in enumerate(landmarks_by_hand):
        categories = handedness_by_hand[index] if index < len(handedness_by_hand) else []
        categories = list(categories or [])
        label = extraction.handedness_label(categories)
        score = extraction.handedness_score(categories)
        infos.append((label, score, _landmark_points(landmarks)))
    return infos


def _mean_movement(
    previous: Sequence[tuple[float, float]],
    current: Sequence[tuple[float, float]],
) -> float | None:
    if len(previous) != len(current) or not previous:
        return None
    total = 0.0
    for (old_x, old_y), (new_x, new_y) in zip(previous, current):
        total += math.hypot(new_x - old_x, new_y - old_y)
    return total / len(previous)


def _draw_hands(frame: Any, result: Any) -> Any:
    display_frame = frame.copy()
    shape = getattr(display_frame, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) < 2:
        return display_frame
    height = int(shape[0])
    width = int(shape[1])
    if height <= 0 or width <= 0:
        return display_frame

    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for landmarks in getattr(result, "hand_landmarks", None) or []:
        points = []
        for landmark in landmarks:
            x = _finite_float(getattr(landmark, "x", None))
            y = _finite_float(getattr(landmark, "y", None))
            if x is None or y is None:
                continue
            pixel_x = max(0, min(width - 1, int(round(x * width))))
            pixel_y = max(0, min(height - 1, int(round(y * height))))
            points.append((pixel_x, pixel_y))

        for connection in connections:
            start = int(connection.start)
            end = int(connection.end)
            if 0 <= start < len(points) and 0 <= end < len(points):
                cv2.line(
                    display_frame,
                    points[start],
                    points[end],
                    LANDMARK_LINE_COLOR,
                    LANDMARK_LINE_THICKNESS,
                    cv2.LINE_AA,
                )
        for point in points:
            cv2.circle(
                display_frame,
                point,
                LANDMARK_POINT_RADIUS,
                LANDMARK_POINT_COLOR,
                -1,
                cv2.LINE_AA,
            )
    return display_frame


def _print_summary(summary: DiagnosticSummary, printer: Callable[[str], None]) -> None:
    printer("\nFinal diagnostic summary")
    printer(
        "actual_resolution="
        f"{_format_float(summary.width, 0)}x{_format_float(summary.height, 0)} "
        f"reported_fps={_format_float(summary.reported_fps)} "
        f"observed_fps={_format_float(summary.observed_fps)} "
        f"elapsed_s={summary.elapsed_s:.3f}"
    )
    printer(
        f"frames_observed={summary.frames_observed} "
        f"no_hand_frames={summary.no_hand_frames} "
        f"any_hand_frames={summary.any_hand_frames} "
        f"both_hands_frames={summary.both_hands_frames} "
        f"left_frames={summary.left_frames} "
        f"right_frames={summary.right_frames}"
    )
    printer(
        "left_confidence count="
        f"{summary.left_confidence.count} mean={_format_float(summary.left_confidence.mean)} "
        f"min={_format_float(summary.left_confidence.minimum)} "
        f"max={_format_float(summary.left_confidence.maximum)}"
    )
    printer(
        "right_confidence count="
        f"{summary.right_confidence.count} mean={_format_float(summary.right_confidence.mean)} "
        f"min={_format_float(summary.right_confidence.minimum)} "
        f"max={_format_float(summary.right_confidence.maximum)}"
    )
    printer(
        f"stability=mean per-landmark normalized-plane displacement between consecutive detected frames; "
        f"threshold={summary.jump_threshold:.6f}; large-movement candidates are review flags, not errors"
    )
    printer(
        "left_stability samples="
        f"{summary.left_stability.samples} mean={_format_float(summary.left_stability.mean)} "
        f"max={_format_float(summary.left_stability.maximum)} "
        f"last={_format_float(summary.left_stability.last_movement)} "
        f"large_movement_candidates={summary.left_stability.large_movement_candidates}"
    )
    printer(
        "right_stability samples="
        f"{summary.right_stability.samples} mean={_format_float(summary.right_stability.mean)} "
        f"max={_format_float(summary.right_stability.maximum)} "
        f"last={_format_float(summary.right_stability.last_movement)} "
        f"large_movement_candidates={summary.right_stability.large_movement_candidates}"
    )
    if summary.interrupted:
        printer("Session interrupted by user.")


def run_diagnostic(
    *,
    camera_index: int = 0,
    requested_fps: float | None = None,
    requested_width: int | None = None,
    requested_height: int | None = None,
    max_frames: int | None = None,
    seconds: float | None = None,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    config_path: str | Path | None = None,
    camera_factory: Any = None,
    landmarker_factory: Any = None,
    printer: Callable[[str], None] = print,
) -> DiagnosticSummary:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if seconds is not None and seconds <= 0.0:
        raise ValueError("seconds must be positive")
    threshold = _finite_float(jump_threshold)
    if threshold is None or threshold <= 0.0:
        raise ValueError("jump_threshold must be positive")

    camera = None
    landmarker = None
    preview_shown = False
    interrupted = False
    summary: DiagnosticSummary | None = None

    try:
        if landmarker_factory is None:
            payload = load_diagnostic_config(config_path)
            model_path = Path(payload["model"]["path"])
            landmarker = extraction.make_landmarker(model_path, payload)
        else:
            landmarker = landmarker_factory(None, None)

        active_camera_factory = camera_factory if camera_factory is not None else cv2.VideoCapture
        camera = active_camera_factory(camera_index)
        if not camera.isOpened():
            raise RuntimeError("camera unavailable")

        if requested_fps is not None:
            camera.set(cv2.CAP_PROP_FPS, float(requested_fps))
        if requested_width is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(requested_width))
        if requested_height is not None:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(requested_height))

        reported_fps = _finite_float(camera.get(cv2.CAP_PROP_FPS))
        width = _finite_float(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _finite_float(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_width = int(width) if width is not None else None
        actual_height = int(height) if height is not None else None

        left_confidence = ConfidenceStats()
        right_confidence = ConfidenceStats()
        confidence_by_side = {"Left": left_confidence, "Right": right_confidence}
        left_stability = StabilityStats()
        right_stability = StabilityStats()
        stability_by_side = {"Left": left_stability, "Right": right_stability}
        last_points: dict[str, tuple[tuple[float, float], ...]] = {}

        frames = 0
        no_hand_frames = 0
        any_hand_frames = 0
        both_hands_frames = 0
        left_frames = 0
        right_frames = 0
        last_timestamp_ms = -1
        started = time.monotonic()
        last_progress = started

        while True:
            elapsed = time.monotonic() - started
            if max_frames is not None and frames >= max_frames:
                break
            if seconds is not None and elapsed >= seconds:
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                if frames == 0:
                    raise RuntimeError("camera returned no readable frames")
                break

            frames += 1
            shape = getattr(frame, "shape", None)
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                actual_width = int(shape[1])
                actual_height = int(shape[0])

            timestamp_ms = extraction.next_timestamp_ms(
                frames - 1,
                reported_fps if reported_fps is not None else 0.0,
                last_timestamp_ms,
            )
            last_timestamp_ms = timestamp_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if getattr(result, "hand_landmarks", None):
                display_frame = _draw_hands(frame, result)
            else:
                display_frame = frame

            infos = _frame_hand_infos(result)
            if not infos:
                no_hand_frames += 1
            else:
                any_hand_frames += 1

            present_sides = set()
            best_by_side: dict[str, tuple[float | None, tuple[tuple[float, float], ...] | None]] = {}
            for label, score, points in infos:
                if label not in SIDES:
                    continue
                present_sides.add(label)
                confidence_by_side[label].add(score)
                current_best = best_by_side.get(label)
                if current_best is None or (score or 0.0) > (current_best[0] or 0.0):
                    best_by_side[label] = (score, points)

            if "Left" in present_sides:
                left_frames += 1
            if "Right" in present_sides:
                right_frames += 1
            if {"Left", "Right"}.issubset(present_sides):
                both_hands_frames += 1

            for label in SIDES:
                current = best_by_side.get(label)
                if current is None:
                    last_points.pop(label, None)
                    continue
                points = current[1]
                if points is None:
                    last_points.pop(label, None)
                    continue
                previous = last_points.get(label)
                if previous is not None:
                    movement = _mean_movement(previous, points)
                    if movement is not None:
                        stability_by_side[label].add(movement, threshold)
                last_points[label] = points

            cv2.imshow(PREVIEW_WINDOW_NAME, display_frame)
            preview_shown = True
            if _is_quit_key(cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF):
                interrupted = True
                break

            elapsed = time.monotonic() - started
            if elapsed - last_progress >= 0.5 or frames == 1:
                observed_fps = frames / elapsed if elapsed > 0.0 else None
                printer(
                    f"frames={frames} actual={actual_width}x{actual_height} "
                    f"reported_fps={_format_float(reported_fps)} "
                    f"observed_fps={_format_float(observed_fps)} "
                    f"no_hand={no_hand_frames} any_hand={any_hand_frames} "
                    f"both_hands={both_hands_frames} left={left_frames} right={right_frames}"
                )
                last_progress = elapsed

        elapsed = time.monotonic() - started
        observed_fps = frames / elapsed if elapsed > 0.0 else None
        summary = DiagnosticSummary(
            reported_fps=reported_fps,
            observed_fps=observed_fps,
            width=actual_width,
            height=actual_height,
            frames_observed=frames,
            no_hand_frames=no_hand_frames,
            any_hand_frames=any_hand_frames,
            both_hands_frames=both_hands_frames,
            left_frames=left_frames,
            right_frames=right_frames,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            left_stability=left_stability,
            right_stability=right_stability,
            jump_threshold=threshold,
            elapsed_s=elapsed,
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if preview_shown:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_NAME)
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    if interrupted:
        elapsed = time.monotonic() - started if "started" in locals() else 0.0
        observed_fps = frames / elapsed if frames and elapsed > 0.0 else None
        summary = DiagnosticSummary(
            reported_fps=reported_fps if "reported_fps" in locals() else None,
            observed_fps=observed_fps,
            width=actual_width if "actual_width" in locals() else None,
            height=actual_height if "actual_height" in locals() else None,
            frames_observed=frames if "frames" in locals() else 0,
            no_hand_frames=no_hand_frames if "no_hand_frames" in locals() else 0,
            any_hand_frames=any_hand_frames if "any_hand_frames" in locals() else 0,
            both_hands_frames=both_hands_frames if "both_hands_frames" in locals() else 0,
            left_frames=left_frames if "left_frames" in locals() else 0,
            right_frames=right_frames if "right_frames" in locals() else 0,
            left_confidence=left_confidence if "left_confidence" in locals() else ConfidenceStats(),
            right_confidence=right_confidence if "right_confidence" in locals() else ConfidenceStats(),
            left_stability=left_stability if "left_stability" in locals() else StabilityStats(),
            right_stability=right_stability if "right_stability" in locals() else StabilityStats(),
            jump_threshold=threshold,
            elapsed_s=elapsed,
            interrupted=True,
        )

    if summary is not None:
        _print_summary(summary, printer)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synora-live-landmark-diagnostic",
        description="SYNORA in-memory live landmark-quality diagnostic; not a sign recognizer.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--jump-threshold", type=float, default=DEFAULT_JUMP_THRESHOLD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("SYNORA LIVE LANDMARK-QUALITY DIAGNOSTIC")
    print("Not part of the recognition pipeline. No frames, landmarks, videos, logs, or recordings will be saved.")
    print("Press Q in the preview window or Ctrl+C to stop early.")
    run_diagnostic(
        camera_index=args.camera_index,
        requested_fps=args.fps,
        requested_width=args.width,
        requested_height=args.height,
        max_frames=args.max_frames,
        seconds=args.seconds,
        jump_threshold=args.jump_threshold,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
