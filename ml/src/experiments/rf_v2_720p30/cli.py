from __future__ import annotations

import argparse
import itertools
from typing import Callable

from .capture import (
    REQUIRED_FPS,
    REQUIRED_HEIGHT,
    REQUIRED_WIDTH,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_QUIT,
    CaptureConfig,
    capture_segment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synora-rf-v2-720p30-capture",
        description="Run isolated SYNORA RF v2 Stage 1 capture at 1280x720 @ 30 FPS. No prediction is run.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Stop after this many successful recordings; omit for continuous capture.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    printer: Callable[[str], None] = print,
    capture_fn: Callable[..., object] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_recordings is not None and args.max_recordings < 0:
        printer("--max-recordings must be non-negative")
        return 2

    printer(
        f"SYNORA RF v2 Stage 1 capture: {REQUIRED_WIDTH}x{REQUIRED_HEIGHT} @ {REQUIRED_FPS:g} FPS; "
        "no prediction model is loaded."
    )
    printer("Press 's' to start a recording, 'e' to end it, and 'q' to quit.")

    active_capture_fn = capture_fn if capture_fn is not None else capture_segment
    config = CaptureConfig(
        camera_index=args.camera_index,
        show_preview=not args.no_preview,
    )

    for attempt in itertools.count(1):
        result = active_capture_fn(config=config)
        status = getattr(result, "status", None)
        if status == STATUS_OK:
            printer(
                f"Recording {attempt}: {getattr(result, 'frame_count', 0)} landmarks buffered in memory; "
                "nothing was saved."
            )
            if args.max_recordings is not None and attempt >= args.max_recordings:
                return 0
            continue
        if status == STATUS_QUIT:
            printer("Capture stopped.")
            return 0
        if status == STATUS_ERROR:
            printer(f"Capture error: {getattr(result, 'reason', 'UNKNOWN_ERROR')}")
            for warning in getattr(result, "warnings", []) or []:
                printer(f"  {warning}")
            return 1
        printer(f"Capture returned unknown status: {status!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
