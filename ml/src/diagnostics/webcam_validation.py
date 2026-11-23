from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import capture as v4_capture
from ml.src.experiments.rf_v4_live_alignment import v4_service

PROTOCOL_VERSION = "synora_webcam_validation_v1"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "ml/reports/validation/webcam"
DEFAULT_ATTEMPTS_PER_CLASS = 10
DEFAULT_SEED = 20261123
QUICK_SIGNS = ("loud", "quiet", "happy", "long", "short")
QUICK_ATTEMPTS_PER_CLASS = 1
MODE_QUICK = "quick"
MODE_DETAILED = "detailed"
VALID_OUTCOMES = {"correct", "incorrect", "rejected"}
FATAL_CAPTURE_REASONS = {
    v4_capture.REASON_CAMERA_UNAVAILABLE,
    v4_capture.REASON_CAMERA_INIT_FAILED,
    v4_capture.REASON_CAMERA_CONFIGURATION_INVALID,
    "LANDMARKER_INIT_FAILED",
}
INSTRUCTIONS = (
    "SYNORA genuine-webcam validation session\n"
    "Use the real webcam and the existing V4 live capture/service path exactly once per attempt.\n"
    "For each attempt: confirm the shown target label, press S, perform one isolated sign, press E.\n"
    "Wait for at least 43 frames; capture stops automatically at 89 frames.\n"
    "Keep hands down between attempts. Do not show the model output before the label is confirmed.\n"
    "Press Q during capture to end the session cleanly."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synora-webcam-validation",
        description="Run genuine-webcam validation against the selected SYNORA V3/V4 live pipeline.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--attempts-per-class",
        type=int,
        default=DEFAULT_ATTEMPTS_PER_CLASS,
        help="Detailed mode only; quick mode always uses one attempt per selected sign.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: five selected signs, one attempt each, ENTER-only confirmation.",
    )
    return parser


def build_labels(service: Any) -> list[str]:
    return [
        str(item.class_label)
        for item in sorted(service.label_mapping, key=lambda item: int(item.model_index))
    ]


def build_attempt_order(labels: list[str], attempts_per_class: int, seed: int) -> list[str]:
    if not labels:
        raise ValueError("labels must not be empty")
    if attempts_per_class <= 0:
        raise ValueError("attempts_per_class must be positive")
    order = [label for label in labels for _ in range(attempts_per_class)]
    random.Random(seed).shuffle(order)
    return order


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp_stats(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "monotonic": True,
            "delta_min_ms": None,
            "delta_max_ms": None,
            "delta_mean_ms": None,
        }
    deltas = [round(later - earlier, 3) for earlier, later in zip(values, values[1:])]
    return {
        "monotonic": all(delta > 0.0 for delta in deltas),
        "delta_min_ms": min(deltas),
        "delta_max_ms": max(deltas),
        "delta_mean_ms": round(sum(deltas) / len(deltas), 3),
    }


def summarize_capture(capture_result: Any) -> dict[str, Any]:
    records = list(getattr(capture_result, "records", []) or [])
    metadata = getattr(capture_result, "metadata", None) or {}
    hands = [int(record.get("num_hands", 0) or 0) for record in records]
    timestamps = [
        value
        for value in (_float(record.get("timestamp_ms")) for record in records)
        if value is not None
    ]
    source_timestamps = [
        value
        for value in (_float(record.get("source_timestamp_ms")) for record in records)
        if value is not None
    ]
    mismatches = sum(
        bool(record.get("handedness_world_landmark_mismatch", False)) for record in records
    )
    frame_count = len(records)
    return {
        "status": getattr(capture_result, "status", None),
        "reason": getattr(capture_result, "reason", None),
        "frame_count": frame_count,
        "capture_frame_count": getattr(capture_result, "frame_count", None),
        "fps": metadata.get("fps"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "duration_s": metadata.get("duration_s"),
        "hand_available_frames": sum(value > 0 for value in hands),
        "hand_available_ratio": round(sum(value > 0 for value in hands) / frame_count, 6)
        if frame_count
        else None,
        "no_hand_frames": sum(value == 0 for value in hands),
        "both_hands_frames": sum(value >= 2 for value in hands),
        "min_hands": min(hands) if hands else 0,
        "max_hands": max(hands) if hands else 0,
        "handedness_mismatch_frames": mismatches,
        "tracking_lifecycle": "reset_at_recording_start",
        "timestamp_stats": _timestamp_stats(timestamps),
        "source_timestamp_stats": _timestamp_stats(source_timestamps),
        "warnings": [str(item) for item in getattr(capture_result, "warnings", []) or []],
    }


def model_identity(service: Any) -> dict[str, Any]:
    training_manifest = service.training_manifest
    expected = training_manifest.get("output_hashes", {}) or {}
    weights_path = service._model_path()
    config_path = service.model_root / "selected/config.json"
    calibration_path = service.report_root / "calibration.json"
    actual = {
        "weights.npz": v2_dataset.sha256_file(weights_path),
        "config.json": v2_dataset.sha256_file(config_path),
        "calibration.json": v2_dataset.sha256_file(calibration_path),
    }
    expected_hashes = {
        "weights.npz": expected.get("selected/weights.npz"),
        "config.json": expected.get("selected/config.json"),
        "calibration.json": expected.get("calibration.json"),
    }
    if any(expected_hashes[key] is not None and expected_hashes[key] != actual[key] for key in actual):
        raise RuntimeError("selected artifact hash mismatch")
    return {
        "service": service.identity,
        "weights_path": str(weights_path),
        "config_path": str(config_path),
        "calibration_path": str(calibration_path),
        "expected_hashes": expected_hashes,
        "actual_hashes": actual,
        "rejection_threshold": float(service.threshold),
        "threshold_source": "existing V4 calibration report; unchanged by validation harness",
        "calibration_temperature": float(service.temperature),
    }


def prompt_confirmed_label(
    target: str,
    labels: list[str],
    sequence_index: int,
    total: int,
    prompter: Callable[[str], str],
    printer: Callable[[str], None],
) -> str | None:
    while True:
        printer(
            f"Attempt {sequence_index}/{total}: intended sign is {target!r} "
            f"({labels.index(target) + 1}/{len(labels)})."
        )
        raw = prompter("Type the sign number to confirm [Enter accepts, Q quits]: ").strip()
        if raw == "":
            return target
        if raw.lower() == "q":
            return None
        try:
            selected = labels[int(raw) - 1]
        except (ValueError, IndexError):
            printer("Invalid sign number. Enter 1-21, Enter, or Q.")
            continue
        if selected == target:
            return target
        printer(f"Entered {selected!r}, which does not match {target!r}. Try again.")


def prompt_quick_label(
    target: str,
    sequence_index: int,
    total: int,
    prompter: Callable[[str], str],
    printer: Callable[[str], None],
) -> str | None:
    while True:
        printer(f"Attempt {sequence_index}/{total}")
        printer(f">>> TARGET SIGN: {target} <<<")
        raw = prompter("Press ENTER to confirm the target [Q quits]: ").strip()
        if raw == "":
            return target
        if raw.lower() == "q":
            return None
        printer("Press ENTER to confirm or Q to quit.")


def classify_outcome(accepted: bool, predicted_label: str, ground_truth: str, has_error: bool) -> str:
    if has_error:
        return "error"
    if not accepted:
        return "rejected"
    return "correct" if predicted_label == ground_truth else "incorrect"


def build_attempt_row(
    *,
    session_id: str,
    sequence_index: int,
    ground_truth: str,
    labels: list[str],
    capture_summary: dict[str, Any],
    prediction: dict[str, Any] | None,
    prediction_error: str | None,
    latency_ms: float | None,
    elapsed_ms: float | None,
    failure_notes: str,
    model_hash: str,
) -> dict[str, Any]:
    status = prediction.get("status") if prediction else None
    errors = list(prediction.get("errors", []) or []) if prediction else []
    accepted = bool(prediction.get("accepted", False)) if prediction else False
    predicted_label = str(prediction.get("predicted_sign", v4_service.UNKNOWN_SIGN)) if prediction else ""
    has_error = bool(prediction_error) or status != v4_service.STATUS_OK
    outcome = classify_outcome(accepted, predicted_label, ground_truth, has_error)
    return {
        "attempt_id": f"{session_id}-{sequence_index:03d}",
        "sequence_index": sequence_index,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_label": ground_truth,
        "ground_truth_model_index": labels.index(ground_truth),
        "confirmation": "operator_entered_or_accepted_before_prediction",
        "outcome": outcome,
        "prediction_status": status,
        "prediction_error": prediction_error,
        "predicted_label": predicted_label,
        "candidate_label": prediction.get("candidate_sign") if prediction else None,
        "candidate_model_class_index": prediction.get("candidate_model_class_index") if prediction else None,
        "accepted": accepted,
        "raw_confidence": _float(prediction.get("raw_confidence")) if prediction else None,
        "calibrated_confidence": _float(prediction.get("calibrated_confidence")) if prediction else None,
        "rejection_threshold": _float(prediction.get("rejection_threshold")) if prediction else None,
        "raw_probabilities": prediction.get("raw_probabilities", {}) if prediction else {},
        "calibrated_probabilities": prediction.get("calibrated_probabilities", {}) if prediction else {},
        "prediction_latency_ms": latency_ms,
        "attempt_elapsed_ms": elapsed_ms,
        "hand_tracking": {
            "hand_available_frames": capture_summary["hand_available_frames"],
            "hand_available_ratio": capture_summary["hand_available_ratio"],
            "no_hand_frames": capture_summary["no_hand_frames"],
            "both_hands_frames": capture_summary["both_hands_frames"],
            "min_hands": capture_summary["min_hands"],
            "max_hands": capture_summary["max_hands"],
            "handedness_mismatch_frames": capture_summary["handedness_mismatch_frames"],
            "tracking_lifecycle": capture_summary["tracking_lifecycle"],
        },
        "sequence_quality": capture_summary,
        "prediction_warnings": [str(item) for item in (prediction.get("warnings", []) or [])] if prediction else [],
        "prediction_errors": errors,
        "failure_notes": failure_notes,
        "model_sha256": model_hash,
    }


def compute_metrics(rows: list[dict[str, Any]], labels: list[str], *, accepted_only: bool) -> dict[str, Any]:
    active = [
        row
        for row in rows
        if row.get("outcome") in VALID_OUTCOMES and (not accepted_only or row.get("accepted"))
    ]
    true_values = [str(row["ground_truth_label"]) for row in active]
    predicted_values = [
        str(row["predicted_label"]) if row.get("accepted") else v4_service.UNKNOWN_SIGN
        for row in active
    ]
    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    correct = 0
    for label in labels:
        tp = sum(true == label and pred == label for true, pred in zip(true_values, predicted_values))
        fp = sum(true != label and pred == label for true, pred in zip(true_values, predicted_values))
        fn = sum(true == label and pred != label for true, pred in zip(true_values, predicted_values))
        support = sum(true == label for true in true_values)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        recalls.append(recall)
        f1_scores.append(f1)
    correct = sum(true == pred for true, pred in zip(true_values, predicted_values))
    return {
        "sample_count": len(active),
        "accuracy": correct / len(active) if active else None,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
    }


def compute_confusion(rows: list[dict[str, Any]], labels: list[str], *, accepted_only: bool) -> dict[str, Any]:
    axis = list(labels)
    if not accepted_only and v4_service.UNKNOWN_SIGN not in axis:
        axis.append(v4_service.UNKNOWN_SIGN)
    index = {label: position for position, label in enumerate(axis)}
    matrix = [[0 for _ in axis] for _ in axis]
    for row in rows:
        if row.get("outcome") not in VALID_OUTCOMES:
            continue
        if accepted_only and not row.get("accepted"):
            continue
        true_label = str(row["ground_truth_label"])
        predicted_label = str(row["predicted_label"]) if row.get("accepted") else v4_service.UNKNOWN_SIGN
        if true_label in index and predicted_label in index:
            matrix[index[true_label]][index[predicted_label]] += 1
    return {"labels": axis, "matrix": matrix}


def build_summary(
    *,
    rows: list[dict[str, Any]],
    labels: list[str],
    model: dict[str, Any],
    session_id: str,
    seed: int,
    attempts_per_class: int,
    started_utc: str,
    finished_utc: str,
    interrupted: bool,
    camera_index: int,
    show_preview: bool,
    mode: str = MODE_DETAILED,
    selected_signs: list[str] | None = None,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("outcome") in VALID_OUTCOMES]
    support = {label: sum(row["ground_truth_label"] == label for row in valid) for label in labels}
    required = len(labels) * attempts_per_class
    sufficient = len(valid) == required and all(support[label] == attempts_per_class for label in labels)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "interrupted": interrupted,
        "model": model,
        "capture": {
            "camera_factory": "cv2.VideoCapture default backend via existing V4 capture_segment",
            "camera_index": camera_index,
            "show_preview": show_preview,
            "prerecorded_input": False,
            "simulated_input": False,
        },
        "protocol": {
            "mode": mode,
            "selected_signs": list(selected_signs or labels),
            "random_seed": seed,
            "attempts_per_class": attempts_per_class,
            "required_valid_attempts": required,
            "ground_truth_recorded_before_prediction": True,
            "thresholds_modified": False,
            "validation_recordings_saved": False,
            "validation_recordings_used_for_training": False,
        },
        "counts": {
            "attempt_rows": len(rows),
            "valid_attempts": len(valid),
            "correct": sum(row["outcome"] == "correct" for row in rows),
            "incorrect": sum(row["outcome"] == "incorrect" for row in rows),
            "rejected": sum(row["outcome"] == "rejected" for row in rows),
            "errors": sum(row["outcome"] == "error" for row in rows),
            "quits": sum(row.get("outcome") == "quit" for row in rows),
            "sufficient_trials": sufficient,
            "complete": sufficient,
        },
        "support": support,
        "metrics_all_with_unknown": compute_metrics(rows, labels, accepted_only=False),
        "metrics_accepted_only": compute_metrics(rows, labels, accepted_only=True),
        "confusion_matrix_all_with_unknown": compute_confusion(rows, labels, accepted_only=False),
        "confusion_matrix_accepted_only": compute_confusion(rows, labels, accepted_only=True),
    }


def format_review(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    model = summary["model"]
    counts = summary["counts"]
    metrics = summary["metrics_all_with_unknown"]
    lines = [
        f"# Webcam validation review: {summary['session_id']}",
        "",
        f"Protocol: {summary['protocol_version']}",
        f"Complete pilot: {counts['complete']}",
        f"Valid attempts: {counts['valid_attempts']}/{summary['protocol']['required_valid_attempts']}",
        f"Correct: {counts['correct']}; incorrect: {counts['incorrect']}; rejected: {counts['rejected']}; errors: {counts['errors']}",
        f"Model: {model['service']['selected_variant']} / {model['service']['model_kind']}",
        f"Weight SHA-256: {model['actual_hashes']['weights.npz']}",
        f"Rejection threshold: {model['rejection_threshold']}",
        "",
        "## Per-class metrics (all valid attempts; rejected counts as unknown)",
        "",
        "| Sign | Support | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in metrics["per_class"].items():
        lines.append(
            f"| {label} | {values['support']} | {values['precision']:.3f} | "
            f"{values['recall']:.3f} | {values['f1']:.3f} | {values['true_positive']} | "
            f"{values['false_positive']} | {values['false_negative']} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix (all valid attempts, including unknown)",
            "",
        ]
    )
    confusion = summary["confusion_matrix_all_with_unknown"]
    lines.append("| true \\ predicted | " + " | ".join(confusion["labels"]) + " |")
    lines.append("|---|" + "|".join("---:" for _ in confusion["labels"]) + "|")
    for label, row in zip(confusion["labels"], confusion["matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Incorrect, rejected, and failed attempts",
            "",
            "| Attempt | Ground truth | Predicted | Candidate | Accepted | Confidence | Outcome | Notes |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    flagged = [
        row
        for row in rows
        if row.get("outcome") in {"incorrect", "rejected", "error"}
    ]
    for row in flagged:
        lines.append(
            f"| {row['attempt_id']} | {row['ground_truth_label']} | "
            f"{row.get('predicted_label') or '-'} | {row.get('candidate_label') or '-'} | "
            f"{row.get('accepted')} | "
            f"{'' if row.get('calibrated_confidence') is None else format(row['calibrated_confidence'], '.6f')} | "
            f"{row['outcome']} | {str(row.get('failure_notes', '')).replace('|', '/')} |"
        )
    if not flagged:
        lines.append("| - | - | - | - | - | - | none | - |")
    lines.append("")
    return "\n".join(lines)


def format_quick_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    model = summary["model"]
    counts = summary["counts"]
    selected_signs = summary["protocol"]["selected_signs"]
    confidence_scale = (
        "calibrated confidence" if any(row.get("calibrated_confidence") is not None for row in rows) else "confidence"
    )
    lines = [
        "# Quick webcam validation summary",
        "",
        f"Session: {summary['session_id']}",
        f"Signs: {', '.join(selected_signs)}",
        f"Attempts: {counts['valid_attempts']}/{summary['protocol']['required_valid_attempts']}",
        f"Correct: {counts['correct']}; incorrect: {counts['incorrect']}; rejected: {counts['rejected']}; errors: {counts['errors']}",
        f"Complete: {counts['complete']}",
        f"Model: {model['service']['selected_variant']} / {model['service']['model_kind']}",
        f"Weight SHA-256: {model['actual_hashes']['weights.npz']}",
        f"Rejection threshold: {model['rejection_threshold']} (unchanged)",
        "",
        f"| Expected | Predicted | {confidence_scale} | Result |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        confidence = row.get("calibrated_confidence")
        lines.append(
            f"| {row['ground_truth_label']} | {row.get('predicted_label') or '-'} | "
            f"{'' if confidence is None else format(confidence, '.6f')} | {row['outcome']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_session(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    session_id: str | None = None,
    attempts_per_class: int = DEFAULT_ATTEMPTS_PER_CLASS,
    seed: int = DEFAULT_SEED,
    camera_index: int = 0,
    show_preview: bool = True,
    service: Any = None,
    capture_config: Any = None,
    capture_fn: Callable[..., Any] | None = None,
    camera_factory: Callable[[int], Any] | None = None,
    landmarker_factory: Callable[[Any, Any], Any] | None = None,
    viewer_factory: Callable[[], Any] | None = None,
    prompter: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
    mode: str = MODE_DETAILED,
    selected_signs: list[str] | None = None,
) -> dict[str, Any]:
    if mode not in {MODE_QUICK, MODE_DETAILED}:
        raise ValueError(f"unsupported validation mode: {mode}")
    if attempts_per_class <= 0:
        raise ValueError("attempts_per_class must be positive")
    active_service = service if service is not None else v4_service.V4RecognitionService()
    active_capture_fn = capture_fn or v4_capture.capture_segment
    active_config = capture_config if capture_config is not None else v4_capture.CaptureConfig()
    if capture_config is None:
        from dataclasses import replace

        active_config = replace(
            active_config,
            camera_index=camera_index,
            show_preview=show_preview,
        )
    labels = build_labels(active_service)
    if len(labels) != 21:
        raise RuntimeError(f"expected 21 labels, found {len(labels)}")
    if mode == MODE_QUICK:
        active_signs = list(QUICK_SIGNS if selected_signs is None else selected_signs)
        if len(active_signs) != len(QUICK_SIGNS) or len(set(active_signs)) != len(active_signs):
            raise ValueError(f"quick mode requires {len(QUICK_SIGNS)} unique selected signs")
        missing = [sign for sign in active_signs if sign not in labels]
        if missing:
            raise ValueError(f"selected signs are not model labels: {', '.join(missing)}")
        attempts_per_class = QUICK_ATTEMPTS_PER_CLASS
    else:
        active_signs = labels
    order = build_attempt_order(active_signs, attempts_per_class, seed)
    identity = model_identity(active_service)
    actual_session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = Path(output_root) / actual_session_id
    if session_dir.exists():
        raise FileExistsError(f"session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    attempts_path = session_dir / "attempts.jsonl"
    rows: list[dict[str, Any]] = []
    interrupted = False
    v2_dataset.write_json(
        session_dir / "session.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": actual_session_id,
            "started_utc": started_utc,
            "model": identity,
            "capture": {
                "camera_factory": "cv2.VideoCapture default backend via existing V4 capture_segment",
                "camera_index": camera_index,
                "show_preview": show_preview,
            },
            "protocol": {
                "mode": mode,
                "random_seed": seed,
                "attempts_per_class": attempts_per_class,
                "selected_signs": active_signs,
                "labels": active_signs,
                "model_labels": labels,
                "attempt_order": order,
                "ground_truth_recorded_before_prediction": True,
                "thresholds_modified": False,
                "validation_recordings_saved": False,
            },
        },
    )
    printer(INSTRUCTIONS)
    printer(f"Session: {actual_session_id}")
    printer(f"Mode: {mode}")
    printer(f"Selected signs: {', '.join(active_signs)}")
    printer(f"Testing weight SHA-256: {identity['actual_hashes']['weights.npz']}")
    printer(f"Rejection threshold: {identity['rejection_threshold']} (unchanged)")
    try:
        for sequence_index, target in enumerate(order, start=1):
            if mode == MODE_QUICK:
                ground_truth = prompt_quick_label(target, sequence_index, len(order), prompter, printer)
            else:
                ground_truth = prompt_confirmed_label(target, labels, sequence_index, len(order), prompter, printer)
            if ground_truth is None:
                interrupted = True
                break
            started = time.perf_counter()
            kwargs: dict[str, Any] = {"config": active_config}
            if camera_factory is not None:
                kwargs["camera_factory"] = camera_factory
            if landmarker_factory is not None:
                kwargs["landmarker_factory"] = landmarker_factory
            if viewer_factory is not None:
                kwargs["viewer_factory"] = viewer_factory
            try:
                capture_result = active_capture_fn(**kwargs)
            except KeyboardInterrupt:
                interrupted = True
                printer("Interrupted during capture. Finishing reports.")
                break
            capture_summary = summarize_capture(capture_result)
            if getattr(capture_result, "status", None) == v4_capture.STATUS_QUIT:
                rows.append(
                    build_attempt_row(
                        session_id=actual_session_id,
                        sequence_index=sequence_index,
                        ground_truth=ground_truth,
                        labels=labels,
                        capture_summary=capture_summary,
                        prediction=None,
                        prediction_error="operator quit during capture",
                        latency_ms=None,
                        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                        failure_notes="operator quit during capture",
                        model_hash=identity["actual_hashes"]["weights.npz"],
                    )
                )
                rows[-1]["outcome"] = "quit"
                v2_dataset.write_jsonl(attempts_path, rows)
                interrupted = True
                break
            prediction: dict[str, Any] | None = None
            prediction_error: str | None = None
            latency_ms: float | None = None
            if getattr(capture_result, "status", None) == v4_capture.STATUS_OK:
                prediction_started = time.perf_counter()
                try:
                    prediction = active_service.predict_capture(capture_result)
                except Exception as exc:
                    prediction_error = f"{type(exc).__name__}: {exc}"
                    printer(f"Prediction failed: {prediction_error}")
                latency_ms = round((time.perf_counter() - prediction_started) * 1000.0, 3)
            else:
                prediction_error = str(getattr(capture_result, "reason", "CAPTURE_ERROR"))
            try:
                notes = prompter("Failure notes (optional; press Enter for none): ").strip()
            except (EOFError, KeyboardInterrupt):
                notes = "operator ended note entry"
            row = build_attempt_row(
                session_id=actual_session_id,
                sequence_index=sequence_index,
                ground_truth=ground_truth,
                labels=labels,
                capture_summary=capture_summary,
                prediction=prediction,
                prediction_error=prediction_error,
                latency_ms=latency_ms,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                failure_notes=notes,
                model_hash=identity["actual_hashes"]["weights.npz"],
            )
            rows.append(row)
            v2_dataset.write_jsonl(attempts_path, rows)
            confidence = row["calibrated_confidence"]
            printer(
                f"Attempt {sequence_index}: expected={ground_truth}, "
                f"predicted={row['predicted_label'] or '-'}, "
                f"confidence={confidence}, result={row['outcome']}"
            )
            if getattr(capture_result, "status", None) == v4_capture.STATUS_ERROR and getattr(
                capture_result, "reason", None
            ) in FATAL_CAPTURE_REASONS:
                printer("Fatal camera/landmarker error. Ending session.")
                break
            if sequence_index < len(order):
                try:
                    prompter("Keep hands down. Press Enter when ready for the next attempt: ")
                except (EOFError, KeyboardInterrupt):
                    interrupted = True
                    printer("Interrupted between attempts. Finishing reports.")
                    break
    except (EOFError, KeyboardInterrupt):
        interrupted = True
        printer("Session interrupted. Finishing reports.")
    finished_utc = datetime.now(timezone.utc).isoformat()
    summary = build_summary(
        rows=rows,
        labels=active_signs,
        model=identity,
        session_id=actual_session_id,
        seed=seed,
        attempts_per_class=attempts_per_class,
        started_utc=started_utc,
        finished_utc=finished_utc,
        interrupted=interrupted,
        camera_index=camera_index,
        show_preview=show_preview,
        mode=mode,
        selected_signs=active_signs,
    )
    v2_dataset.write_json(session_dir / "summary.json", summary)
    (session_dir / "review.md").write_text(format_review(summary, rows), encoding="utf-8", newline="\n")
    if mode == MODE_QUICK:
        quick_summary = format_quick_summary(summary, rows)
        (session_dir / "quick_summary.md").write_text(quick_summary, encoding="utf-8", newline="\n")
        printer(quick_summary)
    printer(f"Results saved to {session_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = MODE_QUICK if args.quick else MODE_DETAILED
    attempts_per_class = QUICK_ATTEMPTS_PER_CLASS if args.quick else args.attempts_per_class
    if attempts_per_class <= 0:
        print("--attempts-per-class must be positive")
        return 2
    session_id = args.session_id
    if args.quick and session_id is None:
        session_id = "quick-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        summary = run_session(
            output_root=args.output_root,
            session_id=session_id,
            attempts_per_class=attempts_per_class,
            seed=args.seed,
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mode=mode,
            selected_signs=list(QUICK_SIGNS) if args.quick else None,
        )
    except Exception as exc:
        print(f"Validation session failed: {type(exc).__name__}: {exc}")
        return 1
    return 0 if summary["counts"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
