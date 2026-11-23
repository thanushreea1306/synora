from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ml.src.diagnostics import webcam_validation
from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import v4_service


class _Label:
    def __init__(self, model_index: int, class_label: str) -> None:
        self.model_index = model_index
        self.class_label = class_label


def _fake_service(root: Path) -> SimpleNamespace:
    model_root = root / "model"
    report_root = root / "report"
    weights_path = model_root / "selected/weights.npz"
    config_path = model_root / "selected/config.json"
    calibration_path = report_root / "calibration.json"
    weights_path.parent.mkdir(parents=True)
    report_root.mkdir(parents=True)
    weights_path.write_bytes(b"weights")
    config_path.write_text("{}", encoding="utf-8")
    calibration_path.write_text("{}", encoding="utf-8")
    expected = v2_dataset.sha256_file(weights_path)
    return SimpleNamespace(
        label_mapping=[
            _Label(index, label)
            for index, label in enumerate(
                list(webcam_validation.QUICK_SIGNS) + [f"sign_{index}" for index in range(16)]
            )
        ],
        identity={
            "selected_variant": "v3_regularized_cv",
            "model_kind": "v3",
            "model_sha256": expected,
        },
        training_manifest={
            "output_hashes": {
                "selected/weights.npz": expected,
                "selected/config.json": v2_dataset.sha256_file(config_path),
                "calibration.json": v2_dataset.sha256_file(calibration_path),
            }
        },
        model_root=model_root,
        report_root=report_root,
        threshold=0.53,
        temperature=0.5322259762856182,
        _model_path=lambda: weights_path,
        predict_capture=lambda capture_result: {
            "status": v4_service.STATUS_OK,
            "accepted": True,
            "predicted_sign": webcam_validation.QUICK_SIGNS[0],
            "candidate_sign": webcam_validation.QUICK_SIGNS[0],
            "candidate_model_class_index": 0,
            "raw_confidence": 0.99,
            "calibrated_confidence": 0.98,
            "rejection_threshold": 0.53,
            "raw_probabilities": {},
            "calibrated_probabilities": {},
            "warnings": [],
            "errors": [],
        },
    )


def _capture_result() -> SimpleNamespace:
    return SimpleNamespace(
        status=v4_service.STATUS_OK,
        records=[
            {"num_hands": 2, "timestamp_ms": 0, "source_timestamp_ms": 0.0, "handedness_world_landmark_mismatch": False},
            {"num_hands": 1, "timestamp_ms": 33, "source_timestamp_ms": 33.0, "handedness_world_landmark_mismatch": False},
        ],
        metadata={"fps": 30.0, "width": 1280, "height": 720, "duration_s": 2.0},
        reason=None,
        warnings=[],
        frame_count=2,
    )


class WebcamValidationTests(unittest.TestCase):
    def test_build_attempt_order_is_balanced_and_deterministic(self) -> None:
        labels = [f"sign_{index}" for index in range(21)]
        first = webcam_validation.build_attempt_order(labels, 10, 20261123)
        second = webcam_validation.build_attempt_order(labels, 10, 20261123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 210)
        for label in labels:
            self.assertEqual(first.count(label), 10)

    def test_prompt_confirmed_label_requires_match(self) -> None:
        labels = [f"sign_{index}" for index in range(3)]
        prompts = iter(["2", "1"])
        result = webcam_validation.prompt_confirmed_label(
            "sign_0", labels, 1, 3, lambda _: next(prompts), lambda _: None
        )
        self.assertEqual(result, "sign_0")

    def test_quick_mode_selects_five_signs_with_one_attempt_each(self) -> None:
        self.assertEqual(len(webcam_validation.QUICK_SIGNS), 5)
        self.assertEqual(webcam_validation.QUICK_ATTEMPTS_PER_CLASS, 1)
        order = webcam_validation.build_attempt_order(
            list(webcam_validation.QUICK_SIGNS),
            webcam_validation.QUICK_ATTEMPTS_PER_CLASS,
            seed=7,
        )
        self.assertEqual(len(order), 5)
        self.assertEqual(sorted(order), sorted(webcam_validation.QUICK_SIGNS))

    def test_prompt_quick_label_accepts_enter_without_number_entry(self) -> None:
        prompts: list[str] = []
        printed: list[str] = []
        result = webcam_validation.prompt_quick_label(
            "loud",
            1,
            5,
            lambda message: prompts.append(message) or "",
            printed.append,
        )
        self.assertEqual(result, "loud")
        self.assertTrue(any("TARGET SIGN: loud" in line for line in printed))
        self.assertTrue(any("Press ENTER" in prompt for prompt in prompts))
        self.assertFalse(any("sign number" in prompt for prompt in prompts))

    def test_format_quick_summary_includes_prediction_details(self) -> None:
        summary = {
            "session_id": "quick-test",
            "protocol": {"selected_signs": list(webcam_validation.QUICK_SIGNS), "required_valid_attempts": 5},
            "counts": {
                "correct": 1,
                "incorrect": 0,
                "rejected": 0,
                "errors": 0,
                "valid_attempts": 1,
                "complete": False,
            },
            "model": {
                "service": {"selected_variant": "v3_regularized_cv", "model_kind": "v3"},
                "actual_hashes": {"weights.npz": "abc"},
                "rejection_threshold": 0.53,
            },
        }
        rows = [
            {
                "ground_truth_label": "loud",
                "predicted_label": "loud",
                "calibrated_confidence": 0.91,
                "outcome": "correct",
            }
        ]
        text = webcam_validation.format_quick_summary(summary, rows)
        self.assertIn("Quick webcam validation summary", text)
        for sign in webcam_validation.QUICK_SIGNS:
            self.assertIn(sign, text)
        self.assertIn("| loud | loud | 0.910000 | correct |", text)

    def test_summarize_capture_reports_tracking_and_quality(self) -> None:
        summary = webcam_validation.summarize_capture(_capture_result())
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["hand_available_frames"], 2)
        self.assertEqual(summary["both_hands_frames"], 1)
        self.assertEqual(summary["no_hand_frames"], 0)
        self.assertTrue(summary["timestamp_stats"]["monotonic"])

    def test_metrics_and_confusion_preserve_unknown(self) -> None:
        labels = ["a", "b", "c"]
        rows = [
            {
                "outcome": "correct",
                "accepted": True,
                "ground_truth_label": "a",
                "predicted_label": "a",
            },
            {
                "outcome": "incorrect",
                "accepted": True,
                "ground_truth_label": "b",
                "predicted_label": "a",
            },
            {
                "outcome": "rejected",
                "accepted": False,
                "ground_truth_label": "c",
                "predicted_label": v4_service.UNKNOWN_SIGN,
            },
        ]
        all_metrics = webcam_validation.compute_metrics(rows, labels, accepted_only=False)
        accepted_metrics = webcam_validation.compute_metrics(rows, labels, accepted_only=True)
        confusion = webcam_validation.compute_confusion(rows, labels, accepted_only=False)
        self.assertEqual(all_metrics["sample_count"], 3)
        self.assertEqual(accepted_metrics["sample_count"], 2)
        self.assertEqual(confusion["labels"][-1], v4_service.UNKNOWN_SIGN)
        self.assertEqual(sum(sum(row) for row in confusion["matrix"]), 3)

    def test_run_session_writes_only_new_validation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = _fake_service(root / "artifacts")
            output_root = root / "validation"
            prompts: list[str] = []
            summary = webcam_validation.run_session(
                output_root=output_root,
                session_id="unit-session",
                attempts_per_class=1,
                seed=7,
                service=service,
                capture_fn=lambda **_: _capture_result(),
                prompter=lambda message: prompts.append(message) or "",
                printer=lambda _: None,
            )
            session_dir = output_root / "unit-session"
            self.assertTrue((session_dir / "session.json").is_file())
            self.assertTrue((session_dir / "attempts.jsonl").is_file())
            self.assertTrue((session_dir / "summary.json").is_file())
            self.assertTrue((session_dir / "review.md").is_file())
            self.assertEqual(len(v2_dataset.read_jsonl(session_dir / "attempts.jsonl")), 21)
            self.assertTrue(summary["counts"]["complete"])
            self.assertEqual(summary["protocol"]["mode"], webcam_validation.MODE_DETAILED)
            self.assertTrue(any("Type the sign number" in prompt for prompt in prompts))
            self.assertEqual(service.threshold, 0.53)
            self.assertEqual(summary["model"]["actual_hashes"]["weights.npz"], service.identity["model_sha256"])

    def test_run_session_quick_mode_writes_concise_five_sign_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = _fake_service(root / "artifacts")
            output_root = root / "validation"
            prompts: list[str] = []
            printed: list[str] = []

            def prompter(message: str) -> str:
                prompts.append(message)
                return ""

            summary = webcam_validation.run_session(
                output_root=output_root,
                session_id="quick-unit-session",
                attempts_per_class=99,
                service=service,
                capture_fn=lambda **_: _capture_result(),
                prompter=prompter,
                printer=printed.append,
                mode=webcam_validation.MODE_QUICK,
                selected_signs=list(webcam_validation.QUICK_SIGNS),
            )
            session_dir = output_root / "quick-unit-session"
            rows = v2_dataset.read_jsonl(session_dir / "attempts.jsonl")
            self.assertEqual(summary["protocol"]["mode"], webcam_validation.MODE_QUICK)
            self.assertEqual(summary["protocol"]["attempts_per_class"], 1)
            self.assertEqual(summary["protocol"]["required_valid_attempts"], 5)
            self.assertEqual(summary["protocol"]["selected_signs"], list(webcam_validation.QUICK_SIGNS))
            self.assertTrue(summary["counts"]["complete"])
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["ground_truth_label"] for row in rows}, set(webcam_validation.QUICK_SIGNS))
            self.assertTrue((session_dir / "quick_summary.md").is_file())
            self.assertIn("Quick webcam validation summary", (session_dir / "quick_summary.md").read_text(encoding="utf-8"))
            self.assertFalse(any("sign number" in prompt for prompt in prompts))
            self.assertTrue(any("Press ENTER" in prompt for prompt in prompts))
            self.assertTrue(any("TARGET SIGN:" in line for line in printed))
            self.assertTrue(any("expected=" in line and "result=" in line for line in printed))

    def test_main_quick_flag_uses_quick_mode(self) -> None:
        original = webcam_validation.run_session
        calls: dict[str, object] = {}

        def fake_run_session(**kwargs: object) -> dict[str, object]:
            calls.update(kwargs)
            return {"counts": {"complete": True}}

        webcam_validation.run_session = fake_run_session
        try:
            result = webcam_validation.main(["--quick"])
        finally:
            webcam_validation.run_session = original
        self.assertEqual(result, 0)
        self.assertEqual(calls["mode"], webcam_validation.MODE_QUICK)
        self.assertEqual(calls["attempts_per_class"], 1)
        self.assertEqual(calls["selected_signs"], list(webcam_validation.QUICK_SIGNS))
        self.assertTrue(str(calls["session_id"]).startswith("quick-"))

    def test_run_session_refuses_existing_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = _fake_service(root / "artifacts")
            output_root = root / "validation"
            (output_root / "unit-session").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                webcam_validation.run_session(
                    output_root=output_root,
                    session_id="unit-session",
                    attempts_per_class=1,
                    service=service,
                    capture_fn=lambda **_: _capture_result(),
                    prompter=lambda _: "",
                    printer=lambda _: None,
                )


if __name__ == "__main__":
    unittest.main()
