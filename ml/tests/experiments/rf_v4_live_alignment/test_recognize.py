import unittest
from dataclasses import dataclass
from typing import Any

from ml.src.experiments.rf_v4_live_alignment import capture, recognize, v4_service


@dataclass
class FakeService:
    result: dict[str, Any]
    calls: int = 0

    def predict_capture(self, capture_result):
        self.calls += 1
        return self.result


def accepted_result():
    return {
        "status": v4_service.STATUS_OK,
        "accepted": True,
        "predicted_sign": "happy",
        "original_label_id": 95,
        "model_class_index": 2,
        "raw_confidence": 0.71,
        "calibrated_confidence": 0.86,
        "rejection_threshold": 0.53,
        "model": {"selected_variant": "v3_regularized_cv"},
        "recording_diagnostics": {"capture_frame_count": 43},
        "warnings": [],
        "confidence_note": "calibrated",
        "limitation_note": "clip-disjoint CV only",
    }


class V4RecognizeCLITests(unittest.TestCase):
    def setUp(self):
        self.output: list[str] = []

    def printer(self, line):
        self.output.append(str(line))

    def test_cli_connects_capture_and_service_without_webcam(self):
        capture_result = capture.CaptureResult(
            status=capture.STATUS_OK,
            records=[{"frame_index": 0}],
            metadata={"duration_s": 1 / 30, "fps": 30.0, "width": 1280.0, "height": 720.0},
            frame_count=1,
        )
        service = FakeService(accepted_result())
        capture_calls = []

        def capture_fn(**kwargs):
            capture_calls.append(kwargs)
            return capture_result

        def viewer_factory():
            return object()

        code = recognize.main(
            argv=["--no-preview", "--max-recordings", "1"],
            printer=self.printer,
            service=service,
            capture_config=capture.CaptureConfig(show_preview=False),
            capture_fn=capture_fn,
            viewer_factory=viewer_factory,
        )
        text = "\n".join(self.output)
        self.assertEqual(code, 0)
        self.assertEqual(service.calls, 1)
        self.assertIs(capture_calls[0]["viewer_factory"], viewer_factory)
        self.assertIn("Predicted sign: happy", text)
        self.assertIn("Model variant: v3_regularized_cv", text)
        self.assertIn("Calibrated confidence: 0.860000", text)

    def test_instructions_show_version_limits_and_tracking_reset(self):
        self.assertIn("v4", recognize.INSTRUCTIONS)
        self.assertIn("minimum 43 frames", recognize.INSTRUCTIONS)
        self.assertIn("maximum 89 frames", recognize.INSTRUCTIONS)
        self.assertIn("tracking resets when recording starts", recognize.INSTRUCTIONS)

    def test_low_confidence_output_says_unknown(self):
        result = dict(
            accepted_result(),
            accepted=False,
            predicted_sign=v4_service.UNKNOWN_SIGN,
            candidate_sign="loud",
            raw_confidence=0.30,
            calibrated_confidence=0.22,
            warnings=["LOW_CONFIDENCE_REJECTED: 0.220000 < 0.530000"],
        )
        text = recognize.format_prediction(result)
        self.assertIn("Predicted sign: unknown", text)
        self.assertIn("Candidate sign: loud", text)
        self.assertIn("LOW_CONFIDENCE_REJECTED", text)


if __name__ == "__main__":
    unittest.main()
