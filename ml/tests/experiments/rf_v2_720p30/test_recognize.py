import unittest
from dataclasses import dataclass
from typing import Any

from ml.src.experiments.rf_v2_720p30 import recognize
from ml.src.experiments.rf_v2_720p30.capture import CaptureConfig, CaptureResult, STATUS_OK as CAPTURE_STATUS_OK
from ml.src.experiments.rf_v2_720p30.v2_service import STATUS_OK, UNKNOWN_SIGN


@dataclass
class FakeService:
    result: dict[str, Any]
    calls: int = 0

    def predict_capture(self, capture_result):
        self.calls += 1
        return self.result


def accepted_result(frame_count=42):
    return {
        "status": STATUS_OK,
        "accepted": True,
        "predicted_sign": "happy",
        "original_label_id": 95,
        "model_class_index": 2,
        "raw_confidence": 0.91,
        "rejection_threshold": 0.59,
        "recording_diagnostics": {"capture_frame_count": frame_count},
        "warnings": [],
        "confidence_note": "validation-derived threshold",
    }


def rejected_result():
    return {
        "status": STATUS_OK,
        "accepted": False,
        "predicted_sign": UNKNOWN_SIGN,
        "candidate_sign": "loud",
        "raw_confidence": 0.2,
        "rejection_threshold": 0.59,
        "recording_diagnostics": {"capture_frame_count": 12},
        "warnings": ["LOW_CONFIDENCE_REJECTED: 0.200000 < 0.590000"],
        "confidence_note": "validation-derived threshold",
    }


class V2RecognizeCLITests(unittest.TestCase):
    def setUp(self):
        self.output: list[str] = []

    def printer(self, line):
        self.output.append(str(line))

    def test_cli_connects_capture_and_service_without_webcam(self):
        capture_calls = []
        capture_result = CaptureResult(
            status=CAPTURE_STATUS_OK,
            records=[{"frame_index": 0}],
            metadata={"duration_s": 1 / 30, "fps": 30.0, "width": 1280.0, "height": 720.0},
            frame_count=1,
        )
        service = FakeService(accepted_result(frame_count=1))

        def capture_fn(**kwargs):
            capture_calls.append(kwargs)
            return capture_result

        code = recognize.main(
            argv=["--no-preview", "--max-recordings", "1"],
            printer=self.printer,
            service=service,
            capture_config=CaptureConfig(show_preview=False),
            capture_fn=capture_fn,
        )
        text = "\n".join(self.output)
        self.assertEqual(code, 0)
        self.assertEqual(service.calls, 1)
        self.assertEqual(len(capture_calls), 1)
        self.assertIn("Predicted sign: happy", text)
        self.assertIn("Raw confidence: 0.910000", text)
        self.assertIn("Rejection threshold: 0.590000", text)
        self.assertIn("capture_frame_count: 1", text)
        self.assertIn("Finished requested recordings", text)

    def test_instructions_show_recording_limits(self):
        self.assertIn("minimum 43 frames", recognize.INSTRUCTIONS)
        self.assertIn("maximum 89 frames", recognize.INSTRUCTIONS)
        self.assertIn("automatically at 89 frames", recognize.INSTRUCTIONS)
        self.assertIn("Press E to end at 43 frames or more", recognize.INSTRUCTIONS)

    def test_low_confidence_output_says_unknown(self):
        text = recognize.format_prediction(rejected_result())
        self.assertIn("Predicted sign: unknown", text)
        self.assertIn("Candidate sign: loud", text)
        self.assertIn("LOW_CONFIDENCE_REJECTED", text)

    def test_failed_prediction_does_not_show_a_sign(self):
        result = {
            "status": "feature_error",
            "accepted": False,
            "predicted_sign": UNKNOWN_SIGN,
            "raw_confidence": None,
            "rejection_threshold": None,
            "recording_diagnostics": {},
            "warnings": [],
            "errors": [{"code": "FEATURE_BUILD_FAILED", "message": "invalid frame"}],
            "confidence_note": "validation-derived threshold",
        }
        text = recognize.format_prediction(result)
        self.assertIn("Prediction failed: feature_error", text)
        self.assertIn("FEATURE_BUILD_FAILED: invalid frame", text)
        self.assertIn("No prediction was produced.", text)


if __name__ == "__main__":
    unittest.main()
