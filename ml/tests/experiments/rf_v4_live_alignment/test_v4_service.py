import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import capture, v4_service


REPO_ROOT = Path(__file__).resolve().parents[4]
V4_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
V4_FEATURES_ROOT = V4_EXTRACTION_ROOT / "features"


def frame_records(count):
    return [{"frame_index": index} for index in range(count)]


def tall_probabilities(value=1.0):
    proba = np.zeros((1, 21), dtype=np.float64)
    proba[0, 5] = value
    if value < 1.0:
        proba[0, 0] = 1.0 - value
    return proba


class V4RecognitionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = v4_service.V4RecognitionService()
        manifest = v2_dataset.read_csv(V4_FEATURES_ROOT / "video_manifest.csv")
        extraction_rows = v2_dataset.read_csv(V4_EXTRACTION_ROOT / "manifest.csv")
        row = next(item for item in manifest if item["split"] == "val")
        extraction_row = next(item for item in extraction_rows if item["video_id"] == row["video_id"])
        cls.records = v2_dataset.read_jsonl(extraction_row["landmark_file"])
        cls.metadata = {
            "duration_s": int(row["frame_count"]) / 30.0,
            "fps": 30.0,
            "width": 1280.0,
            "height": 720.0,
        }

    def test_service_loads_selected_v4_artifacts(self):
        self.assertEqual(self.service.identity["service_version"], "synora_rf_v4_service_v1")
        self.assertEqual(self.service.identity["selected_variant"], "v3_regularized_cv")
        self.assertEqual(self.service.identity["feature_mode"], "base")
        self.assertGreater(self.service.identity["validation_macro_f1"], 0.88)
        self.assertNotIn("test", self.service.identity["validation_protocol"].lower())

    def test_real_v4_record_predicts_with_valid_probabilities(self):
        result = self.service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], v4_service.STATUS_OK)
        self.assertAlmostEqual(sum(result["raw_probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(result["calibrated_probabilities"].values()), 1.0, places=6)
        self.assertGreater(result["calibrated_confidence"], 0.0)
        self.assertIn("no live-webcam accuracy improvement", result["limitation_note"])

    def test_calibration_changes_confidence(self):
        logits = np.zeros((1, 21), dtype=np.float64)
        logits[0, 5] = 2.0
        logits[0, 14] = 1.0
        with patch.object(self.service, "_logits", return_value=logits):
            result = self.service.predict(self.records, self.metadata)
        self.assertNotAlmostEqual(result["raw_confidence"], result["calibrated_confidence"], places=4)

    def test_low_confidence_prediction_is_rejected_as_unknown(self):
        uniform = np.full((1, 21), 1.0 / 21.0, dtype=np.float64)
        with patch.object(self.service, "_probabilities", return_value=(uniform, uniform)):
            result = self.service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], v4_service.STATUS_OK)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["predicted_sign"], v4_service.UNKNOWN_SIGN)
        self.assertTrue(any("LOW_CONFIDENCE_REJECTED" in item for item in result["warnings"]))

    def test_capture_result_warnings_are_preserved(self):
        capture_result = capture.CaptureResult(
            status=capture.STATUS_OK,
            records=self.records,
            metadata=self.metadata,
            frame_count=len(self.records),
            warnings=["capture warning"],
        )
        result = self.service.predict_capture(capture_result)
        self.assertEqual(result["status"], v4_service.STATUS_OK)
        self.assertIn("capture warning", result["warnings"])
        self.assertEqual(result["recording_diagnostics"]["capture_frame_count"], len(self.records))

    def test_non_ok_capture_is_rejected_without_prediction(self):
        result = self.service.predict_capture(capture.CaptureResult(status="error", reason="FRAME_READ_FAILED"))
        self.assertEqual(result["status"], v4_service.STATUS_INVALID_RECORDING)
        self.assertEqual(result["errors"][0]["code"], "CAPTURE_NOT_OK")

    def test_bad_metadata_and_duration_are_invalid(self):
        bad_metadata = self.service.predict(self.records, dict(self.metadata, fps=25.0))
        self.assertEqual(bad_metadata["errors"][0]["code"], "INVALID_RECORDING")
        with patch.object(
            self.service,
            "_probabilities",
            return_value=(tall_probabilities(), tall_probabilities()),
        ) as probabilities:
            short = self.service.predict(frame_records(v4_service.MIN_FRAMES - 1), self.metadata)
        self.assertEqual(short["errors"][0]["code"], "INSUFFICIENT_DURATION")
        probabilities.assert_not_called()

    def test_model_hash_mismatch_fails_initialization(self):
        original = v2_dataset.sha256_file

        def patched(path):
            if Path(path).name == "weights.npz":
                return "bad-hash"
            return original(path)

        with patch.object(v2_dataset, "sha256_file", side_effect=patched):
            with self.assertRaisesRegex(v4_service.V4ServiceConfigurationError, "hash mismatch"):
                v4_service.V4RecognitionService()


if __name__ == "__main__":
    unittest.main()
