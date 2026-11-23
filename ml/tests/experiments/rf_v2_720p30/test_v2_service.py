import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.src.experiments.rf_v2_720p30 import dataset
from ml.src.experiments.rf_v2_720p30.capture import CaptureResult, STATUS_OK as CAPTURE_STATUS_OK
from ml.src.experiments.rf_v2_720p30.v2_service import (
    MAX_FRAMES,
    MIN_FRAMES,
    STATUS_INVALID_RECORDING,
    STATUS_OK,
    UNKNOWN_SIGN,
    V2RecognitionService,
    V2ServiceConfigurationError,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"


def frame_records(count):
    return [{"frame_index": index} for index in range(count)]


def confident_tall_probabilities():
    values = np.zeros((1, 21), dtype=np.float64)
    values[0, 5] = 1.0
    return values


class V2RecognitionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = V2RecognitionService()
        manifest = dataset.read_csv(FEATURES_ROOT / "video_manifest.csv")
        row = next(item for item in manifest if item["split"] == "val")
        cls.records = dataset.read_jsonl(row["landmark_file"])
        cls.metadata = {
            "duration_s": int(row["frame_count"]) / 30.0,
            "fps": 30.0,
            "width": 1280.0,
            "height": 720.0,
        }

    def test_service_loads_selected_v2_model_and_threshold(self):
        self.assertEqual(self.service.identity["service_version"], "synora_rf_v2_service_v1")
        self.assertEqual(self.service.identity["selected_model"], "temporal_mlp_v2")
        self.assertEqual(self.service.identity["rejection_threshold"], 0.59)
        self.assertEqual(len(self.service.feature_names), 831)

    def test_real_validation_record_predicts_with_valid_probabilities(self):
        result = self.service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertTrue(result["accepted"])
        self.assertNotEqual(result["predicted_sign"], UNKNOWN_SIGN)
        self.assertGreaterEqual(result["raw_confidence"], result["rejection_threshold"])
        self.assertEqual(len(result["raw_probabilities"]), 21)
        self.assertAlmostEqual(sum(result["raw_probabilities"].values()), 1.0, places=6)

    def test_low_confidence_prediction_is_rejected_as_unknown(self):
        uniform = np.full((1, 21), 1.0 / 21.0, dtype=np.float64)
        with patch.object(self.service, "_probabilities", return_value=uniform):
            result = self.service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["predicted_sign"], UNKNOWN_SIGN)
        self.assertIsNone(result["model_class_index"])
        self.assertTrue(any("LOW_CONFIDENCE_REJECTED" in warning for warning in result["warnings"]))

    def test_capture_result_uses_record_metadata_and_diagnostics(self):
        capture_result = CaptureResult(
            status=CAPTURE_STATUS_OK,
            records=self.records,
            metadata=self.metadata,
            frame_count=len(self.records),
            warnings=["capture warning"],
        )
        result = self.service.predict_capture(capture_result)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["recording_diagnostics"]["capture_frame_count"], len(self.records))
        self.assertIn("capture warning", result["warnings"])

    def test_non_ok_capture_is_rejected_without_prediction(self):
        result = self.service.predict_capture(CaptureResult(status="error", reason="FRAME_READ_FAILED"))
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertEqual(result["errors"][0]["code"], "CAPTURE_NOT_OK")

    def test_wrong_acquisition_metadata_is_invalid(self):
        bad_metadata = dict(self.metadata, fps=25.0)
        result = self.service.predict(self.records, bad_metadata)
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertEqual(result["errors"][0]["code"], "INVALID_RECORDING")

    def test_minimum_matches_global_manifest_support(self):
        manifest = dataset.read_csv(REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/manifest.csv")
        observed = [int(row["target_frame_count"]) for row in manifest if row["status"] == "complete"]
        self.assertEqual(len(observed), 346)
        self.assertEqual(MIN_FRAMES, min(observed))
        self.assertEqual(MAX_FRAMES, 89)

    def test_below_minimum_duration_is_unknown_without_inference(self):
        with patch.object(
            self.service, "_probabilities", return_value=confident_tall_probabilities()
        ) as probabilities:
            result = self.service.predict(frame_records(MIN_FRAMES - 1), self.metadata)
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["predicted_sign"], UNKNOWN_SIGN)
        self.assertIsNone(result["raw_confidence"])
        self.assertEqual(result["errors"][0]["code"], "INSUFFICIENT_DURATION")
        self.assertEqual(result["recording_diagnostics"]["record_frame_count"], MIN_FRAMES - 1)
        self.assertEqual(result["recording_diagnostics"]["minimum_frame_count"], MIN_FRAMES)
        probabilities.assert_not_called()

    def test_exactly_minimum_duration_is_accepted(self):
        with patch.object(
            self.service, "_probabilities", return_value=confident_tall_probabilities()
        ) as probabilities:
            result = self.service.predict(frame_records(MIN_FRAMES), self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["predicted_sign"], "tall")
        self.assertEqual(result["recording_diagnostics"]["record_frame_count"], MIN_FRAMES)
        probabilities.assert_called_once()

    def test_valid_duration_is_accepted(self):
        with patch.object(
            self.service, "_probabilities", return_value=confident_tall_probabilities()
        ) as probabilities:
            result = self.service.predict(frame_records(MIN_FRAMES + 16), self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["predicted_sign"], "tall")
        probabilities.assert_called_once()

    def test_over_maximum_duration_is_rejected_without_inference(self):
        with patch.object(
            self.service, "_probabilities", return_value=confident_tall_probabilities()
        ) as probabilities:
            result = self.service.predict(frame_records(MAX_FRAMES + 1), self.metadata)
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["predicted_sign"], UNKNOWN_SIGN)
        self.assertIsNone(result["raw_confidence"])
        self.assertEqual(result["errors"][0]["code"], "EXCESSIVE_DURATION")
        self.assertEqual(result["recording_diagnostics"]["maximum_frame_count"], MAX_FRAMES)
        probabilities.assert_not_called()

    def test_model_hash_mismatch_fails_initialization(self):
        original = dataset.sha256_file

        def patched(path):
            if Path(path).name == "weights.npz":
                return "bad-hash"
            return original(path)

        with patch.object(dataset, "sha256_file", side_effect=patched):
            with self.assertRaisesRegex(V2ServiceConfigurationError, "hash mismatch"):
                V2RecognitionService()


if __name__ == "__main__":
    unittest.main()
