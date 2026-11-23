import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.src import prepare_video_features as feature_source
from ml.src.inference import rf_feature_builder as feature_builder
from ml.src.inference.landmark_capture import (
    STATUS_OK as CAPTURE_STATUS_OK,
    CaptureResult,
)
from ml.src.inference.rf_service import (
    CONFIDENCE_NOTE,
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_MODEL_PATH,
    EXPECTED_CLASS_COUNT,
    EXPECTED_FEATURE_COUNT,
    RFInferenceService,
    RFServiceConfigurationError,
    STATUS_FEATURE_ERROR,
    STATUS_INVALID_RECORDING,
    STATUS_MODEL_ERROR,
    STATUS_OK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURE_ROOT = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
FEATURE_CSV_PATH = FEATURE_ROOT / "rf_features.csv"
MANIFEST_PATH = FEATURE_ROOT / "video_manifest.csv"
METADATA_FIELDS = ("duration_s", "fps", "width", "height")


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MockClassifier:
    def __init__(
        self,
        n_features_in_=EXPECTED_FEATURE_COUNT,
        classes_=tuple(range(EXPECTED_CLASS_COUNT)),
        prediction=0,
        probabilities=None,
        predict_error=None,
        proba_error=None,
    ):
        self.n_features_in_ = n_features_in_
        self.classes_ = np.asarray(classes_)
        self.prediction = prediction
        self.probabilities = probabilities
        self.predict_error = predict_error
        self.proba_error = proba_error
        self.predict_calls = []
        self.proba_calls = []

    def _default_probabilities(self):
        values = np.zeros(EXPECTED_CLASS_COUNT, dtype=np.float64)
        values[self.prediction] = 1.0
        return values

    def predict(self, values):
        self.predict_calls.append(np.array(values, copy=True))
        if self.predict_error is not None:
            raise self.predict_error
        return np.array([self.prediction])

    def predict_proba(self, values):
        self.proba_calls.append(np.array(values, copy=True))
        if self.proba_error is not None:
            raise self.proba_error
        probabilities = (
            self._default_probabilities()
            if self.probabilities is None
            else np.asarray(self.probabilities, dtype=np.float64)
        )
        return probabilities.reshape(1, -1)


class FixedFeatureBuilder:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def __call__(self, records, metadata=None, expected_feature_names=None):
        self.calls.append((records, metadata, expected_feature_names))
        return self.row


def make_service(model=None, loader_calls=None, **kwargs):
    model = model if model is not None else MockClassifier()
    active_loader_calls = loader_calls if loader_calls is not None else []

    def loader(path):
        active_loader_calls.append(Path(path))
        return model

    service = RFInferenceService(model_loader=loader, **kwargs)
    return service, active_loader_calls


class RFServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feature_names = feature_builder.load_authoritative_feature_names()
        reference_rows = load_csv_rows(FEATURE_CSV_PATH)
        manifest_rows = {row["video_id"]: row for row in load_csv_rows(MANIFEST_PATH)}
        first = reference_rows[0]
        manifest = manifest_rows[first["video_id"]]
        cls.records = feature_source.read_jsonl(Path(manifest["landmark_file"]))
        cls.metadata = {field: float(manifest[field]) for field in METADATA_FIELDS}
        cls.expected_row = feature_builder.build_rf_feature_row(
            cls.records,
            metadata=cls.metadata,
            expected_feature_names=cls.feature_names,
        )

    def test_model_loads_once_and_repeated_calls_do_not_reload(self):
        loader_calls = []
        service, _ = make_service(loader_calls=loader_calls)
        self.assertEqual(loader_calls, [DEFAULT_MODEL_PATH])

        first = service.predict(self.records, self.metadata)
        second = service.predict(self.records, self.metadata)
        self.assertEqual(first["status"], STATUS_OK)
        self.assertEqual(second["status"], STATUS_OK)
        self.assertEqual(loader_calls, [DEFAULT_MODEL_PATH])

    def test_incompatible_model_feature_count_raises(self):
        model = MockClassifier(n_features_in_=EXPECTED_FEATURE_COUNT - 1)
        with self.assertRaisesRegex(RFServiceConfigurationError, "expects 830 features"):
            make_service(model=model)

    def test_incompatible_model_classes_raises(self):
        model = MockClassifier(classes_=tuple(range(EXPECTED_CLASS_COUNT - 1)))
        with self.assertRaisesRegex(RFServiceConfigurationError, "classes"):
            make_service(model=model)

    def test_incompatible_label_mapping_raises(self):
        payload = json.loads(DEFAULT_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        payload["label_mapping"][8]["original_label_id"] = 999999
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "bad_mapping.json"
            mapping_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RFServiceConfigurationError, "label mapping mismatch"):
                make_service(label_mapping_path=mapping_path)

    def test_correct_label_mapping_uses_authoritative_artifacts(self):
        probabilities = np.zeros(EXPECTED_CLASS_COUNT, dtype=np.float64)
        probabilities[20] = 0.77
        probabilities[0] = 0.23
        model = MockClassifier(prediction=20, probabilities=probabilities)
        service, _ = make_service(model=model)

        result = service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["model_class_index"], 20)
        self.assertEqual(result["predicted_sign"], "wet")
        self.assertEqual(result["original_label_id"], 250)
        self.assertAlmostEqual(result["raw_confidence"], 0.77)
        wet_entry = result["raw_probabilities"][20]
        self.assertEqual(wet_entry["class_label"], "wet")
        self.assertEqual(wet_entry["original_label_id"], 250)
        self.assertAlmostEqual(wet_entry["probability"], 0.77)

    def test_feature_builder_integration_passes_exact_vector_to_model(self):
        model = MockClassifier(prediction=2)
        service, _ = make_service(model=model)

        result = service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(len(model.predict_calls), 1)
        self.assertEqual(len(model.proba_calls), 1)
        np.testing.assert_array_equal(
            model.predict_calls[0],
            self.expected_row.values,
        )
        np.testing.assert_array_equal(
            model.proba_calls[0],
            self.expected_row.values,
        )
        self.assertEqual(result["recording_diagnostics"], self.expected_row.diagnostics)

    def test_prediction_and_probability_output_structure(self):
        probabilities = np.full(EXPECTED_CLASS_COUNT, 1.0 / EXPECTED_CLASS_COUNT)
        model = MockClassifier(prediction=7, probabilities=probabilities)
        service, _ = make_service(model=model)

        result = service.predict(self.records, self.metadata)
        self.assertEqual(set(result), {
            "status",
            "predicted_sign",
            "original_label_id",
            "model_class_index",
            "raw_confidence",
            "raw_probabilities",
            "recording_diagnostics",
            "model",
            "confidence_note",
            "warnings",
            "errors",
        })
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["confidence_note"], CONFIDENCE_NOTE)
        self.assertAlmostEqual(result["raw_confidence"], 1.0 / EXPECTED_CLASS_COUNT)
        self.assertEqual(len(result["raw_probabilities"]), EXPECTED_CLASS_COUNT)
        for entry in result["raw_probabilities"]:
            self.assertEqual(
                set(entry),
                {"model_class_index", "class_label", "original_label_id", "probability"},
            )
        self.assertEqual(result["model"]["service_version"], "synora_rf_service_v1")
        self.assertEqual(result["model"]["experiment_id"], "synora_baseline_v1_20261121")
        self.assertEqual(result["model"]["expected_feature_count"], 831)
        self.assertEqual(result["model"]["model_class_count"], 21)
        self.assertEqual(
            result["model"]["model_sha256"],
            "cdfb6673a6499f999c263c8503e2ae1d48ff888165d1bfa478db73c44bab65c5",
        )

    def test_invalid_records_return_invalid_recording_without_prediction(self):
        model = MockClassifier()
        service, _ = make_service(model=model)

        result = service.predict([], self.metadata)
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertEqual(result["errors"][0]["code"], "FEATURE_BUILD_FAILED")
        self.assertIsNone(result["predicted_sign"])
        self.assertIsNone(result["raw_confidence"])
        self.assertEqual(model.predict_calls, [])
        self.assertEqual(model.proba_calls, [])

    def test_non_finite_feature_vector_returns_feature_error_without_prediction(self):
        model = MockClassifier()
        values = np.zeros((1, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        values[0, 0] = np.nan
        row = feature_builder.RFFeatureRow(
            feature_names=self.feature_names,
            values=values,
            diagnostics={},
        )
        service, _ = make_service(model=model, feature_builder_fn=FixedFeatureBuilder(row))

        result = service.predict([{"frame_index": 0}])
        self.assertEqual(result["status"], STATUS_FEATURE_ERROR)
        self.assertEqual(result["errors"][0]["code"], "FEATURE_VALIDATION_FAILED")
        self.assertIsNone(result["predicted_sign"])
        self.assertIsNone(result["raw_confidence"])
        self.assertEqual(model.predict_calls, [])
        self.assertEqual(model.proba_calls, [])

    def test_invalid_feature_shape_returns_feature_error_without_prediction(self):
        model = MockClassifier()
        row = feature_builder.RFFeatureRow(
            feature_names=self.feature_names,
            values=np.zeros((1, EXPECTED_FEATURE_COUNT - 1), dtype=np.float32),
            diagnostics={},
        )
        service, _ = make_service(model=model, feature_builder_fn=FixedFeatureBuilder(row))

        result = service.predict([{"frame_index": 0}])
        self.assertEqual(result["status"], STATUS_FEATURE_ERROR)
        self.assertIn("feature shape", result["errors"][0]["message"])
        self.assertEqual(model.predict_calls, [])

    def test_model_prediction_error_returns_model_error_without_prediction(self):
        model = MockClassifier(predict_error=RuntimeError("mock predict failure"))
        service, _ = make_service(model=model)

        result = service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_MODEL_ERROR)
        self.assertEqual(result["errors"][0]["code"], "MODEL_PREDICTION_FAILED")
        self.assertIsNone(result["predicted_sign"])
        self.assertIsNone(result["original_label_id"])
        self.assertIsNone(result["model_class_index"])
        self.assertIsNone(result["raw_confidence"])
        self.assertIsNone(result["raw_probabilities"])

    def test_probability_error_returns_model_error_without_prediction_output(self):
        model = MockClassifier(proba_error=RuntimeError("mock predict_proba failure"))
        service, _ = make_service(model=model)

        result = service.predict(self.records, self.metadata)
        self.assertEqual(result["status"], STATUS_MODEL_ERROR)
        self.assertEqual(result["errors"][0]["code"], "MODEL_PREDICTION_FAILED")
        self.assertIsNone(result["raw_confidence"])

    def test_predict_capture_accepts_successful_capture_result(self):
        model = MockClassifier(prediction=5)
        service, _ = make_service(model=model)
        capture = CaptureResult(
            status=CAPTURE_STATUS_OK,
            records=copy.deepcopy(self.records),
            metadata=dict(self.metadata),
            reason=None,
            warnings=["camera fps unavailable; using configured 25.0"],
            frame_count=len(self.records),
        )

        result = service.predict_capture(capture)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["model_class_index"], 5)
        self.assertEqual(result["recording_diagnostics"]["capture_status"], CAPTURE_STATUS_OK)
        self.assertEqual(result["recording_diagnostics"]["frame_count"], len(self.records))
        self.assertTrue(
            any("camera fps unavailable" in warning for warning in result["warnings"])
        )
        np.testing.assert_array_equal(model.predict_calls[0], self.expected_row.values)

    def test_predict_capture_rejects_non_ok_capture_result_without_prediction(self):
        model = MockClassifier()
        service, _ = make_service(model=model)
        capture = CaptureResult(
            status="error",
            records=[],
            metadata=None,
            reason="FRAME_READ_FAILED",
            warnings=[],
            frame_count=0,
        )

        result = service.predict_capture(capture)
        self.assertEqual(result["status"], STATUS_INVALID_RECORDING)
        self.assertEqual(result["errors"][0]["code"], "CAPTURE_NOT_OK")
        self.assertIsNone(result["predicted_sign"])
        self.assertEqual(model.predict_calls, [])


if __name__ == "__main__":
    unittest.main()
