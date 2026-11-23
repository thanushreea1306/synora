import unittest
from pathlib import Path

import numpy as np

from ml.src.experiments.rf_v2_720p30 import dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"
FEATURES_ROOT = EXTRACTION_ROOT / "features"
MODEL_ROOT = REPO_ROOT / "ml/models/experiments/rf_v2_720p30"
REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v2_720p30"


class V2PipelineArtifactTests(unittest.TestCase):
    def test_extraction_uses_train_val_only_and_records_exclusions(self):
        config = dataset.read_json(EXTRACTION_ROOT / "config.json")
        summary = dataset.read_json(EXTRACTION_ROOT / "extraction_summary.json")
        self.assertIs(config["split_policy"]["test_split_read"], False)
        self.assertEqual(config["split_policy"]["allowed_splits"], ["train", "val"])
        self.assertEqual(summary["selected_counts"], {"train": 317, "val": 42})
        self.assertEqual(summary["complete_counts"], {"train": 305, "val": 41})
        self.assertEqual(summary["excluded_counts"], {"train": 12, "val": 1})
        self.assertEqual(summary["failed_video_ids"], [])
        self.assertEqual(len(summary["excluded_video_ids"]), 13)

    def test_feature_matrices_match_frozen_schema(self):
        schema = dataset.read_json(FEATURES_ROOT / "rf_feature_schema.json")
        summary = dataset.read_json(FEATURES_ROOT / "feature_build_summary.json")
        rf = np.load(FEATURES_ROOT / "rf_features.npy")
        labels = np.load(FEATURES_ROOT / "labels.npy")
        tensor = np.load(FEATURES_ROOT / "mlp_tensor.npy")
        flat = np.load(FEATURES_ROOT / "mlp_flat.npy")
        self.assertEqual(len(schema["feature_columns"]), 831)
        self.assertEqual(schema["scaling"], "none")
        self.assertEqual(summary["split_counts"], {"train": 305, "val": 41})
        self.assertEqual(rf.shape, (346, 831))
        self.assertEqual(rf.dtype, np.float32)
        self.assertEqual(labels.shape, (346,))
        self.assertEqual(tensor.shape, (346, 89, 135))
        self.assertEqual(flat.shape, (346, 12015))
        np.testing.assert_array_equal(flat, tensor.reshape(346, -1))
        self.assertTrue(np.isfinite(rf).all())
        self.assertTrue(np.isfinite(tensor).all())

    def test_training_artifacts_and_model_selection(self):
        selection = dataset.read_json(REPORT_ROOT / "model_selection.json")
        manifest = dataset.read_json(REPORT_ROOT / "training_manifest.json")
        threshold = dataset.read_json(REPORT_ROOT / "confidence_threshold.json")
        self.assertIs(manifest["test_split_read"], False)
        self.assertEqual(manifest["selection_split"], "val")
        self.assertEqual(manifest["split_counts"], {"train": 305, "val": 41})
        self.assertEqual(selection["selected_model"], "temporal_mlp_v2")
        self.assertGreater(
            selection["metrics"]["temporal_mlp_v2"]["macro_f1"],
            selection["metrics"]["rf_v2"]["macro_f1"],
        )
        self.assertGreater(
            selection["metrics"]["rf_v2"]["macro_f1"],
            selection["metrics"]["frozen_v1_rf_on_v2_val"]["macro_f1"],
        )
        self.assertGreaterEqual(threshold["chosen"]["coverage"], 0.90)
        for name, expected in manifest["output_hashes"].items():
            self.assertEqual(dataset.sha256_file(MODEL_ROOT / name), expected)


if __name__ == "__main__":
    unittest.main()
