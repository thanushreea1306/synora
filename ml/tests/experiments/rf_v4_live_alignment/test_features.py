import csv
import unittest
from pathlib import Path

import numpy as np

from ml.src import prepare_video_features as feature_source
from ml.src.experiments.rf_v2_720p30 import build_features_v2, dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import features


REPO_ROOT = Path(__file__).resolve().parents[4]
V2_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30/features"
V4_FEATURES_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment/features"
V4_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = v2_dataset.read_csv(V4_FEATURES_ROOT / "video_manifest.csv")
        extraction_rows = v2_dataset.read_csv(V4_EXTRACTION_ROOT / "manifest.csv")
        row = next(item for item in manifest if item["split"] == "val")
        extraction_row = next(item for item in extraction_rows if item["video_id"] == row["video_id"])
        cls.records = v2_dataset.read_jsonl(extraction_row["landmark_file"])
        sample_row = dict(extraction_row)
        sample_row["output_file"] = None
        sample_row["duration_s"] = int(extraction_row["target_frame_count"]) / 30.0
        cls.sample = build_features_v2.build_sample_from_records(sample_row, cls.records)
        with np.load(V2_FEATURES_ROOT / "mlp_preprocessing.npz") as preprocessing:
            cls.mean = preprocessing["mean"].copy()
            cls.std = preprocessing["std"].copy()
        cls.channels = v2_dataset.read_json(V2_FEATURES_ROOT / "mlp_channel_names.json")

    def test_enhanced_schema_has_509_unique_channels(self):
        names = features.ENHANCED_CHANNEL_NAMES
        self.assertEqual(len(names), 509)
        self.assertEqual(len(set(names)), 509)
        self.assertIn("frame_real", names)
        self.assertTrue(any(name.startswith("motion_abs_") for name in names))
        self.assertTrue(any(name.endswith("_valid") for name in names))

    def test_base_tensor_matches_frozen_v2_builder(self):
        actual = features.build_base_tensor(self.sample, self.mean, self.std, self.channels)
        expected = feature_source.build_mlp_tensor(self.sample, self.mean, self.std, self.channels)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_enhanced_tensor_shape_finite_and_padded_frames_are_zero(self):
        tensor = features.build_enhanced_tensor(self.sample)
        self.assertEqual(tensor.shape, (89, 509))
        self.assertTrue(np.isfinite(tensor).all())
        padded = ~np.asarray(self.sample["frame_real"], dtype=bool)
        self.assertTrue(not padded.any() or np.all(tensor[padded] == 0.0))

    def test_manifest_has_only_train_val_and_no_test_access(self):
        rows = v2_dataset.read_csv(V4_FEATURES_ROOT / "video_manifest.csv")
        self.assertEqual({row["split"] for row in rows}, {"train", "val"})
        config = v2_dataset.read_json(V4_EXTRACTION_ROOT / "config.json")
        self.assertIs(config["split_policy"]["test_split_read"], False)


if __name__ == "__main__":
    unittest.main()
