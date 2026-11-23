import json
import unittest
from pathlib import Path

import numpy as np

from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import train_cv


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = REPO_ROOT / "ml/reports/experiments/rf_v4_live_alignment"


class CrossValidationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = train_cv.load_data(train_cv.parse_args([]))
        cls.results = v2_dataset.read_json(REPORT_ROOT / "cv_results.json")
        cls.selection = v2_dataset.read_json(REPORT_ROOT / "selection.json")
        cls.manifest = v2_dataset.read_json(REPORT_ROOT / "training_manifest.json")
        cls.parity = v2_dataset.read_json(REPORT_ROOT / "parity_evidence.json")

    def test_common_subset_uses_333_shared_train_val_clips(self):
        self.assertEqual(len(self.data["common_ids"]), 333)
        self.assertEqual(self.data["split_counts"], {"train": 293, "val": 40})
        self.assertEqual(len(self.data["y"]), 333)
        self.assertTrue(np.array_equal(self.data["v3_tensor"].shape[:1], self.data["base_tensor"].shape[:1]))

    def test_folds_are_deterministic_and_group_disjoint(self):
        first = train_cv.make_folds(self.data["y"], self.data["groups"])
        second = train_cv.make_folds(self.data["y"], self.data["groups"])
        for (train, val), (train_again, val_again) in zip(first, second):
            np.testing.assert_array_equal(train, train_again)
            np.testing.assert_array_equal(val, val_again)
            self.assertFalse(set(self.data["groups"][train]) & set(self.data["groups"][val]))
        covered = np.concatenate([val for _, val in first])
        self.assertEqual(len(covered), 333)

    def test_all_four_variants_were_compared(self):
        self.assertEqual(
            set(self.results["variants"]),
            {
                "v3_regularized_cv",
                "v4_base_mlp_cv",
                "v4_enhanced_tcn_cv",
                "v4_enhanced_attention_cv",
            },
        )
        self.assertEqual(self.selection["selected_variant"], "v3_regularized_cv")
        self.assertIs(self.selection["selection_uses_test"], False)

    def test_selection_penalizes_worst_class_regression(self):
        target_names = [f"class_{index}" for index in range(21)]
        baseline = {
            "classification_report": {name: {"f1-score": 0.8} for name in target_names},
            "metrics": {"macro_f1": 0.8},
            "calibration": {"temperature": {"ece_after": 0.01}},
        }
        strong = {
            "classification_report": {name: {"f1-score": 0.9} for name in target_names},
            "metrics": {"macro_f1": 0.9},
            "calibration": {"temperature": {"ece_after": 0.01}},
        }
        regressed = {
            "classification_report": {
                **{name: {"f1-score": 0.95} for name in target_names[:-1]},
                target_names[-1]: {"f1-score": 0.4},
            },
            "metrics": {"macro_f1": 0.92},
            "calibration": {"temperature": {"ece_after": 0.01}},
        }
        strong_score = train_cv.selection_score(strong, baseline, target_names)
        regressed_score = train_cv.selection_score(regressed, baseline, target_names)
        self.assertGreater(strong_score["score"], regressed_score["score"])
        self.assertGreater(regressed_score["worst_class_regression"], 0.0)

    def test_training_manifest_and_parity_evidence_are_consistent(self):
        self.assertIs(self.manifest["test_split_read"], False)
        self.assertTrue(self.manifest["parity_passed"])
        self.assertTrue(self.parity["passed"])
        self.assertTrue(self.parity["preprocessing"]["passed"])
        self.assertTrue(self.parity["lifecycle"]["passed"])
        calibration = v2_dataset.read_json(REPORT_ROOT / "calibration.json")
        self.assertEqual(calibration["threshold"]["chosen"]["threshold"], 0.53)


if __name__ == "__main__":
    unittest.main()
