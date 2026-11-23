import unittest
from unittest.mock import patch

import numpy as np

from ml.src.experiments.rf_v3_generalisation import cv, model


class CrossValidationTests(unittest.TestCase):
    def test_load_data_uses_only_train_val_features(self):
        data = cv.load_data(cv.parse_args([]))
        self.assertEqual(len(data["manifest"]), 346)
        self.assertEqual({row["split"] for row in data["manifest"]}, {"train", "val"})
        self.assertEqual(len(data["y"]), 346)
        self.assertEqual(data["tensor"].shape, (346, 89, 135))

    def test_make_folds_are_deterministic_group_disjoint_and_stratified(self):
        data = cv.load_data(cv.parse_args([]))
        folds = cv.make_folds(data["y"], data["groups"])
        second = cv.make_folds(data["y"], data["groups"])
        self.assertEqual(len(folds), 5)
        for (train, val), (train_again, val_again) in zip(folds, second):
            np.testing.assert_array_equal(train, train_again)
            np.testing.assert_array_equal(val, val_again)
            self.assertFalse(set(data["groups"][train]) & set(data["groups"][val]))
            support = np.bincount(data["y"][val], minlength=cv.CLASS_COUNT)
            self.assertGreaterEqual(int(support.min()), cv.MIN_CV_CLASS_SUPPORT)
        covered = np.concatenate([val for _, val in folds])
        self.assertEqual(len(covered), len(set(covered.tolist())))
        self.assertEqual(len(covered), 346)

    def test_run_cv_assembles_complete_oof_predictions(self):
        config = model.build_config("test", label_smoothing=0.0, class_weight_power=0.0)
        base_model = model.RegularizedTemporalMLP(config)
        tensor = np.zeros((10, 89, 135), dtype=np.float32)
        tensor[:, :, 130] = 1.0
        y = np.arange(10, dtype=np.int64)
        data = {
            "tensor": tensor,
            "y": y,
            "target_names": [f"class_{index}" for index in range(21)],
            "base_model": base_model,
        }
        folds = [
            (np.arange(5, 10, dtype=np.int64), np.arange(5, dtype=np.int64)),
            (np.arange(5, dtype=np.int64), np.arange(5, 10, dtype=np.int64)),
        ]

        def fake_fit(_config, _prepared, _real, labels, train_index, val_index, fold_number):
            logits = np.full((len(val_index), cv.CLASS_COUNT), -4.0, dtype=np.float64)
            for row, sample_index in enumerate(val_index):
                logits[row, labels[sample_index]] = 4.0
            probabilities = cv.calibration.softmax(logits, 1.0)
            return object(), {"fold": fold_number, "inner_best_epoch": 1}, logits, probabilities

        with patch.object(cv, "_fit_fold", side_effect=fake_fit):
            result = cv.run_cv(config, data, folds)
        self.assertEqual(result["oof_logits"].shape, (10, 21))
        self.assertTrue(np.isfinite(result["oof_logits"]).all())
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertFalse(result["calibration"]["temperature"]["test_split_used"])

    def test_select_variant_prefers_macro_f1(self):
        results = {
            "a": {
                "metrics": {"macro_f1": 0.7, "accuracy": 0.7},
                "calibration": {"temperature": {"ece_after": 0.1}},
            },
            "b": {
                "metrics": {"macro_f1": 0.8, "accuracy": 0.79},
                "calibration": {"temperature": {"ece_after": 0.2}},
            },
        }
        self.assertEqual(cv.select_variant(results), "b")


if __name__ == "__main__":
    unittest.main()
