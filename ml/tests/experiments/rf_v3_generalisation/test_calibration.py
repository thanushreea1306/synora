import unittest

import numpy as np

from ml.src.experiments.rf_v3_generalisation import calibration


class CalibrationTests(unittest.TestCase):
    def test_softmax_sums_to_one_and_temperature_flattens(self):
        logits = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float64)
        sharp = calibration.softmax(logits, 1.0)
        soft = calibration.softmax(logits, 2.0)
        self.assertAlmostEqual(float(sharp.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=9)
        self.assertGreater(sharp[0, 0], soft[0, 0])
        self.assertLess(sharp[0, 2], soft[0, 2])

    def test_softmax_rejects_invalid_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            calibration.softmax(np.zeros((1, 3)), 0.0)

    def test_expected_calibration_error_is_zero_for_perfect_probabilities(self):
        proba = np.eye(4, 21, dtype=np.float64)[:4]
        self.assertAlmostEqual(calibration.expected_calibration_error([0, 1, 2, 3], proba), 0.0, places=9)

    def test_temperature_scale_reduces_nll_on_overconfident_errors(self):
        logits = np.full((6, 21), -4.0, dtype=np.float64)
        for index in range(6):
            logits[index, 7] = 4.0
            logits[index, index] = -1.0
        report = calibration.temperature_scale(list(range(6)), logits, max_iterations=400)
        self.assertGreater(report["temperature"], 1.0)
        self.assertLess(report["nll_after"], report["nll_before"])
        self.assertLessEqual(report["ece_after"], report["ece_before"])

    def test_threshold_sweep_keeps_minimum_coverage(self):
        proba = np.zeros((4, 21), dtype=np.float64)
        proba[0, 0] = 0.40
        proba[0, 2] = 0.30
        proba[0, 3:] = 0.30 / 18
        proba[1, 0] = 0.90
        proba[1, 1:] = 0.10 / 20
        proba[2, 2] = 0.30
        proba[2, 1] = 0.25
        proba[2, 0] = 0.45 / 19
        proba[3, 1] = 0.90
        proba[3, :1] = 0.05
        proba[3, 2:] = 0.05 / 19
        result = calibration.threshold_sweep([0, 0, 1, 1], proba, min_coverage=0.70)
        self.assertGreaterEqual(result["chosen"]["coverage"], 0.70)
        self.assertGreater(result["chosen"]["accepted_accuracy"], result["baseline_accuracy"])
        self.assertFalse(result["test_split_used"])


if __name__ == "__main__":
    unittest.main()
