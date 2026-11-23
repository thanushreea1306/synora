import unittest

import numpy as np

from ml.src.experiments.rf_v3_generalisation import model


class RegularizedTemporalMLPTests(unittest.TestCase):
    def test_class_weight_vector_power_zero_is_unity(self):
        y = np.asarray([0] * 20 + [1] * 5, dtype=np.int64)
        weights = model.class_weight_vector(y, 0.0)
        self.assertEqual(weights.shape, (model.CLASS_COUNT,))
        np.testing.assert_array_equal(weights, np.ones(model.CLASS_COUNT, dtype=np.float32))

    def test_class_weight_vector_inverse_frequency_is_normalized(self):
        y = np.asarray([0] * 20 + [1] * 5, dtype=np.int64)
        weights = model.class_weight_vector(y, 1.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_build_config_contains_regularization_policy(self):
        config = model.build_config("test", label_smoothing=0.1, class_weight_power=0.5)
        self.assertEqual(config["variant"], "test")
        self.assertEqual(config["regularization"]["label_smoothing"], 0.1)
        self.assertEqual(config["regularization"]["class_weight_power"], 0.5)
        self.assertEqual(config["selection_split"], "train_val_group_cv")

    def test_fit_regularized_is_deterministic_and_finite(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(model.CLASS_COUNT, 89, 135)).astype(np.float32)
        real = np.ones((model.CLASS_COUNT, 89), dtype=np.float32)
        y = np.arange(model.CLASS_COUNT, dtype=np.int64)
        config = model.build_config("test", label_smoothing=0.1, class_weight_power=0.5)
        config["max_epochs"] = 1
        config["batch_size"] = 32
        first = model.RegularizedTemporalMLP(config)
        second = model.RegularizedTemporalMLP(config)
        first.fit_regularized(features, real, y, config)
        second.fit_regularized(features, real, y, config)
        for key in first.weights:
            self.assertTrue(np.isfinite(first.weights[key]).all())
            np.testing.assert_allclose(first.weights[key], second.weights[key])

    def test_fit_regularized_reports_fixed_epoch_history(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(model.CLASS_COUNT, 89, 135)).astype(np.float32)
        real = np.ones((model.CLASS_COUNT, 89), dtype=np.float32)
        y = np.arange(model.CLASS_COUNT, dtype=np.int64)
        config = model.build_config("test", label_smoothing=0.0, class_weight_power=0.0)
        config["max_epochs"] = 2
        network = model.RegularizedTemporalMLP(config)
        history = network.fit_regularized(features, real, y, config)
        self.assertEqual(history["epochs_completed"], 2)
        self.assertEqual(history["best_epoch"], 2)
        self.assertIsNone(history["best_validation_macro_f1"])


if __name__ == "__main__":
    unittest.main()
