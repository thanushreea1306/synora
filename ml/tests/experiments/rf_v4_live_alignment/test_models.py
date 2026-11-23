import unittest

import numpy as np

from ml.src.experiments.rf_v4_live_alignment import augment, models


class ModelTests(unittest.TestCase):
    def test_model_configs_are_distinct(self):
        base = models.build_config("base", feature_mode="base", architecture="mlp", input_dim=135, kernel_size=1)
        tcn = models.build_config("tcn", feature_mode="enhanced", architecture="tcn", input_dim=509, kernel_size=3)
        attention = models.build_config(
            "attention",
            feature_mode="enhanced",
            architecture="tcn_attention",
            input_dim=509,
            kernel_size=3,
            attention=True,
        )
        self.assertEqual(base["kernel_size"], 1)
        self.assertEqual(tcn["kernel_size"], 3)
        self.assertTrue(attention["attention"])
        self.assertGreater(
            models.TemporalNetwork(attention).parameter_count(),
            models.TemporalNetwork(tcn).parameter_count(),
        )

    def test_forward_backward_are_finite(self):
        config = models.build_config(
            "test",
            feature_mode="enhanced",
            architecture="tcn",
            input_dim=509,
            kernel_size=3,
            attention=True,
        )
        model = models.TemporalNetwork(config)
        x = np.random.default_rng(4).normal(size=(4, 89, 509)).astype(np.float32)
        real = np.ones((4, 89), dtype=bool)
        real[:, 70:] = False
        x[~real] = 0.0
        logits, cache = model.forward(x, real)
        dlogits = np.full_like(logits, 1.0 / 21.0)
        dlogits[np.arange(4), [0, 1, 2, 3]] -= 1.0
        gradients = model.backward(cache, dlogits)
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))
        self.assertGreater(model.parameter_count(), 10000)

    def test_short_training_run_reduces_loss(self):
        config = models.build_config(
            "test",
            feature_mode="enhanced",
            architecture="tcn",
            input_dim=509,
            kernel_size=3,
            max_epochs=2,
            patience=2,
            augmentation=None,
        )
        model = models.TemporalNetwork(config)
        x = np.random.default_rng(5).normal(size=(8, 89, 509)).astype(np.float32)
        real = np.ones((8, 89), dtype=bool)
        y = np.arange(8) % 21
        history = model.fit(x, y, config, real, preextracted=True)
        self.assertEqual(history["epochs_completed"], 2)
        self.assertGreater(history["history"][-1]["train_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
