import unittest

import numpy as np

from ml.src.experiments.rf_v4_live_alignment import augment, features


class AugmentationTests(unittest.TestCase):
    def test_default_config_has_no_mirroring(self):
        config = augment.default_augmentation_config()
        self.assertIs(config["mirror"], False)
        self.assertEqual(config["scale_range"], [0.98, 1.02])
        self.assertEqual(config["translation_range"], [-0.01, 0.01])

    def test_none_config_returns_inputs_unchanged(self):
        x = np.ones((2, 89, 509), dtype=np.float32)
        real = np.ones((2, 89), dtype=bool)
        actual_x, actual_real = augment.augment_batch(x, real, None, np.random.default_rng(0), feature_mode="enhanced")
        self.assertIs(actual_x, x)
        self.assertIs(actual_real, real)

    def test_enhanced_augmentation_stays_in_bounds(self):
        rng = np.random.default_rng(7)
        real_index = features.ENHANCED_CHANNEL_NAMES.index("frame_real")
        validity_index = features.ENHANCED_CHANNEL_NAMES.index("left_lm00_valid")
        x = rng.uniform(-0.5, 0.5, size=(4, 89, 509)).astype(np.float32)
        x[:, :, real_index] = 1.0
        x[:, 70:, real_index] = 0.0
        x[:, :, validity_index] = 1.0
        real = x[:, :, real_index] > 0.5
        x[~real] = 0.0
        augmented, augmented_real = augment.augment_batch(
            x,
            real,
            augment.default_augmentation_config(),
            rng,
            feature_mode="enhanced",
        )
        self.assertEqual(augmented.shape, x.shape)
        self.assertTrue(np.isfinite(augmented).all())
        self.assertLessEqual(augmented.max(), 1.0)
        self.assertGreaterEqual(augmented.min(), -1.0)
        self.assertTrue(np.all(augmented[~augmented_real] == 0.0))


if __name__ == "__main__":
    unittest.main()
