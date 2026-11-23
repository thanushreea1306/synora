from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.src.experiments.rf_v2_720p30 import dataset
from ml.src.experiments.rf_v3_native_frames import config, train


class TrainTests(unittest.TestCase):
    def test_build_config_matches_selected_architecture(self) -> None:
        result = train.build_config()
        self.assertEqual(result["variant"], config.VARIANT)
        self.assertEqual(result["architecture"], "framewise_shared_mlp_masked_mean_max_pooling_regularized")
        self.assertEqual(result["frame_layers"], [135, 96, 48])
        self.assertEqual(result["head_layers"], [96, 64, 21])
        self.assertEqual(result["seed"], config.SEED)
        self.assertIs(result["preprocessing"]["refit"], False)

    def test_require_equivalence_blocks_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset.write_json(
                root / "equivalence.json",
                {
                    "passed": True,
                    "existing_v4_trainval": {"all_exact": False},
                },
            )
            with self.assertRaises(RuntimeError):
                train._require_equivalence(root)

    def test_require_equivalence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset.write_json(
                root / "equivalence.json",
                {
                    "passed": True,
                    "existing_v4_trainval": {"all_exact": True},
                },
            )
            result = train._require_equivalence(root)
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
