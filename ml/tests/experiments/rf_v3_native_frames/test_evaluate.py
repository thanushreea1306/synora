from __future__ import annotations

import unittest

import numpy as np

from ml.src.experiments.rf_v3_native_frames import evaluate


class EvaluateTests(unittest.TestCase):
    def test_evaluate_model_reports_rejection(self) -> None:
        logits = np.full((3, 21), -10.0, dtype=np.float64)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        result = evaluate.evaluate_model(np.asarray([0, 1, 2]), logits, 1.0, 0.9)
        self.assertEqual(result["rejection"]["accepted_count"], 3)
        self.assertEqual(result["rejection"]["rejected_count"], 0)
        self.assertEqual(result["rejection"]["coverage"], 1.0)
        self.assertEqual(result["rejection"]["accepted_accuracy"], 1.0)
        self.assertTrue(np.array_equal(result["raw_predictions"], np.asarray([0, 1, 2])))

    def test_sample_statistics_reads_native_channels(self) -> None:
        tensor = np.zeros((1, 135), dtype=np.float32)
        tensor[0, 126] = 1.0
        tensor[0, 127] = 1.0
        tensor[0, 130] = 1.0
        tensor[0, 131] = 0.0
        tensor[0, 133] = 0.9
        tensor[0, 134] = 0.8
        row = {
            "video_id": "sample",
            "frame_count": "43",
            "source_frame_count": "43",
        }
        stats = evaluate.sample_statistics(tensor, row)
        self.assertEqual(stats["no_hand_frames"], 0)
        self.assertEqual(stats["both_hands_ratio"], 1.0)
        self.assertAlmostEqual(stats["mean_confidence"], 0.9)
        self.assertEqual(stats["frame_count"], 43)

    def test_grouped_indices_boundaries(self) -> None:
        stats = [
            {
                "video_id": "short",
                "frame_count": 43,
                "source_frame_count": 43,
                "no_hand_frames": 1,
                "both_hands_ratio": 0.9,
                "mean_confidence": 0.5,
                "padded_ratio": 1.0 - 43 / 89,
            },
            {
                "video_id": "long",
                "frame_count": 80,
                "source_frame_count": 80,
                "no_hand_frames": 0,
                "both_hands_ratio": 0.1,
                "mean_confidence": 0.95,
                "padded_ratio": 1.0 - 80 / 89,
            },
        ]
        groups = evaluate.grouped_indices(stats)
        self.assertEqual(groups["missing_hands"]["any_no_hand"], [0])
        self.assertEqual(groups["two_hand_dominant"]["both_hands_majority"], [0])
        self.assertEqual(groups["low_confidence"]["mean_confidence_at_least_0_8"], [1])
        self.assertEqual(groups["duration"]["short_43_49"], [0])
        self.assertEqual(groups["boundary"]["near_cap_80_plus"], [1])
        self.assertEqual(groups["padded_ratio"]["at_least_70_percent_real"], [1])

    def test_group_metrics_empty_bucket(self) -> None:
        result = evaluate.group_metrics([], np.asarray([0]), np.asarray([0]), np.asarray([0]))
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["native"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
