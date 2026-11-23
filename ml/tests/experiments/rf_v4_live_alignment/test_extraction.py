import unittest
from pathlib import Path

from ml.src.experiments.rf_v2_720p30 import dataset as v2_dataset
from ml.src.experiments.rf_v4_live_alignment import preprocessing


REPO_ROOT = Path(__file__).resolve().parents[4]
EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v4_live_alignment"
V2_EXTRACTION_ROOT = REPO_ROOT / "ml/dataset/processed/experiments/rf_v2_720p30"


class ExtractionTests(unittest.TestCase):
    def test_config_certifies_no_test_split_and_no_transforms(self):
        config = v2_dataset.read_json(EXTRACTION_ROOT / "config.json")
        self.assertIs(config["split_policy"]["test_split_read"], False)
        settings = config["preprocessing"]
        self.assertIn("INTER_AREA", settings["resize_interpolation"])
        self.assertIn("no frame duplication", settings["temporal"])
        self.assertIs(settings["no_mirroring"], True)
        self.assertIn("reset", settings["tracking_lifecycle"])

    def test_summary_excludes_short_sources_and_has_no_failures(self):
        summary = v2_dataset.read_json(EXTRACTION_ROOT / "extraction_summary.json")
        self.assertEqual(summary["complete_counts"], {"train": 305, "val": 41})
        self.assertEqual(summary["exclusion_reasons"], {"BELOW_LIVE_MIN_FRAMES": 13})
        self.assertEqual(summary["failed_video_ids"], [])

    def test_manifest_has_no_duplication_and_matches_source_frame_counts(self):
        rows = v2_dataset.read_csv(EXTRACTION_ROOT / "manifest.csv")
        complete = [row for row in rows if row["status"] == "complete"]
        self.assertEqual(len(complete), 346)
        for row in complete:
            self.assertEqual(int(row["duplicate_source_frames"]), 0)
            self.assertEqual(int(row["target_frame_count"]), int(row["source_frame_count"]))
            self.assertGreaterEqual(int(row["source_frame_count"]), preprocessing.MIN_FRAMES)
            self.assertLessEqual(int(row["source_frame_count"]), preprocessing.MAX_FRAMES)
            self.assertEqual(float(row["source_fps"]), 25.0)
            self.assertEqual(row["resize_interpolation"], "cv2.INTER_AREA")
            self.assertEqual(row["tracking_lifecycle"], "reset_at_recording_start")

    def test_v2_resampling_duplicated_frames_while_v4_did_not(self):
        v2_rows = v2_dataset.read_csv(V2_EXTRACTION_ROOT / "manifest.csv")
        v2_duplicates = sum(int(row["repeated_source_frames"]) for row in v2_rows if row["status"] == "complete")
        v4_rows = v2_dataset.read_csv(EXTRACTION_ROOT / "manifest.csv")
        v4_duplicates = sum(int(row["duplicate_source_frames"]) for row in v4_rows if row["status"] == "complete")
        self.assertGreater(v2_duplicates, 0)
        self.assertEqual(v4_duplicates, 0)


if __name__ == "__main__":
    unittest.main()
