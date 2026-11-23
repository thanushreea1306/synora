import copy
import csv
import json
import unittest
from pathlib import Path

import numpy as np

from ml.src import prepare_video_features as feature_source
from ml.src.inference.rf_feature_builder import (
    EXPECTED_FEATURE_COUNT,
    FeatureBuildError,
    build_rf_feature_row,
    load_authoritative_feature_names,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURE_ROOT = REPO_ROOT / "ml/dataset/processed/synora_include21_video_features_v1"
SCHEMA_PATH = FEATURE_ROOT / "rf_feature_schema.json"
FEATURE_CSV_PATH = FEATURE_ROOT / "rf_features.csv"
FEATURE_MATRIX_PATH = FEATURE_ROOT / "rf_features.npy"
MANIFEST_PATH = FEATURE_ROOT / "video_manifest.csv"
PARITY_TOLERANCE = 1e-6
METADATA_FIELDS = ("duration_s", "fps", "width", "height")


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def frame_hand_counts(records):
    counts = {"no_hand": 0, "one_hand": 0, "two_hand": 0, "duplicate": 0}
    for record in records:
        _, presence, _, no_hand, duplicate = feature_source.canonical_frame(record)
        selected = int(presence.sum())
        if selected == 0:
            counts["no_hand"] += 1
        elif selected == 1:
            counts["one_hand"] += 1
        else:
            counts["two_hand"] += 1
        counts["duplicate"] += int(duplicate)
    return counts


class RFBuilderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.feature_names = tuple(load_authoritative_feature_names(SCHEMA_PATH))
        cls.feature_index = {name: index for index, name in enumerate(cls.feature_names)}
        cls.reference_rows = load_csv_rows(FEATURE_CSV_PATH)
        cls.manifest_rows = {row["video_id"]: row for row in load_csv_rows(MANIFEST_PATH)}
        cls.matrix = np.load(FEATURE_MATRIX_PATH)
        cls.results = []
        cls.coverage = {}

        if len(cls.reference_rows) != len(cls.matrix):
            raise AssertionError(
                f"reference row count {len(cls.reference_rows)} does not match matrix rows {len(cls.matrix)}"
            )
        if cls.matrix.shape != (len(cls.reference_rows), EXPECTED_FEATURE_COUNT):
            raise AssertionError(f"unexpected matrix shape: {cls.matrix.shape}")

        for matrix_index, reference_row in enumerate(cls.reference_rows):
            manifest_row = cls.manifest_rows[reference_row["video_id"]]
            landmark_path = Path(manifest_row["landmark_file"])
            records = feature_source.read_jsonl(landmark_path)
            metadata = {field: float(manifest_row[field]) for field in METADATA_FIELDS}
            built = build_rf_feature_row(
                records,
                metadata=metadata,
                expected_feature_names=cls.feature_names,
            )
            expected = cls.matrix[matrix_index].reshape(1, -1)
            max_abs_diff = float(np.max(np.abs(built.values - expected)))
            counts = frame_hand_counts(records)
            entry = {
                "video_id": reference_row["video_id"],
                "records": records,
                "metadata": metadata,
                "expected": expected,
                "built": built,
                "max_abs_diff": max_abs_diff,
                "counts": counts,
            }
            cls.results.append(entry)
            for key in ("no_hand", "one_hand", "two_hand", "duplicate"):
                if counts[key] > 0 and key not in cls.coverage:
                    cls.coverage[key] = entry

        cls.max_abs_diff = max(entry["max_abs_diff"] for entry in cls.results)

    def test_schema_feature_names_are_exact(self):
        self.assertEqual(len(self.feature_names), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(set(self.feature_names)), EXPECTED_FEATURE_COUNT)
        self.assertEqual(self.feature_names, tuple(feature_source.base_feature_names()))
        self.assertEqual(
            self.feature_names,
            tuple(self.schema["feature_columns"]),
        )
        header = list(csv.reader([FEATURE_CSV_PATH.read_text(encoding="utf-8").splitlines()[0]]))[0]
        self.assertEqual(tuple(header[-EXPECTED_FEATURE_COUNT:]), self.feature_names)

    def test_all_443_reference_videos_match_offline_features(self):
        self.assertEqual(len(self.results), 443)
        for entry in self.results:
            self.assertEqual(entry["built"].values.shape, (1, EXPECTED_FEATURE_COUNT))
            self.assertEqual(entry["built"].feature_names, self.feature_names)
            self.assertTrue(bool(np.isfinite(entry["built"].values).all()))
            np.testing.assert_allclose(
                entry["built"].values,
                entry["expected"],
                rtol=0.0,
                atol=PARITY_TOLERANCE,
                err_msg=f"video {entry['video_id']}",
            )
        print(
            "PARITY_SUMMARY "
            f"records={len(self.results)} "
            f"max_abs_diff={self.max_abs_diff:.17g} "
            f"tolerance={PARITY_TOLERANCE:.17g}"
        )

    def test_no_hand_frames_are_reproduced(self):
        entry = self.coverage["no_hand"]
        self.assertGreater(entry["counts"]["no_hand"], 0)
        self.assertEqual(
            entry["built"].diagnostics["no_hand_frame_count"],
            entry["counts"]["no_hand"],
        )
        expected_value = int(entry["expected"][0, self.feature_index["no_hand_frame_count"]])
        self.assertEqual(entry["built"].diagnostics["no_hand_frame_count"], expected_value)

    def test_one_hand_frames_are_reproduced(self):
        entry = self.coverage["one_hand"]
        self.assertGreater(entry["counts"]["one_hand"], 0)
        np.testing.assert_allclose(
            entry["built"].values,
            entry["expected"],
            rtol=0.0,
            atol=PARITY_TOLERANCE,
        )

    def test_two_hand_frames_are_reproduced(self):
        entry = self.coverage["two_hand"]
        self.assertGreater(entry["counts"]["two_hand"], 0)
        self.assertEqual(
            entry["built"].diagnostics["both_hands_frame_count"],
            entry["counts"]["two_hand"],
        )

    def test_duplicate_handedness_is_reproduced(self):
        entry = self.coverage["duplicate"]
        self.assertGreater(entry["counts"]["duplicate"], 0)
        expected_value = int(
            entry["expected"][0, self.feature_index["duplicate_handedness_frames"]]
        )
        self.assertEqual(
            entry["built"].diagnostics["duplicate_handedness_frames"],
            entry["counts"]["duplicate"],
        )
        self.assertEqual(
            entry["built"].diagnostics["duplicate_handedness_frames"],
            expected_value,
        )

    def test_metadata_can_be_derived_from_records(self):
        entry = self.results[0]
        derived = build_rf_feature_row(
            entry["records"],
            expected_feature_names=self.feature_names,
        )
        np.testing.assert_allclose(
            derived.values,
            entry["expected"],
            rtol=0.0,
            atol=PARITY_TOLERANCE,
        )

    def test_invalid_inputs_raise_clear_errors(self):
        base = self.results[0]

        with self.assertRaisesRegex(FeatureBuildError, "at least one frame"):
            build_rf_feature_row([], expected_feature_names=self.feature_names)

        overlong = [copy.deepcopy(base["records"][0]) for _ in range(feature_source.T_VALUE + 1)]
        for index, record in enumerate(overlong):
            record["frame_index"] = index
        with self.assertRaisesRegex(FeatureBuildError, "maximum supported"):
            build_rf_feature_row(overlong, expected_feature_names=self.feature_names)

        gapped = copy.deepcopy(base["records"])
        if len(gapped) < 2:
            self.skipTest("reference video has too few frames to create a gap")
        gapped.pop(1)
        with self.assertRaisesRegex(FeatureBuildError, "non-sequential frame_index"):
            build_rf_feature_row(gapped, expected_feature_names=self.feature_names)

        failed_inference = copy.deepcopy(base["records"])
        failed_inference[0]["infer_ok"] = False
        with self.assertRaisesRegex(FeatureBuildError, "infer_ok"):
            build_rf_feature_row(failed_inference, expected_feature_names=self.feature_names)

        malformed_hands = copy.deepcopy(base["records"])
        selected_hands = None
        for record in malformed_hands:
            if record.get("hands"):
                selected_hands = record
                break
        if selected_hands is None:
            self.skipTest("reference video has no hand records to malform")
        selected_hands["hands"] = [{"handedness_label": "Left"}]
        with self.assertRaisesRegex(FeatureBuildError, "invalid frame"):
            build_rf_feature_row(malformed_hands, expected_feature_names=self.feature_names)

        invalid_metadata = dict(base["metadata"])
        invalid_metadata["duration_s"] = "not-a-number"
        with self.assertRaisesRegex(FeatureBuildError, "duration_s"):
            build_rf_feature_row(
                base["records"],
                metadata=invalid_metadata,
                expected_feature_names=self.feature_names,
            )

        duplicate_names = list(self.feature_names)
        duplicate_names[-1] = duplicate_names[0]
        with self.assertRaisesRegex(FeatureBuildError, "duplicates"):
            build_rf_feature_row(
                base["records"],
                expected_feature_names=duplicate_names,
            )


if __name__ == "__main__":
    unittest.main()
