from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.src.experiments.rf_v3_native_frames import config, prepare_test


class PrepareTestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict, dict]:
        official_path = root / "include_test.txt"
        manifest_path = root / "video_manifest.csv"
        landmark_config_path = root / "landmark_config.json"
        experiment_config_path = root / "experiment_config.json"
        model_path = root / "model.task"
        model_path.write_bytes(b"model")
        label_mapping = []
        for index in range(21):
            label_mapping.append(
                {
                    "model_index": index,
                    "original_label_id": 100 + index,
                    "class_label": f"class_{index}",
                    "class_folder": f"folder_{index}",
                    "folder_id": index,
                }
            )
        official_lines = []
        manifest_lines = [
            "video_id,split,official_rel_path_posix,class_folder,class_label,label_id,source_video_abs,frame_count,fps,width,height,duration_s"
        ]
        for index in range(config.EXPECTED_TEST_COUNT):
            rel_path = f"test/video_{index:03d}.mp4"
            source_path = root / f"video_{index:03d}.mp4"
            source_path.write_bytes(b"video")
            official_lines.append(rel_path)
            label = label_mapping[index % 21]
            manifest_lines.append(
                ",".join(
                    [
                        f"video_{index:03d}",
                        "test",
                        rel_path,
                        label["class_folder"],
                        label["class_label"],
                        str(label["original_label_id"]),
                        str(source_path),
                        "50",
                        "30.0",
                        "1920",
                        "1080",
                        "1.6667",
                    ]
                )
            )
        official_path.write_text("\n".join(official_lines) + "\n", encoding="utf-8")
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        landmark_config = {
            "official_splits": {
                "test": {
                    "path": str(official_path),
                    "sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
                }
            },
            "extraction": {"delegate": "CPU"},
        }
        experiment_config = {"label_mapping": label_mapping}
        return manifest_path, landmark_config_path, landmark_config, experiment_config

    def test_load_test_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _, landmark_config, experiment_config = self._fixture(root)
            entries = prepare_test.load_test_entries(manifest_path, landmark_config, experiment_config)
            self.assertEqual(len(entries), config.EXPECTED_TEST_COUNT)
            self.assertTrue(all(entry["split"] == "test" for entry in entries))
            self.assertEqual(len({entry["video_id"] for entry in entries}), config.EXPECTED_TEST_COUNT)
            self.assertEqual(entries[0]["model_index"], 0)
            self.assertTrue(Path(entries[0]["source_video_abs"]).is_file())

    def test_build_extraction_payload_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, landmark_config_path, landmark_config, experiment_config = self._fixture(root)
            experiment_config_path = root / "experiment_config.json"
            experiment_config_path.write_text("{}", encoding="utf-8")
            landmark_config_path.write_text("{}", encoding="utf-8")
            entries = prepare_test.load_test_entries(manifest_path, landmark_config, experiment_config)
            model_path = root / "model.task"
            payload = prepare_test.build_extraction_payload(
                "native",
                manifest_path,
                landmark_config_path,
                experiment_config_path,
                landmark_config,
                experiment_config,
                entries,
                model_path,
                "code-hash",
            )
            self.assertEqual(payload["split_policy"]["allowed_splits"], ["test"])
            self.assertIs(payload["split_policy"]["test_split_read"], True)
            self.assertIs(payload["split_policy"]["training_split_access"], False)
            self.assertEqual(payload["split_policy"]["selected_test_count"], config.EXPECTED_TEST_COUNT)
            self.assertIn("no frame duplication", payload["preprocessing"]["temporal"])

    def test_official_test_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _, landmark_config, experiment_config = self._fixture(root)
            landmark_config["official_splits"]["test"]["sha256"] = "bad-hash"
            with self.assertRaises(RuntimeError):
                prepare_test.load_test_entries(manifest_path, landmark_config, experiment_config)


if __name__ == "__main__":
    unittest.main()
