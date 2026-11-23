import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from ml.src.diagnostics import live_landmark_diagnostic as diagnostic


@dataclass
class FakeCategory:
    display_name: str
    category_name: str
    score: float
    index: int = 0


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: Any = None
    presence: Any = None
    name: Any = None


class FakeResult:
    def __init__(self, hands=()):
        self.hand_landmarks = []
        self.handedness = []
        for hand_index, (label, score, shift) in enumerate(hands):
            self.hand_landmarks.append(
                [
                    FakeLandmark(
                        x=0.1 + shift + landmark_index * 0.001,
                        y=0.2 + landmark_index * 0.002,
                    )
                    for landmark_index in range(21)
                ]
            )
            self.handedness.append(
                [
                    FakeCategory(
                        display_name=label,
                        category_name=label,
                        score=score,
                        index=hand_index,
                    )
                ]
            )


class FakeCamera:
    def __init__(self, results, opened=True, fps=30.0, width=1280, height=720):
        self.results = list(results)
        self.opened = opened
        self.index = 0
        self.released = False
        self.properties = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.results):
            return False, None
        self.index += 1
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return True, frame

    def get(self, property_id):
        return self.properties.get(property_id, 0.0)

    def set(self, property_id, value):
        self.properties[property_id] = float(value)
        return True

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self, results):
        self.results = list(results)
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        if not self.results:
            raise RuntimeError("no mocked detection result left")
        return self.results.pop(0)

    def close(self):
        self.closed = True


class LiveLandmarkDiagnosticTests(unittest.TestCase):
    def test_mean_movement_and_stability_candidates(self):
        previous = ((0.1, 0.2), (0.2, 0.3))
        current = ((0.11, 0.2), (0.21, 0.3))
        movement = diagnostic._mean_movement(previous, current)
        self.assertAlmostEqual(movement, 0.01, places=12)

        stats = diagnostic.StabilityStats()
        stats.add(movement, 0.08)
        self.assertEqual(stats.samples, 1)
        self.assertEqual(stats.large_movement_candidates, 0)
        self.assertAlmostEqual(stats.mean, 0.01, places=12)

        stats.add(0.2, 0.08)
        self.assertEqual(stats.samples, 2)
        self.assertEqual(stats.large_movement_candidates, 1)
        self.assertAlmostEqual(stats.maximum, 0.2, places=12)
        self.assertAlmostEqual(stats.last_movement, 0.2, places=12)

    def test_frame_hand_infos_extracts_only_labels_scores_and_points(self):
        result = FakeResult([("Left", 0.81, 0.0), ("Right", 0.92, 0.5)])
        infos = diagnostic._frame_hand_infos(result)
        self.assertEqual([label for label, _, _ in infos], ["Left", "Right"])
        self.assertEqual([score for _, score, _ in infos], [0.81, 0.92])
        self.assertEqual([len(points) for _, _, points in infos], [21, 21])
        self.assertEqual(diagnostic._frame_hand_infos(FakeResult()), [])

    def test_quit_key_recognizes_q_and_ignores_other_keys(self):
        self.assertTrue(diagnostic._is_quit_key(ord("q")))
        self.assertTrue(diagnostic._is_quit_key(ord("Q")))
        self.assertFalse(diagnostic._is_quit_key(-1))
        self.assertFalse(diagnostic._is_quit_key(255))
        self.assertFalse(diagnostic._is_quit_key(ord("s")))
        self.assertFalse(diagnostic._is_quit_key(None))

    def test_draw_hands_draws_points_and_connections(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = FakeResult([("Left", 0.90, 0.0)])
        with (
            patch.object(diagnostic.cv2, "line") as line,
            patch.object(diagnostic.cv2, "circle") as circle,
        ):
            drawn = diagnostic._draw_hands(frame, result)

        self.assertIsNot(drawn, frame)
        self.assertEqual(drawn.shape, frame.shape)
        self.assertGreater(line.call_count, 0)
        self.assertEqual(circle.call_count, 21)

    def test_run_diagnostic_counts_detections_and_releases_resources(self):
        results = [
            FakeResult(),
            FakeResult([("Left", 0.80, 0.0), ("Right", 0.90, 0.0)]),
            FakeResult([("Left", 0.82, 0.01)]),
            FakeResult([("Left", 0.70, 0.30), ("Right", 0.85, 0.0)]),
        ]
        camera = FakeCamera(results)
        landmarker = FakeLandmarker(results)
        printed = []

        with (
            patch.object(diagnostic, "_draw_hands", side_effect=lambda frame, result: frame) as draw_hands,
            patch.object(diagnostic.cv2, "imshow") as imshow,
            patch.object(diagnostic.cv2, "waitKey", return_value=-1) as wait_key,
            patch.object(diagnostic.cv2, "destroyWindow") as destroy_window,
        ):
            summary = diagnostic.run_diagnostic(
                max_frames=4,
                seconds=None,
                jump_threshold=0.08,
                camera_factory=lambda index: camera,
                landmarker_factory=lambda model_path, payload: landmarker,
                printer=printed.append,
            )

        self.assertEqual(draw_hands.call_count, 3)
        self.assertEqual(imshow.call_count, 4)
        self.assertEqual(wait_key.call_count, 4)
        destroy_window.assert_called_once_with(diagnostic.PREVIEW_WINDOW_NAME)
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertEqual(summary.frames_observed, 4)
        self.assertEqual(summary.width, 1280)
        self.assertEqual(summary.height, 720)
        self.assertEqual(summary.reported_fps, 30.0)
        self.assertEqual(summary.no_hand_frames, 1)
        self.assertEqual(summary.any_hand_frames, 3)
        self.assertEqual(summary.both_hands_frames, 2)
        self.assertEqual(summary.left_frames, 3)
        self.assertEqual(summary.right_frames, 2)
        self.assertEqual(summary.left_confidence.count, 3)
        self.assertEqual(summary.right_confidence.count, 2)
        self.assertEqual(summary.left_stability.samples, 2)
        self.assertEqual(summary.right_stability.samples, 0)
        self.assertEqual(summary.left_stability.large_movement_candidates, 1)
        self.assertAlmostEqual(summary.left_stability.mean, 0.15, places=12)
        self.assertTrue(any("Final diagnostic summary" in line for line in printed))
        self.assertFalse(any("0.100000" in line for line in printed))

    def test_run_diagnostic_draws_landmarks_only_when_hands_detected(self):
        results = [FakeResult(), FakeResult([("Right", 0.91, 0.0)])]
        camera = FakeCamera(results)
        landmarker = FakeLandmarker(results)

        with (
            patch.object(diagnostic, "_draw_hands", side_effect=lambda frame, result: frame) as draw_hands,
            patch.object(diagnostic.cv2, "imshow"),
            patch.object(diagnostic.cv2, "waitKey", return_value=-1),
            patch.object(diagnostic.cv2, "destroyWindow"),
        ):
            summary = diagnostic.run_diagnostic(
                max_frames=2,
                seconds=None,
                camera_factory=lambda index: camera,
                landmarker_factory=lambda model_path, payload: landmarker,
                printer=lambda line: None,
            )

        self.assertEqual(summary.frames_observed, 2)
        self.assertEqual(summary.no_hand_frames, 1)
        self.assertEqual(summary.any_hand_frames, 1)
        self.assertEqual(draw_hands.call_count, 1)

    def test_preview_quit_key_stops_early_and_destroys_window(self):
        results = [FakeResult() for _ in range(4)]
        camera = FakeCamera(results)
        landmarker = FakeLandmarker(results)

        with (
            patch.object(diagnostic.cv2, "imshow") as imshow,
            patch.object(diagnostic.cv2, "waitKey", side_effect=[-1, ord("q")]) as wait_key,
            patch.object(diagnostic.cv2, "destroyWindow") as destroy_window,
        ):
            summary = diagnostic.run_diagnostic(
                max_frames=10,
                seconds=None,
                camera_factory=lambda index: camera,
                landmarker_factory=lambda model_path, payload: landmarker,
                printer=lambda line: None,
            )

        self.assertEqual(summary.frames_observed, 2)
        self.assertTrue(summary.interrupted)
        self.assertEqual(imshow.call_count, 2)
        self.assertEqual(wait_key.call_count, 2)
        destroy_window.assert_called_once_with(diagnostic.PREVIEW_WINDOW_NAME)
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)

    def test_module_has_no_rf_or_persistence_hooks(self):
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for forbidden in ("rf_service", "rf_feature_builder", "inference.cli"):
            self.assertFalse(any(forbidden in name for name in imported))
        for forbidden in ("VideoWriter", "imwrite", "write_text", "write_bytes", "to_csv", "open("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
