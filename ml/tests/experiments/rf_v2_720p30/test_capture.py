import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.src.experiments.rf_v2_720p30 import capture as capture_module
from ml.src.experiments.rf_v2_720p30.capture import (
    MAX_FRAMES,
    MIN_FRAMES,
    REASON_CAMERA_CONFIGURATION_INVALID,
    REASON_CAMERA_UNAVAILABLE,
    REASON_FRAME_DIMENSION_INVALID,
    REASON_USER_QUIT,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_QUIT,
    CameraConfiguration,
    CameraConfigurationError,
    CaptureConfig,
    capture_segment,
    configure_camera,
)


REQUIRED_RECORD_KEYS = {
    "schema_version",
    "frame_index",
    "timestamp_ms",
    "source_timestamp_ms",
    "fps",
    "width",
    "height",
    "num_hands",
    "hands",
    "handedness_world_landmark_mismatch",
    "decode_ok",
    "infer_ok",
    "error",
}


def key(character):
    return ord(character)


@dataclass(frozen=True)
class FakeCategory:
    display_name: str
    category_name: str
    score: float
    index: int


@dataclass(frozen=True)
class FakePoint:
    x: float
    y: float
    z: float
    visibility: Any = None
    presence: Any = None
    name: Any = None


class FakeDetectionResult:
    def __init__(self, hands=(("Left", 0.91),)):
        self.hand_landmarks = []
        self.hand_world_landmarks = []
        self.handedness = []
        for hand_index, (label, score) in enumerate(hands):
            landmarks = [
                FakePoint(
                    x=0.1 + hand_index * 0.2 + landmark_index * 0.001,
                    y=0.2 + landmark_index * 0.002,
                    z=0.003 + landmark_index * 0.0001,
                )
                for landmark_index in range(21)
            ]
            world = [
                FakePoint(
                    x=point.x + 0.01,
                    y=point.y + 0.01,
                    z=point.z + 0.01,
                )
                for point in landmarks
            ]
            category = FakeCategory(
                display_name=label,
                category_name=label,
                score=score,
                index=hand_index,
            )
            self.hand_landmarks.append(landmarks)
            self.hand_world_landmarks.append(world)
            self.handedness.append([category])


class FakeCamera:
    def __init__(
        self,
        frame_shape=(720, 1280, 3),
        initial_properties=None,
        ignore_set=False,
        set_result=False,
        frame_count=8,
        opened=True,
    ):
        self.frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(frame_count)]
        self.index = 0
        self.opened = opened
        self.properties = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }
        if initial_properties:
            self.properties.update(initial_properties)
        self.ignore_set = ignore_set
        self.set_result = set_result
        self.set_calls = []
        self.get_calls = []
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, property_id):
        self.get_calls.append(property_id)
        return self.properties.get(property_id, 0.0)

    def set(self, property_id, value):
        self.set_calls.append((property_id, float(value)))
        if not self.ignore_set:
            self.properties[property_id] = float(value)
        return self.set_result

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        if not self.results:
            raise RuntimeError("no mocked detection result left")
        return self.results.pop(0)

    def close(self):
        self.closed = True


class FakeViewer:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.shown = 0
        self.statuses = []
        self.closed = False

    def show(self, frame, result, status=None):
        self.shown += 1
        self.statuses.append(status)

    def poll_key(self):
        if not self.keys:
            return None
        return self.keys.pop(0)

    def close(self):
        self.closed = True


class V2CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)
        self.model_path = self.temp_root / "fake_hand_landmarker.task"
        self.model_path.write_bytes(b"fake")
        self.settings_path = self.temp_root / "extraction_config.json"
        payload = {
            "extraction": {
                capture_key: value
                for capture_key, value in zip(
                    capture_module.capture_source.REQUIRED_EXTRACTION_KEYS,
                    (
                        "VIDEO",
                        "CPU",
                        2,
                        0.5,
                        0.5,
                        0.5,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                )
            },
            "model": {"path": str(self.model_path)},
        }
        self.settings_path.write_text(
            __import__("json").dumps(payload),
            encoding="utf-8",
        )
        self.addCleanup(self.temp_dir.cleanup)

    def make_config(self):
        return CaptureConfig(
            extraction_config_path=self.settings_path,
            model_path=self.model_path,
            show_preview=False,
        )

    def run_capture(self, keys, camera, landmarker, viewer):
        return capture_segment(
            config=self.make_config(),
            camera_factory=lambda index: camera,
            landmarker_factory=lambda model_path, payload: landmarker,
            viewer_factory=lambda: viewer,
        )

    def test_configure_camera_uses_readback_and_ignores_set_return_value(self):
        camera = FakeCamera(set_result=False)
        configuration = configure_camera(camera)

        self.assertEqual(configuration, CameraConfiguration(fps=30.0, width=1280, height=720))
        self.assertEqual(
            camera.set_calls,
            [
                (cv2.CAP_PROP_FRAME_WIDTH, 1280.0),
                (cv2.CAP_PROP_FRAME_HEIGHT, 720.0),
                (cv2.CAP_PROP_FPS, 30.0),
            ],
        )
        for property_id in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self.assertIn(property_id, camera.get_calls)

    def test_configure_camera_rejects_incorrect_readback(self):
        camera = FakeCamera(
            initial_properties={cv2.CAP_PROP_FPS: 25.0},
            ignore_set=True,
        )

        with self.assertRaises(CameraConfigurationError) as raised:
            configure_camera(camera)

        message = str(raised.exception)
        self.assertIn("camera reported 1280x720 at 25 FPS", message)
        self.assertIn("required 1280x720 at 30 FPS", message)
        self.assertIn("no fallback attempted", message)

    def test_capture_rejects_unavailable_camera_and_cleans_up(self):
        camera = FakeCamera(opened=False)
        landmarker = FakeLandmarker()
        result = self.run_capture([], camera, landmarker, FakeViewer())

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_UNAVAILABLE)
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)

    def test_capture_rejects_wrong_frame_dimensions_and_cleans_up(self):
        camera = FakeCamera(frame_shape=(1080, 1920, 3))
        landmarker = FakeLandmarker([FakeDetectionResult() for _ in range(8)])
        viewer = FakeViewer()
        result = self.run_capture([], camera, landmarker, viewer)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_FRAME_DIMENSION_INVALID)
        self.assertTrue(any("camera returned a 1920x1080 frame" in item for item in result.warnings))
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_successful_capture_builds_v2_record_and_cleans_up(self):
        camera = FakeCamera(set_result=False, frame_count=MIN_FRAMES + 3)
        landmarker = FakeLandmarker([FakeDetectionResult() for _ in range(MIN_FRAMES + 3)])
        viewer = FakeViewer([None, key("s")] + [None] * MIN_FRAMES + [key("e")])

        result = self.run_capture([], camera, landmarker, viewer)

        self.assertEqual(result.status, STATUS_OK)
        self.assertIsNone(result.reason)
        self.assertEqual(result.frame_count, MIN_FRAMES)
        self.assertEqual(len(result.records), MIN_FRAMES)
        self.assertEqual(
            result.metadata,
            {
                "duration_s": MIN_FRAMES / 30.0,
                "fps": 30.0,
                "width": 1280.0,
                "height": 720.0,
            },
        )
        self.assertEqual(landmarker.timestamps[:4], [0, 33, 67, 100])
        self.assertEqual(len(landmarker.timestamps), MIN_FRAMES + 3)

        record = result.records[0]
        self.assertTrue(REQUIRED_RECORD_KEYS.issubset(record.keys()))
        self.assertEqual(record["frame_index"], 0)
        self.assertEqual(record["timestamp_ms"], 67)
        self.assertEqual(record["fps"], 30.0)
        self.assertEqual(record["width"], 1280)
        self.assertEqual(record["height"], 720)
        self.assertTrue(record["decode_ok"])
        self.assertTrue(record["infer_ok"])
        self.assertEqual(record["num_hands"], 1)
        self.assertEqual(record["hands"][0]["handedness_label"], "Left")
        self.assertEqual(record["hands"][0]["handedness_score"], 0.91)

        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)
        self.assertEqual(viewer.shown, MIN_FRAMES + 3)
        self.assertTrue(any(f"Recording: {MIN_FRAMES}/{MAX_FRAMES} frames" in item for item in viewer.statuses))

    def test_early_e_requires_more_frames(self):
        camera = FakeCamera(set_result=False, frame_count=6)
        landmarker = FakeLandmarker([FakeDetectionResult() for _ in range(6)])
        viewer = FakeViewer([None, key("s"), None, key("e"), None, key("q")])

        result = self.run_capture([], camera, landmarker, viewer)

        self.assertEqual(result.status, STATUS_QUIT)
        self.assertEqual(result.reason, REASON_USER_QUIT)
        self.assertEqual(result.records, [])
        self.assertEqual(result.frame_count, 0)
        self.assertTrue(any(f"Need {MIN_FRAMES - 1} more frame(s)" in item for item in result.warnings))
        self.assertTrue(any(f"Need {MIN_FRAMES - 1} more frame(s)" in item for item in viewer.statuses))
        self.assertTrue(any(f"E is locked until {MIN_FRAMES}" in item for item in viewer.statuses))
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_valid_e_finalizes_at_minimum(self):
        camera = FakeCamera(set_result=False, frame_count=MIN_FRAMES + 3)
        landmarker = FakeLandmarker([FakeDetectionResult() for _ in range(MIN_FRAMES + 3)])
        viewer = FakeViewer([None, key("s")] + [None] * MIN_FRAMES + [key("e")])

        result = self.run_capture([], camera, landmarker, viewer)

        self.assertEqual(result.status, STATUS_OK)
        self.assertIsNone(result.reason)
        self.assertEqual(result.frame_count, MIN_FRAMES)
        self.assertEqual(len(result.records), MIN_FRAMES)
        self.assertFalse(any("Need " in item for item in result.warnings))
        self.assertFalse(any("RECORD_TOO_LONG" in item for item in result.warnings))
        self.assertTrue(any(f"Minimum reached ({MIN_FRAMES}); press E to end" in item for item in viewer.statuses))

    def test_automatic_stop_at_maximum_finalizes_safely(self):
        camera = FakeCamera(set_result=False, frame_count=MAX_FRAMES + 3)
        landmarker = FakeLandmarker([FakeDetectionResult() for _ in range(MAX_FRAMES + 3)])
        viewer = FakeViewer([None, key("s")] + [None] * (MAX_FRAMES + 2))

        result = self.run_capture([], camera, landmarker, viewer)

        self.assertEqual(result.status, STATUS_OK)
        self.assertIsNone(result.reason)
        self.assertEqual(result.frame_count, MAX_FRAMES)
        self.assertEqual(len(result.records), MAX_FRAMES)
        self.assertEqual(result.records[-1]["frame_index"], MAX_FRAMES - 1)
        self.assertTrue(any("AUTO_STOP_AT_MAX_FRAMES" in item for item in result.warnings))
        self.assertFalse(any("RECORD_TOO_LONG" in item for item in result.warnings))
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_configuration_failure_reports_clear_reason(self):
        camera = FakeCamera(
            initial_properties={cv2.CAP_PROP_FRAME_WIDTH: 640.0},
            ignore_set=True,
        )
        landmarker = FakeLandmarker()

        result = self.run_capture([], camera, landmarker, FakeViewer())

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_CONFIGURATION_INVALID)
        self.assertTrue(any("required 1280x720 at 30 FPS" in item for item in result.warnings))
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
