import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.src.inference.landmark_capture import (
    DEFAULT_EXTRACTION_CONFIG_PATH,
    REASON_CAMERA_CONFIGURATION_INVALID,
    REASON_CAMERA_UNAVAILABLE,
    REASON_EMPTY_RECORDING,
    REASON_FRAME_READ_FAILED,
    REASON_LANDMARK_INFERENCE_FAILED,
    REASON_RECORD_TOO_LONG,
    REASON_USER_QUIT,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_QUIT,
    CaptureConfig,
    capture_recording,
)
from ml.src.inference.rf_feature_builder import build_rf_feature_row


REQUIRED_METADATA_KEYS = {"duration_s", "fps", "width", "height"}
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


@dataclass
class FakeCategory:
    display_name: str
    category_name: str
    score: float
    index: int


@dataclass
class FakeLandmarkPoint:
    x: float
    y: float
    z: float
    visibility: Any = None
    presence: Any = None
    name: Any = None


class FakeResult:
    def __init__(self, hands=()):
        self.hand_landmarks = []
        self.hand_world_landmarks = []
        self.handedness = []
        for hand_index, (label, score) in enumerate(hands):
            landmarks = [
                FakeLandmarkPoint(
                    x=0.1 + hand_index * 0.2 + landmark_index * 0.001,
                    y=0.2 + landmark_index * 0.002,
                    z=0.003 + landmark_index * 0.0001,
                )
                for landmark_index in range(21)
            ]
            world = [
                FakeLandmarkPoint(
                    x=point.x + 0.01,
                    y=point.y + 0.01,
                    z=point.z + 0.01,
                )
                for point in landmarks
            ]
            categories = [
                FakeCategory(
                    display_name=label,
                    category_name=label,
                    score=score,
                    index=hand_index,
                )
            ]
            self.hand_landmarks.append(landmarks)
            self.hand_world_landmarks.append(world)
            self.handedness.append(categories)


class FakeCamera:
    def __init__(
        self,
        frame_count=10,
        outcomes=None,
        opened=True,
        frame_width=1920,
        frame_height=1080,
        initial_properties=None,
        reject_settings=False,
        fps_readback=None,
    ):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self.frames = [frame for _ in range(frame_count)]
        self.outcomes = outcomes if outcomes is not None else [True] * frame_count
        self.opened = opened
        self.index = 0
        self.released = False
        self.properties = {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }
        if initial_properties:
            self.properties.update(initial_properties)
        self.reject_settings = reject_settings
        self.fps_readback = fps_readback
        self.set_calls = []
        self.get_calls = []

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.outcomes):
            return False, None
        ok = self.outcomes[self.index]
        frame = self.frames[self.index] if ok and self.index < len(self.frames) else None
        self.index += 1
        return ok, frame

    def get(self, property_id):
        self.get_calls.append(property_id)
        if property_id == cv2.CAP_PROP_FPS and self.fps_readback is not None:
            return self.fps_readback
        return self.properties.get(property_id, 0.0)

    def set(self, property_id, value):
        self.set_calls.append((property_id, float(value)))
        if self.reject_settings:
            return False
        self.properties[property_id] = float(value)
        return True

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self, results=None, raise_on_detect=None):
        self.results = list(results or [])
        self.raise_on_detect = raise_on_detect
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        if self.raise_on_detect is not None:
            raise self.raise_on_detect
        if not self.results:
            raise RuntimeError("no mocked detection result left")
        return self.results.pop(0)

    def close(self):
        self.closed = True


class FakeViewer:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.shown = 0
        self.closed = False

    def show(self, frame, result):
        self.shown += 1

    def poll_key(self):
        if not self.keys:
            return None
        return self.keys.pop(0)

    def close(self):
        self.closed = True


def make_config(**overrides):
    values = {
        "extraction_config_path": DEFAULT_EXTRACTION_CONFIG_PATH,
        "model_path": Path("fake_hand_landmarker.task"),
    }
    values.update(overrides)
    return CaptureConfig(**values)


def run_capture(
    keys,
    results,
    frame_count=None,
    outcomes=None,
    config=None,
    landmarker=None,
    camera=None,
):
    if frame_count is None:
        frame_count = len(results)
    camera = camera or FakeCamera(frame_count=frame_count, outcomes=outcomes)
    landmarker = landmarker or FakeLandmarker(results)
    viewer = FakeViewer(keys)
    active_config = config or make_config()
    result = capture_recording(
        config=active_config,
        camera_factory=lambda index: camera,
        landmarker_factory=lambda model_path, payload: landmarker,
        viewer=viewer,
    )
    return result, camera, landmarker, viewer


class LandmarkCaptureTests(unittest.TestCase):
    def test_start_stop_state_transition_buffers_records_with_monotonic_timestamps(self):
        results = [FakeResult([("Left", 0.91)]) for _ in range(5)]
        result, camera, landmarker, viewer = run_capture(
            keys=[None, key("s"), None, None, key("e")],
            results=results,
        )

        self.assertEqual(result.status, STATUS_OK)
        self.assertIsNone(result.reason)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(len(result.records), 2)
        self.assertEqual([record["frame_index"] for record in result.records], [0, 1])
        self.assertEqual([record["timestamp_ms"] for record in result.records], [80, 120])
        self.assertEqual(landmarker.timestamps, [0, 40, 80, 120, 160])
        self.assertEqual(
            result.metadata,
            {"duration_s": 0.08, "fps": 25.0, "width": 1920.0, "height": 1080.0},
        )
        for record in result.records:
            self.assertTrue(REQUIRED_RECORD_KEYS.issubset(record.keys()))
            self.assertTrue(record["decode_ok"])
            self.assertTrue(record["infer_ok"])
            self.assertEqual(record["num_hands"], 1)
            self.assertEqual(record["hands"][0]["handedness_label"], "Left")
            self.assertEqual(record["hands"][0]["handedness_score"], 0.91)
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)
        self.assertEqual(viewer.shown, 5)

    def test_start_key_is_ignored_while_already_recording(self):
        results = [FakeResult() for _ in range(4)]
        result, _, _, _ = run_capture(
            keys=[key("s"), key("s"), None, key("e")],
            results=results,
        )

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.frame_count, 2)

    def test_missing_hand_frames_are_preserved_and_accepted_by_builder(self):
        results = [FakeResult(), FakeResult(), FakeResult()]
        result, _, _, _ = run_capture(
            keys=[key("s"), None, key("e")],
            results=results,
        )

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record["num_hands"], 0)
        self.assertEqual(record["hands"], [])
        self.assertTrue(record["decode_ok"])
        self.assertTrue(record["infer_ok"])
        built = build_rf_feature_row(result.records, result.metadata)
        self.assertEqual(built.values.shape, (1, 831))

    def test_two_handed_frames_preserve_both_hands_and_scores(self):
        results = [FakeResult(), FakeResult([("Left", 0.81), ("Right", 0.73)]), FakeResult()]
        result, _, _, _ = run_capture(
            keys=[key("s"), None, key("e")],
            results=results,
        )

        self.assertEqual(result.status, STATUS_OK)
        record = result.records[0]
        self.assertEqual(record["num_hands"], 2)
        self.assertEqual(
            [hand["handedness_label"] for hand in record["hands"]],
            ["Left", "Right"],
        )
        self.assertEqual(
            [hand["handedness_score"] for hand in record["hands"]],
            [0.81, 0.73],
        )

    def test_failed_frame_read_returns_clear_error_and_releases_resources(self):
        results = [FakeResult(), FakeResult()]
        result, camera, landmarker, viewer = run_capture(
            keys=[key("s")],
            results=results,
            outcomes=[True, False],
        )

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_FRAME_READ_FAILED)
        self.assertEqual(result.records, [])
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_empty_recording_returns_clear_error(self):
        result, camera, landmarker, viewer = run_capture(
            keys=[key("s"), key("e")],
            results=[FakeResult(), FakeResult()],
        )

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_EMPTY_RECORDING)
        self.assertEqual(result.records, [])
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_camera_unavailable_returns_clear_error_and_releases_camera(self):
        camera = FakeCamera(frame_count=1, opened=False)
        landmarker = FakeLandmarker([FakeResult()])
        viewer = FakeViewer([key("s"), key("e")])
        result = capture_recording(
            config=make_config(),
            camera_factory=lambda index: camera,
            landmarker_factory=lambda model_path, payload: landmarker,
            viewer=viewer,
        )

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_UNAVAILABLE)
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_user_quit_discards_partial_recording(self):
        results = [FakeResult(), FakeResult(), FakeResult()]
        result, camera, landmarker, viewer = run_capture(
            keys=[key("s"), None, key("q")],
            results=results,
        )

        self.assertEqual(result.status, STATUS_QUIT)
        self.assertEqual(result.reason, REASON_USER_QUIT)
        self.assertEqual(result.records, [])
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_resource_safety_cap_rejects_overlong_recording(self):
        config = make_config(max_frames=3)
        results = [FakeResult() for _ in range(5)]
        result, camera, landmarker, viewer = run_capture(
            keys=[key("s"), None, None, None, None],
            results=results,
            config=config,
        )

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_RECORD_TOO_LONG)
        self.assertEqual(result.records, [])
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)

    def test_detection_exception_returns_clear_error_and_releases_resources(self):
        landmarker = FakeLandmarker(raise_on_detect=RuntimeError("mock inference failure"))
        result, camera, _, viewer = run_capture(
            keys=[key("s"), key("e")],
            results=[FakeResult()],
            landmarker=landmarker,
        )

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_LANDMARK_INFERENCE_FAILED)
        self.assertEqual(result.records, [])
        self.assertTrue(camera.released)
        self.assertTrue(landmarker.closed)
        self.assertTrue(viewer.closed)
        self.assertTrue(any("mock inference failure" in item for item in result.warnings))

    def test_configuration_requests_and_verifies_required_camera_properties(self):
        camera = FakeCamera(
            frame_count=3,
            initial_properties={
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            },
        )
        result, camera, landmarker, _ = run_capture(
            keys=[key("s"), None, key("e")],
            results=[FakeResult(), FakeResult(), FakeResult()],
            camera=camera,
        )

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(
            camera.set_calls,
            [
                (cv2.CAP_PROP_FRAME_WIDTH, 1920.0),
                (cv2.CAP_PROP_FRAME_HEIGHT, 1080.0),
                (cv2.CAP_PROP_FPS, 25.0),
            ],
        )
        self.assertIn(cv2.CAP_PROP_FPS, camera.get_calls)
        self.assertEqual(result.metadata["fps"], 25.0)
        self.assertEqual(result.metadata["width"], 1920)
        self.assertEqual(result.metadata["height"], 1080)
        self.assertEqual(len(landmarker.timestamps), 3)

    def test_rejected_camera_settings_return_error_before_prediction(self):
        camera = FakeCamera(
            frame_count=1,
            initial_properties={
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            },
            reject_settings=True,
        )
        landmarker = FakeLandmarker([FakeResult()])
        result, camera, landmarker, _ = run_capture(
            keys=[key("q")],
            results=[FakeResult()],
            camera=camera,
            landmarker=landmarker,
        )
        detail = " ".join(result.warnings)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_CONFIGURATION_INVALID)
        self.assertIn("640x480", detail)
        self.assertIn("30 FPS", detail)
        self.assertIn("1920x1080", detail)
        self.assertEqual(camera.index, 0)
        self.assertEqual(landmarker.timestamps, [])

    def test_incorrect_actual_frame_dimensions_return_error_before_prediction(self):
        camera = FakeCamera(frame_count=1, frame_width=640, frame_height=480)
        landmarker = FakeLandmarker([FakeResult()])
        result, camera, landmarker, _ = run_capture(
            keys=[key("q")],
            results=[FakeResult()],
            camera=camera,
            landmarker=landmarker,
        )
        detail = " ".join(result.warnings)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_CONFIGURATION_INVALID)
        self.assertIn("640x480 frames", detail)
        self.assertIn("required 1920x1080", detail)
        self.assertIn("FPS readback 25", detail)
        self.assertEqual(camera.index, 1)
        self.assertEqual(landmarker.timestamps, [])

    def test_invalid_fps_readback_returns_error_without_fallback(self):
        camera = FakeCamera(frame_count=1, fps_readback=float("nan"))
        landmarker = FakeLandmarker([FakeResult()])
        result, camera, landmarker, _ = run_capture(
            keys=[key("q")],
            results=[FakeResult()],
            camera=camera,
            landmarker=landmarker,
        )
        detail = " ".join(result.warnings)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.reason, REASON_CAMERA_CONFIGURATION_INVALID)
        self.assertIn("FPS readback is invalid", detail)
        self.assertIn("1920x1080 at 25 FPS", detail)
        self.assertEqual(camera.index, 0)
        self.assertEqual(landmarker.timestamps, [])
        self.assertIsNone(result.metadata)


if __name__ == "__main__":
    import unittest

    unittest.main()
