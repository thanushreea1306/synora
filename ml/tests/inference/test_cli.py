import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.src.inference import cli
from ml.src.inference.landmark_capture import (
    REASON_CAMERA_CONFIGURATION_INVALID,
    REASON_CAMERA_UNAVAILABLE,
    REASON_EMPTY_RECORDING,
    REASON_USER_QUIT,
    STATUS_ERROR,
    STATUS_OK as CAPTURE_STATUS_OK,
    STATUS_QUIT,
    CaptureConfig,
    CaptureResult,
)
from ml.src.inference.rf_service import RFInferenceService


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


@dataclass(frozen=True)
class FakeDetectionResult:
    hand_landmarks: list
    hand_world_landmarks: list
    handedness: list


def make_detection_result(label="Left", score=0.93):
    landmarks = [
        FakePoint(x=0.1 + index * 0.001, y=0.2 + index * 0.002, z=0.003)
        for index in range(21)
    ]
    world = [
        FakePoint(x=point.x + 0.01, y=point.y + 0.01, z=point.z + 0.01)
        for point in landmarks
    ]
    category = FakeCategory(
        display_name=label,
        category_name=label,
        score=score,
        index=0,
    )
    return FakeDetectionResult(
        hand_landmarks=[landmarks],
        hand_world_landmarks=[world],
        handedness=[[category]],
    )


class MockClassifier:
    def __init__(self, prediction=2):
        self.prediction = prediction
        self.n_features_in_ = 831
        self.classes_ = np.arange(21)
        self.predict_calls = []
        self.proba_calls = []

    def predict(self, values):
        self.predict_calls.append(np.array(values, copy=True))
        return np.array([self.prediction])

    def predict_proba(self, values):
        self.proba_calls.append(np.array(values, copy=True))
        probabilities = np.zeros(21, dtype=np.float64)
        probabilities[self.prediction] = 1.0
        return probabilities.reshape(1, -1)


class FakeCamera:
    def __init__(self, initial_properties=None):
        self.release_count = 0
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.properties = {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }
        if initial_properties:
            self.properties.update(initial_properties)
        self.set_calls = []

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def get(self, property_id):
        return self.properties.get(property_id, 0.0)

    def set(self, property_id, value):
        self.set_calls.append((property_id, float(value)))
        self.properties[property_id] = float(value)
        return True

    def release(self):
        self.release_count += 1


class FakeLandmarker:
    def __init__(self, results):
        self.results = list(results)
        self.close_count = 0

    def detect_for_video(self, image, timestamp_ms):
        if not self.results:
            raise RuntimeError("no fake detection result left")
        return self.results.pop(0)

    def close(self):
        self.close_count += 1


class FakeViewer:
    def __init__(self, keys):
        self.keys = list(keys)
        self.close_count = 0

    def show(self, frame, result):
        return None

    def poll_key(self):
        if not self.keys:
            return None
        return self.keys.pop(0)

    def close(self):
        self.close_count += 1


def key(character):
    return ord(character)


def make_config():
    return CaptureConfig(camera_index=0, show_preview=False)


def make_service(model):
    loader_calls = []

    def loader(path):
        loader_calls.append(Path(path))
        return model

    return RFInferenceService(model_loader=loader), loader_calls


class CLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = MockClassifier()
        self.service, self.loader_calls = make_service(self.model)
        self.output = []

    def printer(self, line):
        self.output.append(str(line))

    def test_two_recordings_connect_capture_builder_and_service_with_cleanup(self):
        key_sequences = [
            [key("s"), None, None, key("e")],
            [key("s"), None, key("e")],
        ]
        cameras = []
        landmarkers = []
        viewers = []

        def camera_factory(index):
            camera = FakeCamera()
            cameras.append(camera)
            return camera

        def landmarker_factory(model_path, payload):
            landmarker = FakeLandmarker(
                [make_detection_result() for _ in key_sequences[len(landmarkers)]]
            )
            landmarkers.append(landmarker)
            return landmarker

        def viewer_factory():
            viewer = FakeViewer(key_sequences[len(viewers)])
            viewers.append(viewer)
            return viewer

        code = cli.main(
            argv=["--no-preview", "--max-recordings", "2"],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            camera_factory=camera_factory,
            landmarker_factory=landmarker_factory,
            viewer_factory=viewer_factory,
        )

        text = "\n".join(self.output)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.loader_calls), 1)
        self.assertEqual(len(self.model.predict_calls), 2)
        self.assertEqual(len(self.model.proba_calls), 2)
        self.assertEqual(len(cameras), 2)
        self.assertEqual(len(landmarkers), 2)
        self.assertEqual(len(viewers), 2)
        self.assertEqual([camera.release_count for camera in cameras], [1, 1])
        self.assertEqual([item.close_count for item in landmarkers], [1, 1])
        self.assertEqual([item.close_count for item in viewers], [1, 1])
        self.assertIn("Predicted sign: happy", text)
        self.assertIn("Original label ID: 95", text)
        self.assertIn("Model class index: 2", text)
        self.assertIn("Raw confidence: 1.000000", text)
        self.assertIn("frame_count: 2", text)
        self.assertIn("frame_count: 1", text)
        self.assertIn("not calibrated", text)
        self.assertIn("Finished requested recordings", text)

    def test_invalid_recording_shows_error_and_does_not_predict(self):
        capture_calls = []

        def capture_fn(**kwargs):
            capture_calls.append(kwargs)
            return CaptureResult(
                status=STATUS_ERROR,
                reason=REASON_EMPTY_RECORDING,
                warnings=["empty buffer"],
                frame_count=0,
            )

        code = cli.main(
            argv=[],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            capture_fn=capture_fn,
            max_recordings=1,
        )

        text = "\n".join(self.output)
        self.assertEqual(code, 0)
        self.assertEqual(len(capture_calls), 1)
        self.assertEqual(self.model.predict_calls, [])
        self.assertIn("Capture error: EMPTY_RECORDING", text)
        self.assertIn("No prediction was produced.", text)
        self.assertIn("empty buffer", text)

    def test_quit_exits_cleanly_without_prediction(self):
        def capture_fn(**kwargs):
            return CaptureResult(
                status=STATUS_QUIT,
                reason=REASON_USER_QUIT,
                warnings=[],
                frame_count=0,
            )

        code = cli.main(
            argv=[],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            capture_fn=capture_fn,
            max_recordings=1,
        )

        text = "\n".join(self.output)
        self.assertEqual(code, 0)
        self.assertEqual(self.model.predict_calls, [])
        self.assertIn("Q pressed. Exiting cleanly.", text)

    def test_camera_failure_is_fatal_and_does_not_predict(self):
        def capture_fn(**kwargs):
            return CaptureResult(
                status=STATUS_ERROR,
                reason=REASON_CAMERA_UNAVAILABLE,
                warnings=[],
                frame_count=0,
            )

        code = cli.main(
            argv=[],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            capture_fn=capture_fn,
            max_recordings=1,
        )

        text = "\n".join(self.output)
        self.assertEqual(code, 1)
        self.assertEqual(self.model.predict_calls, [])
        self.assertIn("Capture error: CAMERA_UNAVAILABLE", text)

    def test_camera_configuration_failure_is_fatal_and_does_not_predict(self):
        def capture_fn(**kwargs):
            return CaptureResult(
                status=STATUS_ERROR,
                reason=REASON_CAMERA_CONFIGURATION_INVALID,
                warnings=["camera returned 640x480 frames"],
                frame_count=0,
            )

        code = cli.main(
            argv=[],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            capture_fn=capture_fn,
            max_recordings=2,
        )

        text = "\n".join(self.output)
        self.assertEqual(code, 1)
        self.assertEqual(self.model.predict_calls, [])
        self.assertIn("Capture error: CAMERA_CONFIGURATION_INVALID", text)
        self.assertIn("camera returned 640x480 frames", text)

    def test_service_error_result_is_displayed_without_forced_prediction(self):
        result = {
            "status": "model_error",
            "predicted_sign": None,
            "original_label_id": None,
            "model_class_index": None,
            "raw_confidence": None,
            "raw_probabilities": None,
            "recording_diagnostics": {"frame_count": 1},
            "warnings": [],
            "errors": [
                {
                    "code": "MODEL_PREDICTION_FAILED",
                    "message": "RuntimeError: mock failure",
                }
            ],
            "confidence_note": "raw only",
        }

        text = cli.format_prediction(result)
        self.assertIn("Prediction failed: model_error", text)
        self.assertIn("MODEL_PREDICTION_FAILED: RuntimeError: mock failure", text)
        self.assertIn("No prediction was produced.", text)
        self.assertIn("frame_count: 1", text)

    def test_quit_cleanup_releases_fake_resources(self):
        cameras = []
        landmarkers = []
        viewers = []

        def camera_factory(index):
            camera = FakeCamera()
            cameras.append(camera)
            return camera

        def landmarker_factory(model_path, payload):
            landmarker = FakeLandmarker([make_detection_result()])
            landmarkers.append(landmarker)
            return landmarker

        def viewer_factory():
            viewer = FakeViewer([key("s"), key("q")])
            viewers.append(viewer)
            return viewer

        code = cli.main(
            argv=["--no-preview", "--max-recordings", "1"],
            printer=self.printer,
            service=self.service,
            capture_config=make_config(),
            camera_factory=camera_factory,
            landmarker_factory=landmarker_factory,
            viewer_factory=viewer_factory,
        )

        self.assertEqual(code, 0)
        self.assertEqual(self.model.predict_calls, [])
        self.assertEqual(cameras[0].release_count, 1)
        self.assertEqual(landmarkers[0].close_count, 1)
        self.assertEqual(viewers[0].close_count, 1)


if __name__ == "__main__":
    unittest.main()
