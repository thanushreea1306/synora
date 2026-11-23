import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.src.experiments.rf_v2_720p30 import build_features_v2
from ml.src.experiments.rf_v4_live_alignment import capture, parity, preprocessing, v4_service


class FakeCamera:
    def __init__(self, frame_count):
        self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.index = 0
        self.properties = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, property_id):
        return self.properties.get(property_id, 0.0)

    def set(self, property_id, value):
        self.properties[property_id] = float(value)
        return True

    def release(self):
        self.released = True


class FakeViewer:
    def __init__(self, keys):
        self.keys = list(keys)
        self.closed = False

    def show(self, frame, result, status=None):
        return None

    def poll_key(self):
        if not self.keys:
            return -1
        return self.keys.pop(0)

    def close(self):
        self.closed = True


class V4CaptureTests(unittest.TestCase):
    def test_capture_resets_tracking_at_recording_start(self):
        result = parity._fake_result()
        landmarkers = []
        factory_calls = []

        def landmarker_factory(path, payload):
            factory_calls.append((path, payload))
            landmarker = parity.FakeLandmarker(result)
            landmarkers.append(landmarker)
            return landmarker

        keys = [ord("s")] + [-1] * preprocessing.MIN_FRAMES + [ord("e")]
        camera = FakeCamera(frame_count=len(keys) + 2)
        viewer = FakeViewer(keys)
        capture_result = capture.capture_segment(
            config=capture.CaptureConfig(show_preview=False),
            camera_factory=lambda index: camera,
            landmarker_factory=landmarker_factory,
            viewer_factory=lambda: viewer,
        )
        self.assertEqual(capture_result.status, capture.STATUS_OK)
        self.assertEqual(len(capture_result.records), preprocessing.MIN_FRAMES)
        self.assertEqual(len(factory_calls), 2)
        self.assertEqual(capture_result.records[0]["timestamp_ms"], 0)
        expected_last_timestamp = int(round((preprocessing.MIN_FRAMES - 1) * 1000.0 / preprocessing.TARGET_FPS))
        self.assertEqual(capture_result.records[-1]["timestamp_ms"], expected_last_timestamp)
        self.assertTrue(
            all(row["tracking_lifecycle"] == "reset_at_recording_start" for row in capture_result.records)
        )
        self.assertEqual(capture_result.metadata["fps"], 30.0)
        self.assertTrue(camera.released)
        self.assertTrue(viewer.closed)
        self.assertTrue(all(item.closed for item in landmarkers))

    def test_short_recording_is_not_finalized_on_end_key(self):
        landmarker_factory = lambda path, payload: parity.FakeLandmarker(parity._fake_result())
        keys = [ord("s"), -1, ord("e"), ord("q")]
        camera = FakeCamera(frame_count=len(keys) + 2)
        viewer = FakeViewer(keys)
        capture_result = capture.capture_segment(
            config=capture.CaptureConfig(show_preview=False),
            camera_factory=lambda index: camera,
            landmarker_factory=landmarker_factory,
            viewer_factory=lambda: viewer,
        )
        self.assertEqual(capture_result.status, capture.STATUS_QUIT)
        self.assertEqual(capture_result.records, [])

    def test_early_end_key_then_resume_writes_contiguous_frame_indices(self):
        landmarker_factory = lambda path, payload: parity.FakeLandmarker(parity._fake_result())
        frames_before_end = 3
        frames_after_end = preprocessing.MIN_FRAMES - frames_before_end
        keys = (
            [ord("s")]
            + [-1] * frames_before_end
            + [ord("e")]
            + [-1] * frames_after_end
            + [ord("e")]
        )
        camera = FakeCamera(frame_count=len(keys) + 2)
        viewer = FakeViewer(keys)
        capture_result = capture.capture_segment(
            config=capture.CaptureConfig(show_preview=False),
            camera_factory=lambda index: camera,
            landmarker_factory=landmarker_factory,
            viewer_factory=lambda: viewer,
        )

        self.assertEqual(capture_result.status, capture.STATUS_OK)
        self.assertEqual(len(capture_result.records), preprocessing.MIN_FRAMES)
        self.assertEqual(
            [row["frame_index"] for row in capture_result.records],
            list(range(preprocessing.MIN_FRAMES)),
        )
        expected_source_indices = list(range(frames_before_end)) + list(
            range(frames_before_end + 1, preprocessing.MIN_FRAMES + 1)
        )
        self.assertEqual(
            [row["source_frame_index"] for row in capture_result.records],
            expected_source_indices,
        )
        self.assertEqual(capture_result.records[0]["timestamp_ms"], 0)
        self.assertEqual(
            capture_result.records[frames_before_end]["timestamp_ms"],
            int(round(expected_source_indices[frames_before_end] * 1000.0 / preprocessing.TARGET_FPS)),
        )
        self.assertTrue(any("more frame(s)" in item for item in capture_result.warnings))

        row = {
            "video_id": "regression_early_end_key",
            "split": "live",
            "official_rel_path_posix": None,
            "source_video_abs": None,
            "output_file": None,
            "class_folder": None,
            "class_label": None,
            "label_id": -1,
            "duration_s": capture_result.metadata["duration_s"],
        }
        sample = build_features_v2.build_sample_from_records(row, capture_result.records)
        self.assertEqual(sample["frame_count"], preprocessing.MIN_FRAMES)

        prediction = v4_service.V4RecognitionService().predict_capture(capture_result)
        self.assertEqual(prediction["status"], v4_service.STATUS_OK)
        self.assertEqual(prediction["errors"], [])

    def test_auto_stop_writes_warning_at_max_frames(self):
        landmarker_factory = lambda path, payload: parity.FakeLandmarker(parity._fake_result())
        keys = [ord("s")] + [-1] * preprocessing.MAX_FRAMES
        camera = FakeCamera(frame_count=len(keys) + 2)
        viewer = FakeViewer(keys)
        capture_result = capture.capture_segment(
            config=capture.CaptureConfig(show_preview=False),
            camera_factory=lambda index: camera,
            landmarker_factory=landmarker_factory,
            viewer_factory=lambda: viewer,
        )
        self.assertEqual(capture_result.status, capture.STATUS_OK)
        self.assertEqual(len(capture_result.records), preprocessing.MAX_FRAMES)
        self.assertTrue(any("AUTO_STOP_AT_MAX_FRAMES" in item for item in capture_result.warnings))


if __name__ == "__main__":
    unittest.main()
