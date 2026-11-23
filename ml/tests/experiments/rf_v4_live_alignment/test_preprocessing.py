import unittest

import cv2
import numpy as np

from ml.src.experiments.rf_v4_live_alignment import parity, preprocessing


class FakeCategory:
    display_name = "Left"
    category_name = "Left"
    score = 0.95
    index = 0


class PreprocessingTests(unittest.TestCase):
    def test_constants_match_live_capture(self):
        self.assertEqual(preprocessing.TARGET_FPS, 30.0)
        self.assertEqual((preprocessing.TARGET_WIDTH, preprocessing.TARGET_HEIGHT), (1280, 720))
        self.assertEqual((preprocessing.MIN_FRAMES, preprocessing.MAX_FRAMES), (43, 89))
        self.assertEqual(preprocessing.RESIZE_INTERPOLATION, cv2.INTER_AREA)

    def test_preprocess_frame_uses_area_resize_and_bgr_to_rgb(self):
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)
        source[:, :, 0] = 10
        source[:, :, 1] = 20
        source[:, :, 2] = 30
        expected = cv2.cvtColor(
            cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        actual = preprocessing.preprocess_frame(source, resize=True)
        np.testing.assert_array_equal(actual, expected)

    def test_session_reset_restarts_timestamps(self):
        landmarker = parity.FakeLandmarker(parity._fake_result())
        factory_calls = []

        def factory(path, payload):
            factory_calls.append((path, payload))
            return landmarker

        session = preprocessing.LandmarkSession("model", {}, factory)
        session.reset()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        first = [session.detect(frame, resize=False).timestamp_ms for _ in range(2)]
        session.reset()
        second = [session.detect(frame, resize=False).timestamp_ms for _ in range(2)]
        self.assertEqual(first, [0, 33])
        self.assertEqual(second, [0, 33])
        self.assertEqual(len(factory_calls), 2)

    def test_record_records_live_aligned_metadata(self):
        result = parity._fake_result()
        landmarker = parity.FakeLandmarker(result)
        session = preprocessing.LandmarkSession("model", {}, lambda path, payload: landmarker)
        session.reset()
        processed = session.detect(np.zeros((720, 1280, 3), dtype=np.uint8), resize=False)
        record = preprocessing.make_landmark_record(
            processed=processed,
            split=None,
            video_id="live",
            official_rel_path=None,
            source_video_abs=None,
            class_folder=None,
            class_label=None,
            label_id=None,
            source_timestamp_ms=12.0,
            source_frame_index=0,
            source_fps=30.0,
            source_width=1280,
            source_height=720,
        )
        self.assertEqual(record["tracking_lifecycle"], "reset_at_recording_start")
        self.assertEqual(record["resize_interpolation"], "cv2.INTER_AREA")
        self.assertEqual((record["fps"], record["width"], record["height"]), (30.0, 1280, 720))
        self.assertEqual(record["timestamp_ms"], 0)


if __name__ == "__main__":
    unittest.main()
