import os
import sys
import time
import unittest

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera import Camera

try:
    import numpy as np
except ImportError as exc:
    np = None
    numpy_error = exc
else:
    numpy_error = None


class TestCameraTracking(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest(f"NumPy not installed: {numpy_error}")
        try:
            self.cam = Camera(debug=True)
        except Exception as exc:
            self.skipTest(f"Camera unavailable: {exc}")

    def tearDown(self):
        if getattr(self, "cam", None) is not None:
            self.cam.terminate()

    def test_camera_tracking_pipeline(self):
        frame = self.cam.take_picture()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)

        center, offset, found, confidence, fps, last_valid = self.cam.coordinate_with_offset(frame)
        self.assertIsInstance(center, tuple)
        self.assertEqual(len(center), 2)
        self.assertIsInstance(found, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(offset, tuple)
        self.assertEqual(len(offset), 2)

    def test_coordinate_detects_synthetic_ball(self):
        image = np.zeros((150, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 75), 20, (255, 255, 255), -1)

        center = self.cam.coordinate(image)
        self.assertIsInstance(center, tuple)
        self.assertEqual(len(center), 2)
        self.assertTrue(self.cam.ball_found)
        self.assertGreater(self.cam.confidence, 0.0)


def manual_camera_tracking_view(max_frames: int = 500) -> None:
    try:
        cam = Camera(debug=True)
    except Exception as exc:
        raise RuntimeError(f"Camera unavailable: {exc}") from exc

    try:
        print("Camera tracking view started. Press 'q' in the window to quit.")
        frame_count = 0
        while frame_count < max_frames:
            frame = cam.take_picture()
            center, offset, found, confidence, fps, last_valid = cam.coordinate_with_offset(frame)
            status = f"found={found} confidence={confidence:.2f} fps={fps:.1f}"
            cv2.putText(
                frame,
                status,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cam.display_debug(frame)
            cam.display_draw(frame, center, window_name="Camera Tracking")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1
            time.sleep(0.01)
    finally:
        cam.terminate()


if __name__ == "__main__":
    manual_camera_tracking_view()