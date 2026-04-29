import unittest

try:
    import numpy as np
except ImportError as exc:
    np = None
    numpy_error = exc
else:
    numpy_error = None

try:
    from camera import Camera
    camera_available = True
    camera_error = None
except Exception as exc:
    Camera = None
    camera_available = False
    camera_error = exc


class TestCamera(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest(f"NumPy not installed: {numpy_error}")
        if not camera_available:
            self.skipTest(f"Camera unavailable: {camera_error}")
        self.cam = Camera()

    def tearDown(self):
        if getattr(self, "cam", None) is not None:
            self.cam.terminate()

    def test_camera_capture(self):
        frame = self.cam.take_picture()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertGreater(frame.shape[0], 0)
        self.assertGreater(frame.shape[1], 0)

    def test_coordinate_handles_blank_image(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        center = self.cam.coordinate(blank)
        self.assertIsInstance(center, tuple)
        self.assertEqual(len(center), 2)
        self.assertIsInstance(center[0], int)
        self.assertIsInstance(center[1], int)


if __name__ == "__main__":
    unittest.main()
