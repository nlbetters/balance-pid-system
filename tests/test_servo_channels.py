import unittest

from controller import RobotController


class TestServoChannels(unittest.TestCase):
    def test_servo_channel_constants(self):
        self.assertEqual(RobotController.SERVO_CHANNELS, [0, 4, 8, 12])

    def test_servokit_index_access(self):
        try:
            from adafruit_servokit import ServoKit
        except ImportError as ex:
            self.skipTest(f"Adafruit ServoKit not installed: {ex}")

        kit = ServoKit(channels=16)
        for channel in RobotController.SERVO_CHANNELS:
            self.assertIsNotNone(kit.servo[channel], f"Servo channel {channel} should be accessible")


if __name__ == "__main__":
    unittest.main()
