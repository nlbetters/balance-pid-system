import time
import unittest

try:
    from controller import RobotController
    from robotKinematics import RobotKinematics
    servo_available = True
    servo_error = None
except Exception as exc:
    RobotController = None
    RobotKinematics = None
    servo_available = False
    servo_error = exc


class TestServoAngleReporting(unittest.TestCase):
    def test_print_servo_angles_for_10_seconds(self):
        if not servo_available:
            self.skipTest(f"Servo controller unavailable: {servo_error}")

        robot = RobotKinematics()
        rc = RobotController(robot)

        print("Starting servo angle reporting for 10 seconds...")
        start = time.time()
        while time.time() - start < 10:
            angles = rc.get_motor_angles()
            print(f"Servo angles: {angles}")
            time.sleep(1)

        print("Servo angle reporting finished.")


if __name__ == "__main__":
    unittest.main()
