import time
import unittest

# Set the servo angles here before running the test.
SERVO_TEST_ANGLES = [45, 45, 45, 45]  # change these values as needed

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
    def test_command_and_print_servo_angles(self):
        if not servo_available:
            self.skipTest(f"Servo controller unavailable: {servo_error}")

        robot = RobotKinematics()
        rc = RobotController(robot)

        theta1, theta2, theta3, theta4 = SERVO_TEST_ANGLES
        print(f"Commanding servos to: {SERVO_TEST_ANGLES}")
        rc.set_motor_angles(theta1, theta2, theta3, theta4)

        start = time.time()
        while time.time() - start < 10:
            angles = rc.get_motor_angles()
            print(f"Commanded angles: {angles}")
            time.sleep(1)

        print("Servo command test finished.")


if __name__ == "__main__":
    unittest.main()
