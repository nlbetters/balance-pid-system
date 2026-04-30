import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller import RobotController, servo_angles_from_kinematics
from robotKinematics import RobotKinematics


class TestFourArmKinematics(unittest.TestCase):
    def setUp(self):
        self.robot = RobotKinematics()

    def solve_servo_angles(self, theta, phi, h=8.26):
        self.robot.solve_inverse_kinematics_spherical(theta, phi, h)
        return servo_angles_from_kinematics(self.robot)

    def test_channels_are_ordered_as_physical_opposite_pairs(self):
        self.assertEqual(RobotController.SERVO_CHANNELS, [0, 4, 8, 12])

    def test_flat_platform_commands_all_four_arms_equally(self):
        angles = self.solve_servo_angles(0, 0)

        for angle in angles:
            self.assertGreaterEqual(angle, 0)
            self.assertLessEqual(angle, 100)
            self.assertTrue(math.isclose(angle, angles[0], abs_tol=1e-6))

    def test_max_height_maps_to_zero_servo_angle(self):
        angles = self.solve_servo_angles(0, 0, h=self.robot.maxh)

        for angle in angles:
            self.assertTrue(math.isclose(angle, 0, abs_tol=1e-6))

    def test_servo_angles_are_capped_at_safety_limit(self):
        self.robot.theta1 = math.radians(500)
        self.robot.theta2 = math.radians(500)
        self.robot.theta3 = math.radians(500)
        self.robot.theta4 = math.radians(500)

        self.assertEqual(servo_angles_from_kinematics(self.robot), [100, 100, 100, 100])

    def test_x_axis_tilt_moves_opposite_pair_against_each_other(self):
        servo0, servo4, servo8, servo12 = self.solve_servo_angles(5, 0)

        self.assertGreater(servo0, servo8)
        self.assertTrue(math.isclose(servo4, servo12, abs_tol=1e-6))

    def test_y_axis_tilt_moves_opposite_pair_against_each_other(self):
        servo0, servo4, servo8, servo12 = self.solve_servo_angles(5, 90)

        self.assertGreater(servo4, servo12)
        self.assertTrue(math.isclose(servo0, servo8, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
