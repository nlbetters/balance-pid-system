import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set these before running the script.
SERVO_TEST_ANGLES = [45, 45, 45, 45]
SERVO_SWEEP_ANGLES = [20, 45, 70]
RUNTIME_SECONDS = 2

try:
    from controller import RobotController
    from robotKinematics import RobotKinematics
except Exception as exc:
    print("Failed to import project modules:", exc)
    raise

if __name__ == "__main__":
    robot = RobotKinematics()
    rc = RobotController(robot)

    theta1, theta2, theta3, theta4 = SERVO_TEST_ANGLES
    print(f"Commanding servos to: {SERVO_TEST_ANGLES}")
    rc.set_motor_angles(theta1, theta2, theta3, theta4)

    for channel_index, channel in enumerate(RobotController.SERVO_CHANNELS):
        print(f"\nSweeping servo index {channel_index} on PCA9685 channel {channel}")
        for angle in SERVO_SWEEP_ANGLES:
            targets = [45, 45, 45, 45]
            targets[channel_index] = angle
            rc.set_motor_angles(*targets)
            print(f"Targets: {targets}; reported: {rc.get_motor_angles()}")
            time.sleep(RUNTIME_SECONDS)

    print("Servo command script finished.")
