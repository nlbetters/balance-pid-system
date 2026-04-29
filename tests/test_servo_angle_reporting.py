import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the servo angles here before running the script.
SERVO_TEST_ANGLES = [45, 45, 45, 45]  # change these values as needed
RUNTIME_SECONDS = 10

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

    start = time.time()
    while time.time() - start < RUNTIME_SECONDS:
        angles = rc.get_motor_angles()
        print(f"Commanded angles: {angles}")
        time.sleep(1)

    print("Servo command script finished.")
