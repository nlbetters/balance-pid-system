import math
import time
import cv2
from PID import PIDcontroller
from robot import Robot

# Constants
COMMAND_THETA_GAIN = 1.4
MAX_COMMAND_THETA = 8.0
MIN_ACTIVE_THETA = 0.2
PIXEL_DEADBAND = 4.0
COMMAND_PHI_OFFSET_DEG = 0.0
INVERT_X_RESPONSE = True
INVERT_Y_RESPONSE = True

# Asymmetric correction tuning.
# Servo channel 4 is the +Y side of the platform. If the ball gets stuck on that side,
# boost the +Y command so servo 4/12 push harder in that direction.
SERVO4_SIDE_Y_GAIN = 1.45
SERVO4_SIDE_MIN_THETA = 0.55

DEBUG_DIRECTION_TEST = False

def update_robot_pos(robot, pid_controller, target_pos, current_pos):
    # Compute PID output
    theta, phi = pid_controller.pid(target_pos, current_pos)

    # Convert spherical to cartesian
    phi_rad = math.radians(phi + COMMAND_PHI_OFFSET_DEG)
    command_x = theta * math.cos(phi_rad)
    command_y = theta * math.sin(phi_rad)

    # Apply inversion
    if INVERT_X_RESPONSE:
        command_x *= -1
    if INVERT_Y_RESPONSE:
        command_y *= -1

    servo4_boost_active = command_y > 0
    if servo4_boost_active:
        command_y *= SERVO4_SIDE_Y_GAIN

    theta = math.hypot(command_x, command_y) * COMMAND_THETA_GAIN

    error_pixels = math.hypot(current_pos[0] - target_pos[0], current_pos[1] - target_pos[1])
    if error_pixels <= PIXEL_DEADBAND:
        theta = 0.0
    elif servo4_boost_active and theta < SERVO4_SIDE_MIN_THETA:
        theta = SERVO4_SIDE_MIN_THETA
    elif theta < MIN_ACTIVE_THETA:
        theta = MIN_ACTIVE_THETA

    # Limit maximum command theta
    if theta > MAX_COMMAND_THETA:
        theta = MAX_COMMAND_THETA

    # Update robot servo commands
    robot.set_servo_angles(theta, phi)

    if DEBUG_DIRECTION_TEST:
        print(
            f"target=({target_pos[0]:.1f},{target_pos[1]:.1f}) current=({current_pos[0]:.1f},{current_pos[1]:.1f}) "
            f"cmd_xy=({command_x:.2f},{command_y:.2f}) theta={theta:.2f} phi={phi:.1f} "
            f"s4boost={servo4_boost_active} "
        )

def main():
    # Initialize PID controller and robot
    pid_controller = PIDcontroller(kp=1.0, ki=0.1, kd=0.05, alpha=0.7, beta=1.0, max_theta=MAX_COMMAND_THETA)
    robot = Robot()

    # Example target and current positions
    target_pos = (320, 240)
    current_pos = (300, 220)

    while True:
        # Update robot position based on PID output
        update_robot_pos(robot, pid_controller, target_pos, current_pos)

        # Add your main loop code here
        time.sleep(0.01)

if __name__ == "__main__":
    main()
