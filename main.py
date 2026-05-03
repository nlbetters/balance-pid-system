import threading
import time
import cv2
import math
import numpy as np
from camera import Camera
from controller import RobotController
from robotKinematics import RobotKinematics
from PID import PIDcontroller


# Shared variables
latest_frame = np.zeros((150, 200, 3), dtype=np.uint8)
lock = threading.Lock()
running = True

#
# Start mild for direction testing. Increase kp only after the platform moves the ball
# toward the yellow center target instead of away from it.
kp = 0.04
ki = 0.0
kd = 0.01

alpha = 0.1
beta = 2.0


#
# Main control tuning. CONTROL_HEIGHT is the platform operating height used by the
# inverse kinematics. The controller now maps this neutral height to servo angle 45.
CONTROL_HEIGHT = 9.0
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

# Set this True during first tests. It prints the raw ball error and the final tilt command
# so we can quickly flip X/Y direction if the platform pushes the ball away from center.
DEBUG_DIRECTION_TEST = True

# Loop/debug tuning. Vision debug is useful for setup, but it slows the loop down.
CAMERA_HZ = 120
DEBUG_CONTROL = True
DEBUG_INTERVAL_SECONDS = 0.2
DEBUG_VISION = True


# Initialize objects
cam = Camera(debug=DEBUG_VISION)
model = RobotKinematics()
robot = RobotController(model, model.lp, model.l1, model.l2, model.lb)
model.max_theta(CONTROL_HEIGHT)


PID = PIDcontroller(kp, ki, kd, alpha, beta, max_theta=model.maxtheta, conversion="tanh")
last_debug_time = 0.0
pid_was_reset_for_lost_ball = False

# Initialize ball position at the camera target.
x, y = cam.frame_center


def capture():
    global latest_frame
    while running:
        frame = cam.take_picture()
        with lock:
            latest_frame = frame


def process():
    hz = CAMERA_HZ
    global latest_frame, x, y, pid_was_reset_for_lost_ball
    while running:
        with lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        loop_start = time.perf_counter()
        center, offset, found, confidence, fps, last_valid = cam.coordinate_with_offset(frame_copy)
        x_t, y_t = cam.frame_center  # Target position

        if found:
            x, y = center
            pid_was_reset_for_lost_ball = False
        else:
            # If the ball is lost, reset PID and return the platform to neutral.
            # This avoids chasing noise or stale camera positions.
            if not pid_was_reset_for_lost_ball:
                PID.reset()
                robot.Goto_N_time_spherical(0.0, 0.0, CONTROL_HEIGHT)
                pid_was_reset_for_lost_ball = True
                if DEBUG_CONTROL:
                    print("ball lost: returning servos to neutral")

            if DEBUG_VISION:
                cam.display_debug()

            elapsed = time.perf_counter() - loop_start
            sleep_time = (1 / hz) - elapsed
            if sleep_time > 0:
                #print(sleep_time)
                time.sleep(sleep_time)
            continue

        update_robot_pos(robot, model, PID, x_t, y_t, x, y)
        if DEBUG_VISION:
            cam.display_debug()
        #print(f"Coordinates: {x, y}")
        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / hz) - elapsed
        if sleep_time > 0:
            #print(sleep_time)
            time.sleep(sleep_time)


def update_robot_pos(robotcontroller, robotkinematics, pidcontroller, x_t, y_t, x, y):
    # x_t, y_t: target position, x, y: current position
    global last_debug_time
    error_pixels = math.hypot(x - x_t, y - y_t)
    raw_error_x = x - x_t
    raw_error_y = y - y_t

    # Camera axes are swapped here on purpose because of the mounted camera direction.
    # Recheck this if the platform tilts on the wrong axis.
    theta, phi = pidcontroller.pid((y_t, x_t), (y, x))

    command_x = math.cos(math.radians(phi)) * theta
    command_y = math.sin(math.radians(phi)) * theta
    if INVERT_X_RESPONSE:
        command_x *= -1
    if INVERT_Y_RESPONSE:
        command_y *= -1

    servo4_boost_active = command_y > 0
    if servo4_boost_active:
        command_y *= SERVO4_SIDE_Y_GAIN

    theta = math.hypot(command_x, command_y) * COMMAND_THETA_GAIN
    if error_pixels <= PIXEL_DEADBAND:
        theta = 0.0
    elif servo4_boost_active and theta < SERVO4_SIDE_MIN_THETA:
        theta = SERVO4_SIDE_MIN_THETA
    elif theta < MIN_ACTIVE_THETA:
        theta = MIN_ACTIVE_THETA
    theta = min(theta, robotkinematics.maxtheta, MAX_COMMAND_THETA)
    phi = (math.degrees(math.atan2(command_y, command_x)) + COMMAND_PHI_OFFSET_DEG) % 360

    robotcontroller.Goto_N_time_spherical(theta, phi, CONTROL_HEIGHT)

    now = time.perf_counter()
    if DEBUG_CONTROL and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
        last_debug_time = now
        if DEBUG_DIRECTION_TEST:
            print(
                f"ball=({x:.1f},{y:.1f}) target=({x_t:.1f},{y_t:.1f}) "
                f"err_px=({raw_error_x:.1f},{raw_error_y:.1f}) mag={error_pixels:.1f} "
                f"cmd_xy=({command_x:.2f},{command_y:.2f}) theta={theta:.2f} phi={phi:.1f} "
                f"s4boost={servo4_boost_active} "
                f"servos={[round(a, 1) for a in robotcontroller.get_motor_angles()]}"
            )
        else:
            print(
                f"ball=({x:.1f},{y:.1f}) target=({x_t:.1f},{y_t:.1f}) "
                f"err={error_pixels:.1f} theta={theta:.2f} phi={phi:.1f} "
                f"servos={[round(a, 1) for a in robotcontroller.get_motor_angles()]}"
            )


def pid_loop():
    hz = 30  # PID frequency
    while running:
        loop_start = time.perf_counter()
        x_t, y_t = cam.frame_center  # Target position
        update_robot_pos(robot, model, PID, x_t, y_t, x, y)
        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / hz) - elapsed
        if sleep_time > 0:
            #print(sleep_time)
            time.sleep(sleep_time)


# Move servos to the neutral position before starting the control loop.
robot.initialize()

# Start threads
threading.Thread(target=capture, daemon=True).start()
threading.Thread(target=process, daemon=True).start()
time.sleep(2)
#threading.Thread(target=pid_loop).start()


# Keep running until manually stopped
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    running = False
    print("\n")
    print("Exiting...")
