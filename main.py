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
kp = 0.035
ki = 0.0
kd = 0.020

alpha = 0.1
beta = 2.0


#
# Main control tuning. CONTROL_HEIGHT is the platform operating height used by the
# inverse kinematics. The controller now maps this neutral height to servo angle 45.
CONTROL_HEIGHT = 9.0
COMMAND_THETA_GAIN = 2.6
MAX_COMMAND_THETA = 16.0
MIN_ACTIVE_THETA = 0.45
PIXEL_DEADBAND = 5.0
COMMAND_PHI_OFFSET_DEG = 0.0
INVERT_X_RESPONSE = True
INVERT_Y_RESPONSE = True

# Direct camera-error control is easier to tune than the polar PID direction while testing.
# Screen left/right error maps to servo pair 4/12. Screen up/down error maps to servo pair 0/8.
USE_DIRECT_ERROR_CONTROL = True
DIRECT_X_TO_LR_GAIN = 0.070
DIRECT_Y_TO_UD_GAIN = 0.028
DIRECT_LR_SIGN = -1.0
DIRECT_UD_SIGN = -1.0

# Velocity damping.
# The platform angle controls ball acceleration, so position-only control overshoots.
# These terms brake the ball based on how fast it is moving in the camera frame.
VELOCITY_DAMPING_ENABLED = True
VELOCITY_X_TO_LR_GAIN = 0.0080
VELOCITY_Y_TO_UD_GAIN = 0.0060
MAX_VELOCITY_PIXELS_PER_SECOND = 350.0

# Axis response tuning.
# In the camera view, servo motors 4 and 12 are the left/right motors.
# Because the camera axes are swapped in the PID call below, screen horizontal error
# mainly shows up as command_y. Boost that whole left/right pair, not just one side.
LEFT_RIGHT_PAIR_GAIN = 1.45
UP_DOWN_PAIR_GAIN = 0.55
LEFT_RIGHT_MIN_THETA = 0.80
LEFT_RIGHT_ERROR_THRESHOLD = 20.0

# Corner correction tuning.
# If the ball is stuck in the bottom-right corner between motors 4 and 8,
# both the left/right and up/down axes need to respond together instead of forcing
# a pure left/right tilt.
CORNER_ERROR_THRESHOLD = 24.0
BOTTOM_RIGHT_CORNER_GAIN = 1.00
CORNER_MIN_THETA = 1.20

# Dynamic stuck response.
# If the ball is far from center and the error is not improving, slowly increase tilt.
DYNAMIC_TILT_ENABLED = False
STUCK_ERROR_THRESHOLD = 20.0
STUCK_IMPROVEMENT_THRESHOLD = 0.75
STUCK_BOOST_RATE = 0.12
STUCK_BOOST_MAX = 1.25

# Set this True during first tests. It prints the raw ball error and the final tilt command
# so we can quickly flip X/Y direction if the platform pushes the ball away from center.
DEBUG_DIRECTION_TEST = True

# Loop/debug tuning. Keep vision debug off during balancing because display rendering slows the response.
CAMERA_HZ = 120
DEBUG_CONTROL = True
DEBUG_INTERVAL_SECONDS = 0.50
DEBUG_VISION = True


# Initialize objects
cam = Camera(debug=DEBUG_VISION)
model = RobotKinematics()
robot = RobotController(model, model.lp, model.l1, model.l2, model.lb)
model.max_theta(CONTROL_HEIGHT)


PID = PIDcontroller(kp, ki, kd, alpha, beta, max_theta=model.maxtheta, conversion="tanh")
last_debug_time = 0.0
pid_was_reset_for_lost_ball = False
stuck_error_previous = None
stuck_boost = 1.0
last_stuck_time = None
prev_ball_x = None
prev_ball_y = None
prev_ball_time = None

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
                reset_stuck_response()
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


def reset_stuck_response():
    global stuck_error_previous, stuck_boost, last_stuck_time
    global prev_ball_x, prev_ball_y, prev_ball_time
    stuck_error_previous = None
    stuck_boost = 1.0
    last_stuck_time = None
    prev_ball_x = None
    prev_ball_y = None
    prev_ball_time = None

def update_robot_pos(robotcontroller, robotkinematics, pidcontroller, x_t, y_t, x, y):
    # x_t, y_t: target position, x, y: current position
    global last_debug_time, stuck_error_previous, stuck_boost, last_stuck_time
    global prev_ball_x, prev_ball_y, prev_ball_time
    error_pixels = math.hypot(x - x_t, y - y_t)
    raw_error_x = x - x_t
    raw_error_y = y - y_t

    now_velocity = time.perf_counter()
    if prev_ball_time is None:
        velocity_x = 0.0
        velocity_y = 0.0
    else:
        dt_velocity = max(0.001, min(now_velocity - prev_ball_time, 0.1))
        velocity_x = (x - prev_ball_x) / dt_velocity
        velocity_y = (y - prev_ball_y) / dt_velocity
        velocity_x = max(-MAX_VELOCITY_PIXELS_PER_SECOND, min(velocity_x, MAX_VELOCITY_PIXELS_PER_SECOND))
        velocity_y = max(-MAX_VELOCITY_PIXELS_PER_SECOND, min(velocity_y, MAX_VELOCITY_PIXELS_PER_SECOND))

    prev_ball_x = x
    prev_ball_y = y
    prev_ball_time = now_velocity

    if USE_DIRECT_ERROR_CONTROL:
        # Direct mapping from camera error to platform command.
        # raw_error_x > 0 means the ball is right of center in the camera view.
        # raw_error_y > 0 means the ball is below center in the camera view.
        command_y = DIRECT_LR_SIGN * raw_error_x * DIRECT_X_TO_LR_GAIN
        command_x = DIRECT_UD_SIGN * raw_error_y * DIRECT_Y_TO_UD_GAIN

        if VELOCITY_DAMPING_ENABLED:
            # Brake the ball's velocity so it does not shoot through the center. .
            # This intentionally opposes velocity, not position error.
            command_y -= DIRECT_LR_SIGN * velocity_x * VELOCITY_X_TO_LR_GAIN
            command_x -= DIRECT_UD_SIGN * velocity_y * VELOCITY_Y_TO_UD_GAIN

        bottom_right_corner_active = (
            raw_error_x >= CORNER_ERROR_THRESHOLD
            and raw_error_y >= CORNER_ERROR_THRESHOLD
        )
    else:
        # Camera axes are swapped here on purpose because of the mounted camera direction.
        # Recheck this if the platform tilts on the wrong axis.
        theta, phi = pidcontroller.pid((y_t, x_t), (y, x))
        command_x = math.cos(math.radians(phi)) * theta
        command_y = math.sin(math.radians(phi)) * theta
        if INVERT_X_RESPONSE:
            command_x *= -1
        if INVERT_Y_RESPONSE:
            command_y *= -1
        bottom_right_corner_active = False

    # Screen horizontal error is handled by the left/right servo pair 4/12.
    # Boost left/right correction and calm the up/down pair so the platform does not
    # waste motion oscillating vertically while the ball is stuck on the side.
    left_right_boost_active = abs(raw_error_x) >= LEFT_RIGHT_ERROR_THRESHOLD and abs(raw_error_x) > abs(raw_error_y)
    if bottom_right_corner_active:
        command_y *= LEFT_RIGHT_PAIR_GAIN * BOTTOM_RIGHT_CORNER_GAIN
        command_x *= UP_DOWN_PAIR_GAIN * BOTTOM_RIGHT_CORNER_GAIN
    elif left_right_boost_active:
        command_y *= LEFT_RIGHT_PAIR_GAIN
        command_x *= UP_DOWN_PAIR_GAIN
    else:
        command_y *= 1.50
        command_x *= 1.00

    if DYNAMIC_TILT_ENABLED:
        now_stuck = time.perf_counter()
        if last_stuck_time is None:
            dt_stuck = 0.0
        else:
            dt_stuck = max(0.0, min(now_stuck - last_stuck_time, 0.2))

        if stuck_error_previous is None or error_pixels <= PIXEL_DEADBAND:
            stuck_boost = 1.0
        else:
            error_improvement = stuck_error_previous - error_pixels
            if error_pixels >= STUCK_ERROR_THRESHOLD and error_improvement < STUCK_IMPROVEMENT_THRESHOLD:
                stuck_boost = min(STUCK_BOOST_MAX, stuck_boost + STUCK_BOOST_RATE * dt_stuck)
            else:
                stuck_boost = max(1.0, stuck_boost - STUCK_BOOST_RATE * dt_stuck)

        command_x *= stuck_boost
        command_y *= stuck_boost
        stuck_error_previous = error_pixels
        last_stuck_time = now_stuck
    else:
        stuck_boost = 1.0

    theta = math.hypot(command_x, command_y) * COMMAND_THETA_GAIN
    if error_pixels <= PIXEL_DEADBAND:
        theta = 0.0
    elif bottom_right_corner_active and theta < CORNER_MIN_THETA:
        theta = CORNER_MIN_THETA
    elif left_right_boost_active and abs(command_y) > abs(command_x) and theta < LEFT_RIGHT_MIN_THETA:
        theta = LEFT_RIGHT_MIN_THETA
    elif theta < MIN_ACTIVE_THETA:
        theta = MIN_ACTIVE_THETA
    # Use MAX_COMMAND_THETA as the main software limit while tuning.
    # robotkinematics.maxtheta can be overly conservative and may cap the tilt too early.
    theta = min(theta, MAX_COMMAND_THETA)
    phi = (math.degrees(math.atan2(command_y, command_x)) + COMMAND_PHI_OFFSET_DEG) % 360
    if left_right_boost_active and not bottom_right_corner_active and abs(command_y) > 2.5 * abs(command_x):
        # Force a clean left/right correction only when the ball is very clearly off to the side,
        # not when it is near center or in a corner and needs a blended correction.
        phi = 90.0 if command_y > 0 else 270.0

    robotcontroller.Goto_N_time_spherical(theta, phi, CONTROL_HEIGHT)

    now = time.perf_counter() #BOMBA
    if DEBUG_CONTROL and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
        last_debug_time = now
        if DEBUG_DIRECTION_TEST:
            print(
                f"ball=({x:.1f},{y:.1f}) target=({x_t:.1f},{y_t:.1f}) "
                f"err_px=({raw_error_x:.1f},{raw_error_y:.1f}) mag={error_pixels:.1f} "
                f"vel=({velocity_x:.1f},{velocity_y:.1f}) "
                f"cmd_xy=({command_x:.2f},{command_y:.2f}) theta={theta:.2f} phi={phi:.1f} "
                f"maxtheta={robotkinematics.maxtheta:.2f} stuckboost={stuck_boost:.2f} "
                f"lrboost={left_right_boost_active} corner={bottom_right_corner_active} "
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