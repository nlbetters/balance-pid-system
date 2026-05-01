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

kp = 0.06
ki = 0.005
kd = 0.015

alpha = 0.1
beta = 2.0

CONTROL_HEIGHT = 7.5
COMMAND_THETA_GAIN = 3.0
MAX_COMMAND_THETA = 18.0
MIN_ACTIVE_THETA = 0.2
PIXEL_DEADBAND = 2.0
COMMAND_PHI_OFFSET_DEG = 0.0
INVERT_X_RESPONSE = True
INVERT_Y_RESPONSE = True
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

# Initialize ball position at the camera target.
x, y = cam.frame_center

def capture():

    global latest_frame
    while True:
        frame = cam.take_picture()
        with lock:
            latest_frame = frame 

def process():
    hz = 120
    global latest_frame, x, y
    while True:
        with lock:
            if latest_frame is None:
                continue 
            frame_copy = latest_frame.copy()
        
        loop_start = time.perf_counter()
        center, offset, found, confidence, fps, last_valid = cam.coordinate_with_offset(frame_copy)
        x_t, y_t = cam.frame_center  # Target position

        if found or cam.last_valid_position is not None:
            x, y = center
        else:
            # No trustworthy ball yet: command the neutral target position.
            x, y = x_t, y_t

        update_robot_pos(robot, model, PID, x_t, y_t, x, y)
        if DEBUG_VISION:
            cam.display_debug()
        #print(f"Coordinates: {x, y}")
        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / hz) - elapsed
        if sleep_time > 0:
            #print(sleep_time)
            time.sleep(sleep_time)

def update_robot_pos(robotcontroller, robotkinematics, pidcontroller, x_t, y_t, x, y): #x_t, y_t: target position, x, y: current position, t: duration 

    global last_debug_time
    error_pixels = math.hypot(x - x_t, y - y_t)
    theta, phi = pidcontroller.pid((y_t, x_t), (y, x))
    command_x = math.cos(math.radians(phi)) * theta
    command_y = math.sin(math.radians(phi)) * theta
    if INVERT_X_RESPONSE:
        command_x *= -1
    if INVERT_Y_RESPONSE:
        command_y *= -1

    theta = math.hypot(command_x, command_y) * COMMAND_THETA_GAIN
    if error_pixels <= PIXEL_DEADBAND:
        theta = 0.0
    elif theta < MIN_ACTIVE_THETA:
        theta = MIN_ACTIVE_THETA
    theta = min(theta, robotkinematics.maxtheta, MAX_COMMAND_THETA)
    phi = (math.degrees(math.atan2(command_y, command_x)) + COMMAND_PHI_OFFSET_DEG) % 360

    robotcontroller.Goto_N_time_spherical(theta, phi, CONTROL_HEIGHT)

    now = time.perf_counter()
    if DEBUG_CONTROL and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
        last_debug_time = now
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
