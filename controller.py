import time
import math
import random
from robotKinematics import RobotKinematics

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_servokit import ServoKit
except ImportError:
    board = None
    busio = None
    PCA9685 = None
    ServoKit = None

MIN_SERVO_ANGLE = 0
MAX_SERVO_ANGLE = 100
DEFAULT_SERVO_ANGLE = 45
SERVO_OFFSETS = [0, 0, 0, 0]
SERVO_DIRECTIONS = [1, 1, 1, 1]
SERVO_DIRECTION_PIVOT = DEFAULT_SERVO_ANGLE

# Servo pair layout:
# channel 0 is opposite channel 8, and channel 4 is opposite channel 12.
# Leave SERVO_DIRECTIONS as [1, 1, 1, 1] unless a physical servo is mounted backward.
MAX_SERVO_STEP_PER_COMMAND = 6.0

def clamp(value, lower=MIN_SERVO_ANGLE, upper=MAX_SERVO_ANGLE):
    return max(lower, min(value, upper))

def apply_servo_calibration(angle, servo_index):
    directed_angle = SERVO_DIRECTION_PIVOT + (
        SERVO_DIRECTIONS[servo_index] * (angle - SERVO_DIRECTION_PIVOT)
    )
    return clamp(directed_angle + SERVO_OFFSETS[servo_index])

def reference_kinematic_theta(robot, h):
    ax, az = robot.lp, h
    bx, bz = robot.lb, 0.0
    dx = bx - ax
    dz = bz - az
    d = math.hypot(dx, dz)
    a = (robot.l1**2 - robot.l2**2 + d**2) / (2 * d)
    leg_height = math.sqrt(max(0.0, robot.l1**2 - a**2))
    p2x = ax + a * dx / d
    p2z = az + a * dz / d
    perp_x = -dz / d
    perp_z = dx / d
    c1 = [p2x + leg_height * perp_x, p2z + leg_height * perp_z]
    c2 = [p2x - leg_height * perp_x, p2z - leg_height * perp_z]
    if robot.invert:
        cx, cz = min((c1, c2), key=lambda c: c[0])
    else:
        cx, cz = max((c1, c2), key=lambda c: c[0])
    return math.degrees(math.pi / 2 - math.atan2(cx - robot.lb, cz))

def servo_angles_from_kinematics(robot):
    neutral_theta = reference_kinematic_theta(robot, robot.h)
    direction = 1 if robot.invert else -1
    raw_angles = [
        DEFAULT_SERVO_ANGLE + direction * (math.degrees(robot.theta1) - neutral_theta),
        DEFAULT_SERVO_ANGLE + direction * (math.degrees(robot.theta2) - neutral_theta),
        DEFAULT_SERVO_ANGLE + direction * (math.degrees(robot.theta3) - neutral_theta),
        DEFAULT_SERVO_ANGLE + direction * (math.degrees(robot.theta4) - neutral_theta)
    ]
    return [clamp(angle) for angle in raw_angles]

class RobotController:
    SERVO_CHANNELS = [0, 4, 8, 12]

    def __init__(self, model, lp=7.125, l1=6.20, l2=4.50, lb=4.00, servo_channels=None):
        self.robot = model
        self.servo_channels = servo_channels or self.SERVO_CHANNELS

        # Initialize the ServoKit and assign servos
        self.Controller = ServoKit(channels=16)
        self.s1 = self.Controller.servo[self.servo_channels[0]]
        self.s2 = self.Controller.servo[self.servo_channels[1]]
        self.s3 = self.Controller.servo[self.servo_channels[2]]
        self.s4 = self.Controller.servo[self.servo_channels[3]]
        self.last_commanded_angles = [DEFAULT_SERVO_ANGLE] * 4

        # Configure servos
        for s in (self.s1, self.s2, self.s3, self.s4):
            s.actuation_range = 270
            s.set_pulse_width_range(500, 2500)

        self.initialize()

    def initialize(self):
        print("Initializing ...")
        neutral_angles = [DEFAULT_SERVO_ANGLE] * 4
        self.last_commanded_angles = neutral_angles[:]
        self.set_motor_angles(*neutral_angles, rate_limit=False)
        self.interpolate_time(neutral_angles, duration=0.25)
        time.sleep(1)
        print("Initialized!")
    
    def set_motor_angles(self, theta1, theta2, theta3, theta4=None, rate_limit=True):
        commanded = [theta1, theta2, theta3, self.last_commanded_angles[3] if theta4 is None else theta4]
        commanded = [clamp(angle) for angle in commanded]

        if rate_limit:
            limited = []
            for previous, target in zip(self.last_commanded_angles, commanded):
                delta = target - previous
                delta = max(-MAX_SERVO_STEP_PER_COMMAND, min(delta, MAX_SERVO_STEP_PER_COMMAND))
                limited.append(previous + delta)
            commanded = limited

        self.last_commanded_angles = commanded[:]
        self.s1.angle = apply_servo_calibration(commanded[0], 0)
        self.s2.angle = apply_servo_calibration(commanded[1], 1)
        self.s3.angle = apply_servo_calibration(commanded[2], 2)
        self.s4.angle = apply_servo_calibration(commanded[3], 3)

    def get_motor_angles(self):
        return [
            getattr(self.s1, "angle", None),
            getattr(self.s2, "angle", None),
            getattr(self.s3, "angle", None),
            getattr(self.s4, "angle", None)
        ]

    def interpolate_time(self, target_angles, steps=100, duration=0.3, individual_durations=None):
        current_angles = [self.s1.angle, self.s2.angle, self.s3.angle, self.s4.angle]
        target_angles = list(target_angles)
        if len(target_angles) == 3:
            target_angles.append(self.s4.angle)
        if len(target_angles) != 4:
            raise ValueError("target_angles must be length 3 or 4")
        if individual_durations is None:
            individual_durations = [duration] * 4
        max_duration = max(individual_durations)
        steps = max(1, int(max_duration / 0.01))
        for i in range(steps + 1):
            t = i * max_duration / steps
            angles = [
                c + (t_angle - c) * min(t / d, 1) if d > 0 else t_angle
                for c, t_angle, d in zip(current_angles, target_angles, individual_durations)
            ]
            self.set_motor_angles(*angles, rate_limit=False)
            time.sleep(max_duration / steps)

    def interpolate_speed(self, target_angles, speed=30, individual_speeds=None):
        current_angles = [self.s1.angle, self.s2.angle, self.s3.angle, self.s4.angle]
        target_angles = list(target_angles)
        if len(target_angles) == 3:
            target_angles.append(self.s4.angle)
        if len(target_angles) != 4:
            raise ValueError("target_angles must be length 3 or 4")
        if individual_speeds is None:
            individual_speeds = [speed] * 4
        durations = [
            abs(t - c) / s if s > 0 else 0
            for c, t, s in zip(current_angles, target_angles, individual_speeds)
        ]
        max_duration = max(durations)
        steps = max(1, int(max_duration / 0.01))
        for i in range(steps + 1):
            t = i * max_duration / steps
            angles = [
                c + (t_angle - c) * min(t / d, 1) if d > 0 else t_angle
                for c, t_angle, d in zip(current_angles, target_angles, durations)
            ]
            self.set_motor_angles(*angles, rate_limit=False)
            time.sleep(max_duration / steps)

    def Goto_time_spherical(self, theta, phi, h, t=0.5):
        self.robot.solve_inverse_kinematics_spherical(theta, phi, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.interpolate_time(target_angles, duration=t)

    def Goto_time_vector(self, a, b, c, h, t=0.5):
        self.robot.solve_inverse_kinematics_vector(a, b, c, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.interpolate_time(target_angles, duration=t)

    def Goto_N_time_vector(self, a, b, c, h):
        self.robot.solve_inverse_kinematics_vector(a, b, c, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.set_motor_angles(*target_angles, rate_limit=True)
    
    def Goto_N_time_spherical(self, theta, phi, h):
        self.robot.solve_inverse_kinematics_spherical(theta, phi, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.set_motor_angles(*target_angles, rate_limit=True)

    def return_to_neutral(self, h=None, rate_limit=True):
        if h is None:
            h = self.robot.h
        self.robot.solve_inverse_kinematics_spherical(0.0, 0.0, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.set_motor_angles(*target_angles, rate_limit=rate_limit)

    def Dance1(self):
        self.Goto_time_vector(0.258819045103, 0, 0.965925826289, 8)
        for _ in range(3):
            for i in range(100):
                t = (2 * math.pi / 100) * i
                x = math.cos(math.pi * 5 / 12) * math.cos(t)
                y = math.cos(math.pi * 5 / 12) * math.sin(t)
                z = math.sin(math.pi * 5 / 12)
                print(x, y, z, math.sqrt(x**2 + y**2 + z**2))
                self.Goto_N_time_vector(x, y, z, 8)
                time.sleep(1/100)
        self.Goto_time_vector(0, 0, 1, 8)


if __name__ == "__main__":
    model = RobotKinematics()
    rc = RobotController(model)
    time.sleep(0.5)
