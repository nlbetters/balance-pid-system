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
DEFAULT_SERVO_ANGLE = 35

def clamp(value, lower=MIN_SERVO_ANGLE, upper=MAX_SERVO_ANGLE):
    return max(lower, min(value, upper))

def max_height_kinematic_theta(robot):
    ax, az = robot.lp, robot.maxh
    bx, bz = robot.lb, 0.0
    dx = bx - ax
    dz = bz - az
    d = math.hypot(dx, dz)
    a = (robot.l1**2 - robot.l2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0.0, robot.l1**2 - a**2))
    p2x = ax + a * dx / d
    p2z = az + a * dz / d
    perp_x = -dz / d
    perp_z = dx / d
    c1 = [p2x + h * perp_x, p2z + h * perp_z]
    c2 = [p2x - h * perp_x, p2z - h * perp_z]
    cx, cz = max((c1, c2), key=lambda c: c[1])
    return math.degrees(math.pi / 2 - math.atan2(cx - robot.lb, cz))

def servo_angles_from_kinematics(robot):
    zero_height_theta = max_height_kinematic_theta(robot)
    return [
        clamp(math.degrees(robot.theta1) - zero_height_theta),
        clamp(math.degrees(robot.theta2) - zero_height_theta),
        clamp(math.degrees(robot.theta3) - zero_height_theta),
        clamp(math.degrees(robot.theta4) - zero_height_theta)
    ]

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

        # Configure servos
        for s in (self.s1, self.s2, self.s3, self.s4):
            s.actuation_range = 270
            s.set_pulse_width_range(500, 2500)

        self.initialize()

    def initialize(self):
        print("Initializing ...")
        self.set_motor_angles(DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE)
        self.interpolate_time([DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE, DEFAULT_SERVO_ANGLE], duration=0.25)
        time.sleep(1)
        self.Goto_time_spherical(0, 0, 8.26, t=0.25)
        time.sleep(1)
        print("Initialized!")
    
    def set_motor_angles(self, theta1, theta2, theta3, theta4=None):
        self.s1.angle = clamp(theta1 - 4)
        self.s2.angle = clamp(theta2)
        self.s3.angle = clamp(theta3)
        if theta4 is not None:
            self.s4.angle = clamp(theta4)

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
            self.set_motor_angles(*angles)
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
            self.set_motor_angles(*angles)
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
        self.set_motor_angles(*target_angles)
    
    def Goto_N_time_spherical(self, theta, phi, h):
        self.robot.solve_inverse_kinematics_spherical(theta, phi, h)
        target_angles = servo_angles_from_kinematics(self.robot)
        self.set_motor_angles(*target_angles)

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
