"""Utility script to verify servo channels and camera capture."""

from controller import RobotController


def verify_servos():
    print("Servo mapping:", RobotController.SERVO_CHANNELS)
    try:
        from adafruit_servokit import ServoKit
    except Exception as exc:
        print("Cannot import ServoKit:", exc)
        return

    kit = ServoKit(channels=16)
    for channel in RobotController.SERVO_CHANNELS:
        servo = kit.servo[channel]
        print(f"Channel {channel}: {'OK' if servo is not None else 'MISSING'}")


def verify_camera():
    try:
        from camera import Camera
    except Exception as exc:
        print("Cannot import camera module:", exc)
        return

    try:
        cam = Camera()
        frame = cam.take_picture()
        print("Camera capture OK:", frame.shape, frame.dtype)
    except Exception as exc:
        print("Camera capture failed:", exc)
    finally:
        if 'cam' in locals():
            cam.terminate()


if __name__ == '__main__':
    print("===== Servo Channel Check =====")
    verify_servos()
    print("\n===== Camera Check =====")
    verify_camera()
