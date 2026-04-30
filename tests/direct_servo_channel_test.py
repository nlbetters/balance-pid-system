import sys
import time

try:
    from adafruit_servokit import ServoKit
except Exception as exc:
    print("Failed to import ServoKit:", exc)
    raise


CHANNELS = [0, 4, 8, 12]
ANGLES = [20, 45, 70]
RUNTIME_SECONDS = 2


if __name__ == "__main__":
    channels = [int(arg) for arg in sys.argv[1:]] or CHANNELS
    kit = ServoKit(channels=16)

    for channel in channels:
        servo = kit.servo[channel]
        servo.actuation_range = 270
        servo.set_pulse_width_range(500, 2500)

        print(f"\nDirectly sweeping PCA9685 channel {channel}")
        for angle in ANGLES:
            servo.angle = angle
            print(f"channel {channel}: commanded {angle}, reported {servo.angle}")
            time.sleep(RUNTIME_SECONDS)

    print("Direct channel test finished.")
