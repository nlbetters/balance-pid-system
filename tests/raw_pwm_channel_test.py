import sys
import time

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
except Exception as exc:
    print("Failed to import PCA9685 dependencies:", exc)
    raise


CHANNELS = [0, 4, 8, 12]
PULSE_WIDTHS_US = [500, 870, 1240]
RUNTIME_SECONDS = 2
PWM_FREQUENCY_HZ = 50
PERIOD_US = 1_000_000 / PWM_FREQUENCY_HZ


def pulse_width_to_duty_cycle(pulse_width_us):
    return int((pulse_width_us / PERIOD_US) * 0xFFFF)


if __name__ == "__main__":
    channels = [int(arg) for arg in sys.argv[1:]] or CHANNELS

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = PWM_FREQUENCY_HZ

    try:
        for channel in channels:
            print(f"\nRaw PWM sweep on PCA9685 channel {channel}")
            for pulse_width_us in PULSE_WIDTHS_US:
                duty_cycle = pulse_width_to_duty_cycle(pulse_width_us)
                pca.channels[channel].duty_cycle = duty_cycle
                print(
                    f"channel {channel}: pulse={pulse_width_us}us "
                    f"duty_cycle={duty_cycle}"
                )
                time.sleep(RUNTIME_SECONDS)
    finally:
        for channel in channels:
            pca.channels[channel].duty_cycle = 0
        pca.deinit()

    print("Raw PWM channel test finished.")
