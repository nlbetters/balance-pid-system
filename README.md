# Ball-Balancing Robot

This project is a camera-based ball-balancing robot that uses computer vision, inverse kinematics, and feedback control to keep a ball centered on a moving platform. The system tracks the ball with a camera, calculates how far the ball is from the center target, and moves four servo motors to tilt the platform and push the ball back toward the middle.

The main goal of the project is to create a real-time closed-loop control system. Instead of manually controlling the platform, the robot continuously watches the ball position and automatically adjusts the platform angle. This makes the project a practical demonstration of robotics concepts like sensing, actuation, kinematics, and PID-style feedback control.

## Project Overview

The robot uses a 4-arm servo-driven platform. Each servo controls one side of the platform through a linkage arm. By raising and lowering opposite sides of the platform, the robot can tilt the surface in different directions and move the ball toward the center.

This version uses servo channels `0`, `4`, `8`, and `12` on a PCA9685 servo driver:

| Servo Channel | Platform Position |
|---|---|
| `0` | Opposite servo `8` |
| `8` | Opposite servo `0` |
| `4` | Opposite servo `12` |
| `12` | Opposite servo `4` |

In the camera view, servos `4` and `12` control the left/right response, while servos `0` and `8` control the up/down response.

## How It Works

The system works as a loop:

1. The camera captures a frame of the platform.
2. The vision code identifies the ball and calculates its `(x, y)` position.
3. The program compares the ball position to the center target.
4. The controller calculates the direction and amount of tilt needed.
5. The inverse kinematics code converts the desired platform tilt into servo angles.
6. The servo controller sends angle commands to the PCA9685.
7. The platform tilts and moves the ball back toward the center.
8. The loop repeats continuously.

If the camera does not confidently detect the ball, the controller resets and sends the platform back to its neutral/default position. This prevents the robot from chasing random camera noise.

## Main Files

| File | Purpose |
|---|---|
| `main.py` | Runs the main camera/control loop and connects all components together. |
| `camera.py` | Handles ball detection, filtering, confidence scoring, and camera debug view. |
| `PID.py` | Contains the PID-style controller logic and output smoothing. |
| `controller.py` | Sends servo commands and maps inverse kinematics output to motor angles. |
| `robotKinematics.py` | Calculates the 4-arm platform inverse kinematics. |
| `GUI.py` | Optional visualization/debug interface. |
| `app.py` | Optional app/server interface depending on the setup. |

## Camera Tracking

The camera system is designed to identify the ball and reject background noise. The current setup is tuned to track a green ball because it is easier to distinguish from the platform and background than a white or gold ball.

The camera pipeline uses:

- HSV color filtering
- platform region masking
- contour detection
- area and radius limits
- confidence scoring
- frame-to-frame consistency checks
- rejection of small noisy blobs

When the ball is found, `camera.py` returns its center position. When the ball is not found, it returns `found=False`, and the robot returns to neutral.

## Control System

The controller compares the ball position to the center of the camera frame. The error is used to calculate how much the platform should tilt.

The current control loop uses direct camera-error control while testing:

- screen left/right error maps to servo pair `4` and `12`
- screen up/down error maps to servo pair `0` and `8`

This direct mapping makes it easier to tune the robot response before moving to a more advanced PID control setup.

The important tuning constants are in `main.py`:

```python
COMMAND_THETA_GAIN
MAX_COMMAND_THETA
MIN_ACTIVE_THETA
DIRECT_X_TO_LR_GAIN
DIRECT_Y_TO_UD_GAIN
LEFT_RIGHT_PAIR_GAIN
UP_DOWN_PAIR_GAIN
```

If the platform does not move enough, increase the gain values slowly. If the ball overshoots or oscillates, lower the gains or increase damping in `PID.py`.

## Servo Control

The servo controller uses a neutral operating angle around `45` degrees. The servo range is limited between `0` and `100` degrees for safety and consistency.

Opposite motors should move against each other. For example, when servo `4` increases, servo `12` should generally decrease to create a tilt across that axis.

The servo channels are defined in `controller.py`:

```python
SERVO_CHANNELS = [0, 4, 8, 12]
```

The servo response can be adjusted using:

```python
SERVO_OFFSETS
SERVO_DIRECTIONS
MAX_SERVO_STEP_PER_COMMAND
```

`SERVO_OFFSETS` can be used to level the platform if one side sits higher or lower than the others. `SERVO_DIRECTIONS` can be changed if a servo is physically mounted backward.

## Setup

### Install Dependencies

On the Raspberry Pi, use a virtual environment or install the required packages system-wide depending on your setup.

Typical dependencies include:

```bash
pip install flask Flask-Cors numpy opencv-python picamera2 adafruit-circuitpython-servokit adafruit-circuitpython-pca9685 adafruit-blinka pyqt5 pyqtgraph
```

If the Pi gives an externally managed environment error, create and activate a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Hardware

Main hardware used:

- Raspberry Pi
- Pi camera or compatible camera
- PCA9685 servo driver
- 4 servo motors
- external 5V servo power supply
- 4-arm ball-balancing platform
- green tracking ball

The servos should be powered from an external supply, not directly from the Raspberry Pi. The Pi and the servo power supply should share a common ground.

## Running the Project

From the project directory, run:

```bash
python3 main.py
```

The robot will initialize the servos to the neutral position, start the camera loop, and begin tracking the ball. If the ball is detected, the platform will move to push it toward the center. If the ball is not detected, the platform returns to neutral.

## Calibration

### Motor Angle Calibration

Each of the four motors must be calibrated so the platform is level at the neutral position.

1. Set all motor offsets to `0` in `controller.py`.
2. Run the controller and let the servos move to neutral.
3. Check whether the platform is level.
4. Adjust `SERVO_OFFSETS` for any motor that is too high or too low.
5. Repeat until the platform is flat.

Example:

```python
SERVO_OFFSETS = [0, 0, 0, 0]
```

Increase or decrease individual offsets until the platform sits level.

### Camera Calibration

The camera should be mounted above the platform so the center of the image matches the center target on the platform. The green ball should be clearly visible against the platform.

Useful camera tuning values are in `camera.py`:

```python
BALL_COLOR_PROFILES
PLATFORM_MASK_RADIUS
MIN_CONTOUR_AREA
MIN_RADIUS
MIN_COLOR_CONFIDENCE
MIN_SATURATION_SCORE
```

If the camera detects too much noise, increase the minimum area, radius, color confidence, or saturation score. If the camera loses the ball, lower those values slightly.

### Direction Testing

Before full balancing, manually place the ball on each side of the platform and confirm the robot tilts in the correct direction.

If the platform pushes the ball away from the center instead of toward it, change the signs in `main.py`:

```python
DIRECT_LR_SIGN
DIRECT_UD_SIGN
```

If one servo moves the wrong way, adjust `SERVO_DIRECTIONS` in `controller.py`.

## Verification Tests

Run servo mapping checks with:

```bash
python -m unittest tests.test_servo_channels
```

Run camera checks with:

```bash
python -m unittest tests.test_camera
```

Run a quick hardware smoke test with:

```bash
python verify_system.py
```

## Notes

This project is still tuning-heavy because the real robot depends on camera lighting, ball color, servo mounting, linkage geometry, and platform friction. The most important debugging steps are:

- make sure the camera only tracks the ball
- make sure the platform returns to neutral when the ball is lost
- make sure each servo pair moves in opposite directions
- make sure the platform pushes the ball toward the center, not away from it
- tune gains slowly to avoid oscillation

## Original Reference

This project was based on an educational ball-balancing robot concept and adapted into a 4-arm design using servo channels `0`, `4`, `8`, and `12`.

Video reference:

<div align="center">
  <a href="https://www.youtube.com/watch?v=l92hJUUjWb0&t=6s">
    <img src="https://img.youtube.com/vi/l92hJUUjWb0/0.jpg" alt="Watch the video" width="640">
  </a>
</div>
