import time
from dataclasses import dataclass

import cv2
import numpy as np
from picamera2 import Picamera2


# Keep this at 200 x 150 unless the PID target in main.py is also changed.
FRAME_SIZE = (200, 150)
CAMERA_FORMAT = "RGB888"
FRAME_DURATION_US = 8333  # about 120 FPS if exposure/light allow it

# Tune these first. White ping-pong balls are usually low saturation/high value.
# The orange range is included for orange balls; remove it if it causes clutter.
BALL_HSV_RANGES = (
    (np.array([0, 0, 145]), np.array([180, 70, 255])),    # white/light ball
    (np.array([5, 80, 80]), np.array([28, 255, 255])),    # orange ball
)

# Ball size limits in the 200 x 150 tracking image.
MIN_CONTOUR_AREA = 40
MAX_CONTOUR_AREA = 2600
MIN_RADIUS = 4
MAX_RADIUS = 35
EDGE_MARGIN = 3

# Shape/mask quality checks.
MIN_CIRCULARITY = 0.62
MIN_FILL_RATIO = 0.52
MAX_FILL_RATIO = 1.18
MIN_ASPECT_RATIO = 0.70
MAX_ASPECT_RATIO = 1.30
MIN_CONFIDENCE = 0.68

# Tracking checks and PID-friendly smoothing.
MAX_FRAME_JUMP = 55
SMOOTHING_ALPHA = 0.35
MAX_MISSED_FRAMES = 4

# Mask cleanup. Blur is intentionally off by default because it costs time.
USE_GAUSSIAN_BLUR = False
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 1


@dataclass
class Detection:
    center: tuple[int, int]
    radius: float
    area: float
    confidence: float
    circularity: float
    fill_ratio: float
    contour: np.ndarray


class Camera:
    def __init__(self, resolution=FRAME_SIZE, format=CAMERA_FORMAT, debug=False):
        self.resolution = resolution
        self.debug = debug
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": format},
            buffer_count=2,
            controls={"FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US)},
        )
        self.picam2.configure(config)

        self.frame_center = (resolution[0] // 2, resolution[1] // 2)
        self.last_valid_position = None
        self.last_center = self.frame_center
        self.last_offset = (0, 0)
        self.ball_found = False
        self.confidence = 0.0
        self.missed_frames = 0

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        self._last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.debug_frame = None
        self.debug_mask = None

        self.picam2.start()

    def take_picture(self):
        return self.picam2.capture_array()

    def display(self, image, window_name="Camera Output"):
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def display_draw(self, image, center=None, window_name="Tracked Output"):
        if center is None:
            center = self.last_center
        output = image.copy()
        if self.ball_found:
            self._draw_crosshair(output, center, (0, 255, 0))
        else:
            self._draw_crosshair(output, center, (255, 0, 0))
        cv2.imshow(window_name, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def display_debug(self, image=None, window_name="Ball Tracking Debug"):
        if self.debug_frame is None:
            return
        frame = self.debug_frame if image is None else image
        mask_rgb = cv2.cvtColor(self.debug_mask, cv2.COLOR_GRAY2RGB)
        debug = np.hstack((frame, mask_rgb))
        cv2.imshow(window_name, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def terminate(self):
        self.picam2.stop()
        self.picam2.close()
        cv2.destroyAllWindows()

    def coordinate(self, image):
        detection, mask, rejected = self._detect_ball(image)
        self._update_timing()

        if detection is not None:
            smoothed = self._smooth_position(detection.center)
            self.last_center = smoothed
            self.last_valid_position = smoothed
            self.last_offset = self.offset_from_center(smoothed)
            self.ball_found = True
            self.confidence = detection.confidence
            self.missed_frames = 0
        else:
            self.ball_found = False
            self.confidence = 0.0
            self.missed_frames += 1
            # Keep the last valid position briefly so the PID loop never sees noise.
            if self.missed_frames > MAX_MISSED_FRAMES:
                self.last_valid_position = None

        if self.debug:
            self._build_debug_frame(image, mask, detection, rejected)

        return self.last_center

    def coordinate_with_offset(self, image):
        center = self.coordinate(image)
        return center, self.last_offset, self.ball_found, self.confidence

    def offset_from_center(self, center=None):
        if center is None:
            center = self.last_center
        return (center[0] - self.frame_center[0], center[1] - self.frame_center[1])

    def _detect_ball(self, image):
        source = image
        if USE_GAUSSIAN_BLUR:
            source = cv2.GaussianBlur(source, (3, 3), 0)

        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in BALL_HSV_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        if MORPH_OPEN_ITERATIONS:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self._kernel, iterations=MORPH_OPEN_ITERATIONS
            )
        if MORPH_CLOSE_ITERATIONS:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self._kernel, iterations=MORPH_CLOSE_ITERATIONS
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        rejected = []

        for contour in contours:
            detection, reason = self._score_contour(contour, image.shape)
            if detection is None:
                if self.debug:
                    rejected.append((contour, reason))
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection

        if best is not None and best.confidence >= MIN_CONFIDENCE:
            return best, mask, rejected
        return None, mask, rejected

    def _score_contour(self, contour, image_shape):
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            return None, "area"

        (x_float, y_float), radius = cv2.minEnclosingCircle(contour)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            return None, "radius"

        height, width = image_shape[:2]
        x, y = int(round(x_float)), int(round(y_float))
        if (
            x - radius < EDGE_MARGIN
            or x + radius > width - EDGE_MARGIN
            or y - radius < EDGE_MARGIN
            or y + radius > height - EDGE_MARGIN
        ):
            return None, "edge"

        if self.last_valid_position is not None:
            jump = np.hypot(x - self.last_valid_position[0], y - self.last_valid_position[1])
            if jump > MAX_FRAME_JUMP:
                return None, "jump"

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None, "perimeter"
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            return None, "circularity"

        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h else 0.0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return None, "aspect"

        enclosing_area = np.pi * radius * radius
        fill_ratio = area / enclosing_area if enclosing_area else 0.0
        if fill_ratio < MIN_FILL_RATIO or fill_ratio > MAX_FILL_RATIO:
            return None, "fill"

        confidence = self._confidence_score(
            area=area,
            radius=radius,
            circularity=circularity,
            fill_ratio=fill_ratio,
            aspect_ratio=aspect_ratio,
            center=(x, y),
        )
        return Detection(
            center=(x, y),
            radius=radius,
            area=area,
            confidence=confidence,
            circularity=circularity,
            fill_ratio=fill_ratio,
            contour=contour,
        ), "accepted"

    def _confidence_score(self, area, radius, circularity, fill_ratio, aspect_ratio, center):
        circularity_score = np.clip((circularity - MIN_CIRCULARITY) / 0.35, 0.0, 1.0)
        fill_score = np.clip(1.0 - abs(fill_ratio - 0.78) / 0.35, 0.0, 1.0)
        aspect_score = np.clip(1.0 - abs(1.0 - aspect_ratio) / 0.30, 0.0, 1.0)

        radius_mid = (MIN_RADIUS + MAX_RADIUS) / 2.0
        radius_span = (MAX_RADIUS - MIN_RADIUS) / 2.0
        radius_score = np.clip(1.0 - abs(radius - radius_mid) / radius_span, 0.0, 1.0)

        motion_score = 1.0
        if self.last_valid_position is not None:
            jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
            motion_score = np.clip(1.0 - jump / MAX_FRAME_JUMP, 0.0, 1.0)

        # Area is a light tie-breaker: shape and motion matter more than size.
        area_score = np.clip((area - MIN_CONTOUR_AREA) / (MAX_CONTOUR_AREA - MIN_CONTOUR_AREA), 0.0, 1.0)
        return float(
            0.30 * circularity_score
            + 0.22 * fill_score
            + 0.18 * aspect_score
            + 0.15 * motion_score
            + 0.10 * radius_score
            + 0.05 * area_score
        )

    def _smooth_position(self, center):
        if self.last_valid_position is None:
            return (int(center[0]), int(center[1]))

        x = (SMOOTHING_ALPHA * center[0]) + ((1.0 - SMOOTHING_ALPHA) * self.last_valid_position[0])
        y = (SMOOTHING_ALPHA * center[1]) + ((1.0 - SMOOTHING_ALPHA) * self.last_valid_position[1])
        return (int(round(x)), int(round(y)))

    def _update_timing(self):
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = instant_fps if self.fps == 0.0 else (0.85 * self.fps + 0.15 * instant_fps)

    def _build_debug_frame(self, image, mask, detection, rejected):
        debug = image.copy()
        cv2.drawContours(debug, [item[0] for item in rejected], -1, (255, 0, 0), 1)
        if detection is not None:
            cv2.circle(debug, detection.center, int(detection.radius), (0, 255, 0), 2)
            self._draw_crosshair(debug, detection.center, (0, 255, 0))
        self._draw_crosshair(debug, self.frame_center, (255, 255, 0))
        cv2.putText(
            debug,
            f"fps {self.fps:4.1f} conf {self.confidence:.2f}",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0) if self.ball_found else (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        self.debug_frame = debug
        self.debug_mask = mask

    @staticmethod
    def _draw_crosshair(image, center, color):
        x, y = int(center[0]), int(center[1])
        cv2.line(image, (x - 6, y), (x + 6, y), color, 1)
        cv2.line(image, (x, y - 6), (x, y + 6), color, 1)


if __name__ == "__main__":
    cam = Camera(debug=True)

    try:
        while True:
            img = cam.take_picture()
            center, offset, found, confidence = cam.coordinate_with_offset(img)
            cam.display_debug()
            print(f"center={center} offset={offset} found={found} confidence={confidence:.2f}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.terminate()
