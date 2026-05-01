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
MIN_COLOR_CONFIDENCE = 0.45

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
    color_confidence: float
    circularity: float
    fill_ratio: float
    edge_distance: float
    rejection_reason: str
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
        self.debug_detections = []
        self._debug_frame_index = 0

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
        detection, mask, debug_detections = self._detect_ball(image)
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
            self._build_debug_frame(image, mask, detection, debug_detections)

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
        debug_detections = []

        if self.debug:
            self._debug_frame_index += 1
            print(
                f"vision frame {self._debug_frame_index} "
                f"mask_pixels={cv2.countNonZero(mask)} contours={len(contours)}"
            )

        for contour in contours:
            detection = self._score_contour(contour, image.shape, mask)
            if self.debug:
                debug_detections.append(detection)
                self._print_detection_debug(detection)
            if detection.rejection_reason != "accepted":
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection

        if best is not None and best.confidence >= MIN_CONFIDENCE:
            return best, mask, debug_detections
        return None, mask, debug_detections

    def _score_contour(self, contour, image_shape, mask):
        area = cv2.contourArea(contour)
        center = (0, 0)
        radius = 0.0
        circularity = 0.0
        fill_ratio = 0.0
        aspect_ratio = 0.0
        edge_distance = 0.0
        color_confidence = 0.0
        confidence = 0.0

        if area > 0:
            (x_float, y_float), radius = cv2.minEnclosingCircle(contour)
            center = (int(round(x_float)), int(round(y_float)))
            edge_distance = self._edge_distance(center, radius, image_shape)
            color_confidence = self._color_confidence(contour, mask)

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            enclosing_area = np.pi * radius * radius
            fill_ratio = area / enclosing_area if enclosing_area else 0.0

            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h else 0.0
            confidence = self._confidence_score(
                area=area,
                radius=radius,
                circularity=circularity,
                fill_ratio=fill_ratio,
                aspect_ratio=aspect_ratio,
                center=center,
                color_confidence=color_confidence,
            )

        def result(reason):
            return Detection(
                center=center,
                radius=radius,
                area=area,
                confidence=confidence,
                color_confidence=color_confidence,
                circularity=circularity,
                fill_ratio=fill_ratio,
                edge_distance=edge_distance,
                rejection_reason=reason,
                contour=contour,
            )

        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            return result("area")
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            return result("radius")

        if edge_distance < EDGE_MARGIN:
            return result("edge")

        if self.last_valid_position is not None:
            jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
            if jump > MAX_FRAME_JUMP:
                return result("jump")

        if circularity < MIN_CIRCULARITY:
            return result("circularity")
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return result("aspect")
        if fill_ratio < MIN_FILL_RATIO or fill_ratio > MAX_FILL_RATIO:
            return result("fill")
        if color_confidence < MIN_COLOR_CONFIDENCE:
            return result("color")
        if confidence < MIN_CONFIDENCE:
            return result("confidence")

        return result("accepted")

    def _confidence_score(self, area, radius, circularity, fill_ratio, aspect_ratio, center, color_confidence):
        circularity_score = np.clip((circularity - MIN_CIRCULARITY) / 0.35, 0.0, 1.0)
        fill_score = np.clip(1.0 - abs(fill_ratio - 0.78) / 0.35, 0.0, 1.0)
        aspect_score = np.clip(1.0 - abs(1.0 - aspect_ratio) / 0.30, 0.0, 1.0)

        motion_score = 1.0
        if self.last_valid_position is not None:
            jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
            motion_score = np.clip(1.0 - jump / MAX_FRAME_JUMP, 0.0, 1.0)

        radius_score = 1.0 if MIN_RADIUS <= radius <= MAX_RADIUS else 0.0
        area_score = 1.0 if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA else 0.0
        return float(
            0.24 * circularity_score
            + 0.20 * fill_score
            + 0.16 * aspect_score
            + 0.16 * color_confidence
            + 0.12 * motion_score
            + 0.07 * radius_score
            + 0.05 * area_score
        )

    def _edge_distance(self, center, radius, image_shape):
        height, width = image_shape[:2]
        x, y = center
        return float(min(x - radius, y - radius, width - x - radius, height - y - radius))

    @staticmethod
    def _color_confidence(contour, mask):
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_pixels = cv2.countNonZero(contour_mask)
        if contour_pixels == 0:
            return 0.0
        matched_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=contour_mask))
        return float(matched_pixels / contour_pixels)

    @staticmethod
    def _print_detection_debug(detection):
        print(
            "vision contour "
            f"center={detection.center} "
            f"area={detection.area:.1f} "
            f"radius={detection.radius:.1f} "
            f"circularity={detection.circularity:.2f} "
            f"fill={detection.fill_ratio:.2f} "
            f"edge={detection.edge_distance:.1f} "
            f"color={detection.color_confidence:.2f} "
            f"confidence={detection.confidence:.2f} "
            f"reason={detection.rejection_reason}"
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

    def _build_debug_frame(self, image, mask, detection, debug_detections):
        debug = image.copy()
        for item in debug_detections:
            accepted = item.rejection_reason == "accepted"
            color = (0, 255, 0) if accepted else (255, 0, 0)
            cv2.drawContours(debug, [item.contour], -1, color, 1)
            if item.radius > 0:
                cv2.circle(debug, item.center, int(item.radius), color, 1)
            label = f"{item.rejection_reason} {item.confidence:.2f}"
            cv2.putText(
                debug,
                label,
                (max(0, item.center[0] - 24), max(10, item.center[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                color,
                1,
                cv2.LINE_AA,
            )
        if detection is not None:
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
        self.debug_detections = debug_detections

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
