import time
from dataclasses import dataclass

import cv2
import numpy as np
from picamera2 import Picamera2


# Keep this at 200 x 150 unless the PID target in main.py is also changed.
FRAME_SIZE = (200, 150)
CAMERA_FORMAT = "RGB888"
FRAME_DURATION_US = 8333  # about 120 FPS if exposure/light allow it

# Ball color profiles. Each profile has one or more HSV ranges.
# These are intentionally broad because the camera exposure changes a lot.
BALL_COLOR_PROFILES = {
    # For real balancing, only track the green ball. White/gold are too close
    # to the platform/background and create false positives.
    "green": [
        # Slightly wider green range so the darker/shadowed parts of the ball still pass.
        (np.array([30, 35, 35]), np.array([95, 255, 255])),
    ],
}
WHITE_LAB_RANGE = (np.array([155, 105, 105]), np.array([255, 165, 165]))

# Platform ROI. Use almost the full frame so the ball is not accidentally masked out.
# A smaller ROI can be used later once the camera is mounted permanently.
PLATFORM_MASK_RADIUS = 92

# Ball size limits in the 200 x 150 tracking image.
MIN_CONTOUR_AREA = 90
MAX_CONTOUR_AREA = 7000
MIN_RADIUS = 5
MAX_RADIUS = 55
EDGE_MARGIN = -8

# Shape/mask quality checks.
MIN_CIRCULARITY = 0.30
MIN_FILL_RATIO = 0.20
MAX_FILL_RATIO = 1.45
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 1.75
MIN_CONFIDENCE = 0.58
MIN_INITIAL_CONFIDENCE = 0.70
MIN_COLOR_CONFIDENCE = 0.35
MIN_BRIGHTNESS_SCORE = 0.18
MIN_SATURATION_SCORE = 0.10

# Strong green blobs are allowed even when the visible contour is not a perfect circle.
# This helps when the ball is partly cut off by the frame/ROI but is still clearly green.
ALLOW_STRONG_COLOR_BLOB = True
STRONG_COLOR_CONFIDENCE = 0.75
STRONG_COLOR_MIN_SATURATION = 0.18
STRONG_COLOR_MIN_AREA = 90

#
# Hough is useful for testing, but it sees platform rings, screws, and shadows as circles.
# Keep it off during real balancing so the tracker only trusts color-mask ball candidates.
USE_HOUGH_FALLBACK = False
RUN_HOUGH_ONLY_IF_NO_MASK_CANDIDATE = True
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 28
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 18
HOUGH_MIN_RADIUS = MIN_RADIUS
HOUGH_MAX_RADIUS = 45
MIN_HOUGH_COLOR_CONFIDENCE = 0.35
MIN_HOUGH_BRIGHTNESS_SCORE = 0.40

# Tracking checks and PID-friendly smoothing.
MAX_FRAME_JUMP = 90
SMOOTHING_ALPHA = 0.35
MAX_MISSED_FRAMES = 4

# A new object must look like the ball for a couple frames before the control loop trusts it.
# This prevents the tracker from immediately choosing a random circle when the ball is gone.
REQUIRED_INITIAL_HITS = 3
INITIAL_MATCH_DISTANCE = 18

# Mask cleanup.
USE_GAUSSIAN_BLUR = True
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2
USE_CLAHE = False


@dataclass
class Detection:
    center: tuple[int, int]
    radius: float
    area: float
    confidence: float
    color_confidence: float
    color_name: str
    circularity: float
    fill_ratio: float
    edge_distance: float
    brightness_score: float
    saturation_score: float
    movement_score: float
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
        self.pending_center = None
        self.pending_hits = 0

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._platform_mask = self._create_platform_mask(resolution)
        self._last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.debug_frame = None
        self.debug_mask = None
        self.debug_detections = []
        self._debug_frame_index = 0

        self.picam2.start()

    def _create_platform_mask(self, resolution):
        mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
        center = (resolution[0] // 2, resolution[1] // 2)
        cv2.circle(mask, center, PLATFORM_MASK_RADIUS, 255, -1)
        return mask

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
            if self.last_valid_position is None:
                if self.pending_center is not None:
                    jump = np.hypot(
                        detection.center[0] - self.pending_center[0],
                        detection.center[1] - self.pending_center[1],
                    )
                    if jump <= INITIAL_MATCH_DISTANCE:
                        self.pending_hits += 1
                    else:
                        self.pending_hits = 1
                else:
                    self.pending_hits = 1

                self.pending_center = detection.center

                if detection.confidence < MIN_INITIAL_CONFIDENCE or self.pending_hits < REQUIRED_INITIAL_HITS:
                    self.ball_found = False
                    self.confidence = 0.0
                    self.missed_frames += 1
                    self.last_center = self.frame_center
                    self.last_offset = (0, 0)
                    if self.missed_frames > MAX_MISSED_FRAMES:
                        self.last_valid_position = None
                    if self.debug:
                        self._build_debug_frame(image, mask, None, debug_detections)
                    return self.last_center

            self.pending_center = None
            self.pending_hits = 0
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
            self.pending_center = None
            self.pending_hits = 0
            self.last_center = self.frame_center
            self.last_offset = (0, 0)
            if self.missed_frames > MAX_MISSED_FRAMES:
                self.last_valid_position = None

        if self.debug:
            self._build_debug_frame(image, mask, detection if self.ball_found else None, debug_detections)
        return self.last_center

    def _apply_clahe(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = self._clahe.apply(v)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

    def _build_ball_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        masks = {}
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_name, ranges in BALL_COLOR_PROFILES.items():
            profile_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                profile_mask = cv2.bitwise_or(profile_mask, cv2.inRange(hsv, lower, upper))
            masks[color_name] = profile_mask
            combined = cv2.bitwise_or(combined, profile_mask)

        # Do not add the broad white LAB mask during real balancing.
        # It tends to include the platform, glare, walls, and other light objects.

        return combined, masks

    def coordinate_with_offset(self, image):
        center = self.coordinate(image)
        return center, self.last_offset, self.ball_found, self.confidence, self.fps, self.last_valid_position

    def offset_from_center(self, center=None):
        if center is None:
            center = self.last_center
        return (center[0] - self.frame_center[0], center[1] - self.frame_center[1])

    def _detect_ball(self, image):
        source = image
        if USE_GAUSSIAN_BLUR:
            source = cv2.GaussianBlur(source, (5, 5), 0)

        if USE_CLAHE:
            source = self._apply_clahe(source)

        mask, profile_masks = self._build_ball_mask(source)
        mask = cv2.bitwise_and(mask, mask, mask=self._platform_mask)
        for name in profile_masks:
            profile_masks[name] = cv2.bitwise_and(profile_masks[name], profile_masks[name], mask=self._platform_mask)

        if MORPH_OPEN_ITERATIONS:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self._kernel, iterations=MORPH_OPEN_ITERATIONS
            )
        if MORPH_CLOSE_ITERATIONS:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self._kernel, iterations=MORPH_CLOSE_ITERATIONS
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        best = None
        debug_detections = []

        if self.debug:
            self._debug_frame_index += 1
            print(
                f"vision frame {self._debug_frame_index} "
                f"mask_pixels={cv2.countNonZero(mask)} contours={len(contours)}"
            )

        for contour in contours:
            detection = self._score_contour(contour, image.shape, mask, profile_masks, source)
            if self.debug:
                debug_detections.append(detection)
                self._print_detection_debug(detection)
            if detection.rejection_reason != "accepted":
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection

        should_run_hough = USE_HOUGH_FALLBACK and (
            best is None or not RUN_HOUGH_ONLY_IF_NO_MASK_CANDIDATE
        )
        if should_run_hough:
            for detection in self._hough_detections(source, image.shape, profile_masks):
                if self.debug:
                    debug_detections.append(detection)
                    self._print_detection_debug(detection)
                if detection.rejection_reason != "accepted":
                    continue
                if best is None or detection.confidence > best.confidence:
                    best = detection

        if best is not None:
            required_confidence = MIN_CONFIDENCE
            if self.last_valid_position is None:
                required_confidence = MIN_INITIAL_CONFIDENCE
            if best.confidence >= required_confidence:
                return best, mask, debug_detections
        return None, mask, debug_detections

    def _score_contour(self, contour, image_shape, mask, profile_masks, image):
        area = cv2.contourArea(contour)
        center = (0, 0)
        radius = 0.0
        circularity = 0.0
        fill_ratio = 0.0
        aspect_ratio = 0.0
        edge_distance = 0.0
        color_confidence = 0.0
        color_name = "none"
        brightness_score = 0.0
        saturation_score = 0.0
        movement_score = 1.0
        confidence = 0.0

        if area > 0:
            (x_float, y_float), radius = cv2.minEnclosingCircle(contour)
            center = (int(round(x_float)), int(round(y_float)))
            edge_distance = self._edge_distance(center, radius, image_shape)
            color_confidence, color_name = self._best_color_confidence(contour, profile_masks)
            brightness_score, saturation_score = self._brightness_saturation_scores(contour, image)

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            enclosing_area = np.pi * radius * radius
            fill_ratio = area / enclosing_area if enclosing_area else 0.0

            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h else 0.0

            if self.last_valid_position is not None:
                jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
                movement_score = np.clip(1.0 - jump / MAX_FRAME_JUMP, 0.0, 1.0)

            confidence = self._confidence_score(
                area=area,
                radius=radius,
                circularity=circularity,
                fill_ratio=fill_ratio,
                aspect_ratio=aspect_ratio,
                center=center,
                color_confidence=color_confidence,
                brightness_score=brightness_score,
                saturation_score=saturation_score,
                movement_score=movement_score,
            )

        def result(reason):
            return Detection(
                center=center,
                radius=radius,
                area=area,
                confidence=confidence,
                color_confidence=color_confidence,
                color_name=color_name,
                circularity=circularity,
                fill_ratio=fill_ratio,
                edge_distance=edge_distance,
                brightness_score=brightness_score,
                saturation_score=saturation_score,
                movement_score=movement_score,
                rejection_reason=reason,
                contour=contour,
            )

        strong_green_blob = (
            ALLOW_STRONG_COLOR_BLOB
            and color_name == "green"
            and color_confidence >= STRONG_COLOR_CONFIDENCE
            and saturation_score >= STRONG_COLOR_MIN_SATURATION
            and area >= STRONG_COLOR_MIN_AREA
            and radius <= MAX_RADIUS
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
        if not strong_green_blob:
            if circularity < MIN_CIRCULARITY:
                return result("circularity")
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                return result("aspect")
            if fill_ratio < MIN_FILL_RATIO or fill_ratio > MAX_FILL_RATIO:
                return result("fill")
        if color_confidence < MIN_COLOR_CONFIDENCE:
            return result("color")
        if brightness_score < MIN_BRIGHTNESS_SCORE:
            return result("brightness")
        if saturation_score < MIN_SATURATION_SCORE:
            return result("saturation")
        if confidence < MIN_CONFIDENCE and not strong_green_blob:
            return result("confidence")

        return result("accepted")

    def _hough_detections(self, image, image_shape, profile_masks):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=self._platform_mask)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS,
        )
        if circles is None:
            return []

        detections = []
        for x_float, y_float, radius_float in np.round(circles[0, :]).astype("int"):
            center = (int(x_float), int(y_float))
            radius = float(radius_float)
            contour = cv2.ellipse2Poly(center, (int(radius), int(radius)), 0, 0, 360, 8)
            contour = contour.reshape((-1, 1, 2))
            detections.append(self._score_circle_candidate(center, radius, contour, image_shape, profile_masks, image))
        return detections

    def _score_circle_candidate(self, center, radius, contour, image_shape, profile_masks, image):
        area = float(np.pi * radius * radius)
        edge_distance = self._edge_distance(center, radius, image_shape)
        color_confidence, color_name = self._circle_color_confidence(center, radius, profile_masks)
        brightness_score, saturation_score = self._circle_brightness_saturation_scores(center, radius, image)
        movement_score = 1.0
        if self.last_valid_position is not None:
            jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
            movement_score = np.clip(1.0 - jump / MAX_FRAME_JUMP, 0.0, 1.0)

        circularity = 1.0
        fill_ratio = 0.95
        aspect_ratio = 1.0
        confidence = self._confidence_score(
            area=area,
            radius=radius,
            circularity=circularity,
            fill_ratio=fill_ratio,
            aspect_ratio=aspect_ratio,
            center=center,
            color_confidence=color_confidence,
            brightness_score=brightness_score,
            saturation_score=saturation_score,
            movement_score=movement_score,
        )

        def result(reason):
            return Detection(
                center=center,
                radius=radius,
                area=area,
                confidence=confidence,
                color_confidence=color_confidence,
                color_name=color_name,
                circularity=circularity,
                fill_ratio=fill_ratio,
                edge_distance=edge_distance,
                brightness_score=brightness_score,
                saturation_score=saturation_score,
                movement_score=movement_score,
                rejection_reason=reason,
                contour=contour,
            )

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            return result("hough_radius")
        if edge_distance < EDGE_MARGIN:
            return result("hough_edge")
        if self._platform_mask[center[1], center[0]] == 0:
            return result("hough_roi")
        if self.last_valid_position is not None:
            jump = np.hypot(center[0] - self.last_valid_position[0], center[1] - self.last_valid_position[1])
            if jump > MAX_FRAME_JUMP:
                return result("hough_jump")
        if color_confidence < MIN_HOUGH_COLOR_CONFIDENCE:
            return result("hough_color")
        if brightness_score < MIN_HOUGH_BRIGHTNESS_SCORE:
            return result("hough_dark")
        if confidence < MIN_CONFIDENCE:
            return result("hough_confidence")
        return result("accepted")

    def _confidence_score(self, area, radius, circularity, fill_ratio, aspect_ratio, center, color_confidence, brightness_score, saturation_score, movement_score):
        circularity_score = np.clip((circularity - MIN_CIRCULARITY) / 0.35, 0.0, 1.0)
        fill_score = np.clip(1.0 - abs(fill_ratio - 0.78) / 0.35, 0.0, 1.0)
        aspect_score = np.clip(1.0 - abs(1.0 - aspect_ratio) / 0.30, 0.0, 1.0)
        radius_score = 1.0 if MIN_RADIUS <= radius <= MAX_RADIUS else 0.0
        area_score = 1.0 if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA else 0.0
        return float(
            0.12 * circularity_score
            + 0.10 * fill_score
            + 0.08 * aspect_score
            + 0.38 * color_confidence
            + 0.08 * brightness_score
            + 0.16 * saturation_score
            + 0.03 * movement_score
            + 0.025 * radius_score
            + 0.025 * area_score
        )

    def _edge_distance(self, center, radius, image_shape):
        height, width = image_shape[:2]
        x, y = center
        return float(min(x - radius, y - radius, width - x - radius, height - y - radius))

    @staticmethod
    def _best_color_confidence(contour, profile_masks):
        contour_mask = np.zeros(next(iter(profile_masks.values())).shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_pixels = cv2.countNonZero(contour_mask)
        if contour_pixels == 0:
            return 0.0, "none"

        best_score = 0.0
        best_name = "none"
        for name, mask in profile_masks.items():
            matched_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=contour_mask))
            score = float(matched_pixels / contour_pixels)
            if score > best_score:
                best_score = score
                best_name = name
        return best_score, best_name

    @staticmethod
    def _circle_color_confidence(center, radius, profile_masks):
        circle_mask = np.zeros(next(iter(profile_masks.values())).shape, dtype=np.uint8)
        cv2.circle(circle_mask, center, int(radius * 0.80), 255, -1)
        circle_pixels = cv2.countNonZero(circle_mask)
        if circle_pixels == 0:
            return 0.0, "none"

        best_score = 0.0
        best_name = "none"
        for name, mask in profile_masks.items():
            matched_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=circle_mask))
            score = float(matched_pixels / circle_pixels)
            if score > best_score:
                best_score = score
                best_name = name
        return best_score, best_name

    @staticmethod
    def _circle_brightness_saturation_scores(center, radius, image):
        circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, center, int(radius * 0.80), 255, -1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, s, v = cv2.split(hsv)
        mean_v = cv2.mean(v, mask=circle_mask)[0] / 255.0
        mean_s = cv2.mean(s, mask=circle_mask)[0] / 255.0
        return mean_v, mean_s

    @staticmethod
    def _brightness_saturation_scores(contour, image):
        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        mean_v = cv2.mean(v, mask=contour_mask)[0] / 255.0
        mean_s = cv2.mean(s, mask=contour_mask)[0] / 255.0
        return mean_v, mean_s

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
            f"color={detection.color_confidence:.2f}/{detection.color_name} "
            f"brightness={detection.brightness_score:.2f} "
            f"saturation={detection.saturation_score:.2f} "
            f"movement={detection.movement_score:.2f} "
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
            if not accepted and not self.debug:
                continue
            cv2.drawContours(debug, [item.contour], -1, color, 1)
            if item.radius > 0:
                cv2.circle(debug, item.center, int(item.radius), color, 1)
            if accepted:
                label = f"BALL {item.confidence:.2f} {item.color_name}"
            else:
                label = f"reject {item.rejection_reason} {item.confidence:.2f}"
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
            center, offset, found, confidence, fps, last_valid = cam.coordinate_with_offset(img)
            cam.display_debug()
            print(f"center={center} offset={offset} found={found} confidence={confidence:.2f}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.terminate()
