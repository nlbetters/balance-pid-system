import time
import math
import cv2
from picamera2 import Picamera2, Preview
import numpy as np
from collections import deque
import threading

class Camera:
    #1640, 1232
    def __init__(self, resolution=(1640, 1232), format="RGB888"):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": resolution, "format": format}, controls={"FrameDurationLimits": (8333, 8333)})
        self.picam2.configure(config)

        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])
        self.gray_threshold = 80

        self.queue = deque(maxlen=16)
        self.queue.append((100, 75))
        self.last_center = (100, 75)

        self.picam2.start()    

    def take_picture(self):
        image = self.picam2.capture_array()
        scale = 200.0 / image.shape[1]
        frame_resized = cv2.resize(image, (200, int(image.shape[0] * scale)))
        return frame_resized

    def display(self, image, window_name="Camera Output"):
        cv2.imshow(window_name, image)
        cv2.waitKey(1) 

    def display_draw(self, image, center, window_name="Tracked Output"):
        x, y = center
        cv2.line(image, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
        cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 255), 2)
        cv2.imshow(window_name, image)
        cv2.waitKey(1) 


    def terminate(self):
        self.picam2.stop()
        self.picam2.close()
        cv2.destroyAllWindows()

    def coordinate(self, image):
        
        prev_time = time.time()

        # Apply Gaussian blur.
        frame_blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Convert from RGB to HSV / GRAY because Picamera2 returns RGB data.
        frame_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_RGB2HSV)
        frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_RGB2GRAY)

        # Filter based on dark ball detection in HSV and grayscale.
        mask_dark = cv2.inRange(frame_hsv, self.lower_black, self.upper_black)
        mask_gray = cv2.threshold(frame_gray, self.gray_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        mask_combined = cv2.bitwise_or(mask_dark, mask_gray)

        # Clean up the mask and keep only well-defined blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

        # --- Find Contours (circles)
        valid_detections = []
        contours, _ = cv2.findContours(mask_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Minimum Enclosing Circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)

            # Ignore small objects
            if radius < 5 or radius > 100:  # Adjust min/max radius based on expected size
                continue

            #Compute Circularity 4π(Area / Perimeter²)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < 0.6:  # Threshold to eliminate non-circular objects
                continue

            # Compute Aspect Ratio of Bounding Box
            x, y, w, h = cv2.boundingRect(contour)

            # If the contour passes all filters
            valid_detections.append((area, (int(x + w / 2), int(y + h / 2))))

        if valid_detections:
            if self.last_center is not None:
                def score(candidate):
                    center = candidate[1]
                    dist = math.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
                    return candidate[0] - dist * 5
                best_center = max(valid_detections, key=score)[1]
            else:
                best_center = max(valid_detections, key=lambda item: item[0])[1]
            self.last_center = best_center
            self.queue.append(best_center)
        else:
            blurred_gray = cv2.GaussianBlur(frame_gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=40,
                param1=100,
                param2=35,
                minRadius=10,
                maxRadius=100,
            )
            if circles is not None and self.last_center is not None:
                circles = np.uint16(np.around(circles[0]))
                candidates = []
                for cx, cy, r in circles:
                    dist = math.hypot(cx - self.last_center[0], cy - self.last_center[1])
                    if dist < 80:
                        candidates.append((r, (int(cx), int(cy))))
                if candidates:
                    best_circle = max(candidates, key=lambda item: item[0])
                    best_center = best_circle[1]
                    self.last_center = best_center
                    self.queue.append(best_center)
                    return self.queue[-1]

            self.queue.append(self.last_center)

        return self.queue[-1]
        

if __name__ == "__main__":
    cam = Camera()

    try:
        while True:
            img = cam.take_picture()
            c = cam.coordinate(img)
            cam.display_draw(img, c)
            print(c)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.terminate()