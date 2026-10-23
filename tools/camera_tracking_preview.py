import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from camera import Camera

cam = Camera()
try:
    while True:
        frame = cam.take_picture()
        center = cam.coordinate(frame)
        cam.display_draw(frame, center, window_name="Camera Processed")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.terminate()