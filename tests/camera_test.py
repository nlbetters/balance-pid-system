import cv2
from camera import Camera

cam = Camera()
try:
    while True:
        frame = cam.take_picture()
        cam.display(frame, window_name="Camera Preview")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.terminate()