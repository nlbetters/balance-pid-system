import cv2
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