import cv2
import os

def capture_video(source=0, width=640, height=480):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise Exception("Could not open video source.")

    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def release_capture(cap):
    cap.release()
    cv2.destroyAllWindows()