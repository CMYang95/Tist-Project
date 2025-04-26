import cv2
def initialize_tracker(frame, bbox):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return tracker

def update_tracker(tracker, frame):
    success, bbox = tracker.update(frame)
    return success, bbox

def draw_tracking_box(frame, bbox):
    (x, y, w, h) = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def calculate_center(bbox):
    (x, y, w, h) = [int(v) for v in bbox]
    cx, cy = int(x + w / 2), int(y + h / 2)
    return cx, cy