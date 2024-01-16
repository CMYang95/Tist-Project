import cv2
import torch
import dlib
import numpy as np

cap = cv2.VideoCapture("D:\TIST_PROJECT\Wrong_act\moveFR.mp4")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.iou = 0.3
model.conf = 0.5

# Initialize dlib correlation tracker
tracker = dlib.correlation_tracker()

# Initialize variables for tracking
trackers = []
selected_object = None
path = []

def select_object(event, x, y, flags, param):
    global selected_object,path
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_object = None
        for i, tracker in enumerate(trackers):
            pos = tracker.get_position()
            if x >= pos.left() and x <= pos.right() and y >= pos.top() and y <= pos.bottom():
                if tracker in trackers:
                    print(f"Selected object: {i}")
                    selected_object = tracker
                    path = []  # 清空路徑列表
                    break
        if selected_object is not None:
            path.clear()
            pos = selected_object.get_position()
            x1 = int(pos.left())
            y1 = int(pos.top())
            x2 = int(pos.right())
            y2 = int(pos.bottom())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            path.append((cx, cy))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', select_object)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 416))

    # Detect license plates using YOLOv5
    results = model(frame)
    plates = results.xyxy[0]

    # Update trackers
    for i, tracker in enumerate(trackers):
        if not tracker.update(frame):
            trackers.remove(tracker)
        else:
            pos = tracker.get_position()
            x1 = int(pos.left())
            y1 = int(pos.top())
            x2 = int(pos.right())
            y2 = int(pos.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"Plate {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Add new trackers for newly detected plates
    for plate in plates:
        if plate[-1] == 0:  # Only track license plates
            x1, y1, x2, y2 = map(int, plate[:4])
            # Check if there is already a tracker for this plate
            matched_tracker = False
            for tracker in trackers:
                pos = tracker.get_position()
                if x1 <= pos.right() and x2 >= pos.left() and y1 <= pos.bottom() and y2 >= pos.top():
                    tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
                    matched_tracker = True
                    break
            if not matched_tracker:
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
                trackers.append(tracker)
    if selected_object is not None:
        tracker = selected_object
        pos = tracker.get_position()
        x1 = int(pos.left())
        y1 = int(pos.top())
        x2 = int(pos.right())
        y2 = int(pos.bottom())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        path.append((cx, cy))  # 添加新的位置到路徑列表中
        if len(path) > 1:
            for i in range(len(path)-1):
                cv2.line(frame, path[i], path[i+1], (0, 0, 255), 2)
    else:
        path=[]

    # Display tracking information
    cv2.putText(frame, f"Tracking: {len(trackers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()