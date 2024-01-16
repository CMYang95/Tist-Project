import cv2
import torch
import dlib
import numpy as np
cap = cv2.VideoCapture('D:\TIST_PROJECT\Tist_vide\SUN_SQ.mp4')
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.iou = 0.3
model.conf = 0.5
# Initialize dlib correlation tracker
tracker = dlib.correlation_tracker()
# Initialize variables for tracking
trackers = []
track_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (416, 416))
    # Detect license plates using YOLOv5
    results = model(frame)
    plates = results.xyxy[0]
    # Update trackers
    for tracker in trackers:
        if not tracker.update(frame):
            trackers.remove(tracker)
        else:
            pos = tracker.get_position()
            x1 = int(pos.left())
            y1 = int(pos.top())
            x2 = int(pos.right())
            y2 = int(pos.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
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
                track_id += 1
    # Display tracking information
    cv2.putText(frame, f"Tracking: {len(trackers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()