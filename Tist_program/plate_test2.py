import cv2
import torch
import numpy as np
import dlib

model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.iou = 0.3
model.conf = 0.5
re_width,re_height= 480 ,720
tracker = dlib.correlation_tracker()
cap = cv2.VideoCapture('D:\TIST_PROJECT\Tist_vide\Sun_High_ExploSnatch.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (re_width, re_height))
    if not ret:
        break
    detections = model(frame)
    detections = detections.xyxy[0].cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        # print("detection:",detection)
        if conf < model.conf:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = model.names[cls]
        tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    tracker.update(frame)
    pos = tracker.get_position()
    # print("pos",pos)
    x1 = int(pos.left())
    y1 = int(pos.top())
    x2 = int(pos.right())
    y2 = int(pos.bottom())
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
