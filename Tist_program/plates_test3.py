import cv2
import torch

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.conf = 0.4
model.iou = 0.5

# 加载 CSRT 跟踪器并初始化跟踪
tracker = cv2.TrackerCSRT_create()
init_frame = cv2.imread('init_frame.jpg')
bbox = cv2.selectROI(init_frame, False)
tracker.init(init_frame, bbox)

# 打开视频文件
cap = cv2.VideoCapture('video.mp4')

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLOv5 进行目标检测
    results = model(frame)
    for result in results.xyxy[0].tolist():
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.4 and cls == 0:
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            ok = tracker.init(frame, bbox)
            break

    # 使用 CSRT 跟踪器进行目标跟踪
    ok, bbox = tracker.update(frame)
    if ok:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Tracking failure detected', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 显示视频帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放视频文件并关闭窗口
cap.release()
cv2.destroyAllWindows()