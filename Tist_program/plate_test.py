import cv2
import torch
import dlib
import numpy as np
cap = cv2.VideoCapture('D:\TIST_PROJECT\Tist_vide\Sun_High_ExploSnatch.mp4')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.iou = 0.3
model.conf = 0.5
while True:
    ret, frame = cap.read()
    if not ret:
        # 如果影片讀取完畢，就退出迴圈
        break
    frame = cv2.resize(frame, (416, 416))  # 調整影像大小
    # 使用 YOLOv5 模型在影片中偵測物件
    result = model(frame)
    # 把 PyTorch 格式的預測結果轉換成 Numpy 陣列
    result = result.xyxy[0].cpu().numpy()
    # 繪製偵測框和標籤
    for r in result:
        x1, y1, x2, y2, conf, cls = r
        if conf < model.conf:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = model.names[cls]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
    # 把影格顯示出來
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # 如果按下 q 鍵，就退出迴圈
        break
# 釋放資源和關閉視窗
cap.release()
cv2.destroyAllWindows()