import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose =mp.solutions.pose
def caulate_angle(a,b,c):
    a=np.array(a) #第一個點
    b=np.array(b) #第二個點
    c=np.array(c) #第三個點

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle
    return angle


#抓鏡頭或是影片
cap= cv2.VideoCapture(r'D:\JTC\hip_to_knee.mp4')        #可改成影片放入


#抓影片的大小fps等資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#設置影片寫入
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = os.path.join('D:\JTC', time.strftime('%Y%m%d-%H%M%S') + '.mp4')
out = cv2.VideoWriter(filename,fourcc,fps,(width,height))


                                    #抓取的可信度 和追蹤可信度
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, fream = cap.read() #ret 表示讀取影片偵數 fream讀取偵數資料 
        #顏色處裡
        image = cv2.cvtColor(fream,cv2.COLOR_BGR2RGB) #把fream 的BGR 轉成RGB
        image.flags.writeable=False #把數組這為設定成不可改寫
        #動作偵測抓取
        results= pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #轉回到BGR
        #
        try:
            landmarks = results.pose_landmarks.landmark           
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]         
            #右邊hip
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            #右邊膝蓋
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            #右邊腳踝
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate distance 
            hip_knee_distance = math.sqrt((right_hip[0] - right_knee[0])**2 + (right_hip[1] - right_knee[1])**2)
            centimeter_distance = hip_knee_distance * 100 # assuming 1 pixel = 1 cm
            # Print distance in centimeters
            print(f"右髖關節和右膝關節之間的距離: {centimeter_distance:.2f} cm")
 
        except: #如果有錯誤就pass掉
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=1,circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(245,33,240), thickness=1,circle_radius=1))

        # out.write(image)

        cv2.imshow("Feed",image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()