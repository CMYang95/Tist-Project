import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt# 資料視覺化套件
import matplotlib.animation as animation
import tensorflow as tf
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose =mp.solutions.pose
# KG_Force=input()
all_time_angles=[]
all_time_speed=[]
def caulate_angle(a,b,c):
    a=np.array(a) #第一個點
    b=np.array(b) #第二個點
    c=np.array(c) #第三個點

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle
plt.ion()


legend_shown = False

def speed_force(timestamp,all_speed):
    global legend_shown,all_time_speed
    all_time_speed.append((timestamp,[int(all_speed[0]),int(all_speed[1]),
                                int(all_speed[2])]))
    
    speed_time_x=[item[0] for item in all_time_speed]
    speed=[item[1][0] for item in all_time_speed]
    force=[item[1][1] for item in all_time_speed] 
    acceleration=[item[1][2] for item in all_time_speed]
    plt.plot(speed_time_x, speed, color='b', label='Speed')
    plt.plot(speed_time_x, force, color='r', label='Force')
    plt.plot(speed_time_x, acceleration, color='g', label='Acceleration')
    plt.title('Angles of Joints')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Value')
    if not legend_shown:
            plt.legend(loc='lower right')
            legend_shown = True
    plt.draw

def plot_joint_angles(timestamp,angles):
    global all_time_angles,legend_shown
    all_time_angles.append((timestamp, [int(angles[0]),int(angles[1]),
                                int(angles[2]),int(angles[3])]))
    time_x=[item[0] for item in all_time_angles]
    angle_elbow_y=[item[1][0] for item in all_time_angles]
    angle_shoulder_y=[item[1][1] for item in all_time_angles] 
    angle_Knee_y=[item[1][2] for item in all_time_angles]
    angle_hip_y=[item[1][3] for item in all_time_angles] 
    plt.plot(time_x, angle_elbow_y,color='red', label="Elbow")
    plt.plot(time_x, angle_shoulder_y ,color='green', label="Shoulder")
    plt.plot(time_x, angle_Knee_y,color='blue', label="Knee")
    plt.plot(time_x, angle_hip_y, color='purple',label="Hip")
    plt.title('Angles of Joints')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    if not legend_shown:
            plt.legend(loc='lower right')
            legend_shown = True
    plt.draw

name_video="Left_stranger3"    

#抓鏡頭或是影片
cap= cv2.VideoCapture(r'D:\JTC\Tist_vide\Left_stranger3.mp4')      
#抓影片的大小fps等資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
re_width,re_height= 540 ,640
#設置影片寫入
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = os.path.join('D:\JTC\Tist_outp', name_video+time.strftime('-%H%M%S') + '.mp4')
out = cv2.VideoWriter(filename,fourcc,fps,(width,height))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #影片偵數

video_time= frame_count / fps #影片時間
per_sencond_to_fps= frame_count/video_time #每一秒偵數量

UPPER_BODY_PARTS = {"right_shoulder": 12, "left_shoulder": 11, "right_elbow": 13,"left_elbow": 14,}
LOWER_BODY_PARTS = {"right_ankle": 28, "left_ankle": 29, "right_hip": 24, "left_hip": 23, "right_knee":25,"left_knee":26}

timestamps=[] #放入時間點
current_time=0.0 #目前時間

for i in range(frame_count*2):
    timestamps.append(i/per_sencond_to_fps)

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose: 
    ret, frame = cap.read()
    frame = cv2.resize(frame, (re_width, re_height))
    # 創建追蹤 CSRT
    tracker = cv2.TrackerCSRT_create()
    bbox = cv2.selectROI(frame, False)
    # 初始化追蹤器
    tracking = tracker.init(frame, bbox)
    
    for timestamp in timestamps:
         # 讀取畫面
        ret, frame = cap.read()
         # 調整影像大小
        #frame = cv2.resize(frame, (re_width, re_height))
        # 檢查畫面是否讀取成功
        if not ret:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)

        
        #顏色處裡
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #把fream 的BGR 轉成RGB
        image.flags.writeable=False #把數組這為設定成不可改寫
        
        #動作偵測抓取
        results= pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #轉回到BGR
        try:
            landmarks = results.pose_landmarks.landmark
            #右手腕 #LEFT_WRIST
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            #右手肩膀
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]           
            #右手肘
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]           
            #右邊hip
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            #右邊膝蓋
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            #右邊腳踝
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]  
            
            #upper body
            right_shoulder_balance = results.pose_landmarks.landmark[UPPER_BODY_PARTS["right_shoulder"]]
            left_shoulder_balance = results.pose_landmarks.landmark[UPPER_BODY_PARTS["left_shoulder"]]
            right_elbow_balance = results.pose_landmarks.landmark[UPPER_BODY_PARTS["right_elbow"]]
            left_elbow_balance = results.pose_landmarks.landmark[UPPER_BODY_PARTS["left_elbow"]]
            ## lower body
    
            right_hip_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["right_hip"]]
            left_hip_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["left_hip"]]
            right_ankle_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["right_ankle"]]
            left_ankle_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["left_ankle"]]
            right_knee_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["right_knee"]]
            left_knee_balance = results.pose_landmarks.landmark[LOWER_BODY_PARTS["left_knee"]]
            
        except: #如果有錯誤就pass掉    
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=1,circle_radius=1),
                                mp_drawing.DrawingSpec(color=(245,33,240), thickness=1,circle_radius=1))
        
        #右邊肩膀
        angle_of_right_shoulder= caulate_angle(right_elbow,right_shoulder,right_hip)
        #右邊手肘角度
        angle_of_right_elbow= caulate_angle(right_wrist,right_elbow,right_shoulder)
        #右腿膝蓋角度
        angle_of_right_Knee= caulate_angle(right_hip,right_knee,right_ankle)
        
        #右邊髖關節角度
        angle_of_right_hip= caulate_angle(right_shoulder,right_hip,right_knee)
        

        starting_shoulder=int(angle_of_right_shoulder)
        starting_elbow=int(angle_of_right_elbow)

        ending_elbow=int(angle_of_right_elbow)
        ending_knee=int(angle_of_right_Knee)


    #肩膀平衡等等可改
        if right_shoulder_balance.y > left_shoulder_balance.y and right_hip_balance.y > left_hip_balance.y:
            cv2.putText(image, "Balanced ", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unbalanced", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        #放字串
        cv2.putText(image, "R-Shoudler"+str(round(angle_of_right_shoulder,2))+"Dg", #右邊肩膀
                    (39,158),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(64,224,208),2, cv2.LINE_AA)
        
        cv2.putText(image, "R-Elbow"+str(round(angle_of_right_elbow,2))+"Dg", #右邊手肘
                    (39,185), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(64,224,208),2, cv2.LINE_AA)
        
        cv2.putText(image,"R-Hip"+ str(round(angle_of_right_hip,2))+"Dg", #右邊hip
                    (39,215), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(64,224,208),2, cv2.LINE_AA)

        cv2.putText(image, "R-Knee"+str(round(angle_of_right_Knee,2))+"Dg",#右邊膝蓋
                    (39,245), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(64,224,208),2, cv2.LINE_AA)

        
        
        cv2.imshow("Feed",image)
        angles = [angle_of_right_elbow,angle_of_right_shoulder,angle_of_right_Knee,angle_of_right_hip]
        out.write(image)
        plot_joint_angles(timestamp,angles)
        #speed_force(timestamp,all_speed)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


plt.savefig(os.path.join('D:\JTC\Tist_outp', name_video))
cap.release()
cv2.destroyAllWindows()
