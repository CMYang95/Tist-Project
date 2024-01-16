#畫折線圖check!
#手腕路線 3/20 #python ankleline.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt# 資料視覺化套件
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose =mp.solutions.pose

all_time_angles=[]

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
def plot_joint_angles(timestamp,angles):
    global all_time_angles,legend_shown
    all_time_angles.append((timestamp, [int(angles[0]),int(angles[1]),
                                int(angles[2]),int(angles[3])]))
    # 設定x軸資料 (時間)
    time_x=[item[0] for item in all_time_angles]
    #print("time_x:",time_x)

    # 設定y軸資料 (角度)
    angle_elbow_y=[item[1][0] for item in all_time_angles]  # elbow
    #print("angle_elbow_y:",angle_elbow_y)
    angle_shoulder_y=[item[1][1] for item in all_time_angles]  # shoulder
    angle_Knee_y=[item[1][2] for item in all_time_angles]  # Knee
    angle_hip_y=[item[1][3] for item in all_time_angles]  # hip

    # 使用plot()畫出折線圖(每個參數畫一條線並用不同顏色)
    plt.plot(time_x, angle_elbow_y, color='red', label="Elbow")
    plt.plot(time_x, angle_shoulder_y, color='green', label="Shoulder")
    plt.plot(time_x, angle_Knee_y, color='blue', label="Knee")
    plt.plot(time_x, angle_hip_y, color='purple',label="Hip")

    # 設定標題與x,y軸的label名稱
    plt.title('Angles of Joints')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    if not legend_shown:
            plt.legend(loc='lower right')
            legend_shown = True    
    plt.draw

#抓鏡頭或是影片
cap= cv2.VideoCapture(r'D:\TIST_PROJECT\Tist_vide\Y_ankle_line2.mp4')        #可改成影片放入
#抓影片的大小fps等資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#re_image
re_width,re_height= 480 ,720

name_video="Mark_B" 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = os.path.join(r'D:\TIST_PROJECT\Tist_outp', name_video+time.strftime('-%H%M%S') + '.mp4')
out = cv2.VideoWriter(filename,fourcc,fps,(width,height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #影片偵數
video_time= frame_count / fps #影片時間
per_sencond_to_fps= frame_count/video_time #每一秒偵數量

#紀錄空字參數
wrist_point=[] #紀錄手腕軌跡座標
Ankle_point=[]
timestamps=[] #放入時間點
current_time=0.0 #目前時間

for i in range(frame_count*2):
    timestamps.append(i/per_sencond_to_fps)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:#model_complexity=1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (re_width, re_height))

    for timestamp in timestamps:
        
        ret, frame = cap.read()
        if not ret:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        #frame = cv2.resize(frame, (re_width, re_height))
        
        #顏色處裡
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #把fream 的BGR 轉成RGB
        image.flags.writeable=False #把數組這為設定成不可改寫
        #動作偵測抓取
        results= pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #轉回到BGR
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]           
            
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]           
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]           
        except: 
                pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=1,circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(245,33,240), thickness=1,circle_radius=1))
        
        #手腕座標
        wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        x = int(wrist_landmark.x * image.shape[1])
        y = int(wrist_landmark.y * image.shape[0])
        wrist_point.append((x, y))
        #腳踝座標
        right_ankle_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        right_heel_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        right_food_index_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        
        # 計算中心點
        Ankle_center_x = int((right_ankle_landmark.x + right_heel_landmark.x + right_food_index_landmark.x) / 3 * image.shape[1])
        Ankle_center_y = int((right_ankle_landmark.y + right_heel_landmark.y + right_food_index_landmark.y) / 3 * image.shape[0])

        # 將中心點添加到 Ankle_point 列表中
        Ankle_point.append((Ankle_center_x, Ankle_center_y))
        
        
                
        #右邊肩膀
        angle_of_right_shoulder= caulate_angle(right_elbow,right_shoulder,right_hip)
        # 右邊手肘角度
        angle_of_right_elbow= caulate_angle(right_wrist,right_elbow,right_shoulder)

        #右腿膝蓋角度
        angle_of_right_Knee= caulate_angle(right_hip,right_knee,right_ankle)
        
        #右邊髖關節角度
        angle_of_right_hip= caulate_angle(right_shoulder,right_hip,right_knee)
        
        
        starting_shoulder=int(angle_of_right_shoulder)
        starting_elbow=int(angle_of_right_elbow)

        ending_elbow=int(angle_of_right_elbow)
        ending_knee=int(angle_of_right_Knee)
        angles = [angle_of_right_elbow,angle_of_right_shoulder,angle_of_right_Knee,angle_of_right_hip]  #把座標放進去  
        
        
        #放字串
        cv2.putText(image, "R_shoudler"+str(angle_of_right_shoulder), #右邊肩膀
                    (0,110), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
        
        cv2.putText(image, "R_elbow"+str(angle_of_right_elbow), #右邊手肘
                    (0,130), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
        
        cv2.putText(image,"R_hip"+ str(angle_of_right_hip), #右邊hip
                    (0,150), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)

        cv2.putText(image, "R_knee"+str(angle_of_right_Knee),
                    (0,175), #右邊膝蓋
                    #tuple(np.multiply(right_knee,[640,480]).astype(int)), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)

        # for i in range(1,len(wrist_point)):
        #      cv2.line(image, wrist_point[i-1],wrist_point[i],(0,255,0),2)
        
        for i in range(1,len(Ankle_point)):
                cv2.line(image, Ankle_point[i-1],Ankle_point[i],(0,255,0),2)
        
        # 在影像上繪製右腳中心點
        cv2.circle(image, (Ankle_center_x, Ankle_center_y), 5, (255, 0, 0), -1)

        # for i in range(1,len(Ankle_point)):
        #         cv2.line(image, Ankle_point[i-1],Ankle_point[i],(0,0,0),0)
      
        cv2.imshow("Feed",image)
        

        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    
    #plt.savefig(time.strftime('%Y%m%d-%H%M%S'))
    cap.release()
    #out.release()
    cv2.destroyAllWindows()