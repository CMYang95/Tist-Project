
#python Tist.py
#重心 左右
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt# 資料視覺化套件
import matplotlib.animation as animation

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
    plt.plot(time_x, angle_elbow_y,color='red', label="Elbow")
    plt.plot(time_x, angle_shoulder_y ,color='green', label="Shoulder")
    plt.plot(time_x, angle_Knee_y,color='blue', label="Knee")
    plt.plot(time_x, angle_hip_y, color='purple',label="Hip")

    # 設定標題與x,y軸的label名稱
    plt.title('Angles of Joints')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    if not legend_shown:
            plt.legend(loc='lower right')
            legend_shown = True
    # 設定 legend 顯示每個參數代表的線的名稱
    # plt.savefig(time.strftime('%Y%m%d-%H%M%S'))  # 檔名可自行指定
    # 顯示出圖形
    #plt.draw
    

#抓鏡頭或是影片
cap= cv2.VideoCapture(r"C:\Users\User\Desktop\workoutppt\S__57786442.jpg")        #可改成影片放入

#抓影片的大小fps等資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

re_width= 640
re_height=540


#設置影片寫入
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#filename = os.path.join('D:\JTC', time.strftime('%Y%m%d-%H%M%S') + '.mp4')
#out = cv2.VideoWriter(filename,fourcc,fps,(width,height))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #影片偵數

video_time= frame_count / fps #影片時間
per_sencond_to_fps= frame_count/video_time #每一秒偵數量

UPPER_BODY_PARTS = {"right_shoulder": 12, "left_shoulder": 11, "right_elbow": 13,"left_elbow": 14,}
LOWER_BODY_PARTS = {"right_ankle": 28, "left_ankle": 29, "right_hip": 24, "left_hip": 23, "right_knee":25,"left_knee":26}


timestamps=[] #放入時間點
current_time=0.0 #目前時間
for i in range(frame_count*2):
    timestamps.append(i/per_sencond_to_fps)


with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.7) as pose:
    for timestamp in timestamps:
        
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        
        ret, frame = cap.read()

        # 調整影像大小
        frame = cv2.resize(frame, (re_width, re_height))
        
        #顏色處裡
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #把fream 的BGR 轉成RGB
        image.flags.writeable=False #把數組這為設定成不可改寫
        #動作偵測抓取
        results= pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #轉回到BGR
        
        
        try:
                landmarks = results.pose_landmarks.landmark
                #右手腕
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
                
                #追加旋轉
                right_hip_3d=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                              ,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                left_hip_3d=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                              ,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]  
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

        #算 hip 和 spine 的 3D 座標差
        #計算人體的骨盆部位（hip）相對於脊椎部位（spine）的三維向量差
        diff_3d = tuple(map(lambda i, j: i - j, right_hip_3d, left_hip_3d))

        # 判斷 hip 是否向左旋轉或向右旋轉
        if diff_3d[0] < -0.1: # 左旋轉
            cv2.putText(image, "Hip is turning left", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif diff_3d[0] > 0.1: # 右旋轉
            cv2.putText(image, "Hip is turning right", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: # 平衡
            cv2.putText(frame, "Balanced", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


       
        if right_shoulder_balance.y > left_shoulder_balance.y and right_hip_balance.y > left_hip_balance.y:
            cv2.putText(image, "Balanced", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unbalanced", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)




        #放字串
        cv2.putText(image, "R_shoudler"+str(angle_of_right_shoulder), #右邊肩膀
                    (370,110), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),1, cv2.LINE_AA)
        
        cv2.putText(image, "R_elbow"+str(angle_of_right_elbow), #右邊手肘
                    (370,130), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),1, cv2.LINE_AA)
        
        cv2.putText(image,"R_hip"+ str(angle_of_right_hip), #右邊hip
                    (370,150), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),1, cv2.LINE_AA)

        cv2.putText(image, "R_knee"+str(angle_of_right_Knee),
                    (370,175), #右邊膝蓋
                    #tuple(np.multiply(right_knee,[640,480]).astype(int)), #轉成整數 因為cv2.puttext 沒辦法處理小數點
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),1, cv2.LINE_AA)

        cv2.imshow("Feed",image)
        angles = [angle_of_right_elbow,angle_of_right_shoulder,angle_of_right_Knee,angle_of_right_hip]
        
        #plot_joint_angles(timestamp,angles)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
    #plt.savefig(time.strftime('%Y%m%d-%H%M%S'))
    cap.release()
    cv2.destroyAllWindows()

