import cv2
import torch
import dlib
import numpy as np
import mediapipe as mp
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt# 資料視覺化套件
#成功!!
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\sunvideo\7.mp4")

# YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\TIST_PROJECT\Tist_program\plate.pt')
model.iou = 0.3
model.conf = 0.5

# dlib追蹤
tracker = dlib.correlation_tracker()
trackers = []
selected_object = None
path = []

UPPER_BODY_PARTS = {"right_shoulder": 12, "left_shoulder": 11, "right_elbow": 13,"left_elbow": 14,}
LOWER_BODY_PARTS = {"right_ankle": 28, "left_ankle": 29, "right_hip": 24, "left_hip": 23, "right_knee":25,"left_knee":26}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose =mp.solutions.pose

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

#__影片資訊_____
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
re_width,re_height= 540 ,640

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #影片偵數
video_time= frame_count / fps #影片時間
per_sencond_to_fps= frame_count/video_time #每一秒偵數量

track_move=[]
timestamps=[] #放入時間點
current_time=0.0 #目前時間
prev_cx, prev_cy=None,None
speed=0
prev_speed = 0.0
acceleration = 0.0
time_delta = 0
force=0
#_______儲存影片_______
name_video="sun7"   
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = os.path.join(r'D:\TIST_PROJECT\Tist_outp', name_video+time.strftime('-%H%M%S') + '.mp4')
out = cv2.VideoWriter(filename,fourcc,fps,(width,height))

for i in range(frame_count*2):
    timestamps.append(i/per_sencond_to_fps)
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose: 
    ret, frame = cap.read()
    frame = cv2.resize(frame, (re_width, re_height))

    for timestamp in timestamps:
        ret, frame = cap.read()
        if not ret:
            break

        #frame = cv2.resize(frame, (460,520))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
            # Update tracker for selected object
            if not tracker.update(frame):
                selected_object = None
            if prev_cx is not None and prev_cy is not None:
                    # 計算移動距離
                
                distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                # 計算移動速率，假設每個 frame 間隔為 0.033 秒
                speed = (distance / 0.033)/100
                #print(f"Speed: {speed:.2f} m/s")
                
                # 計算加速度
                
                time_delta = (timestamp - current_time)
                #print("1timestamp",timestamp,"time_delta",time_delta)
                acceleration = (speed - prev_speed) / time_delta if time_delta > 0 else 0
                prev_speed = speed
                current_time = timestamp
                #print("time_delta",current_time)
                force = 60 * acceleration #50是公斤數
                
            prev_cx, prev_cy = cx, cy       
        else:
            path = []
            


        #人體辨識角度    
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #轉回到BGR
        # if results.pose_landmarks is not None:
            # Extract landmarks for upper and lower body parts
        #landmarks = results.pose_landmarks.landmark
        try:
            landmarks = results.pose_landmarks.landmark
        #左面
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]           
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]           
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]  
            
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
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(245,33,240), thickness=1,circle_radius=1))
        
        #左邊肩膀
        angle_of_left_shoulder= caulate_angle(left_elbow,left_shoulder,left_hip)
        #左邊手肘角度
        angle_of_left_elbow= caulate_angle(left_wrist,left_elbow,left_shoulder)

        #左腿膝蓋角度
        angle_of_left_Knee= caulate_angle(left_hip,left_knee,left_ankle)
        
        #左邊髖關節角度
        angle_of_left_hip= caulate_angle(left_shoulder,left_hip,left_knee)

        #放字串
        Str_speed= round(prev_speed, 2) #round()處理 取小數點第幾位
        cv2.putText(image, "Bar Speed :"+str(Str_speed)+"m/s", #槓鈴速度
                    (39,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)
        cv2.putText(image, f"Acceleration: {acceleration:.2f} m/s^2", 
                    (39,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)
        cv2.putText(image, f"Force: {force:.2f} N", 
                    (39,135), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)
        
        cv2.putText(image, "L-Shoudler"+str(round(angle_of_left_shoulder,2))+"Dg", #右邊肩膀
                    (39,160),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)
        
        cv2.putText(image, "L-Elbow"+str(round(angle_of_left_elbow,2))+"Dg", #右邊手肘
                    (39,190), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)
        
        cv2.putText(image,"L-Hip"+ str(round(angle_of_left_hip,2))+"Dg", #右邊hip
                    (39,225), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)

        cv2.putText(image, "L-Knee"+str(round(angle_of_left_Knee,2))+"Dg",#右邊膝蓋
                    (39,260), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,165,0),2, cv2.LINE_AA)



        cv2.imshow('frame', image)
        angles = [angle_of_left_elbow,angle_of_left_shoulder,angle_of_left_Knee,angle_of_left_hip]
        out.write(image)
        plot_joint_angles(timestamp,angles)
        if cv2.waitKey(1) == ord('q'):
            break
plt.savefig(os.path.join('D:\TIST_PROJECT\Tist_blt', name_video+time.strftime('-%H%M%S'+'new')))
cap.release()
cv2.destroyAllWindows()