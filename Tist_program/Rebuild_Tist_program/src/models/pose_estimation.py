import cv2
import mediapipe as mp
import numpy as np
import torch
from ..utils.calculations import calculate_speed, calculate_acceleration, calculate_force

class PoseEstimator:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8, model_path='\\plate.pt'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                       min_tracking_confidence=min_tracking_confidence)
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.iou = 0.3
        self.model.conf = 0.5

        # Tracking variables
        self.prev_cx, self.prev_cy = None, None
        self.prev_speed = 0.0
        self.current_time = 0.0
        self.speed_force_data = []  # 儲存速度、加速度和力的數據
    def yolov5_inference(self, image):
        results = self.model(image)
        return results

    def process_frame(self, frame, timestamp):
        # Perform YOLOv5 inference
        results = self.yolov5_inference(frame)
        for detection in results.xyxy:  # Iterate through detections
            x1, y1, x2, y2, conf, cls = detection
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  # Convert to (x, y, w, h)
            cx, cy = int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)  # Calculate center

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Tracking logic
            if self.prev_cx is not None and self.prev_cy is not None:
                # Calculate distance
                distance = ((cx - self.prev_cx) ** 2 + (cy - self.prev_cy) ** 2) ** 0.5
                # Calculate speed
                speed = calculate_speed(distance, 0.033)
                # Calculate acceleration
                time_delta = timestamp - self.current_time
                acceleration = calculate_acceleration(speed, self.prev_speed, time_delta)
                # Calculate force
                force = calculate_force(60, acceleration)  # Assume mass = 60 kg

                # Update tracking variables
                self.prev_speed = speed
                self.current_time = timestamp
                self.speed_force_data.append((timestamp, speed, force, acceleration))  # 儲存數據
                # Display tracking info
                cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Acceleration: {acceleration:.2f} m/s^2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Force: {force:.2f} N", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update previous position
            self.prev_cx, self.prev_cy = cx, cy

        return frame