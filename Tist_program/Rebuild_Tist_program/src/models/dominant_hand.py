import cv2
import time
import logging

logger = logging.getLogger(__name__)

class DominantHandDetector:
    def __init__(self, pose, mp_pose):
        self.pose = pose
        self.mp_pose = mp_pose
        self.dominant_hand = None
        self.hand_timer = None
        self.tracking_enabled = False

    def is_point_in_circle(self, point, circle_center, radius):
        """Check if a point is inside a circle."""
        distance = ((point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2) ** 0.5
        return distance <= radius

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = (a[0] - b[0], a[1] - b[1])
        c = (c[0] - b[0], c[1] - b[1])
        dot_product = a[0] * c[0] + a[1] * c[1]
        magnitude_a = (a[0] ** 2 + a[1] ** 2) ** 0.5
        magnitude_c = (c[0] ** 2 + c[1] ** 2) ** 0.5
        angle = abs(dot_product / (magnitude_a * magnitude_c))
        return angle

    def detect_dominant_hand(self, frame):
        """Detect the dominant hand."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            height, width = frame.shape[:2]
            left_wrist_pos = (int(left_wrist.x * width), int(left_wrist.y * height))
            right_wrist_pos = (int(right_wrist.x * width), int(right_wrist.y * height))

            circle_right = (int(width * 0.2), int(height * 0.5))
            circle_left = (int(width * 0.8), int(height * 0.5))
            radius = int(height * 0.1)

            if self.dominant_hand is None:
                cv2.circle(frame, circle_left, radius, (0, 255, 0), 2)
                cv2.circle(frame, circle_right, radius, (0, 0, 255), 2)

                in_left = self.is_point_in_circle(left_wrist_pos, circle_left, radius)
                in_right = self.is_point_in_circle(right_wrist_pos, circle_right, radius)

                current_time = time.time()

                if in_left or in_right:
                    if self.hand_timer is None:
                        self.hand_timer = current_time
                    elif current_time - self.hand_timer >= 3.0:
                        self.dominant_hand = "left" if in_left else "right"
                        self.tracking_enabled = True
                        logger.info(f"Dominant hand detected: {self.dominant_hand}")
                    else:
                        remaining = 3.0 - (current_time - self.hand_timer)
                        cv2.putText(frame, f"Hold: {remaining:.1f}s",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                else:
                    self.hand_timer = None

            else:
                if self.dominant_hand == "right":
                    angle = self.calculate_angle(
                        (right_shoulder.x * width, right_shoulder.y * height),
                        (right_elbow.x * width, right_elbow.y * height),
                        (right_wrist.x * width, right_wrist.y * height)
                    )
                else:
                    angle = self.calculate_angle(
                        (left_shoulder.x * width, left_shoulder.y * height),
                        (left_elbow.x * width, left_elbow.y * height),
                        (left_wrist.x * width, left_wrist.y * height)
                    )

        return frame