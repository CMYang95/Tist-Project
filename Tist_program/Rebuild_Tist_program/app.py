import cv2
import time
from src.video_processing.capture import capture_video
from src.video_processing.analysis import process_frame
from src.models.dominant_hand import DominantHandDetector
from src.models.pose_estimation import PoseEstimator
from src.utils.plotting import plot_speed_force
from src.config.settings import VIDEO_PATH

def main():
    # 初始化影片捕捉
    # cap = capture_video(VIDEO_PATH)
    cap = cv2.VideoCapture(0)

    # 初始化 PoseEstimator 和 DominantHandDetector
    pose_estimator = PoseEstimator()
    dominant_hand_detector = DominantHandDetector(pose_estimator.pose, pose_estimator.mp_pose)

    # 判斷使用者的慣用手
    dominant_hand = None
    while cap.isOpened() and dominant_hand is None:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 DominantHandDetector 判斷慣用手
        frame = dominant_hand_detector.detect_dominant_hand(frame)
        dominant_hand = dominant_hand_detector.dominant_hand

        # 顯示畫面
        cv2.imshow('Detecting Dominant Hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 確認慣用手後，進行分析
    if dominant_hand:
        print(f"Detected dominant hand: {dominant_hand}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 獲取當前時間戳
            timestamp = time.time()

            # 使用 process_frame 處理每一幀
            frame = process_frame(frame)

            # 使用 PoseEstimator 計算關節角度與追蹤
            frame = pose_estimator.process_frame(frame, timestamp)

            # 顯示處理後的畫面
            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 繪製速度、加速度和力的圖表
        plot_speed_force(pose_estimator.speed_force_data)

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()