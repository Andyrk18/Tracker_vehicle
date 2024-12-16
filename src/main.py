from load_video import load_video
from get_fps import get_fps, calculate_frame_skip_interval
from yolo_handler import load_yolo_model, perform_yolo, process_frame
from object_tracker import predict_next_bbox, update_tracked_data
from prediction_evaluator import evaluate_predictions
from frame_visualizer import draw_detections_with_predictions
import cv2

# 設定
YOLO_MODEL_PATH = "../models/yolov8x.pt"
YOLO_CLASSES = [2]
INPUT_VIDEO_PATH = "../videos/video_01.mp4"
TARGET_FPS = 5
SHOW_FRAME = True
SAVE_VIDEO = False

YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)

def main():
    cap =   load_video(INPUT_VIDEO_PATH)
    original_fps = get_fps(cap)
    original_fps = 20
    target_fps = TARGET_FPS
    frame_skip_interval = calculate_frame_skip_interval(original_fps, target_fps)

    tracked_data = {}

    pause = False

    total_frames = 0
    processed_frames = 0

    while cap.isOpened():
        if not pause:
            total_frames += 1
            ret, frame = process_frame(cap, frame_skip_interval)
            if not ret:
                break
            processed_frames += 1
            detections = perform_yolo(frame, YOLO_MODEL, YOLO_CLASSES)
            update_tracked_data(tracked_data, detections)
            predictions = {track_id: predict_next_bbox(tracked_data, track_id) for track_id in tracked_data}
            evaluate_predictions(detections, predictions)
            frame = draw_detections_with_predictions(frame, detections, predictions)
        if SHOW_FRAME:
            cv2.imshow("Result", frame)
            key =  cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                pause = not pause
    cap.release()
    cv2.destroyAllWindows()

    print(f"総フレーム:{total_frames}")
    print(f"処理フレーム:{processed_frames}")

if __name__ == "__main__":
    main()
