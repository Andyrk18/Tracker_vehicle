from load_video import load_video
from get_fps import get_fps, calculate_frame_skip_interval
from yolo_handler import load_yolo_model, perform_yolo, process_frame
from object_tracker import predict_next_bbox, update_tracked_data
from prediction_evaluator import log_frame_iou, summarize_iou_results
from frame_visualizer import draw_detections_with_predictions
import cv2
from collections import defaultdict

# 設定
YOLO_MODEL_PATH = "../models/yolov8x.pt"
YOLO_CLASSES = [2]
INPUT_VIDEO_PATH = "../videos/street1_sample_01.mp4"
TARGET_FPS = 5
SHOW_FRAME = True
SAVE_VIDEO = False

YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)

def main():
    cap =   load_video(INPUT_VIDEO_PATH)
    original_fps = get_fps(cap)
    original_fps = 20
    frame_skip_interval = calculate_frame_skip_interval(original_fps, TARGET_FPS)

    tracked_data = {}
    iou_history = defaultdict(list)
    frame_number = 0
    pause = False
    while cap.isOpened():
        if not pause:
            ret, frame = process_frame(cap, frame_skip_interval)
            if not ret:
                break
            frame_number += 1
            detections = perform_yolo(frame, YOLO_MODEL, YOLO_CLASSES)
            update_tracked_data(tracked_data, detections)
            predictions = {track_id: predict_next_bbox(tracked_data, track_id) for track_id in tracked_data}
            # フレームごとのIoU評価
            frame_iou_results = log_frame_iou(detections, predictions, frame_number)
            for track_id, iou in frame_iou_results:
                iou_history[track_id].append(iou)
            # 描画と表示
            frame = draw_detections_with_predictions(frame, detections, predictions)
        if SHOW_FRAME:
            cv2.imshow("Result", frame)
            key =  cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                pause = not pause
                if pause:
                    print(f" PAUSE: Frame-{frame_number}")
    cap.release()
    cv2.destroyAllWindows()

    summarize_iou_results(iou_history)

if __name__ == "__main__":
    main()
