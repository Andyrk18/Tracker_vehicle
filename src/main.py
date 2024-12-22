from load_video import load_video       # 動画の読み込み
from get_fps import get_fps, calculate_frame_skip_interval      # フレームスキップ関連
from yolo_handler import load_yolo_model, perform_yolo, process_frame       # YOLO実行関連
from object_tracker import predict_next_bbox, predict_next_bbox_quadratic, update_tracked_data   # 予測
from anomaly_handler import save_anomaly_frame, log_anomaly_info
from anomaly_detectors import detect_combined_anomalies
from frame_visualizer import draw_detections_with_predictions
from histogram_generator import save_all_histograms

import cv2
from collections import defaultdict

# 設定
YOLO_MODEL_PATH = "../models/yolov8x.pt"
YOLO_CLASSES = [2]
INPUT_VIDEO_PATH = "../videos/street1_sample_01.mp4"
OUTPUT_FOLDER = "../results"
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
    metrics = {"iou":{}, "area":{}, "aspect":{}}
    frame_number = 0
    pause = False

    while cap.isOpened():
        if not pause:
            ret, frame = process_frame(cap, frame_skip_interval)
            if not ret:
                break
            frame_number += 1
            # YOLOによる検出
            detections = perform_yolo(frame, YOLO_MODEL, YOLO_CLASSES)
            detection_dict = {track_id: bbox for track_id, bbox in detections}  # りすとを辞書型に変換
            update_tracked_data(tracked_data, detections)
            # 予測
            predictions = {
                track_id: predict_next_bbox_quadratic(tracked_data, track_id)
                for track_id in tracked_data
            }
            # 各トラックIDに対して異常検知処理
            for track_id in tracked_data:
                current_bbox = detection_dict.get(track_id)
                predicted_bbox = predictions.get(track_id)
                previous_bbox = (
                    tracked_data[track_id][-2]
                    if len(tracked_data[track_id]) > 1
                    else None
                )
                if current_bbox and predicted_bbox:
                    is_anomaly, anomalies = detect_combined_anomalies(
                        current_bbox, previous_bbox, predicted_bbox
                    )
                    if is_anomaly:
                        save_anomaly_frame(frame, frame_number, track_id, OUTPUT_FOLDER)
                        log_anomaly_info(
                            frame_number, track_id, current_bbox, predicted_bbox, anomalies
                        )
            # フレームの描画
            frame = draw_detections_with_predictions(frame, detections, predictions)
        # フレームの表示
        if SHOW_FRAME:
            cv2.imshow("Result", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                pause = not pause
                if pause:
                    print(f"PAUSE : Frame {frame_number}")
            if key == ord('q'):
                break

    # ヒストグラムの作成と保存
    save_all_histograms(metrics, "../histograms")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()