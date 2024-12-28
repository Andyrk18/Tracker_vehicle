from load_video import load_video                                   # 動画の読み込み
from get_fps import get_fps, calculate_frame_skip_interval          # フレームスキップ関連
from yolo_handler import load_yolo_model, perform_yolo, process_frame # YOLO実行関連
from prediction import get_prediction_function, update_tracked_data # 予測
import prediction_evaluator as pe                                   # 予測の評価と各基準の計算
from anomaly_detectors import detect_combined_anomalies             # 異常検出
from anomaly_handler import save_anomaly_frame, log_anomaly_info    # 異常判定時の保存
from frame_visualizer import draw_detections_with_predictions       # フレーム表示
from histogram_generator import save_all_histograms                 # ヒストグラム作成, 保存

import cv2
import os

# 設定
INPUT_VIDEO_NAME = "minokamo_01.MOV"
RESULTS_FOLDER = "../results"
ANOMALIES_FOLDER = os.path.join(RESULTS_FOLDER, "anomalies")
HISTOGRAMS_FOLDER = os.path.join(RESULTS_FOLDER, "histograms")

TARGET_FPS = 2

PREDICTION_METHOD = "linear"        # 線形:"linear", 曲線:"quadratic", カルマン:"kalman"

SHOW_FRAME = True
SAVE_VIDEO = False

YOLO_MODEL_PATH = "../models/yolov8x.pt"
YOLO_CLASSES = [2]
YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)

def main():
    os.makedirs(ANOMALIES_FOLDER, exist_ok=True)
    os.makedirs(HISTOGRAMS_FOLDER, exist_ok=True)

    input_video_path = f"../videos/{INPUT_VIDEO_NAME}"
    cap =   load_video(input_video_path)
    original_fps = get_fps(cap)
    original_fps = 20
    frame_skip_interval = calculate_frame_skip_interval(original_fps, TARGET_FPS)

    tracked_data = {}
    metrics = {"iou":{}, "area":{}, "aspect":{}}    # 初期化
    frame_number = 0
    pause = False
    predict_bbox = get_prediction_function(PREDICTION_METHOD)
    while cap.isOpened():
        if not pause:
            ret, frame = process_frame(cap, frame_skip_interval)
            if not ret:
                break
            frame_number += 1
            # YOLOによる検出
            detections = perform_yolo(frame, YOLO_MODEL, YOLO_CLASSES)
            detection_dict = {track_id: bbox for track_id, bbox in detections}  # リストを辞書型に変換
            update_tracked_data(tracked_data, detections)
            # 予測
            predictions = {
                track_id: predict_bbox(tracked_data, track_id)
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
                    iou = pe.calculate_iou(current_bbox, predicted_bbox)
                    area_ratio = pe.calculate_area_ratio(current_bbox, previous_bbox)
                    aspect_ratio = pe.calculate_aspect_ratio(current_bbox, previous_bbox)
                    if track_id not in metrics["iou"]:
                        metrics["iou"][track_id] = []
                        metrics["area"][track_id] = []
                        metrics["aspect"][track_id] = []
                    if iou is not None:
                        metrics["iou"][track_id].append(iou)
                    if area_ratio is not None:
                        metrics["area"][track_id].append(area_ratio)
                    if aspect_ratio is not None:
                        metrics["aspect"][track_id].append(aspect_ratio)
                    # 異常検知
                    is_anomaly, anomalies = detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox)
                    if is_anomaly:
                        save_anomaly_frame(frame, frame_number, track_id, ANOMALIES_FOLDER)
                        log_anomaly_info(frame_number, track_id, current_bbox, predicted_bbox, anomalies)
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
    save_all_histograms(metrics, HISTOGRAMS_FOLDER)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()