import os
import cv2

from src.load_video import load_video                                               # 動画の読み込み
from src.get_fps import get_fps, calculate_frame_skip_interval                      # フレームスキップ関連
from src.yolo_handler import load_yolo_model, process_frame, process_frame_data     # YOLO実行関連
from src.prediction import get_prediction_function                                  # 予測
from src.anomaly_detectors import detect_combined_anomalies                         # 異常検出
import src.anomaly_handler as anomaly                                               # 異常判定時
from src.frame_visualizer import annotate_frame_with_tracking, display_frame, draw_predictions, draw_tracking_data  # フレーム表示
from src.histogram_generator import save_all_histograms, update_metrics             # ヒストグラム作成, 保存
import src.log as log                                                               # log


# 設定
INPUT_VIDEO_NAME = "street1_sample_01.mp4"
INPUT_VIDEO_NAME = "minokamo_08_hide2.mp4"

RESULTS_FOLDER = "../results"
ANOMALIES_FOLDER = os.path.join(RESULTS_FOLDER, "anomalies")
HISTOGRAMS_FOLDER = os.path.join(RESULTS_FOLDER, "histograms")

TARGET_FPS = 10

PREDICTION_METHOD = "quadratic_weight"        # 線形:"linear", 曲線:"quadratic", カルマン:"kalman"

SHOW_FRAME = True
SAVE_VIDEO = False

YOLO_MODEL_PATH = "../models/yolov8x.pt"
YOLO_CLASSES = [2]
YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)

MAX_MISSED_FRAME = TARGET_FPS * 1.5

TARGET_IDS = []

def main():
    os.makedirs(ANOMALIES_FOLDER, exist_ok=True)
    os.makedirs(HISTOGRAMS_FOLDER, exist_ok=True)

    input_video_path = f"../videos/{INPUT_VIDEO_NAME}"
    cap =   load_video(input_video_path)
    original_fps = get_fps(cap)
    frame_skip_interval = calculate_frame_skip_interval(original_fps, TARGET_FPS)

    tracked_data = {}
    missed_frames = {}
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
            print(f"=== Frame {frame_number} ===")

            # 検出と予測
            detection_dict, predictions = process_frame_data(frame, YOLO_MODEL, YOLO_CLASSES, tracked_data, predict_bbox)

            # 検出値と予測値を確認
            log.log_predictions_and_detections(detection_dict, predictions)

            anomalies = {}
            for track_id in list(tracked_data.keys()):
                # 各トラックIDに対して未検出フレームをカウント
                if track_id not in detection_dict:  # 現在検出されていない場合
                    missed_frames[track_id] = missed_frames.get(track_id, 0) + 1
                    if missed_frames[track_id] > MAX_MISSED_FRAME:
                        # print(f"Track ID {track_id} removed after {MAX_MISSED_FRAME} missed frames")
                        del tracked_data[track_id]  # `tracked_data`から削除
                        del missed_frames[track_id]  # `missed_frames`からも削除
                        continue  # 次のトラックIDの処理に移る
                else:
                    missed_frames[track_id] = 0  # 検出された場合はカウンタをリセット
                if track_id not in TARGET_IDS:
                    continue
                current_bbox = detection_dict.get(track_id)
                predicted_bbox = predictions.get(track_id)
                previous_bbox = (
                    tracked_data[track_id][-2]
                    if len(tracked_data[track_id]) > 1
                    else None
                )
                if current_bbox and predicted_bbox:
                    target_ids = [1]
                    update_metrics(metrics, track_id, current_bbox, predicted_bbox, previous_bbox, target_ids)
                    # 異常検知
                    is_anomaly, _ = detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox)
                    anomalies[track_id] = is_anomaly

                # 異常時の置き換え処理
                anomaly.handle_replace(detection_dict, predictions, anomalies, tracked_data)

            # 確定情報の確認
            log.display_latest_tracked_data(tracked_data, frame_number)

            # フレームの描画
            if SHOW_FRAME or SAVE_VIDEO:
                # frame = annotate_frame_with_tracking(frame, detection_dict, predictions, tracked_data)  # すべて描画
                # frame = annotate_frame_with_tracking(frame, detection_dict)     # YOLO検出値のみ描画
                # frame = draw_predictions(frame, predictions)                    # 予測値のみ描画
                # frame = draw_tracking_data(frame, tracked_data)                 # 確定値のみ描画
                frame = annotate_frame_with_tracking(frame, detection_dict, predictions)
        # フレームの表示
        if SHOW_FRAME:
            pause, should_exit = display_frame(
                frame, frame_number, pause, metrics, HISTOGRAMS_FOLDER
            )
            if should_exit:
                break

    save_all_histograms(metrics, HISTOGRAMS_FOLDER)     # ヒストグラムの作成と保存

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
