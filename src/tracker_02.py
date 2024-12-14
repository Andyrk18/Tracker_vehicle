import cv2
from collections import defaultdict

from ultralytics import YOLO
import process_video as pv


# YOLO設定
YOLO_CLASSES = [2]      # 検出対象
YOLO_MODEL_PATH = "../models/yolov8n.pt"
YOLO_MODEL = YOLO(YOLO_MODEL_PATH)

# 設定
SAVE_VIDEO = False      # 動画保存
DISPLAY_VIDEO = True    # 動画表示

# データ構造
tracked_data = defaultdict(list)
predicted_boxes = {}

def track_video(file_path, target_fps, output_path=None):
    """動画を処理し、検出結果を保存"""
    global predicted_boxes
    cap, frame_skip, video_writer = pv.initialize_processing(file_path, target_fps, output_path)
    frame_count = 0

    while cap.isOpened():
        success, frame = pv.process_frame(cap, frame_skip)
        if not success:
            break
        # YOLO推論
        detections = pv.perform_yolo(frame, YOLO_MODEL, YOLO_CLASSES)
        # 検出結果を保存
        for track_id, bbox in detections:
            tracked_data[track_id].append(bbox)
        new_predicted_boxes = {}
        for track_id in tracked_data.keys():
            predicted_bbox = predict_next_bbox(tracked_data, track_id, num_frames=2)
            if predicted_bbox:
                new_predicted_boxes[track_id] = predicted_bbox

        # 検出結果を描画
        if DISPLAY_VIDEO:
            # frame = pv.draw_detections(frame, detections)
            frame = draw_detections_with_predictions(frame ,detections, predicted_boxes)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        predicted_boxes = new_predicted_boxes

        # 動画に書き込み（オプション）
        if SAVE_VIDEO and video_writer:
            video_writer.write(frame)

        frame_count += 1

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    return tracked_data

def predict_next_bbox(tracked_data, track_id, num_frames=3):
    """T+1フレームの位置を予測"""
    if track_id not in tracked_data or len(tracked_data[track_id]) < num_frames:
        if track_id not in tracked_data:
            print(f"未検出：{track_id}")
        elif len(tracked_data[track_id]) < num_frames:
            print(f"データ不足：{track_id}")
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]  # 最新のnum_frames分を取得
    deltas = [
        [recent_bboxes[i][j] - recent_bboxes[i - 1][j] for j in range(4)]
        for i in range(1, num_frames)
    ]
    avg_delta = [sum(delta[j] for delta in deltas) / len(deltas) for j in range(4)]
    # 予測
    last_bbox = recent_bboxes[-1]
    predicted_bbox = [last_bbox[j] + avg_delta[j] for j in range(4)]
    return predicted_bbox

def evaluate_predictions(detections, predictions):
    """予測と検出結果の一致率を評価"""
    total = len(detections)
    matched = 0
    iou_threshold = 0.5

    for track_id, actual_bbox in detections:
        if track_id in predictions:
            predicted_bbox = predictions[track_id]
            iou = calculate_iou(actual_bbox, predicted_bbox)
            if iou > iou_threshold:
                matched += 1
                print(f"Track ID {track_id}: IoU = {iou:.2f} -> Matched")
            else:
                print(f"Track ID {track_id}: IoU = {iou:.2f} -> Not Matched")

    match_rate = matched / total if total > 0 else 0
    print(f"Match Rate: {match_rate:.2f}")
    return match_rate

def calculate_iou(box1, box2):
    """IoU (Intersection over Union) を計算"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0





if __name__ == "__main__":
    input_file = "../videos/street1_sample_01.mp4"
    output_file = "../videos/output_video_01.mp4" if SAVE_VIDEO else None

    data = track_video(input_file, target_fps=5, output_path=output_file)
    # 結果を表示
    print(f"Total Tracked Vehicles: {len(data)}")
