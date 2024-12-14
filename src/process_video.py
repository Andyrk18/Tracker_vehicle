import cv2
from ultralytics import YOLO
from load_video import load_video
from get_fps import get_fps, calculate_frame_skip_interval


def load_yolo_model(model_path, classes=None, conf=0.3):
    model = YOLO(model_path)
    if classes is not None:
        model.classes = classes
    model.conf = conf
    return model

def initialize_processing(file_path, target_fps, output_file=None):
    """動画の初期化"""
    cap = load_video(file_path)
    original_fps = get_fps(cap)
    frame_skip = calculate_frame_skip_interval(original_fps, target_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = Non

    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    return cap, frame_skip, video_writer

def process_frame(cap, frame_skip):
    """指定されたフレームスキップを考慮してフレームを取得"""
    for _ in range(frame_skip - 1):
        cap.grab()
    success, frame = cap.read()
    return success, frame

def perform_yolo(frame, model, classes):
    """YOLOを使った推論"""
    results = model.track(frame, persist=True, verbose=False)
    detections = []
    if len(results) != 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        coords = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        for track_id, xyxy in zip(ids, coords):
            detections.append((track_id, xyxy.tolist()))
    return detections

def draw_detections(frame, detections):
    """フレームに検出結果を描画"""
    for track_id, bbox in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def draw_detections_with_predictions(frame, detections, predictions=None):
    """検出されたバウンディングボックスと予測したバウンディングボックスを表示"""
    for track_id, bbox in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 緑色
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if predictions:
        for track_id, predicted_bbox in predictions.items():
            px1, py1, px2, py2 = predicted_bbox
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)  # 青色
            cv2.putText(frame, f"Pred: {track_id}", (int(px1), int(py1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame
