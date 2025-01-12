from ultralytics import YOLO


def load_yolo_model(model_path):
    return YOLO(model_path)

def process_frame(cap, frame_skip_interval):
    """指定されたフレームスキップを考慮してフレームを取得"""
    for _ in range(frame_skip_interval - 1):
        cap.grab()
    success, frame = cap.read()
    return success, frame

def perform_yolo(frame, model, classes=None, conf=0.3):
    """YOLOを使った推論"""
    results = model.track(frame, persist=True, conf=conf, classes=classes, verbose=False)
    detections = []
    if len(results) != 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        coords = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        for track_id, xyxy in zip(ids, coords):
            detections.append((track_id, xyxy.tolist()))
    return detections

def update_tracked_data(tracked_data, detections):
    for track_id, bbox in detections:
        if track_id not in tracked_data:
            tracked_data[track_id] = []
        tracked_data[track_id].append(bbox)

def process_frame_data(frame, model, classes, tracked_data, predict_bbox):
    # YOLOによる検出
    detections = perform_yolo(frame, model, classes)
    detection_dict = {track_id: bbox for track_id, bbox in detections}  # リストを辞書型に変換
    update_tracked_data(tracked_data, detections)
    # 予測
    predictions = {
        track_id: predict_bbox(tracked_data, track_id, 10)
        for track_id in tracked_data
    }
    return detection_dict, predictions