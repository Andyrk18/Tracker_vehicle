def display_latest_tracked_data(tracked_data, frame_number):
    print(f"=== Final Tracked Data for Frame {frame_number} ===")
    for track_id, history in tracked_data.items():
        if history:  # 履歴が空でない場合
            latest_bbox = history[-1]
            print(f"  Track ID: {track_id}, Final BBox: {latest_bbox}")

def log_predictions_and_detections(detection_dict, predictions):
    """フレームごとの検出値と予測値を表示"""
    print(f"Detections: {detection_dict}")
    print(f"Predictions: {predictions}")

def log_frame_info(frame_number, detection_dict, predictions, anomalies, tracked_data):
    """フレームごとの情報を整理して表示"""
    print(f"=== Frame {frame_number} ===")

    print("Detections:")
    for track_id, bbox in detection_dict.items():
        print(f"  Track ID: {track_id}, BBox: {bbox}")

    print("Predictions:")
    for track_id, bbox in predictions.items():
        print(f"  Track ID: {track_id}, BBox: {bbox}")

    print("Anomalies:")
    for track_id, status in anomalies.items():
        anomaly_status = "Anomaly" if status else "Normal"
        print(f"  Track ID: {track_id}, Status: {anomaly_status}")

    print("Tracked Data:")
    for track_id, history in tracked_data.items():
        print(f"  Track ID: {track_id}, History: {history[-3:]}")  # 最新3フレームだけ表示