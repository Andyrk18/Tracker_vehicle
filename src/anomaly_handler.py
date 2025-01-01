import os
import cv2

def save_anomaly_frame(frame, frame_number, track_id, output_folder):
    """異常検知フレームを画像データとして保存"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, f"frame_{frame_number}_track_{track_id}.jpg")
    cv2.imwrite(filename, frame)

def log_anomaly_info(frame_number, track_id, current_bbox, predicted_bbox, anomalies):
    """異常検知結果をターミナルに出力"""
    print(f"\n[Frame {frame_number}] Track ID: {track_id}")
    print(f"  Current BBox: {current_bbox}")
    print(f"  Predicted BBox: {predicted_bbox}")
    for anomaly_type, is_detected in anomalies.items():
        status = "Detected" if is_detected else "Normal"
        print(f"  {anomaly_type}: {status}")

def replace_with_prediction(detections, predictions, anomalies):
    """異常検知されたトラックIDの検出結果を予測値に置き換える"""
    updated_detections = []
    for track_id, bbox in detections.items():
        if anomalies.get(track_id, False):  # 異常と判定された場合
            updated_bbox = predictions.get(track_id, bbox)  # 予測値があれば置き換える
            updated_detections.append((track_id, updated_bbox))
        else:
            updated_detections.append((track_id, bbox))  # 異常がない場合はそのまま
    return updated_detections
