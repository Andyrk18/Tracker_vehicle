import os
import cv2

from yolo_handler import update_tracked_data


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
            print(f"異常検出： {track_id}: {updated_bbox}")  # デバッグ
        else:
            updated_detections.append((track_id, bbox))  # 異常がない場合はそのまま
            print(f"異常なし: {track_id}")              # デバッグ
    # 予測値の中で現在の検出に含まれないトラックIDの追加
    for track_id, predicted_bbox in predictions.items():
        if track_id not in detections and predicted_bbox is not None:
            updated_detections.append((track_id, predicted_bbox))
            print(f"未検出車両： {track_id}: {predicted_bbox}")           # デバッグ
    return updated_detections

def handle_replace(detections, predictions, anomalies, tracked_data):
    updated_detections = replace_with_prediction(detections, predictions, anomalies)
    update_tracked_data(tracked_data, updated_detections)
    # print(f"異常補完完了。更新されたデータ数: {len(updated_detections)}")

def test_replace_with_prediction():
    # サンプルデータ
    detections = {1: [10, 10, 50, 50], 2: [20, 20, 60, 60]}  # 実測値
    predictions = {1: [11, 11, 51, 51], 2: [21, 21, 61, 61], 3: [30, 30, 70, 70]}  # 予測値
    anomalies = {1: True, 2: False}  # 異常判定
    # 関数実行
    updated_detections = replace_with_prediction(detections, predictions, anomalies)
    # 結果確認
    print("=== Updated Detections ===")
    for track_id, bbox in updated_detections:
        print(f"Track ID: {track_id}, BBox: {bbox}")

# 実行
if __name__ == "__main__":
    test_replace_with_prediction()
