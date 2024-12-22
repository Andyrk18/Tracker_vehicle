import numpy as np

def update_tracked_data(tracked_data, detections):
    for track_id, bbox in detections:
        if track_id not in tracked_data:
            tracked_data[track_id] = []
        tracked_data[track_id].append(bbox)

def predict_next_bbox(tracked_data, track_id, num_frames=2):
    """線形補完を用いた次フレームの予測"""
    if len(tracked_data[track_id]) < num_frames:
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]
    deltas = [recent_bboxes[i+1][j] - recent_bboxes[i][j] for i in range(num_frames-1) for j in range(4)]
    avg_delta = [sum(deltas[i::4]) / (num_frames-1) for i in range(4)]
    predicted_bbox = [recent_bboxes[-1][j] + avg_delta[j] for j in range(4)]
    return predicted_bbox

def predict_next_bbox_quadratic(tracked_data, track_id, num_frames=3):
    """二次補完を使用して次のフレームのバウンディングボックスを予測"""
    # 過去のバウンディングボックス情報を取得
    if len(tracked_data[track_id]) < num_frames:
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]  # 最新の num_frames 分の情報
    frames = np.arange(-num_frames + 1, 1)  # フレーム番号（例: [-2, -1, 0]）
    predicted_bbox = []
    for j in range(4):  # x1, y1, x2, y2 の順に処理
        coords = [recent_bboxes[i][j] for i in range(num_frames)]  # 各座標の値を取得
        # 二次関数の係数を計算
        coeffs = np.polyfit(frames, coords, 2)  # 2次多項式でフィット
        # 次フレーム（フレーム番号 = 1）の値を予測
        next_value = np.polyval(coeffs, 1)  # フレーム番号1を代入
        predicted_bbox.append(next_value)
    return predicted_bbox
