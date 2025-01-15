import numpy as np


def predict_bbox_linear(tracked_data, track_id, num_frames=4):
    """線形補完を用いた次フレームの予測"""
    if track_id not in tracked_data or len(tracked_data[track_id]) < num_frames:
        print(f"データ不足：{track_id}")
        return None     # データが不足している場合, 予測不可
    recent_bboxes = tracked_data[track_id][-num_frames:]
    # print(f"ID-{track_id}, 参照データ：{recent_bboxes}")
    if any(bbox is None for bbox in recent_bboxes):
        print(f"Noneが含まれている：{track_id}")
        return None     # Noneが含まれている場合, 予測不可
    deltas = [recent_bboxes[i + 1][j] - recent_bboxes[i][j] for i in range(num_frames - 1) for j in range(4)]
    avg_delta = [sum(deltas[i::4]) / (num_frames - 1) for i in range(4)]
    predicted_bbox = [recent_bboxes[-1][j] + avg_delta[j] for j in range(4)]
    return predicted_bbox

def predict_bbox_quadratic(tracked_data, track_id, num_frames=10):
    """二次補完を使用して次のフレームのバウンディングボックスを予測"""
    if track_id not in tracked_data or len(tracked_data[track_id]) < num_frames:
        print(f"データ不足；{track_id}")
        return None     # データが不足している場合, 予測不可
    recent_bboxes = tracked_data[track_id][-num_frames:]
    if any(bbox is None for bbox in recent_bboxes):
        print(f"Noneが含まれている：{track_id}")
        return None     # Noneが含まれている場合, 予測不可
    frames = np.arange(-num_frames + 1, 1)  # フレーム番号（例: [-2, -1, 0]）
    predicted_bbox = []
    for j in range(4):                              # x1, y1, x2, y2 の順に処理
        coords = [recent_bboxes[i][j] for i in range(num_frames)]
        coeffs = np.polyfit(frames, coords, 2)     # 2次多項式でフィット
        next_value = np.polyval(coeffs, 1)          # 次フレームの予測
        predicted_bbox.append(next_value)
    return predicted_bbox

def predict_bbox_kalman():
    """カルマンフィルターを利用してバウンディングボックスを予測"""
    pass

def get_prediction_function(method):
    """予測方法を動的に切り替える"""
    if method == "linear":
        return predict_bbox_linear
    if method == "quadratic":
        return predict_bbox_quadratic
    if method == "kalman":
        return predict_bbox_kalman
    else:
        raise ValueError(f"Unknown:{method}")

