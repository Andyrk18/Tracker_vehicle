import numpy as np

def update_tracked_data(tracked_data, detections):
    for track_id, bbox in detections:
        if track_id not in tracked_data:
            tracked_data[track_id] = []
        tracked_data[track_id].append(bbox)

def predict_next_bbox(tracked_data, track_id, num_frames=2):
    if len(tracked_data[track_id]) < num_frames:
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]
    deltas = [recent_bboxes[i+1][j] - recent_bboxes[i][j] for i in range(num_frames-1) for j in range(4)]
    avg_delta = [sum(deltas[i::4]) / (num_frames-1) for i in range(4)]
    predicted_bbox = [recent_bboxes[-1][j] + avg_delta[j] for j in range(4)]
    return predicted_bbox

def predict_next_bbox_curve_fit(tracked_data, track_id, num_frames=3):
    """2次近似曲線を使ったT+1フレームの予測"""
    if len(tracked_data[track_id]) < num_frames:
        return None

    # 過去のバウンディングボックスを取得
    recent_bboxes = np.array(tracked_data[track_id][-num_frames:])
    # 各座標のフィッティング
    x1 = recent_bboxes[:, 0]
    y1 = recent_bboxes[:, 1]
    x2 = recent_bboxes[:, 2]
    y2 = recent_bboxes[:, 3]
    # 2次近似曲線をフィッティング
    t = np.arange(num_frames)
    p_x1 = np.polyfit(t, x1, 2)
    p_y1 = np.polyfit(t, y1, 2)
    p_x2 = np.polyfit(t, x2, 2)
    p_y2 = np.polyfit(t, y2, 2)
    # 次のフレームの時間を計算
    t_next = num_frames
    # 次のフレームの座標を予測
    predicted_bbox = [
        np.polyval(p_x1, t_next),
        np.polyval(p_y1, t_next),
        np.polyval(p_x2, t_next),
        np.polyval(p_y2, t_next),
    ]
    return predicted_bbox
