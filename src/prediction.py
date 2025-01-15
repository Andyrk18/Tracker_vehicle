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


def test_prediction_functions():
    # テストデータ
    tracked_data = {
        1: [
            [10, 20, 50, 60],  # Frame 1
            [15, 25, 55, 65],  # Frame 2
            [20, 30, 60, 70],  # Frame 3
            [25, 35, 65, 75],  # Frame 4
            [30, 40, 70, 80],  # Frame 5
            [35, 45, 75, 85],  # Frame 6
            [40, 50, 80, 90],  # Frame 7
            [45, 55, 85, 95],  # Frame 8
            [50, 60, 90, 100], # Frame 9
            [55, 65, 95, 105]  # Frame 10
        ]
    }

    # 線形補完
    print("Testing Linear Prediction:")
    predicted_bbox_linear = predict_bbox_linear(tracked_data, 1, num_frames=2)
    print(f"Predicted BBox (Linear): {predicted_bbox_linear}")

    # 二次補完
    print("\nTesting Quadratic Prediction:")
    predicted_bbox_quadratic = predict_bbox_quadratic(tracked_data, 1, num_frames=3)
    print(f"Predicted BBox (Quadratic): {predicted_bbox_quadratic}")

    # カルマンフィルタ（未実装の場合は確認スキップ）
    print("\nTesting Kalman Prediction:")
    try:
        predicted_bbox_kalman = predict_bbox_kalman()
        print(f"Predicted BBox (Kalman): {predicted_bbox_kalman}")
    except NotImplementedError:
        print("Kalman Filter prediction is not implemented yet.")

if __name__ == "__main__":
    test_prediction_functions()