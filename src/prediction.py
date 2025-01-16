import numpy as np

def get_prediction_function(method):
    """予測方法を動的に切り替える"""
    if method == "linear":
        return predict_bbox_linear
    if method == "quadratic":
        return predict_bbox_quadratic
    if method == "kalman":
        return predict_bbox_kalman
    if method == "quadratic_weight":
        return predict_bbox_quadratic_weighted_with_outlier_removal
    else:
        raise ValueError(f"Unknown:{method}")

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

def predict_bbox_quadratic_weighted(tracked_data, track_id, num_frames=10):
    """重み付けを使用して二次補完による次フレームのバウンディングボックスを予測"""
    if track_id not in tracked_data or len(tracked_data[track_id]) < num_frames:
        print(f"データ不足；{track_id}")
        return None     # データが不足している場合, 予測不可
    recent_bboxes = tracked_data[track_id][-num_frames:]
    if any(bbox is None for bbox in recent_bboxes):
        print(f"Noneが含まれている：{track_id}")
        return None     # Noneが含まれている場合, 予測不可
    frames = np.arange(-num_frames + 1, 1)  # フレーム番号（例: [-9, ..., 0]）
    weights = np.exp(-np.abs(frames))       # 過去のフレームほど重視（例: 指数関数で重み付け）
    predicted_bbox = []
    for j in range(4):  # x1, y1, x2, y2 の順に処理
        coords = np.array([recent_bboxes[i][j] for i in range(num_frames)])
        coeffs = np.polyfit(frames, coords, 2, w=weights)  # 重み付けを追加
        next_value = np.polyval(coeffs, 1)  # 次フレームの予測
        predicted_bbox.append(next_value)
    return predicted_bbox

def detect_outliers(data, threshold=1.5):
    """IQRを用いた外れ値検出"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def predict_bbox_quadratic_weighted_with_outlier_removal(tracked_data, track_id, num_frames=10):
    """二次補完 + 重み付け + 外れ値除去"""
    if track_id not in tracked_data or len(tracked_data[track_id]) < num_frames:
        print(f"データ不足；{track_id}")
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]
    if any(bbox is None for bbox in recent_bboxes):
        print(f"Noneが含まれている：{track_id}")
        return None
    frames = np.arange(-num_frames + 1, 1)  # フレーム番号

    predicted_bbox = []
    for j in range(4):  # x1, y1, x2, y2 の順に処理
        coords = [recent_bboxes[i][j] for i in range(num_frames)]
        # 外れ値除去
        filtered_coords = detect_outliers(coords)
        if len(filtered_coords) < 3:  # データが少なすぎる場合
            print(f"有効なデータが不足しています：{track_id}, {filtered_coords}")
            return None
        # 重み計算（最新フレームが最小の重み）
        weights = np.linspace(1.0, 2.0, len(filtered_coords))
        coeffs = np.polyfit(frames[:len(filtered_coords)], filtered_coords, 2, w=weights)
        # 次フレームの予測
        next_value = np.polyval(coeffs, 1)
        predicted_bbox.append(next_value)
    return predicted_bbox

def test_predict_bbox_quadratic_weighted():
    """重み付けを使用した predict_bbox_quadratic_weighted のテスト"""
    # サンプルデータ：トラックID 1 の過去 20 フレーム分のデータ
    tracked_data = {
        1: [
            [100 + i, 100 + i, 200 + i, 200 + i] for i in range(20)
        ]
    }
    track_id = 1
    num_frames = 10

    # 予測を実行
    predicted_bbox = predict_bbox_quadratic_weighted(tracked_data, track_id, num_frames=num_frames)

    # テストの結果を表示
    print("=== 重み付け二次予測のテスト ===")
    print(f"トラックID: {track_id}")
    print(f"過去 {num_frames} フレームのデータ: {tracked_data[track_id][-num_frames:]}")
    print(f"予測された次フレームのバウンディングボックス: {predicted_bbox}")

    # 簡単な期待値テスト
    expected_bbox = [110 + num_frames, 110 + num_frames, 210 + num_frames, 210 + num_frames]
    for p, e in zip(predicted_bbox, expected_bbox):
        assert abs(p - e) < 1e-3, f"予測値が期待値と一致しません: Predicted={p}, Expected={e}"

    print("テスト成功！")

def test_predict_bbox_quadratic_weighted_with_anomalies():
    """異常データを含む場合の重み付け二次予測のテスト"""
    # 通常データ
    tracked_data_normal = {
        1: [
            [100 + i, 100 + i, 200 + i, 200 + i] for i in range(19)
        ]
    }

    # 異常データ（最新フレームだけ異常）
    tracked_data_anomalous = {
        1: tracked_data_normal[1] + [[500, 500, 600, 600]]  # 異常値
    }

    track_id = 1
    num_frames = 10

    print("=== 異常データを含む場合のテスト ===")
    print(f"過去データ（通常）: {tracked_data_normal[1][-num_frames:]}")
    print(f"過去データ（異常あり）: {tracked_data_anomalous[1][-num_frames:]}")

    # 通常データで予測
    predicted_normal = predict_bbox_quadratic_weighted(
        tracked_data_normal, track_id, num_frames=num_frames
    )

    # 異常データで予測
    predicted_anomalous = predict_bbox_quadratic_weighted(
        tracked_data_anomalous, track_id, num_frames=num_frames
    )

    # 期待される結果（通常データ）
    expected_bbox = [119, 119, 219, 219]

    print("=== 結果 ===")
    print(f"通常データでの予測: {predicted_normal}")
    print(f"異常データでの予測: {predicted_anomalous}")
    print(f"期待される結果: {expected_bbox}")

    # 検証
    for p, e in zip(predicted_normal, expected_bbox):
        assert abs(p - e) < 1e-3, f"通常データの予測が期待値と一致しません: Predicted={p}, Expected={e}"

    for p, e in zip(predicted_anomalous, expected_bbox):
        assert abs(p - e) < 1e-3, f"異常データの予測が期待値と一致しません: Predicted={p}, Expected={e}"

    print("異常データを含む場合のテスト成功！")

def test_predict_bbox_quadratic_weighted_with_outlier_removal():
    """異常データを含む場合の重み付け二次予測 + 外れ値除去のテスト"""
    # 通常データ
    tracked_data_normal = {
        1: [[100 + i, 100 + i, 200 + i, 200 + i] for i in range(19)]
    }

    # 異常データ（最新フレームだけ異常）
    tracked_data_anomalous = {
        1: tracked_data_normal[1] + [[500, 500, 600, 600]]  # 異常値
    }

    track_id = 1
    num_frames = 10

    print("=== 異常データを含む場合のテスト（外れ値除去） ===")
    print(f"過去データ（通常）: {tracked_data_normal[1][-num_frames:]}")
    print(f"過去データ（異常あり）: {tracked_data_anomalous[1][-num_frames:]}")

    # 通常データで予測
    predicted_normal = predict_bbox_quadratic_weighted_with_outlier_removal(
        tracked_data_normal, track_id, num_frames=num_frames
    )

    # 異常データで予測
    predicted_anomalous = predict_bbox_quadratic_weighted_with_outlier_removal(
        tracked_data_anomalous, track_id, num_frames=num_frames
    )

    # 期待される結果（通常データ）
    expected_bbox = [119, 119, 219, 219]

    print("=== 結果 ===")
    print(f"通常データでの予測: {predicted_normal}")
    print(f"異常データでの予測: {predicted_anomalous}")
    print(f"期待される結果: {expected_bbox}")

    # 検証
    for p, e in zip(predicted_normal, expected_bbox):
        assert abs(p - e) < 1e-3, f"通常データの予測が期待値と一致しません: Predicted={p}, Expected={e}"

    for p, e in zip(predicted_anomalous, expected_bbox):
        assert abs(p - e) < 1e-3, f"異常データの予測が期待値と一致しません: Predicted={p}, Expected={e}"

    print("異常データを含む場合のテスト成功！")

if __name__ == "__main__":
    # test_predict_bbox_quadratic_weighted()
    # test_predict_bbox_quadratic_weighted_with_anomalies()
    test_predict_bbox_quadratic_weighted_with_outlier_removal()