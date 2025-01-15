import unittest

from src.yolo_handler import update_tracked_data
from src.prediction import predict_bbox_linear, predict_bbox_quadratic
import src.anomaly_handler as anomaly

class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        """テストケースごとの初期化"""
        self.detections   = {1: [10,10,50,50], 2: [20,20,60,60]}
        self.predictions  = {1: [11,11,51,51], 2: [21,21,61,61], 3: [30,30,70,70]}
        self.anomalies    = {1: True, 2: False}
        self.tracked_data = {1: [[5,5,45,45]], 2: [[15,15,55,55]]}

    def test_handle_replace(self):
        """handle_replace 関数の詳細テスト"""
        print("=== handle_replace 関数のテスト ===")
        anomaly.handle_replace(self.detections, self.predictions, self.anomalies, self.tracked_data)
        expected_tracked_data = {
            1: [[5, 5, 45, 45],   [11, 11, 51, 51]],      # 異常検知で予測値に置き換え
            2: [[15, 15, 55, 55], [20, 20, 60, 60]],    # 正常時、実測値を保持
            3: [[30, 30, 70, 70]]                       # 未検出で予測値を追加
        }
        self.assertEqual(self.tracked_data, expected_tracked_data)

    def test_update_tracked_data(self):
        """update_tracked_data 関数の詳細テスト"""
        print("=== updata_tracked_data 関数のテスト ===")
        new_detections = [(1, [50, 50, 100, 100]), (3, [30, 30, 70, 70])]
        update_tracked_data(self.tracked_data, new_detections)
        expected_tracked_data = {
            1: [[5, 5, 45, 45], [50, 50, 100, 100]],
            2: [[15, 15, 55, 55]],
            3: [[30, 30, 70, 70]]  # 新しい ID が追加される
        }
        self.assertEqual(self.tracked_data, expected_tracked_data)

    def test_predict_bbox_linear(self):
        """predict_bbox_linear 関数のテスト"""
        print("=== predict_bbox_linear 関数のテスト ===")
        self.tracked_data[1] = [[5, 5, 45, 45], [10, 10, 50, 50], [15, 15, 55, 55]]
        track_id = 1
        predicted_bbox = predict_bbox_linear(self.tracked_data, track_id, num_frames=3)
        expected_bbox = [20, 20, 60, 60]
        self.assertAlmostEqual(predicted_bbox, expected_bbox)

    def test_predict_bbox_quadratic(self):
        """predict_bbox_quadratic 関数のテスト"""
        print("=== predict_bbox_quadratic 関数のテスト ===")
        self.tracked_data[1] = [[5, 5, 45, 45], [10, 10, 50, 50], [16, 16, 56, 56], [23, 23, 63, 63], [30, 30, 70, 70]]
        # self.tracked_data[1] = [[5, 5, 45, 45], [10, 10, 50, 50], [15, 15, 55, 55], [20, 20, 60, 60], [25, 25, 65, 65]]
        track_id = 1
        predicted_bbox = predict_bbox_quadratic(self.tracked_data, track_id, num_frames=5)
        expected_bbox = [38, 38, 78, 78]
        # expected_bbox = [30, 30, 70, 70]
        tolerance = 5
        for p, e in zip(predicted_bbox, expected_bbox):
            # self.assertAlmostEqual(p, e, places=5)
            self.assertTrue(abs(p - e) < tolerance)

    def test_edge_cse_empty_data(self):
        """空のデータに対する予測テスト"""
        print("=== 空データに対する予測のテスト ===")
        self.tracked_data = {}
        track_id = 1
        predicted_bbox = predict_bbox_linear(self.tracked_data, track_id, num_frames=3)
        # predicted_bbox = predict_bbox_quadratic(self.tracked_data, track_id, num_frames=5)
        self.assertIsNone(predicted_bbox)

    def test_edge_case_insufficient_data(self):
        """データ不足時の予測テスト"""
        print("=== データ不足時の予測のテスト ===")
        self.tracked_data[1] = [[5, 5, 45, 45]]
        track_id = 1
        predicted_bbox = predict_bbox_linear(self.tracked_data, track_id, num_frames=3)
        # predicted_bbox = predict_bbox_quadratic(self.tracked_data, track_id, num_frames=5)
        self.assertIsNone(predicted_bbox)


if __name__ == '__main__':
    unittest.main()