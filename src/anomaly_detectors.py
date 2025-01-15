import src.prediction_evaluator as evaluator

"""　0に近いほど検出結果と予測値（あるいは過去値）は異なる　"""
""" 1に近いほど両者は一致　"""

def detect_iou_anomaly(current_bbox, predicted_bbox, threshold=0.90):
    """
    IoUを用いた異常検知
    異常があればTrue, 無ければFalse
    """
    if not current_bbox or not predicted_bbox:
        return False
    iou = evaluator.calculate_iou(current_bbox, predicted_bbox)
    # print(f"Iou: {iou}")                        # debug
    return iou < threshold

def detect_area_anomaly(current_bbox, previous_bbox, threshold=0.90):
    """
    面積比較による異常検知
    異常があればTrue, 無ければFalse
    """
    if not current_bbox or not previous_bbox:
        return False
    area_ratio = evaluator.calculate_area_ratio(current_bbox, previous_bbox)
    # print(f"Area_Ratio: {area_ratio}")          # debug
    return area_ratio < threshold

def detect_aspect_ratio_anomaly(current_bbox, previous_bbox, threshold=0.90):
    """
    縦横比比較による異常検知
    異常があればTrue, 無ければFalse
    """
    if not current_bbox or not previous_bbox:
        return False
    aspect_ratio = evaluator.calculate_aspect_ratio(current_bbox, previous_bbox)
    # print(f"Aspect_Raito: {aspect_ratio}")      # debug
    return aspect_ratio < threshold

def detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox, iou_threshold=0.90, area_threshold=0.90, ratio_threshold=0.90):
    """
    異常検知を統合
    異常があれば最終的にTrueを返す
    """
    iou_anomaly = detect_iou_anomaly(current_bbox, predicted_bbox, iou_threshold)
    area_anomaly = detect_area_anomaly(current_bbox, previous_bbox, area_threshold)
    aspect_ratio_anomaly = detect_aspect_ratio_anomaly(current_bbox, previous_bbox, ratio_threshold)
    anomalies = {
        "IoU Anomaly": iou_anomaly,
        "Area Anomaly": area_anomaly,
        "Aspect Ratio Anomaly": aspect_ratio_anomaly
    }
    anomaly_count = sum(anomalies.values())
    is_anomaly = anomaly_count >= 2     # Trueが２以上かどうか
    return is_anomaly, anomalies

if __name__ == "__main__":
    current_bbox = [10, 10, 50, 50]
    predicted_bbox = [15, 15, 55, 55]
    print(detect_iou_anomaly(current_bbox, predicted_bbox))