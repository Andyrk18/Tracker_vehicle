import prediction_evaluator as pe

"""　0に近いほど検出結果と予測値（あるいは過去値）は異なる　"""
""" 1に近いほど両者は一致　"""

def detect_iou_anomaly(current_bbox, predicted_bbox, threshold=0.85):
    """IoUを用いた異常検知"""
    if not current_bbox or not predicted_bbox:
        return False
    iou = pe.calculate_iou(current_bbox, predicted_bbox)
    return iou < threshold

def detect_area_anomaly(current_bbox, previous_bbox, threshold=0.5):
    """面積比較による異常検知"""
    if not current_bbox or not previous_bbox:
        return False
    area_ratio = pe.calculate_area_ratio(current_bbox, previous_bbox)
    return area_ratio < threshold

def detect_aspect_ratio_anomaly(current_bbox, previous_bbox, threshold=0.3):
    """縦横比比較による異常検知"""
    if not current_bbox or not previous_bbox:
        return False
    aspect_ratio = pe.calculate_aspect_ratio(current_bbox, previous_bbox)
    return aspect_ratio < threshold

def detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox, iou_threshold=0.5, area_threshold=0.2, ratio_threshold=0.2):
    """異常検知を統合"""
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