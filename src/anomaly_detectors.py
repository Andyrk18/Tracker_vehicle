from numpy.ma.core import anomalies

from prediction_evaluator import calculate_iou

def detect_iou_anomaly(current_bbox, predicted_bbox, threshold=0.85):
    """IoUを用いた異常検知"""
    if not current_bbox or not predicted_bbox:
        return False
    iou = calculate_iou(current_bbox, predicted_bbox)
    return iou < threshold

def detect_area_anomaly(current_bbox, previous_bbox, threshold=0.5):
    """面積比較による異常検知"""
    if not current_bbox or not previous_bbox:
        return False
    current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    previous_area = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
    change_rate = abs(current_area - previous_area) / max(previous_area, 1e-5)
    return change_rate > threshold

def detect_aspect_ratio_anomaly(current_bbox, previous_bbox, threshold=0.3):
    """縦横比比較による異常検知"""
    if not current_bbox or not previous_bbox:
        return False
    current_ratio = (current_bbox[2] - current_bbox[0]) / max((current_bbox[3] - current_bbox[1]), 1e-5)
    previous_ratio = (previous_bbox[2] - previous_bbox[0]) / max((previous_bbox[3] - previous_bbox[1]), 1e-5)
    change_rate = abs(current_ratio - previous_ratio) / max(previous_ratio, 1e-5)
    return change_rate > threshold

def detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox, iou_threshold=0.5, ratio_threshold=0.2, area_threshold=0.2):
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
    is_anomaly = anomaly_count >= 2     # Trueが２以上ならば
    return is_anomaly, anomalies