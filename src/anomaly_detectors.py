from prediction_evaluator import calculate_iou

def detect_iou_anomaly(current_bbox, predicted_bbox, iou_threshold=0.85):
    """IoUを用いた異常検知"""
    iou = calculate_iou(current_bbox, predicted_bbox)
    return iou < iou_threshold

def detect_area_anomaly(current_bbox, previous_bbox, area_threshold=0.5):
    """面積比較による異常検知"""
    current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    previous_area = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
    area_change_ratio = abs(current_area - previous_area) / max(previous_area, 1e-5)
    return area_change_ratio > area_threshold

def detect_aspect_ratio_anomaly(current_bbox, previous_bbox, aspect_ratio_threshold=0.3):
    """縦横比比較による異常検知"""
    current_aspect_ratio = (current_bbox[2] - current_bbox[0]) / max((current_bbox[3] - current_bbox[1]), 1e-5)
    previous_aspect_ratio = (previous_bbox[2] - previous_bbox[0]) / max((previous_bbox[3] - previous_bbox[1]), 1e-5)
    aspect_ratio_change = abs(current_aspect_ratio - previous_aspect_ratio)
    return aspect_ratio_change > aspect_ratio_threshold

def detect_combined_anomalies(current_bbox, previous_bbox, predicted_bbox, iou_threshold=0.5, ratio_threshold=0.2, area_threshold=0.2):
    """異常検知を統合"""
