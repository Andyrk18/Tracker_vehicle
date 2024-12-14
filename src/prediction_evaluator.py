def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(detections, predictions):
    total = len(detections)
    matched = 0
    for track_id, actual_bbox in detections:
        if track_id in predictions and predictions[track_id] is not None:
            predicted_bbox = predictions[track_id]
            iou = calculate_iou(actual_bbox, predicted_bbox)
            if iou > 0.5:
                matched += 1
    match_rate = matched / total if total > 0 else 0
    print(f"Match Rate: {match_rate:.2f}")
