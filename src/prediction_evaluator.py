def calculate_iou(current_bbox, predicted_bbox):
    """実測値と予測値からIoUを計算"""
    if not current_bbox or not predicted_bbox:
        return None
    x1_inter = max(current_bbox[0], predicted_bbox[0])
    y1_inter = max(current_bbox[1], predicted_bbox[1])
    x2_inter = min(current_bbox[2], predicted_bbox[2])
    y2_inter = min(current_bbox[3], predicted_bbox[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    box2_area = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_area_ratio(current_bbox, previous_bbox):
    """面積比を計算"""
    if not current_bbox or not previous_bbox:
        return None
    area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
    area2 = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
    return min(area1, area2) / max(area1, area2) if area2 > 0 else 0

def calculate_aspect_ratio(current_bbox, previous_bbox):
    """縦横比を計算"""
    if not current_bbox or not previous_bbox:
        return None
    current_aspect = (current_bbox[2] - current_bbox[0]) / max((current_bbox[3] - current_bbox[1]), 1e-5)
    previous_aspect = (previous_bbox[2] - previous_bbox[0]) / max((previous_bbox[3] - previous_bbox[1]), 1e-5)
    return min(current_aspect, previous_aspect) / max(current_aspect, previous_aspect) if previous_aspect > 0 else 0

def log_frame_iou(detections, predictions, frame_number):
    """フレームごとのIoUを計算し、ログに出力"""
    print(f"\nFrame: {frame_number}")
    frame_iou_results = []
    for track_id, actual_bbox in detections:
        if track_id in predictions and predictions[track_id] is not None:
            predicted_bbox = predictions[track_id]
            iou = calculate_iou(actual_bbox, predicted_bbox)
            print(f"Track ID: {track_id}, IoU: {iou:.2f}")
            frame_iou_results.append((track_id, iou))
        else:
            print(f"Track ID: {track_id}, IoU: N/A")
    return frame_iou_results

def summarize_iou_results(iou_history):
    """全フレームのIoUデータを基に車両ごとの一致率を計算し、出力"""
    print("\n--- IoU Summary ---")
    for track_id, iou_list in iou_history.items():
        average_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        print(f"Track ID: {track_id}, Average IoU: {average_iou:.2f}")
