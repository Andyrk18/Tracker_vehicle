def calculate_iou(box1, box2):
    """実測値と予測値からIoUを計算"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

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
