import cv2


def draw_detections_with_predictions(frame, detections, predictions=None):
    """検出されたバウンディングボックスと予測したバウンディングボックスを表示"""
    for track_id, bbox in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # 緑色
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if predictions:
        for track_id, predicted_bbox in predictions.items():
            if predicted_bbox is None:
                continue
            px1, py1, px2, py2 = predicted_bbox
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)  # 青色
            cv2.putText(frame, f"Pred: {track_id}", (int(px1), int(py1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame