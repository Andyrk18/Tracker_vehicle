import cv2

from src.histogram_generator import save_all_histograms

def draw_detections_with_predictions(frame, detections, predictions=None):
    """検出されたバウンディングボックスと予測したバウンディングボックスを表示"""
    # 検出されたバウンディングボックスを描画
    for track_id, bbox in detections.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  # 緑色
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # 予測されたバウンディングボックスを描画
    if predictions:
        for track_id, predicted_bbox in predictions.items():
            if predicted_bbox is None:
                continue
            px1, py1, px2, py2 = predicted_bbox
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)  # 青色
            cv2.putText(frame, f"Pred: {track_id}", (int(px1), int(py1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

def display_frame(frame, frame_number, pause, metrics, histograms_folder):
    cv2.imshow("Result", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # スペースキーで一時停止
        pause = not pause
        if pause:
            print(f"PAUSE : Frame {frame_number}")
    elif key == ord('q'):  # 'q'キーで強制終了
        save_all_histograms(metrics, histograms_folder)
        print("強制終了しました。")
        return pause, True  # 終了フラグをTrueに
    return pause, False  # 終了フラグをFalseに