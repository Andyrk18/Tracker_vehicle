import cv2
from load_video import load_video
from get_fps import get_fps, calculate_frame_skip_interval


def initialize_processing(file_path, target_fps, output_file):
    """初期化処理：動画読み込み、パラメータ設定"""
    cap = load_video(file_path)
    original_fps = get_fps(cap)
    frame_skip_interval = calculate_frame_skip_interval(original_fps, target_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))
    return cap, frame_skip_interval, video_writer


def process_frame(cap, frame_skip_interval):
    """フレームの読み込みとスキップ処理"""
    for _ in range(frame_skip_interval - 1):
        cap.grab()  # フレームスキップ
    success, frame = cap.read()
    return success, frame


def perform_yolo_inference(frame, model, classes):
    """YOLO推論の実行"""
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    results = model.track(frame, persist=True, conf=0.5,
                          classes=classes,
                          verbose=False)
    frame_vehicles = []
    if len(results) != 0 and results[0].boxes is not None:
        result = results[0]
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        xyxys = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        for track_id, xyxy in zip(ids, xyxys):
            frame_vehicles.append((track_id, xyxy))
    return frame_vehicles


def remove_old_tracks(vehicle_positions, tracked_vehicles, vehicle_frame_count, last_frame, current_frame,
                      retention_threshold):
    """一定時間検出されていない車両データを削除"""
    tracks_to_remove = [
        track_id for track_id, last_seen in last_frame.items()
        if current_frame - last_seen > retention_threshold
    ]
    for track_id in tracks_to_remove:
        del vehicle_positions[track_id]
        del tracked_vehicles[track_id]
        del vehicle_frame_count[track_id]
        del last_frame[track_id]


def draw_frame(frame, track_id, xyxy):
    """車両情報をフレームに描画する　確認用　"""
    x1, y1, x2, y2 = xyxy
    display_text = f"{track_id}"
    color = (0, 255, 0)
    cv2.putText(frame, display_text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
