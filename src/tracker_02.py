from collections import defaultdict, deque
from tkinter.messagebox import showerror

from pykalman import KalmanFilter

import cv2
import torch.cuda
from ultralytics import YOLO

from get_fps import get_fps, calculate_frame_skip_interval
from load_video import load_video

YOLO_Classes = [2, 3, 5, 7]  # 2:car, 3:motorcycle, 5:bus, 7:truck
STATIC_FRAME_THRESHOLD = 3
POSITION_THRESHOLD_WAITING = 10

YOLO_Model = "../models/yolov8x.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = YOLO(YOLO_Model)
MODEL.to(device)

Vehicle_Positions = defaultdict(list)
Tracked_Vehicles = defaultdict(dict)
Vehicle_Frame_Count = defaultdict(int)
Last_Frame = {}

save_video = False

def process_video(file_path, target_fps,
                  show_window=False):
    """動画フレームを読み込んで、YOLOで解析、結果をCSVに保存"""
    cap, frame_skip_interval, video_writer = initialize_processing(file_path, target_fps, output_file=None)
    frame_cnt = 0
    retention_threshold = 100
    frame_buffer = deque(maxlen=3)  # 直近3フレーム分のデータを保持するバッファ
    success, sample_frame = cap.read()

    if not success:
        raise RuntimeError("動画の読み込みに失敗しました。")
    frame_height, frame_width, _ = sample_frame.shape  # フレームサイズを取得
    cap.release()
    cap = load_video(file_path)  # キャプチャを再初期化

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存フォーマット（例: MP4）
        video_writer = cv2.VideoWriter("../../data/proccessed_03.mp4", fourcc, target_fps, (1920, 1080))

    while cap.isOpened():
        success, frame = process_frame(cap, frame_skip_interval)
        if not success:
            break
        frame_cnt += 1
        # YOLO推論
        frame_vehicles = perform_yolo_inference(frame)
        # バッファに追加
        frame_buffer.append(frame_vehicles)
        # バッファが満杯の場合に補完を実行
        if len(frame_buffer) == 3:
            vehicles_t_minus_1 = frame_buffer[0]
            vehicles_t = frame_buffer[1]
            vehicles_t_plus_1 = frame_buffer[2]
            # 補完ロジックの実行
            vehicles_t = check_frames(vehicles_t_minus_1, vehicles_t, vehicles_t_plus_1, frame_cnt)
            # 補完済みのデータでフレームtを処理
            for track_id, xyxy in vehicles_t:
                if show_window:
                    draw_frame(frame, track_id, xyxy)
            if save_video:
                video_writer.write(frame)
            # フレーム表示
            if show_window:
                cv2.imshow("検出結果", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        # 古いトラックの削除
        remove_old_tracks(Vehicle_Positions, Tracked_Vehicles, Vehicle_Frame_Count, Last_Frame, frame_cnt,
                          retention_threshold)
    if save_video:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


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

def initialize_kalman_filter():
    """カルマンフィルタの初期化"""
    return KalmanFilter(
        initial_state_mean=[0, 0, 0, 0],
        transition_matrices=[[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]],
        observation_matrices=[[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]
    )

def process_frame(cap, frame_skip_interval):
    """フレームの読み込みとスキップ処理"""
    for _ in range(frame_skip_interval - 1):
        cap.grab()  # フレームスキップ
    success, frame = cap.read()
    return success, frame

def perform_yolo_inference(frame):
    """YOLO推論の実行"""
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    results = MODEL.track(frame, persist=True, conf=0.5,
                          classes=YOLO_Classes,
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

def update_tracked_vehicles(frame_vehicles, frame_cnt):
    """現在のフレームの情報をトラッキングデータに反映"""
    global Tracked_Vehicles
    for track_id, bbox in frame_vehicles:
        if track_id not in Tracked_Vehicles:
            Tracked_Vehicles[track_id] = {
                'bbox_history': deque(maxlen=5),  # 履歴の保存（最大5フレーム）
                'last_seen_frame': frame_cnt
            }
        Tracked_Vehicles[track_id]['bbox_history'].append(bbox)
        Tracked_Vehicles[track_id]['last_seen_frame'] = frame_cnt

def check_frames(vehicles_t_minus_1, vehicles_t, vehicles_t_plus_1, frame_cnt):
    """フレーム間の補正"""
    global Tracked_Vehicles
    corrected_vehicles_t = vehicles_t.copy()
    all_track_ids = {v[0] for v in vehicles_t_minus_1} | {v[0] for v in vehicles_t} | {v[0] for v in vehicles_t_plus_1}

    for track_id in all_track_ids:
        if not any(v[0] == track_id for v in vehicles_t):
            # フレームtに存在しない車両を補完
            if track_id in Tracked_Vehicles:
                last_bbox = Tracked_Vehicles[track_id]['bbox_history'][-1]
                corrected_vehicles_t.append((track_id, last_bbox))

    # トラッキング情報を更新
    update_tracked_vehicles(corrected_vehicles_t, frame_cnt)
    return corrected_vehicles_t

def complement_missing_vehicles(frame_buffer, current_frame_idx):
    """見失われた車両を補完"""
    current_frame_vehicles = frame_buffer[current_frame_idx]
    for track_id, bbox in Vehicle_Positions.items():
        if track_id not in current_frame_vehicles:
            if Last_Frame[track_id] == current_frame_idx - 1:  # 直前のフレームで見つかっている場合
                frame_buffer[current_frame_idx][track_id] = bbox[-1]

def interpolate_bbox(bbox1, bbox2, alpha=0.5):
    """加重線形補間"""
    return [(1 - alpha) * bbox1[i] + alpha * bbox2[i] for i in range(4)]

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
    color = (0, 255 ,0)
    cv2.putText(frame, display_text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

if __name__ == "__main__":
    input_file = "../videos/street1_sample_03.mp4"
    # save_video = True
    process_video(input_file, 5, True)
