import cv2


def get_fps(cap):
    """動画の元のFPSを取得"""
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"元のFPS： {original_fps}")
    return original_fps


def calculate_frame_skip_interval(original_fps, target_fps):
    """指定されたターゲットFPSに基づいてフレームスキップ間隔を計算"""
    frame_skip_interval = int(original_fps // target_fps)
    print(f"フレームスキップ間隔　: {frame_skip_interval}")
    return frame_skip_interval