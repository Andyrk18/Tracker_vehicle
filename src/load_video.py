import os
import cv2


def load_video_files_from_folder(folder_path):
    """指定されたフォルダ内の動画ファイルを読み取り、名前順にリストで返す"""
    if not os.path.exists(folder_path):
        raise ValueError(f"フォルダ {folder_path} が見つかりません。")

    video_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.mp4', '.AVI', '.mov'))]
    video_files.sort()  # ファイル名順にソート

    if not video_files:
        raise ValueError("指定されたフォルダには動画ファイルが含まれていません。")

    return video_files


def load_video(video_path):
    """動画をロードし、VideoCaptureオブジェクトを返す"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"動画が見つかりません。： {video_path}")
        return None

    return cap

