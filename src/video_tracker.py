from tabnanny import verbose

import cv2
import os

from torch import classes
from ultralytics import YOLO


class VideoAnalyzer:
    def __init__(self, model_path, input_video, output_video, show_video=False, save_video=False, target_classes=None):
        """
        動画解析クラスの初期化
        """
        self.model = YOLO(model_path)  # YOLOモデルをロード
        self.input_video = input_video
        self.output_video = output_video
        self.show_video = show_video
        self.save_video = save_video
        self.target_classes = target_classes if target_classes else [2, 3, 5, 7]  # デフォルトは車両クラス

    def load_video(self):
        """
        動画を読み込む
        """
        video = cv2.VideoCapture(self.input_video)
        if not video.isOpened():
            raise ValueError(f"動画を開けません: {self.input_video}")
        return video

    def process_frame(self, frame):
        """
        フレームを解析し、車両の検出・追跡結果を返す
        """
        results = self.model.track(frame, persist=True, classes=self.target_classes, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックス座標
            conf = float(box.conf[0])  # 信頼度
            track_id = int(box.id[0]) if box.id is not None else -1  # トラックID
            detections.append((track_id, x1, y1, x2, y2, conf))
        return detections

    def draw_detections(self, frame, detections):
        """
        フレームに検出結果を描画する
        """
        for det in detections:
            track_id, x1, y1, x2, y2, conf = det
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id} Conf: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def analyze_video(self):
        """
        動画を解析し、結果を保存または表示する
        """
        video = self.load_video()
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval = fps // 5
        frame_cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None
        if self.save_video:
            writer = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        previous_detections = []
        all_results = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_cnt % frame_interval == 0:
                detections = self.process_frame(frame)
                current_detections = {det[0]: det[2:] for det in detections if det[0] != -1}
                all_results.append(current_detections)
                # 前後フレーム比較
                missing_ids = [tid for tid in previous_detections if tid not in current_detections]
                if missing_ids:
                    print(f"検出されなくなった車両: {missing_ids}")

                # フレームに描画
                if self.show_video or self.save_video:
                    frame = self.draw_detections(frame, detections)

                previous_detections = current_detections

            # フレームを表示
            if self.show_video:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 結果を保存
            if self.save_video and writer:
                writer.write(frame)

            frame_cnt += 1

        video.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return all_results  # 全フレームの結果を返す


if __name__ == "__main__":
    # 入力動画と出力動画のパスを設定
    input_video_path = "../videos/street1_sample_01.mp4"  # 入力動画
    output_video_path = "output.mp4"  # 出力動画

    # 動画解析クラスをインスタンス化
    analyzer = VideoAnalyzer(
        model_path="../model/yolov8n.pt",  # YOLOv8モデル
        input_video=input_video_path,
        output_video=output_video_path,
        show_video=True,
        # save_video=True
    )

    # 動画を解析
    results = analyzer.analyze_video()
    print(f"解析結果：{results}")
