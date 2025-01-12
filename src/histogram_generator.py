import json
import os

import matplotlib.pyplot as plt
import numpy as np
import prediction_evaluator as evaluator

def update_metrics(metrics, track_id, current_bbox, predicted_bbox, previous_bbox, target_ids=None):
    if target_ids is not None and track_id not in target_ids:
        return
    if track_id not in metrics["iou"]:
        metrics["iou"][track_id] = []
        metrics["area"][track_id] = []
        metrics["aspect"][track_id] = []
    iou = evaluator.calculate_iou(current_bbox, predicted_bbox)
    if iou is not None:
        metrics["iou"][track_id].append(iou)
    area_ratio = evaluator.calculate_area_ratio(current_bbox, previous_bbox)
    if area_ratio is not None:
        metrics["area"][track_id].append(area_ratio)
    aspect_ratio = evaluator.calculate_aspect_ratio(current_bbox, previous_bbox)
    if aspect_ratio is not None:
        metrics["aspect"][track_id].append(aspect_ratio)
    # print(metrics)  # デバッグ


def generate_histogram(data, title, xlabel, ylabel, output_path, stats_path):
    """
    ヒストグラムを生成して保存し、統計データを記録する関数
    :param data: ヒストグラム用データ（リスト形式）
    :param title: ヒストグラムのタイトル
    :param xlabel: 横軸のラベル
    :param ylabel: 縦軸のラベル
    :param output_path: 保存先の画像パス
    :param stats_path: 保存先の統計データパス
    """
    bins = [i / 100 for i in range(101)]  # 0.00~1.00を0.01刻みにするビン
    hist, bin_edges = np.histogram(data, bins=bins)
    # 累積頻度の計算
    # print(f"HIST: {hist}, SUM_HIST: {sum(hist)}")     # デバッグ
    # print(f"cumsum: {np.cumsum(hist)}")               # デバッグ
    cdf = np.cumsum(hist) / sum(hist)   # CDF(昇順)
    # print(f"CDF  : {cdf}")
    # 閾値を設定
    percentages = [0.5, 0.4, 0.3, 0.2, 0.1]
    thresholds = {f"{int(p * 100)}%": find_threshold_bin(cdf, bin_edges, p) for p in percentages}
    # print(f"BIN_EDGE: {thresholds}")
    # 統計情報をjsonファイルに記録
    stats = {
        "mean": np.mean(data),
        "median": np.median(data),
        "std_dev": np.std(data),
        "threshold_top_10%": thresholds,
        "histogram": {f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": int(hist[i]) for i in range(len(hist))},
        "cdf": cdf.tolist()
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"統計情報を保存しました: {stats_path}")
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor="black", color="blue", alpha=0.7)
    # 各閾値をプロット
    colors = ["maroon", "firebrick", "red", "coral", "gold"]
    for i, (label, value) in enumerate(thresholds.items()):
        if value is not None:
            # print(f"Label: {label}, Value: {value}")        # デバッグ
            plt.axvline(value, color=colors[i], linestyle="--", label=f"{label}: {value:.2f}")
    # グラフ設定
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.5)
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"ヒストグラムを保存しました: {output_path}")

def find_threshold_bin(cdf, bin_edges, target_percent):
    threshold_idx = np.searchsorted(cdf, target_percent, side="left")
    return bin_edges[threshold_idx]

def save_all_histograms(metrics, output_folder):
    """
    各トラックIDのヒストグラムと統計データを保存
    :param metrics: 各トラックIDのメトリクス辞書
                    {"iou": {"id1": [...], ...}, "area": {...}, "aspect": {...}}
    :param output_folder: 保存先のフォルダ
    """
    for metric_name, track_data in metrics.items():
        for track_id, values in track_data.items():
            if not values:
                continue
            output_path = os.path.join(output_folder, f"{metric_name}_{track_id}.png")
            stats_path = os.path.join(output_folder, f"{metric_name}_{track_id}_stats.json")
            title = f"Histogram for {metric_name.upper()} - Track ID: {track_id}"
            xlabel = metric_name.upper()
            ylabel = "Frame Count"
            generate_histogram(values, title, xlabel, ylabel, output_path, stats_path)


if __name__ == "__main__":
    # テストデータ生成
    np.random.seed(42)  # 再現性のために乱数シードを固定
    test_data = np.random.uniform(0, 1, 100)  # 0~1の間で一様分布から1000個のサンプルを生成

    # テスト用パス
    output_path = "../results/test_histogram.png"
    stats_path = "../results/test_histogram_stats.json"

    # ヒストグラム生成と統計計算
    generate_histogram(
        data=test_data,
        title="Test Histogram for Top Percentiles",
        xlabel="Value",
        ylabel="Frequency",
        output_path=output_path,
        stats_path=stats_path
    )

    print("テスト用ヒストグラムと統計情報が作成されました。確認してください。")