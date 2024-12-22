import os
import matplotlib.pyplot as plt

def generate_histogram(data, title, xlabel, ylabel, output_path):
    """
    ヒストグラムを生成して保存する関数

    :param data: ヒストグラム用データ（リスト形式）
    :param title: ヒストグラムのタイトル
    :param xlabel: 横軸のラベル
    :param ylabel: 縦軸のラベル
    :param output_path: 保存先のパス
    """
    # 0.00~1.00を0.01刻みにするビンを定義
    bins = [i / 100 for i in range(101)]
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.5)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"ヒストグラムを保存しました: {output_path}")

def save_all_histograms(metrics, output_folder):
    """
    全てのトラックIDに対してヒストグラムを作成して保存する関数

    :param metrics: 各トラックIDのメトリクス辞書
                    例: {"iou": {"id1": [0.5, 0.6, ...], ...},
                         "area": {"id1": [...], ...},
                         "aspect": {"id1": [...], ...}}
    :param output_folder: ヒストグラムを保存するフォルダ
    """
    for metric_name, track_data in metrics.items():
        for track_id, values in track_data.items():
            output_path = os.path.join(output_folder, f"{track_id}_{metric_name}.png")
            title = f"Histogram for {metric_name.upper()} - Track ID: {track_id}"
            xlabel = metric_name.upper()
            ylabel = "Frame Count"
            generate_histogram(values, title, xlabel, ylabel, output_path)
