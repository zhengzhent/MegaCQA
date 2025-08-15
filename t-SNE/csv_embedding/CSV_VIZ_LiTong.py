import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# 设置全局字体（推荐放在类外或 __init__ 中初始化）
plt.rcParams["font.family"] = "Times New Roman"  # ✅ 设置全局字体
plt.rcParams["font.size"] = 14  # 可选：设置默认字体大小

class TSNEVisualizer:
    def __init__(self, json_files, figsize=(12, 8)):
        """
        :param json_files: 明确指定要加载的 JSON 文件路径列表
        :param figsize: 图像大小（默认宽 > 高，突出横向分布）
        """
        self.json_files = json_files
        self.figsize = figsize
        self.data = []

        # ✅ 自定义高对比度颜色（RGB 十六进制格式）
        # 推荐使用色盲友好的高对比配色
        self.custom_colors = [
            "#969696",  # 深灰色
            "#F89B30",  # 明亮金黄色
            "#009E73",  
        ]

    def load_data(self):
        """
        全局归一化：所有文件共享相同的 Min-Max 缩放参数，确保尺度一致
        """
        if not self.json_files:
            print("⚠️ No JSON files provided.")
            return

        all_points = []
        file_data_temp = []

        # 第一步：收集所有 t-SNE 点
        for idx, file_path in enumerate(self.json_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                tsne_points = [item["tsne"] for item in records if "tsne" in item and isinstance(item["tsne"], list)]
                if not tsne_points:
                    print(f"⚠️ No valid t-SNE points in {file_path}")
                    continue

                all_points.extend(tsne_points)
                label = os.path.splitext(os.path.basename(file_path))[0]
                file_data_temp.append({
                    "path": file_path,
                    "points": tsne_points,
                    "label": label
                })
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        if not all_points:
            print("⚠️ No valid t-SNE points found across all files.")
            return

        # 全局归一化到 [-1, 1]
        all_points = np.array(all_points)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(all_points)

        # 第二步：为每个文件分配自定义颜色
        for idx, file_info in enumerate(file_data_temp):
            normalized_points = scaler.transform(np.array(file_info["points"]))
            color = self.custom_colors[idx % len(self.custom_colors)]  # 循环使用颜色

            self.data.append({
                "label": file_info["label"],
                "points": normalized_points.tolist(),
                "color": color
            })

        print(f"✅ Global normalization completed. Total files processed: {len(self.data)}")

    def plot(self, save_file=None):
        """
        绘制全局归一化后的 t-SNE 散点图
        - 无标题、无坐标轴标签
        - 使用自定义高对比度颜色
        - 增大图例
        """
        if not self.data:
            print("⚠️ No data to plot.")
            return

        plt.figure(figsize=self.figsize)

        for group in self.data:
            xs, ys = zip(*group["points"])
            plt.scatter(
                xs, ys,
                label=group["label"],
                alpha=0.8,
                s=50,                    # ✅ 增大散点大小（原为25）
                color=group["color"],
                edgecolors='black',      # ✅ 添加黑色边框
                linewidths=0.2,          # ✅ 细边框，非常轻微的描边效果
                marker='o'
            )

        # 固定坐标范围
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)

        # ✅ 增大图例字体大小
        # plt.legend(
        #     fontsize=30,                    # 明显增大
        #     loc="best",
        #     frameon=True,
        #     fancybox=True,
        #     shadow=True,
        #     ncol=1,
        #     handlelength=1.5,
        #     handletextpad=1.0,
        #     markerscale=4  # 增大图例中点的大小
        # )

        # 可选：移除网格
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=900, bbox_inches='tight')
            print(f"✅ Plot saved to {save_file}")
        else:
            plt.show()


def main():
    json_files = [
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/ChartX.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/ChartQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/MegaCQA.json",
        # 可继续添加更多...
    ]

    visualizer = TSNEVisualizer(
        json_files=json_files,
        figsize=(12, 8)  # 宽屏比例，突出横向分布
    )
    visualizer.load_data()
    visualizer.plot(
        save_file="X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/CSV_LT.png"
    )


if __name__ == "__main__":
    main()