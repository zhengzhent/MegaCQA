import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TSNEVisualizer:
    def __init__(self, json_files, figsize=(12, 8)):
        """
        :param json_files: 明确指定要加载的 JSON 文件路径列表
        :param figsize: 图像大小（默认宽屏，突出横向分布）
        """
        self.json_files = json_files
        self.figsize = figsize
        self.data = []

        # ✅ 自定义高对比度、色盲友好颜色（十六进制 RGB）
        self.custom_colors = [
            "#0D48C8",  # 深青绿色
            "#FFBD6C",  # 明亮金黄色
            "#FC50CF",  # 紫色（备用）
            "#8ABDBD",  # 深灰色
            "#D93915",  # 橙红色

        ]

    def load_data(self):
        """
        加载所有 JSON 文件中的 t-SNE 数据，并执行全局 Min-Max 归一化到 [-1, 1]
        所有数据共享相同的缩放参数，确保可视化尺度一致
        """
        if not self.json_files:
            print("⚠️ No JSON files provided.")
            return

        all_points = []
        file_data_temp = []

        # 第一步：遍历所有文件，收集 t-SNE 坐标
        for file_path in self.json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                # 提取 "tsne" 字段，确保是二维列表
                tsne_points = [
                    item["tsne"] for item in records
                    if "tsne" in item and isinstance(item["tsne"], list) and len(item["tsne"]) == 2
                ]
                if not tsne_points:
                    print(f"⚠️ No valid t-SNE points in {file_path}")
                    continue

                all_points.extend(tsne_points)
                label = os.path.splitext(os.path.basename(file_path))[0]  # 如 CSV_ChartX
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

        # 第二步：归一化每组数据并分配自定义颜色
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
        绘制 t-SNE 散点图
        - 无标题、无坐标轴标签
        - 固定坐标范围 [-1.05, 1.05]
        - 使用自定义高对比度颜色
        - 图例字体加大至 20
        - 保留轻微网格
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
                s=30,  # 增大点大小
                color=group["color"],
                edgecolors='none'
            )

        # 固定坐标轴范围
        plt.xlim(-1.15, 1.15)
        plt.ylim(-1.15, 1.15)

        # 可选：保持等比缩放（取消注释以启用）
        # plt.gca().set_aspect('equal', adjustable='box')

        # ✅ 增大图例
        plt.legend(
            fontsize=25,
            loc="upper left",  # 设置图例位置为左上角
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=1,
            handlelength=1,
            handletextpad=0.5,
            markerscale=3.5,  # 增大图例中点的大小
        )

        # 轻微网格
        plt.grid(True, linestyle='--', alpha=0.5)

        # 紧凑布局
        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_file}")
        else:
            plt.show()


def main():
    # 示例：CSV 数据集对比
    json_files = [
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/ChartX.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/ChartQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/ChartBench.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/NovaChart.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/MegaCQA.json",
        # 可继续添加更多文件...
    ]

    visualizer = TSNEVisualizer(
        json_files=json_files,
        figsize=(12, 8)  # 宽屏布局
    )
    visualizer.load_data()
    visualizer.plot(
        # save_file="X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/PNG_1000_4_40.png"
    )


if __name__ == "__main__":
    main()