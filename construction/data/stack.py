# -*- coding: utf-8 -*-
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


class StackedBarChart:
    """集成数据生成、CSV 保存和堆叠柱状图绘制的单一类"""
    # 全局常量
    FONT = "Times New Roman"
    EDGE_COLOR = "#000000"
    DPI = 300
    # 主体图像大小
    MAIN_WIDTH = 8.4
    MAIN_HEIGHT = 4.8
    # 额外留给图例的宽度
    LEGEND_MARGIN = 0
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12

    DISTRIBUTIONS = ["random", "normal", "exponential", "long_tail", "skewed"]
    PALETTES = ["tab20", "Set3", "Paired", "Set2", "Pastel1", "Set1"]

    def __init__(
            self,
            base_name: str,
            x_pool: List[str],
            label_pool: List[str],
            topic: str,
            theme: str,
            unit: str,
            index_name: str,
            csv_dir: str,
            png_dir: str,
            svg_dir: str
    ):
        self.base_name = base_name
        self.x_pool = x_pool
        self.label_pool = label_pool
        self.topic = topic
        self.theme = theme
        self.unit = unit
        self.index_name = index_name
        self.csv_dir = csv_dir
        self.png_dir = png_dir
        self.svg_dir = svg_dir

        self.apply_style()
        self.palette = self.build_color_palette()

    def apply_style(self) -> None:
        mpl.rcParams.update({
            "font.family": self.FONT,
            "text.color": self.EDGE_COLOR,
            "axes.labelcolor": self.EDGE_COLOR,
            "xtick.color": self.EDGE_COLOR,
            "ytick.color": self.EDGE_COLOR,
            "axes.edgecolor": self.EDGE_COLOR,
            "axes.linewidth": 1.0,
            "axes.grid": False
        })

    def build_color_palette(self) -> List[Tuple]:
        palette: List[Tuple] = []
        for name in self.PALETTES:
            try:
                cmap = plt.get_cmap(name)
                if hasattr(cmap, 'colors'):
                    palette.extend(cmap.colors)
                else:
                    palette.extend([cmap(i / 19) for i in range(20)])
            except ValueError:
                continue
        return palette

    def generate_values(self, n: int, dist: str) -> np.ndarray:
        """
        根据指定分布生成 n 个值并裁剪到合理范围（100–1000 tonnes）
        """
        if dist not in self.DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {dist}")
        if dist == "random":
            arr = np.random.randint(100, 1000, size=n)
        elif dist == "normal":
            arr = np.random.normal(550, 150, size=n)
        elif dist == "exponential":
            arr = np.random.exponential(scale=300, size=n)
        elif dist == "long_tail":
            arr = (np.random.pareto(2.0, size=n) + 1) * 200
        else:
            arr = np.random.gamma(shape=2, scale=250, size=n)
        arr = np.clip(arr, 150, 1000)
        return np.round(arr).astype(int)

    def sample(self, pool: List[str], k: int) -> List[str]:
        """从给定池中随机抽取 k 个元素"""
        return random.sample(pool, k)

    def save_csv(self, df: pd.DataFrame, idx: int) -> None:
        """保存 CSV，首行写入主题和单位，去除空行"""
        os.makedirs(self.csv_dir, exist_ok=True)
        name = f"{self.base_name}_{idx}"
        path = os.path.join(self.csv_dir, f"{name}.csv")
        header = f"{self.topic},{self.theme}({self.unit})"
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(header + "\n")
            df.to_csv(f, index=True)

    def plot(self, df: pd.DataFrame, idx: int) -> None:
        """绘制堆叠柱状图并保存到 PNG/SVG"""
        name = f"{self.base_name}_{idx}"
        png_path = os.path.join(self.png_dir, f"{name}.png")
        svg_path = os.path.join(self.svg_dir, f"{name}.svg")
        os.makedirs(self.png_dir, exist_ok=True)
        os.makedirs(self.svg_dir, exist_ok=True)

        x = list(df.index)
        cats = list(df.columns)
        n = len(x)
        bottoms = np.zeros(n)

        colors = random.sample(self.palette, len(cats))
        total_width = self.MAIN_WIDTH + self.LEGEND_MARGIN
        fig, ax = plt.subplots(figsize=(total_width, self.MAIN_HEIGHT), dpi=self.DPI)
        for i, cat in enumerate(cats):
            ax.bar(range(n), df[cat].values, bottom=bottoms,
                   width=0.6, facecolor=colors[i],
                   edgecolor=self.EDGE_COLOR, linewidth=1)
            bottoms += df[cat].values

        ax.set_title(f"{self.theme} ({self.unit})", fontsize=self.TITLE_SIZE, pad=12)
        ax.set_xticks(range(n))
        ax.set_xticklabels(x, fontsize=self.TICK_SIZE)
        ax.tick_params(axis='y', labelsize=self.TICK_SIZE)
        ax.set_xlabel('', fontsize=self.LABEL_SIZE)
        ax.set_ylabel(self.unit, fontsize=self.LABEL_SIZE)
        ax.legend(cats, fontsize=self.LABEL_SIZE,
                  loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=[0, 0, (self.MAIN_WIDTH / total_width), 1])

        for path, fmt in [(png_path, 'png'), (svg_path, 'svg')]:
            fig.savefig(path, format=fmt, dpi=self.DPI)
        plt.close(fig)

    def run(self, num_sets: int) -> None:
        """主流程：批量生成数据、保存 CSV、绘图"""
        for idx in range(1, num_sets + 1):
            k_cat = random.randint(2, min(7, len(self.label_pool)))
            cats = self.sample(self.label_pool, k_cat)
            k_x = random.randint(3, min(15, len(self.x_pool)))
            xs = self.sample(self.x_pool, k_x)

            dist = random.choice(self.DISTRIBUTIONS)
            data = {c: self.generate_values(k_x, dist) for c in cats}
            df = pd.DataFrame(data, index=xs)
            df.index.name = self.index_name

            self.save_csv(df, idx)
            self.plot(df, idx)

            if idx % 100 == 0:
                print(f"[✔] Saved #{idx}")


if __name__ == '__main__':
    X_POOL = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    LABEL_POOL = [
        'Electronics', 'Automotive Parts', 'Pharmaceuticals', 'Furniture',
        'Apparel', 'Grain', 'Steel', 'Oil', 'Coal', 'Chemicals',
        'Vehicles', 'Textiles', 'Machinery', 'Paper', 'Plastics'
    ]
    TOPIC = 'Transportation and Logistics'
    THEME = 'Monthly Shipment Volume of Different Commodities'
    UNIT = 'tonnes'

    chart = StackedBarChart(
        base_name='stacked_bar_topic1',
        x_pool=X_POOL,
        label_pool=LABEL_POOL,
        topic=TOPIC,
        theme=THEME,
        unit=UNIT,
        index_name='Month',
        csv_dir='csv/stacked_bar_chart/',
        png_dir='png/stacked_bar_chart/',
        svg_dir='svg/stacked_bar_chart/'
    )
    chart.run(num_sets=800)
    print('Done')

