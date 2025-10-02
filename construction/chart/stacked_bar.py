# -*- coding: utf-8 -*-
"""
批量读取、修改并可视化堆叠柱状图数据脚本
"""
import os
import csv
import random
from typing import Tuple, List

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 样式及调色板初始化
def init_style() -> List[Tuple]:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.0,
        "axes.grid": False
    })
    palettes = ["tab20","Set2"]
    colors: List[Tuple] = []
    for name in palettes:
        try:
            cmap = plt.get_cmap(name)
            if hasattr(cmap, 'colors'):
                colors.extend(cmap.colors)
            else:
                colors.extend([cmap(i/19) for i in range(20)])
        except ValueError:
            continue
    return colors

# 绘制随机方向堆叠图
def plot_stacked(df: pd.DataFrame, theme: str, unit: str,
                 colors: List[Tuple], out_png: str, out_svg: str, orientation:str) -> None:
    """
    随机选择竖向(70%)或横向(30%)堆叠图，其他样式保持一致。
    """
    x = df.index.tolist()
    categories = df.columns.tolist()
    n = len(x)
    # 初始化基线
    bottoms = np.zeros(n)
    lefts = np.zeros(n)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=300)
    # 随机取色
    cset = random.sample(colors, len(categories))

    # 根据方向绘制
    for i, cat in enumerate(categories):
        values = df[cat].values
        if orientation == "vertical":
            ax.bar(
                range(n), values,
                bottom=bottoms,
                width=0.6,
                facecolor=cset[i],
                edgecolor="#000000",
                linewidth=1
            )
            bottoms += values
        else:
            ax.barh(
                range(n), values,
                left=lefts,
                height=0.6,
                facecolor=cset[i],
                edgecolor="#000000",
                linewidth=1
            )
            lefts += values

    # 标题
    ax.set_title(f"{theme} ({unit})", fontsize=16, pad=12)

    # 坐标轴和标签
    if orientation == "vertical":
        ax.set_xticks(range(n))
        ax.set_xticklabels(x, fontsize=12, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel(unit, fontsize=14)
    else:
        ax.set_yticks(range(n))
        ax.set_yticklabels(x, fontsize=12, rotation=0, va='center')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xlabel(unit, fontsize=14)

    # 图例
    ax.legend(categories,
              fontsize=14,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 1, 1])

    # 保存
    fig.savefig(out_png, format='png', dpi=300)
    fig.savefig(out_svg, format='svg', dpi=300)
    plt.close(fig)

# 主流程
def main():
    Topic = ["Transportation_and_Logistics", "Tourism_and_Hospitality","Business_and_Finance",
             "Real_Estate_and_Housing_Market","Healthcare_and_Health", "Retail_and_E-commerce",
             "Human_Resources_and_Employee_Management", "Sports_and_Entertainment", "Education_and_Academics",
             "Food_and_Beverage_Industry", "Science_and_Engineering", "Agriculture_and_Food_Production",
             "Energy_and_Utilities", "Cultural_Trends_and_Influences", "Social_Media_and_Digital_Media_and_Streaming"]
    # 初始化样式与调色板
    colors = init_style()

    # 创建输出目录
    out_dirs = {
        'png': './png/stacked_bar_chart/',
        'svg': './svg/stacked_bar_chart/'
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    j = 1
    for t in range(1, 16):  # 主题 1~15
        for i in range(1, 1001):  # 文件 1~800
    # for t in [2]:  # 主题 1-15
        # for i in range(1, 8):  # 文件编号 1-800
            in_path = f'./csv/stacked_bar_chart/stacked_{Topic[t-1]}_{i}.csv'
            if not os.path.exists(in_path):
                continue

            # 读取第一行，解析 topic_name, theme, unit, dist_type
            with open(in_path, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
            topic_name, theme, unit, orientation = [p.strip() for p in header_line.split(',')]

            df = pd.read_csv(in_path, skiprows=1, index_col=0)


            # 绘图
            out_png = os.path.join(out_dirs['png'], f'stacked_{Topic[t-1]}_{i}.png')
            out_svg = os.path.join(out_dirs['svg'], f'stacked_{Topic[t-1]}_{i}.svg')
            plot_stacked(df, theme, unit, colors, out_png, out_svg, orientation)

            j += 1
            if j % 100 == 0:
                print(f"[✔] Saved #{j}")

if __name__ == '__main__':
    main()
