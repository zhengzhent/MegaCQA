import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm

# =============================================
# 脚本：CSV 后处理并生成小提琴图
# 功能：
#   1. 读取已有 box_plot CSV (1–4800)
#   2. 修改并保存为 violin_plot CSV
#   3. 生成小提琴图 (PNG + SVG)
#   4. 保持原样式，仅保证同图中小提琴颜色互异
#
# 目录结构：
#   输入:  ./box_plot/csv/box_plot_{i}.csv      (i = 1..4800)
#   输出CSV:  ./violin_plot/csv/violin_plot_{i}.csv
#   输出PNG:  ./violin_plot/png/violin_plot_{i}.png
#   输出SVG:  ./violin_plot/svg/violin_plot_{i}.svg
# =============================================

# 配置
# INPUT_CSV_DIR   = './box_plot/csv'
# OUTPUT_CSV_DIR  = './violin_plot/csv'
# OUTPUT_PNG_DIR  = './violin_plot/png'
# OUTPUT_SVG_DIR  = './violin_plot/svg'
INPUT_CSV_DIR   = './csv/box_plot/'
OUTPUT_CSV_DIR  = './csv/violin_plot/'
OUTPUT_PNG_DIR  = './png/violin_plot/'
OUTPUT_SVG_DIR  = './svg/violin_plot/'

BOX_WIDTH_MIN   = 0.075
# 取自 tab20 调色板，保证足够丰富的颜色池
RICH_COLORS = list(cm.get_cmap('tab20').colors)
RICH_COLORS.extend(list(cm.get_cmap('Set2').colors))

# 全局 Matplotlib 样式
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
    'axes.edgecolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.linewidth': 1.0,
})


def ensure_dirs():
    """创建输出目录"""
    for d in [OUTPUT_CSV_DIR, OUTPUT_PNG_DIR, OUTPUT_SVG_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_header(header_line: str):
    """
    拆分第一行，返回 topic, theme, unit, mode
    例: "Education and Academics, Employment Rate by School, %, normal"
    """
    parts = [p.strip() for p in header_line.strip().split(',')]
    if len(parts) != 4:
        raise ValueError(f'表头字段数异常: {header_line}')
    topic, theme, unit, mode = parts
    return topic, theme, unit, mode


def plot_violin(df: pd.DataFrame, labels: list, theme: str, unit: str, mode: str,
                png_path: str, svg_path: str):
    """
    基于 DataFrame 绘制小提琴图并保存
    保持原脚本样式，仅确保同图中小提琴颜色互异
    """
    num = len(labels)
    # 动态画布宽度
    fig_width = max(6.4, num * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=300)
    ax.set_title(f"{theme} ({unit})", fontsize=16)

    # 小提琴宽度设置
    width = 0.8 if num <= 5 else 0.8 * 5 / num

    # 绘制小提琴
    parts = ax.violinplot(
        [df[label] for label in labels],
        showmeans=False,
        showextrema=False,
        showmedians=False,
        widths=width
    )
    # 为每个 violin 分配不同颜色
    colors = random.sample(RICH_COLORS, num)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('#000000')
        pc.set_alpha(0.7)

    # 叠加透明箱线图
    box_w = max(width * 0.15, BOX_WIDTH_MIN)
    ax.boxplot(
        [df[label] for label in labels],
        positions=np.arange(1, num+1),
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='none', edgecolor='#000000', linewidth=1),
        whiskerprops=dict(color='#000000', linewidth=1),
        capprops=dict(color='#000000', linewidth=1),
        medianprops=dict(color='#000000', linewidth=1.5)
    )

    # 坐标轴
    ax.set_xticks(np.arange(1, num+1))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel(unit, fontsize=14)

    plt.tight_layout(pad=2)
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_dirs()
    Topic = [
        "Transportation_and_Logistics", "Tourism_and_Hospitality","Business_and_Finance",
        "Real_Estate_and_Housing_Market","Healthcare_and_Health", "Retail_and_E-commerce",
        "Human_Resources_and_Employee_Management", "Sports_and_Entertainment", "Education_and_Academics",
        "Food_and_Beverage_Industry", "Science_and_Engineering", "Agriculture_and_Food_Production",
        "Energy_and_Utilities", "Cultural_Trends_and_Influences", "Social_Media_and_Digital_Media_and_Streaming"
    ]
    # for t in [1,2]:
    for t in range(1,16):
        for j in range(1, 1001):
            in_csv = os.path.join(INPUT_CSV_DIR, f'box_plot_{Topic[t-1]}_{j}.csv')
            if not os.path.isfile(in_csv):
                print(f'[跳过] 文件不存在: {in_csv}')
                continue

            # 1. 读取并解析表头
            with open(in_csv, 'r', encoding='utf-8') as f:
                header_line = f.readline()
            topic, theme, unit, mode = parse_header(header_line)

            # 2. 读取数据 (跳过首行)
            df = pd.read_csv(in_csv, skiprows=1)
            labels = df.columns.tolist()

            # 3. 保存新的 CSV
            out_csv = os.path.join(OUTPUT_CSV_DIR, f'violin_{Topic[t-1]}_{j}.csv')
            with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                f.write(f'{topic}, {theme}, {unit}, {mode}\n')
                df.to_csv(f, index=False)

            # 4. 绘制并保存小提琴图
            out_png = os.path.join(OUTPUT_PNG_DIR, f'violin_{Topic[t-1]}_{j}.png')
            out_svg = os.path.join(OUTPUT_SVG_DIR, f'violin_{Topic[t-1]}_{j}.svg')
            plot_violin(df, labels, theme, unit, mode, out_png, out_svg)
            
            if(j%100==0):
                print(f'[完成] violin_plot_{j}')

        print('所有文件处理完毕。')

if __name__ == '__main__':
    main()
