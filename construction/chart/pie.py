import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# 确保输出目录存在
os.makedirs('./png/pie', exist_ok=True)
os.makedirs('./svg/pie', exist_ok=True)

# 配色方案
color_schemes = [
    ['#8c510a', '#d8b365', '#bf812d',  '#dfc27d', '#f6e8c3','#c7eae5', '#80cdc1'],
    ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#e6f5d0', '#b8e186','#7fbc41'],
    ['#d6604d', '#9970ab', '#c2a5cf', '#d1e5f0', '#92c5de', '#4393c3','#2166ac'],
    ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0','#5aae61'],
    ['#d53e4f', '#fc8d59', '#fee08b', '#ffffbf','#e6f598', '#99d594', '#3288bd']
]

# 基础配置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
    'axes.edgecolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
})

# 遍历所有CSV文件
data_dir = './csv/pie/'
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)

        # 读取第一行（标题行）
        with open(file_path, 'r', encoding='utf-8') as f:
            title_line = f.readline().strip()

        # 提取第一个逗号和第二个逗号之间的字段作为标题
        comma_parts = title_line.split(',')
        if len(comma_parts) >= 3:
            chart_title = comma_parts[1].strip()
        else:
            chart_title = title_line  # 如果逗号不足，回退使用整行

        # 最终标题添加 "(Percentage)"
        final_title = f"{chart_title} %"

        # 继续读数据
        df = pd.read_csv(file_path, skiprows=1)
        labels = df['Category']
        sizes = df['Proportion']

        # 随机选择一种配色方案
        colors = random.choice(color_schemes)

        # 创建画布
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)

        # 绘制饼图（注意：删掉了 labels=labels）
        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize':12, 'color':'#000000'},
        )

        # 确保饼图为正圆
        ax.axis('equal')

        # 添加标题
        ax.set_title(final_title, fontsize=16, pad=20)

        # 设置图例（保留图例）
        ax.legend(
            wedges, labels,
            title='Category',
            loc='upper left',
            bbox_to_anchor=(0.95, 1),
            fontsize=12,
            title_fontsize=12,
            edgecolor='#000000',
            ncol=1,
            handletextpad=1.2,
            borderaxespad=0.5
        )

        # 去除背景和边框线
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 修改基础线宽度为1pt
        for line in ax.get_lines():
            line.set_linewidth(1)

        # 保存图表
        base_filename = os.path.splitext(file_name)[0]
        fig.savefig(f'./png/pie/{base_filename}.png', dpi=300, bbox_inches='tight', transparent=False)
        fig.savefig(f'./svg/pie/{base_filename}.svg', dpi=300, bbox_inches='tight', transparent=False)

        plt.close(fig)
