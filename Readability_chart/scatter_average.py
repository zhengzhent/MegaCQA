# --------------------------------散点图-----------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Constructing the data
# data = {
#     "Chart Type": [
#         "Bar", "Box", "Bubble", "Chord", "Fill Bubble", "Funnel", "Heatmap", "Line",
#         "Node Link", "Parallel", "Pie", "Radar", "Ridgeline", "Sankey", "Scatter",
#         "Stacked Area", "Stacked Bar", "Stream", "Surburst", "Treemap",  "Violin"
#     ],
#     "Qwen2.5-VL-72B-Instruct": [
#         4.66, 4.0, 3.56, 3.0, 4.0, 4.0, 3.99, 4.02, 3.46, 2.16, 4.17, 3.9,
#         3.97, 3.25, 3.87, 3.87, 3.68, 3.88, 3.63, 3.81, 3.85
#     ],
#     "gpt-4o-mini": [
#         4.63, 4.44, 3.25, 2.0, 2.60, 4.7, 3.42, 4.59, 3.09, 2.0, 3.81, 2.99,
#         3.3, 2.9, 4.02, 3.93, 3.88, 3.79, 2.31, 2.32, 4.32
#     ]
# }

# df = pd.DataFrame(data)

# # Set style
# sns.set(style="whitegrid")

# # Increase figure size while maintaining aspect ratio
# plt.figure(figsize=(12, 9))  # Increase figure size

# # Plot the scatter plot
# sns.scatterplot(
#     x='Qwen2.5-VL-72B-Instruct',
#     y='gpt-4o-mini',
#     data=df,
#     marker='o',           # Use circles
#     color='dodgerblue',
#     s=80,                 # Point size
#     edgecolor='black'
# )

# # Add the diagonal (best score line)
# plt.plot([1, 5], [1, 5], 'r--', label='Best Score')

# # Add text labels for each point, placing "Stacked Bar" and "Funnel" labels on the left
# for i in range(len(df)):
#     if df["Chart Type"][i] in ["Stacked Area", "Funnel", "Stacked Bar"]:
#         plt.text(df["Qwen2.5-VL-72B-Instruct"][i] - 0.1, df["gpt-4o-mini"][i], df["Chart Type"][i], fontsize=8, ha='right')
#     else:
#         plt.text(df["Qwen2.5-VL-72B-Instruct"][i], df["gpt-4o-mini"][i] - 0.1, df["Chart Type"][i], fontsize=8, ha='center')


# # Add axis labels and title in English
# plt.xlabel("Qwen2.5-VL-72B-Instruct Average Score")
# plt.ylabel("gpt-4o-mini Average Score")
# plt.title("Qwen2.5-VL-72B-Instruct VS gpt-4o-mini Average Score Scatter Plot")
# plt.legend()

# # Set the same aspect ratio to avoid distortion
# plt.gca().set_aspect('equal', adjustable='box')

# # Display the plot with tight layout
# plt.tight_layout()
# # 导出图像为 PNG 格式，分辨率为 300 DPI
# plt.savefig('../figure/scatter_plot.png', dpi=300)
# # Show the plot
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 构造数据
data = {
    "Chart Type": [
        "Bar", "Box", "Bubble", "Chord", "Fill Bubble", "Funnel", "Heatmap", "Line",
        "Node Link", "Parallel", "Pie", "Radar", "Ridgeline", "Sankey", "Scatter",
        "Stacked Area", "Stacked Bar", "Stream", "Surburst", "Treemap", "Violin"
    ],
    "Qwen2.5-VL-72B-Instruct": [
        4.66, 4.0, 3.56, 3.0, 4.0, 4.0, 3.99, 4.02, 3.46, 2.16, 4.17, 3.9,
        3.97, 3.25, 3.87, 3.87, 3.68, 3.88, 3.63, 3.81, 3.85
    ],
    "gpt-4o-mini": [
        4.63, 4.44, 3.25, 2.0, 2.60, 4.7, 3.42, 4.59, 3.09, 2.0, 3.81, 2.99,
        3.3, 2.9, 4.02, 3.93, 3.88, 3.79, 2.31, 2.32, 4.32
    ]
}

df = pd.DataFrame(data)

# 绘制
sns.set(style="whitegrid")
plt.figure(figsize=(12, 9))

# 绘制散点
plt.scatter(
    df['Qwen2.5-VL-72B-Instruct'],
    df['gpt-4o-mini'],
    color='dodgerblue',
    s=80,
    edgecolor='black'
)

# 对角线
plt.plot([1, 5], [1, 5], 'r--', label='Best Score')

# 添加标签：偶数点右移，奇数点左移 + 特殊调整
for i, row in df.iterrows():
    x = row['Qwen2.5-VL-72B-Instruct']
    y = row['gpt-4o-mini']

    # 特殊偏移
    if row['Chart Type'] == "Treemap":
        y += 0.08  # 向上
    elif row['Chart Type'] == "Surburst":
        y -= 0.08  # 向下

    # 水平偏移
    if i % 2 == 0:
        plt.text(
            x + 0.08, y,
            row['Chart Type'],
            ha='left', va='center', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=0.5)
        )
    else:
        plt.text(
            x - 0.08, y,
            row['Chart Type'],
            ha='right', va='center', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=0.5)
        )

# 细节设置
plt.xlabel("Qwen2.5-VL-72B-Instruct Average Score")
plt.ylabel("gpt-4o-mini Average Score")
plt.title("Qwen2.5-VL-72B-Instruct VS gpt-4o-mini Average Score Scatter Plot")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.savefig('../figure/scatter_plot_treemap_surburst_adjust.png', dpi=300)
plt.show()



# ------------------柱状图---------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Constructing the data
# data = {
#     "Chart Type": [
#         "Bar", "Box", "Bubble", "Chord", "Fill Bubble", "Funnel", "Heatmap", "Line",
#         "Node Link", "Parallel", "Pie", "Radar", "Ridgeline", "Sankey", "Scatter",
#         "Stacked Area", "Stacked Bar", "Stream", "Surburst", "Treemap",  "Violin"
#     ],
#     "Qwen2.5-VL-72B-Instruct": [
#         4.66, 4.0, 3.56, 3.0, 4.0, 4.0, 3.99, 4.02, 3.46, 2.16, 4.17, 3.9,
#         3.97, 3.25, 3.87, 3.87, 3.68, 3.88, 3.63, 3.81, 3.85
#     ],
#     "gpt-4o-mini": [
#         4.63, 4.44, 3.25, 2.0, 2.60, 4.7, 3.42, 4.59, 3.09, 2.0, 3.81, 2.99,
#         3.3, 2.9, 4.02, 3.93, 3.88, 3.79, 2.31, 2.32, 4.32
#     ]
# }

# df = pd.DataFrame(data)

# # Set style
# sns.set(style="whitegrid")

# # Increase figure size for better readability
# plt.figure(figsize=(14, 8))

# # Melt the data to make it easier to plot
# df_melted = df.melt(id_vars="Chart Type", value_vars=["Qwen2.5-VL-72B-Instruct", "gpt-4o-mini"], var_name="Model", value_name="Score")

# # Set a custom color palette with two shades of blue (dark blue and light blue)
# palette = ['#1f3a69', '#5c8bb5']  # Dark blue and light blue

# # Plot with seaborn's barplot, using the custom palette
# sns.barplot(x="Chart Type", y="Score", hue="Model", data=df_melted, palette=palette)

# # Rotate the x-axis labels by 45 degrees clockwise
# plt.xticks(rotation=45)

# # Set y-axis limits to be between 1 and 5
# plt.ylim(1, 5)

# # Add axis labels and title
# plt.xlabel("Chart Type")
# plt.ylabel("Average Score")
# plt.title("Qwen2.5-VL-72B-Instruct and gpt-4o-mini Average Scores by Bar Chart")
# plt.legend(title="Model")

# # Show the plot
# plt.tight_layout()
# plt.savefig('../figure/bar.png', dpi=300)
# plt.show()
# --------------------------------雷达图-----------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 数据构建
# data = {
#     "Chart Type": [
#         "Bar", "Box", "Bubble", "Chord", "Fill Bubble", "Funnel", "Heatmap", "Line",
#         "Node Link", "Parallel", "Pie", "Radar", "Ridgeline", "Sankey", "Scatter",
#         "Stacked Area", "Stacked Bar", "Stream", "Surburst", "Treemap",  "Violin"
#     ],
#     "Qwen2.5-VL-72B-Instruct": [
#         4.66, 4.0, 3.56, 3.0, 4.0, 4.0, 3.99, 4.02, 3.46, 2.16, 4.17, 3.9,
#         3.97, 3.25, 3.87, 3.87, 3.68, 3.88, 3.63, 3.81, 3.85
#     ],
#     "gpt-4o-mini": [
#         4.63, 4.44, 3.25, 2.0, 2.60, 4.7, 3.42, 4.59, 3.09, 2.0, 3.81, 2.99,
#         3.3, 2.9, 4.02, 3.93, 3.88, 3.79, 2.31, 2.32, 4.32
#     ]
# }

# df = pd.DataFrame(data)

# # 准备雷达图数据
# labels = df['Chart Type']
# qwen_scores = df['Qwen2.5-VL-72B-Instruct']
# gpt_scores = df['gpt-4o-mini']
# num_vars = len(labels)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# # 闭合图形
# qwen_scores = np.concatenate((qwen_scores, [qwen_scores[0]]))
# gpt_scores = np.concatenate((gpt_scores, [gpt_scores[0]]))
# angles += angles[:1]

# # 画图
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# # 数据绘制
# ax.fill(angles, qwen_scores, color='darkblue', alpha=0.25)
# ax.fill(angles, gpt_scores, color='#4682B4', alpha=0.25)
# ax.plot(angles, qwen_scores, color='darkblue', linewidth=2, label='Qwen2.5-VL-72B-Instruct')
# ax.plot(angles, gpt_scores, color='#4682B4', linewidth=2, label='gpt-4o-mini')

# # 调整标签距离中心的半径位置
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(labels)
# for label, angle in zip(ax.get_xticklabels(), angles):
#     label.set_horizontalalignment('center')
#     label.set_rotation(np.degrees(angle))
#     label.set_rotation_mode('anchor')
#     label.set_fontsize(10)
#     # 控制label离中心的距离（这句是关键）
#     label.set_position((1.1, 0.0008))  # 1.1是半径比例，适当调整即可

# # 隐藏极轴刻度
# ax.set_yticklabels([])

# # 移动标题更靠上
# plt.title("Qwen2.5-VL-72B-Instruct VS gpt-4o-mini Scores - Radar Chart", size=16, y=1.15)

# # 图例
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# plt.tight_layout()
# plt.savefig('../figure/radar.png', dpi=300)
# plt.show()








