import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 📁 修改为你本地txt文件所在的文件夹路径
folder_path = "../bar"  # <-- 修改这里

# 🧾 提取 Readability Scores
scores = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
            content = file.read()
            matches = re.findall(r"Readability Score:\s*(\d)", content)
            if len(matches) >= 2:
                scores.append((int(matches[0]), int(matches[1])))

# 📊 构建数据表
df = pd.DataFrame(scores, columns=["Model1_Score", "Model2_Score"])
score_counts = df.groupby(["Model1_Score", "Model2_Score"]).size().reset_index(name="Count")

# 🔵 归一化气泡大小
max_size = 3000
min_size = 300
normalized_sizes = (score_counts["Count"] - score_counts["Count"].min()) / (score_counts["Count"].max() - score_counts["Count"].min())
bubble_sizes = normalized_sizes * (max_size - min_size) + min_size

# 🎨 绘制图形
fig, ax = plt.subplots(figsize=(8, 6))

# ✅ 先手动绘制浅色网格线（zorder 1）
for x in range(1, 6):
    ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.5, zorder=1)
for y in range(1, 6):
    ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.5, zorder=1)

# ✅ 再绘制气泡图（zorder 2）
bubble = ax.scatter(
    score_counts["Model1_Score"],
    score_counts["Model2_Score"],
    s=bubble_sizes,
    c=score_counts["Count"],
    cmap="viridis",
    alpha=1.0,
    edgecolors="black",
    marker='o',
    zorder=2  # 确保气泡在网格线之上
)

# ✏️ 添加计数标签
for _, row in score_counts.iterrows():
    ax.text(row["Model1_Score"], row["Model2_Score"], str(row["Count"]),
            ha='center', va='center', color='black', fontsize=8, weight='bold', zorder=3)

# ⚙️ 设置外观
ax.set_title("Bubble Chart of Readability Score Combinations")
ax.set_xlabel("Qwen2.5-VL-72B-Instruct Readability Score")
ax.set_ylabel("gpt-4o-mini Readability Score")
ax.set_xticks(range(1, 6))
ax.set_yticks(range(1, 6))
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)

# ✅ 添加颜色条
cbar = plt.colorbar(bubble, ax=ax)
cbar.set_label("Sample Count")

plt.tight_layout()
plt.show()
