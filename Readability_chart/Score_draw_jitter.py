import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📁 设置你的 txt 文件夹路径
folder_path = "../bar"  # ← 替换为你的路径

# 🧾 提取评分
scores = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
            content = file.read()
            matches = re.findall(r"Readability Score:\s*(\d)", content)
            if len(matches) >= 2:
                scores.append((int(matches[0]), int(matches[1])))

# 📊 构建评分表
df = pd.DataFrame(scores, columns=["Model1_Score", "Model2_Score"])

# ✨ 添加抖动（防止重叠）
jitter_strength_y = 0.45
jitter_strength_x = 0.3
df["Model1_Jitter"] = df["Model1_Score"] + np.random.uniform(-jitter_strength_x, jitter_strength_x, size=len(df))
df["Model2_Jitter"] = df["Model2_Score"] + np.random.uniform(-jitter_strength_y, jitter_strength_y, size=len(df))

# 🎨 绘图
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Model1_Jitter"],
    df["Model2_Jitter"],
    alpha=0.6,
    edgecolor="black",
    s=80
)

# 📐 设置坐标轴与样式
plt.title("Scatter Plot with Jittered of Readability Scores")
plt.xlabel("Qwen2.5-VL-72B-Instruct Readability Score")
plt.ylabel("gpt-4o-mini Readability Score")
plt.xticks(range(1, 6))
plt.yticks(range(1, 6))
plt.xlim(0.5, 5.5)
plt.ylim(0.5, 5.5)
plt.grid(True)
plt.tight_layout()
plt.show()
