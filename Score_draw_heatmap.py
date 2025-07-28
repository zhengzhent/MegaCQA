import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 修改为你的TXT文件所在路径
folder_path = "./bar"  # ← 替换为你本地的文件夹路径

# 🧾 提取两个 Readability Score
scores = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
            content = file.read()
            matches = re.findall(r"Readability Score:\s*(\d)", content)
            if len(matches) >= 2:
                scores.append((int(matches[0]), int(matches[1])))

# 📊 构建 DataFrame
df = pd.DataFrame(scores, columns=["Model1_Score", "Model2_Score"])

# 📐 创建完整5x5索引，并填补缺失组合为0
full_index = pd.MultiIndex.from_product(
    [range(1, 6), range(1, 6)],
    names=["Model2_Score", "Model1_Score"]
)
heatmap_data = df.groupby(["Model2_Score", "Model1_Score"]) \
                 .size().reindex(full_index, fill_value=0).unstack()

# 🎨 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar_kws={"label": "Sample Count"},
    linewidths=0.5,
    linecolor='gray',
    square=True
)
plt.gca().invert_yaxis()
# 🧭 图表样式设置
plt.title("Heatmap of Readability Score Combinations")
plt.xlabel("Qwen2.5-VL-72B-Instruct Readability Score")
plt.ylabel("gpt-4o-mini Readability Score")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
