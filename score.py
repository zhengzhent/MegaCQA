import numpy as np
from scipy.stats import pearsonr

# 假设您有两组数据
qwen_scores = [4.66, 4.0, 3.56, 3.0, 4.0, 4.0, 3.99, 4.02, 3.46, 2.16, 4.17, 3.9, 3.97, 3.25, 3.87, 3.87, 3.68, 3.88, 3.63, 3.81, 3.85]  # Qwen模型分数
gpt_scores = [4.63, 4.44, 3.25, 2.0, 2.6, 4.7, 3.42, 4.59, 3.09, 2.0, 3.81, 2.99, 3.3, 2.9, 4.02, 3.93, 3.88, 3.79, 2.31, 2.32, 4.32]   # GPT模型分数

# 计算皮尔逊相关系数
correlation, p_value = pearsonr(qwen_scores, gpt_scores)
print(f"相关系数: {correlation:.3f}")
print(f"p值: {p_value:.3f}")