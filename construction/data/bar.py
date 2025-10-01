# -*- coding: utf-8 -*-
"""
柱状图虚拟数据批量生成与可视化脚本（颜色 / 图案再次增强版）
=============================================================
此次变更
--------
1. **HATCH_CHOICES 差异更大**：扩充至 20 种、包括粗细/密度各异的组合以提高辨识度。
2. **图案模式**：柱体采用 *纯白填充*（facecolor="white"），仅用黑色边框 + 不同 hatch 图案区分，不再叠加颜色。
"""
import os
import random
from typing import List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ======== Matplotlib 全局样式 ========
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "text.color":  "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#000000",
    "xtick.color":  "#000000",
    "ytick.color":  "#000000",
    "axes.linewidth": 1.0,
})

# ======== 分布名称常量 ========
DISTRIBUTIONS = ["random", "long_tail", "normal", "bimodal", "step", "spike"]

# ======== 填充模式 ========
FILL_MODES   = ["solid", "gradient", "pattern"]
FILL_WEIGHTS = [0.7, 0.15, 0.15]

# ======== Hatch 样式（20 种，差异更大） ========
HATCH_CHOICES = [
    "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*",
    "//", "\\\\", "||", "++", "xx", "oo", "***", "---", "///", "|||"
]

# ======== 连续型同色系 cmap ========
SEQUENTIAL_CMAPS = [
    "Blues", "Reds", "Greens", "Purples", "Oranges", "Greys",
    "PuBu", "BuPu", "YlGn", "YlOrBr"
]

# --------------------------------------------------
# 工具
# --------------------------------------------------

def get_random_num(min_n: int = 3, max_n: int = 15) -> int:
    return random.randint(min_n, max_n)


# --------------------------------------------------
# 1. 生成数据
# --------------------------------------------------

def generate_values(n: int, dist_type: str) -> np.ndarray:
    if dist_type not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    if dist_type == "random":
        data = np.random.randint(300, 5000, n)
    elif dist_type == "long_tail":
        data = (np.random.pareto(a=2.0, size=n) + 1) * 600
    elif dist_type == "normal":
        data = np.random.normal(2500, 1000, n)
    elif dist_type == "bimodal":
        half = n // 2
        data = np.concatenate([
            np.random.normal(1200, 300, half),
            np.random.normal(3800, 600, n - half)
        ])
    elif dist_type == "step":
        base = np.linspace(800, 4200, n)
        noise = np.random.normal(0, 150, n)
        data = base + noise
    else:  # spike
        peak = np.random.uniform(2200, 3200)
        data = np.random.normal(peak, 200, n)
        idx = np.random.choice(n, size=max(1, n // 10), replace=False)
        data[idx] = np.random.uniform(300, 800, size=idx.shape[0])

    data = np.clip(data, 300, 5000)
    data = np.round(data)
    for i in range(len(data)):
        if data[i] < 700:
            data[i] += random.randint(300, 500)
    return data.astype(int)


# --------------------------------------------------
# 2. 保存 CSV
# --------------------------------------------------

def save_as_csv(df: pd.DataFrame, meta_line: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(meta_line + "\n")
        df.to_csv(f, index=False)


# --------------------------------------------------
# 3. 绘图
# --------------------------------------------------

def plot_bar(df: pd.DataFrame, unit: str, fill_mode: str, png_path: str, svg_path: str) -> None:
    if fill_mode not in FILL_MODES:
        raise ValueError(f"Unknown fill mode: {fill_mode}")

    labels: List[str] = df.iloc[:, 0].tolist()
    values: np.ndarray = df.iloc[:, 1].to_numpy()
    n_labels = len(labels)

    # ------ 颜色 / 图案 ------
    if fill_mode == "solid":
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(n_labels)]
        hatches = [None] * n_labels
    elif fill_mode == "gradient":
        cmap = plt.get_cmap(random.choice(SEQUENTIAL_CMAPS))
        colors = [cmap((i + 1) / (n_labels + 1)) for i in range(n_labels)]
        hatches = [None] * n_labels
    else:  # pattern
        colors = ["white"] * n_labels  # 纯白填充
        hatches = random.sample(HATCH_CHOICES, k=n_labels)

    # ------ 绘制 ------
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    x_pos = np.arange(n_labels)

    for idx, (lab, val) in enumerate(zip(labels, values)):
        bar_kwargs = dict(
            x=x_pos[idx], height=val, width=0.6,
            facecolor=colors[idx], edgecolor="#000000", linewidth=1,
        )
        if hatches[idx] is not None:
            bar_kwargs["hatch"] = hatches[idx]
        ax.bar(**bar_kwargs)
        ax.text(x_pos[idx], val + values.max() * 0.01, str(val),
                ha="center", va="bottom", fontsize=11, fontfamily="Times New Roman")

    theme, unit_label = df.columns[1].split(" (", 1)[0], unit
    ax.set_title(f"{theme} ({unit_label})", fontsize=16, pad=12)
    ax.set_xlabel(df.columns[0], fontsize=14)
    ax.set_ylabel(f"{theme} ({unit_label})", fontsize=14)
    ax.set_xticks(x_pos, labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    for path, fmt in [(png_path, "png"), (svg_path, "svg")]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=fmt, dpi=300)
    plt.close(fig)


# --------------------------------------------------
# 4. 综合执行
# --------------------------------------------------

def generate_and_save(num_files: int, theme: str, unit: str,
                       label_col: str, label_pool: List[str],
                       csv_dir: str, png_dir: str, svg_dir: str) -> None:
    for idx in range(1, num_files + 1):
        n_labels = get_random_num()
        categories = random.sample(label_pool, n_labels)
        dist_type = random.choice(DISTRIBUTIONS)
        values = generate_values(n_labels, dist_type)
        df = pd.DataFrame({label_col: categories, f"{theme} ({unit})": values})
        fill_mode = random.choices(FILL_MODES, weights=FILL_WEIGHTS, k=1)[0]
        basename = f"bar_topic1_{idx}"
        csv_path = os.path.join(csv_dir, f"{basename}.csv")
        png_path = os.path.join(png_dir, f"{basename}.png")
        svg_path = os.path.join(svg_dir, f"{basename}.svg")
        save_as_csv(df, f"{theme}({unit})", csv_path)
        plot_bar(df, unit, fill_mode, png_path, svg_path)
        print(f"[✔] Saved -> {csv_path}, {png_path}, {svg_path} | Dist={dist_type}, Fill={fill_mode}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    LABEL_COL = "Country"
    LABEL_POOL = [
        "United States", "China", "India", "Brazil", "Russia", "Germany", "Canada",
        "Australia", "Mexico", "Indonesia", "Japan", "United Kingdom", "France",
        "Italy", "South Korea"
    ]

    generate_and_save(
        num_files=10,
        theme="Fruit Transportation Flow",
        unit="tons",
        label_col=LABEL_COL,
        label_pool=LABEL_POOL,
        csv_dir="./csv/bar/",
        png_dir="./png/bar/",
        svg_dir="./svg/bar/",
    )
