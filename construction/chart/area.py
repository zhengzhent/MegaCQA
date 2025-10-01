# drawArea.py – 修正版：解决首行数据缺失、图例位置、Y 轴范围及单位解析问题
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 全局字体和颜色
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "figure.autolayout": False,
})

# 颜色循环（tab20）
COLOR_CYCLE = plt.cm.tab20.colors

# 图表尺寸与分辨率
FIGSIZE = (8, 5)
DPI = 300


def _parse_theme_and_units(first_line: str):
    """解析首行并返回 `(theme, field_name, unit)`。

    兼容三种常见写法：
        1. 逗号分隔 —— ``Theme, Field, Unit``
        2. 括号标注 —— ``Theme, Field (Unit)``
        3. 点号分隔 —— ``Theme, Field.Unit``
    """
    parts = [p.strip() for p in first_line.split(',')]
    theme = parts[0] if parts else ""

    # 首选：逗号分隔形式  Theme, Field, Unit
    if len(parts) >= 3:
        field_name = parts[1]
        unit = parts[2]
        return theme, field_name, unit

    # ➜ 兼容旧格式  Field (Unit) 或 Field.Unit
    field_unit_part = parts[1] if len(parts) > 1 else ""

    # 括号提取单位
    unit_match = re.search(r"\((.*?)\)", field_unit_part)
    if unit_match:
        unit = unit_match.group(1).strip()
        field_name = re.sub(r"\s*\(.*?\)", "", field_unit_part).strip()
    elif "." in field_unit_part:
        field_name, unit = [s.strip() for s in field_unit_part.split(".", 1)]
    else:
        field_name, unit = field_unit_part.strip(), ""

    return theme, field_name, unit


def _parse_headers(header_line: str):
    """解析第二行列头（跳过首列 Year）"""
    return [col.strip() for col in header_line.split(',')[1:]]


def plot_area_from_csv(csv_path: str, base_save_dir: str):
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()      # 行 0：主题 + 字段 / 单位
            header_line = f.readline().strip()     # 行 1：Year + 各主体名称
            f.readline()                           # 行 2：趋势行（忽略）

        # 元数据解析
        theme, field_name, unit = _parse_theme_and_units(first_line)
        y_headers = _parse_headers(header_line)

        # 读取数据（跳过前三行），保留列头
        #   跳过行 0 与行 2，让行 1 成为表头
        df = pd.read_csv(csv_path, skiprows=[0, 2], encoding="utf-8")  # header=0 默认

        # 处理年份列，防止 BOM 或空格导致的 NaN
        year_raw = df.iloc[:, 0].astype(str).str.strip()
        year_numeric = pd.to_numeric(year_raw.str.extract(r"(\d+)")[0], errors="coerce")
        valid_mask = ~year_numeric.isna()

        x = year_numeric[valid_mask].astype(int)
        y_data = df.iloc[valid_mask.index[valid_mask], 1:].apply(pd.to_numeric, errors="coerce").fillna(0)

        # 标题与坐标轴标签
        min_year, max_year = int(x.min()), int(x.max())
        title = f"{field_name} from {min_year} to {max_year}({unit})" if unit else f"{field_name} "
        ylabel = f"{field_name} ({unit})" if unit else field_name

        # 绘图
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        stack = ax.stackplot(x, y_data.T, colors=COLOR_CYCLE, labels=y_headers)

        # 坐标轴与刻度
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel(df.columns[0], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x[::step].astype(int), rotation=45, ha="right")

        # Y 轴上限：总量 * 1.15，留出 15% 空间
        y_max = y_data.sum(axis=1).max() * 1.15
        ax.set_ylim(0, y_max)

        # 图例放在右侧
        fig.subplots_adjust(right=0.8)  # 给右侧预留空间
        ax.legend(
            handles=stack,
            labels=y_headers,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=10,
        )

        # 保存
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        for ext in ["png", "svg"]:
            save_dir = os.path.join(base_save_dir, ext)
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{stem}.{ext}"), bbox_inches="tight", dpi=DPI)
        plt.close()
        print(f"生成图表: {stem}")

    except Exception as e:
        print(f"处理 {csv_path} 时出错: {e}")
        if "fig" in locals():
            plt.close()


def batch_plot_area(csv_dir="csv", output_dir="charts"):
    """批量生成堆叠面积图"""
    if not os.path.exists(csv_dir):
        print(f"目录不存在: {csv_dir}")
        return

    for csv_path in glob.glob(os.path.join(csv_dir, "*.csv")):
        print(f"正在处理: {os.path.basename(csv_path)}")
        plot_area_from_csv(csv_path, output_dir)


if __name__ == "__main__":
    batch_plot_area()
