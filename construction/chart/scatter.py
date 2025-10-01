import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import numpy as np
from typing import Tuple, Dict, Any, List
import random
import traceback
import glob
import re
import matplotlib.ticker as mticker

# 定义颜色和标记样式 (保持不变)
CATEGORICAL_CMAPS = [
    'tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2',
]
MARKER_STYLES = ['o']

def parse_name_unit(info_string: str) -> Tuple[str, str]:
    """从 "名称 (单位)" 格式中解析名称和单位 (保持不变)"""
    name = info_string.strip()
    match = re.search(r'^(.*)\s*\(([^)]*)\)$', name)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return name, ""

def parse_scatter_csv(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    读取散点图CSV文件并解析其元数据。
    优化：此函数已更新，以匹配新的CSV文件头格式。
    """
    metadata: Dict[str, Any] = {}
    df = pd.DataFrame()

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件未找到: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            header_line_1 = f.readline().strip()
            header_line_2 = f.readline().strip()

        # --- 优化开始: 适配新的三列元数据格式 ---
        parts_1 = [part.strip() for part in header_line_1.split(',')]
        expected_parts_1 = 3  # 期望3个部分: Topic, Little_Theme, Generation_Type
        if len(parts_1) != expected_parts_1:
            raise ValueError(f"CSV第一行元数据格式错误。期望 {expected_parts_1} 个部分，但找到 {len(parts_1)}。行内容: '{header_line_1}'")

        metadata['Topic'] = parts_1[0]
        metadata['Little_Theme'] = parts_1[1]
        metadata['Generation_Type'] = parts_1[2]  # 使用新的 'Generation_Type' 键
        # --- 优化结束 ---

        # 解析第二行坐标轴信息 (逻辑保持不变)
        parts_2 = [part.strip() for part in header_line_2.split(',')]
        if len(parts_2) != 2:
            raise ValueError(f"CSV第二行坐标轴格式错误。期望 2 个部分，但找到 {len(parts_2)}。行内容: '{header_line_2}'")

        x_info_str, y_info_str = parts_2[0], parts_2[1]
        x_col_name_cleaned, x_unit_display = parse_name_unit(x_info_str)
        y_col_name_cleaned, y_unit_display = parse_name_unit(y_info_str)

        metadata['X_label_display'] = x_info_str
        metadata['Y_label_display'] = y_info_str
        metadata['X_col'] = x_col_name_cleaned
        metadata['Y_col'] = y_col_name_cleaned
        metadata['X_unit_display'] = x_unit_display
        metadata['Y_unit_display'] = y_unit_display

        # 读取数据 (逻辑保持不变)
        df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8', index_col=False, names=[metadata['X_col'], metadata['Y_col']])
        if df.empty:
            print(f"警告: 从 {filepath} 读取数据后DataFrame为空。")
            return df, metadata

        for col in [metadata['X_col'], metadata['Y_col']]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].round(2)
                if df[col].isnull().any():
                    print(f"警告: 文件 {filepath} 的列 '{col}' 包含非数值。")
            else:
                raise ValueError(f"逻辑错误: 列 '{col}' 在DataFrame中未找到。")

    except (FileNotFoundError, ValueError) as e:
        raise e
    except Exception as e:
        raise Exception(f"读取或解析CSV文件 {filepath} 时出错: {e}") from e

    return df, metadata

def visualize_scatter_plot(csv_file_path: str, png_filepath: str, svg_filepath: str) -> None:
    """从单个CSV文件可视化散点图数据，并保存为PNG和SVG。"""
    try:
        df, metadata = parse_scatter_csv(csv_file_path)

        x_col, y_col = metadata.get('X_col'), metadata.get('Y_col')
        if x_col is None or y_col is None or df.empty or df.dropna(subset=[x_col, y_col]).empty:
            print(f"跳过文件 {csv_file_path}: 缺少列信息或无有效数据点。")
            return

        x_col_cleaned = metadata.get('X_col', 'X')
        y_col_cleaned = metadata.get('Y_col', 'Y')
        x_unit_display = metadata.get('X_unit_display', '')
        y_unit_display = metadata.get('Y_unit_display', '')
        
        # --- 优化: 移除对旧元数据'Trend'和'Correlation_Type'的引用 ---
        little_theme = metadata.get('Little_Theme', metadata.get('Topic', 'Scatter Plot'))

        # 设置绘图样式 (逻辑保持不变)
        plt.style.use('default')
        rcParams['font.family'] = 'Times New Roman'
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 1
        # ... (其他样式设置保持不变)
        rcParams['axes.edgecolor'] = '#000000'
        rcParams['axes.labelcolor'] = '#000000'
        rcParams['xtick.color'] = '#000000'
        rcParams['ytick.color'] = '#000000'
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        # 随机化绘图属性 (颜色、大小、透明度、标记) (逻辑保持不变)
        # --- 随机选择一个非黄色的颜色 ---
        selected_color_rgba = '#1f77b4' # 默认蓝色
        yellow_r_threshold, yellow_g_threshold, yellow_b_threshold, yellow_sum_threshold = 0.9, 0.9, 0.5, 2.4
        for _ in range(10):
            cmap = plt.cm.get_cmap(random.choice(CATEGORICAL_CMAPS))
            candidate_colors = getattr(cmap, 'colors', [cmap(i) for i in np.linspace(0, 1, 15)])
            non_yellow_colors = [c for c in candidate_colors if not (c[0] > yellow_r_threshold and c[1] > yellow_g_threshold and c[2] > yellow_b_threshold and sum(c[:3]) > yellow_sum_threshold)]
            if non_yellow_colors:
                selected_color_rgba = random.choice(non_yellow_colors)
                break
        
        # --- 其他随机属性 ---
        scatter_area = random.uniform(np.pi * 3**2, np.pi * 6**2)
        random_alpha = random.uniform(0.8, 1.0)
        selected_marker = random.choice(MARKER_STYLES)

        # 绘制散点图
        ax.scatter(df[x_col], df[y_col], s=scatter_area, c=[selected_color_rgba], alpha=random_alpha, marker=selected_marker, edgecolors='w', linewidths=0.5)

        # 强化坐标轴 (逻辑保持不变)
        for spine in ['bottom', 'left']: ax.spines[spine].set_linewidth(2); ax.spines[spine].set_color('#333333')
        for spine in ['top', 'right']: ax.spines[spine].set_visible(True); ax.spines[spine].set_linewidth(1); ax.spines[spine].set_color('#000000')
        ax.tick_params(axis='both', which='both', direction='out', top=True, right=True)
        
        # 设置标题
        ax.set_title(little_theme, fontsize=14, pad=20, color='#000000')
        
        # 自动处理坐标轴标签、单位和科学计数法偏移量 (逻辑保持不变)
        plt.tight_layout()
        x_offset_text, y_offset_text = "", ""
        if isinstance(ax.xaxis.get_major_formatter(), mticker.ScalarFormatter):
            offset_artist = ax.xaxis.get_offset_text()
            if offset_artist and offset_artist.get_text().strip() and offset_artist.get_text().strip() != '\x08':
                x_offset_text = offset_artist.get_text().strip()
                offset_artist.set_visible(False)
        if isinstance(ax.yaxis.get_major_formatter(), mticker.ScalarFormatter):
            offset_artist = ax.yaxis.get_offset_text()
            if offset_artist and offset_artist.get_text().strip() and offset_artist.get_text().strip() != '\x08':
                y_offset_text = offset_artist.get_text().strip()
                offset_artist.set_visible(False)

        label_parts_x = [part for part in [x_offset_text, x_unit_display] if part]
        x_label_final = f"{x_col_cleaned} ({' '.join(label_parts_x)})" if label_parts_x else x_col_cleaned
        
        label_parts_y = [part for part in [y_offset_text, y_unit_display] if part]
        y_label_final = f"{y_col_cleaned} ({' '.join(label_parts_y)})" if label_parts_y else y_col_cleaned

        ax.set_xlabel(x_label_final, fontsize=12)
        ax.set_ylabel(y_label_final, fontsize=12)

        # 保存图像 (逻辑保持不变)
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
        plt.savefig(svg_filepath, bbox_inches='tight', format='svg')
        plt.close(fig)
        print(f"已保存图像: {png_filepath} 和 {svg_filepath}")

    except (FileNotFoundError, ValueError) as e:
        print(f"跳过文件 {csv_file_path}，原因: {e}")
    except Exception as e:
        print(f"处理文件 {csv_file_path} 时发生意外错误: {e}")
        traceback.print_exc()

def process_all_csv_files(input_dir: str, png_output_dir: str, svg_output_dir: str):
    """可视化目录中的所有CSV文件 (逻辑保持不变)"""
    if not os.path.isdir(input_dir):
        print(f"输入目录未找到: {input_dir}")
        return

    files_to_process = glob.glob(os.path.join(input_dir, "scatter_*.csv"))
    if not files_to_process:
        print(f"在 {input_dir} 中未找到 'scatter_*.csv' 文件。")
        return

    print(f"找到 {len(files_to_process)} 个 'scatter_*.csv' 文件进行处理。")
    files_to_process.sort() # 简单排序

    os.makedirs(png_output_dir, exist_ok=True)
    os.makedirs(svg_output_dir, exist_ok=True)

    for csv_path in files_to_process:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        png_output_path = os.path.join(png_output_dir, f"{base_name}.png")
        svg_output_path = os.path.join(svg_output_dir, f"{base_name}.svg")
        visualize_scatter_plot(csv_file_path=csv_path, png_filepath=png_output_path, svg_filepath=svg_output_path)

    print("\n可视化处理完成。")

if __name__ == "__main__":
    INPUT_CSV_DIR = './csv'
    OUTPUT_PNG_DIR = './png'
    OUTPUT_SVG_DIR = './svg'

    process_all_csv_files(input_dir=INPUT_CSV_DIR, png_output_dir=OUTPUT_PNG_DIR, svg_output_dir=OUTPUT_SVG_DIR)
