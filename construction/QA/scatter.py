import os
import glob
import json
import re
import random
import math
import traceback
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ==============================================================================
# 1. UTILITY FUNCTIONS (整合自两个文件)
#    - read_metadata 已更新以适应新的CSV格式
# ==============================================================================

def read_metadata(filepath: str) -> dict:
    """
    读取CSV文件的第一行元数据。
    优化：适配新的三列格式: topic,little_theme,generation_type
    """
    try:
        meta_df = pd.read_csv(filepath, header=None, nrows=1, encoding='utf-8')
        if meta_df.empty:
            return {}
        meta = meta_df.iloc[0].tolist()
        # 更新：使用新的键
        keys = ['topic', 'little_theme', 'generation_type']
        return dict(zip(keys, [str(item).strip() if pd.notna(item) else "" for item in (meta + [""]*len(keys))[:len(keys)]]))
    except Exception as e:
        print(f"Error reading metadata from {filepath}: {e}")
        return {}

def read_axis_labels(filepath: str) -> tuple[str, str, str, str]:
    """读取CSV文件的第二行，解析坐标轴标签和单位。"""
    x_label, x_unit, y_label, y_unit = "", "", "", ""
    try:
        labels_df = pd.read_csv(filepath, header=None, skiprows=1, nrows=1, encoding='utf-8')
        if labels_df.empty:
            return x_label, x_unit, y_label, y_unit

        labels = labels_df.iloc[0].tolist()

        def parse_label_unit(label_str):
            if isinstance(label_str, str):
                match = re.match(r'(.+)\s*\((.+)\)', label_str)
                if match:
                    return match.group(1).strip(), match.group(2).strip()
                return label_str.strip(), ""
            return str(label_str).strip(), ""

        if len(labels) >= 1: x_label, x_unit = parse_label_unit(labels[0])
        if len(labels) >= 2: y_label, y_unit = parse_label_unit(labels[1])
    except Exception as e:
        print(f"Error reading axis labels from {filepath}: {e}")
    return x_label, x_unit, y_label, y_unit

def read_scatter_data_df(filepath: str) -> Optional[pd.DataFrame]:
    """读取散点图的数据部分到一个DataFrame中。"""
    try:
        df = pd.read_csv(filepath, header=None, skiprows=2, encoding='utf-8', on_bad_lines='warn').apply(pd.to_numeric, errors='coerce')
        if df.shape[1] < 2:
            print(f"Warning: Not enough data columns in {filepath} (need at least 2).")
            return None
        df = df.rename(columns={0: 'x', 1: 'y'})
        df = df.dropna(subset=['x', 'y'])
        if df.empty:
            print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.")
            return None
        return df
    except Exception as e:
        print(f"Error reading scatter data from {filepath}: {e}")
        return None

def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]], qa_dir: str = './scatter/QA'):
    """将QA条目写入JSON文件，会加载、更新和保存，并避免重复。"""
    if not qa_items: return # 如果没有生成QA，则不执行任何操作

    json_dir = qa_dir
    os.makedirs(json_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    json_path = os.path.join(json_dir, base_name + '.json')

    template_data: Dict[str, List] = {
        "CTR": [], "VEC": [], "SRP": [], "VPR": [], "VE": [],
        "EVJ": [], "SC": [], "NF": [], "NC": [], "MSR": [], "VA": []
    }

    existing_data: Dict[str, List] = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict): existing_data = loaded_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse JSON from {json_path}: {e}. Starting with template.")

    data_to_save = template_data.copy()
    for key in template_data:
        if key in existing_data and isinstance(existing_data.get(key), list):
            data_to_save[key] = existing_data[key]

    # 避免重复添加
    existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save.get(qa_type, []) if isinstance(item, dict)}
    new_items_to_add = [item for item in qa_items if (item.get('Q'), item.get('A')) not in existing_qa_pairs]
    data_to_save[qa_type].extend(new_items_to_add)

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing QA to {json_path} for type {qa_type}: {e}")

# ==============================================================================
# 2. CALCULATION FUNCTIONS (整合自两个文件)
# ==============================================================================

def task_count_points(df: pd.DataFrame) -> int:
    """计算数据点总数。"""
    if df is None or df.empty: return 0
    return len(df)

def task_get_min_max(df: pd.DataFrame) -> Dict[str, float]:
    """计算X和Y轴的最小值和最大值。"""
    results = {}
    if df is None or df.empty: return results
    if 'x' in df.columns:
        results['x_min'] = df['x'].min()
        results['x_max'] = df['x'].max()
    if 'y' in df.columns:
        results['y_min'] = df['y'].min()
        results['y_max'] = df['y'].max()
    return results

def task_get_averages(df: pd.DataFrame) -> Dict[str, float]:
    """计算X和Y轴的平均值。"""
    results = {}
    if df is None or df.empty: return results
    if 'x' in df.columns: results['x_avg'] = df['x'].mean()
    if 'y' in df.columns: results['y_avg'] = df['y'].mean()
    return results

def task_get_extreme_points(df: pd.DataFrame, n: int = 3) -> Dict[str, List[float]]:
    """寻找与最大/最小X/Y值对应的点。"""
    results: Dict[str, List[float]] = {'top_x_y': [], 'bottom_x_y': [], 'top_y_x': [], 'bottom_y_x': []}
    if df is None or df.empty or len(df) < n: return results
    
    top_x_points = df.nlargest(n, columns='x')
    results['top_x_y'] = top_x_points['y'].tolist()

    bottom_x_points = df.nsmallest(n, columns='x')
    results['bottom_x_y'] = bottom_x_points['y'].tolist()

    top_y_points = df.nlargest(n, columns='y')
    results['top_y_x'] = top_y_points['x'].tolist()

    bottom_y_points = df.nsmallest(n, columns='y')
    results['bottom_y_x'] = bottom_y_points['x'].tolist()
    return results

def task_count_points_in_range(df: pd.DataFrame, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> int:
    """计算在指定矩形范围内的数据点数量。"""
    if df is None or df.empty: return 0
    min_x, max_x = x_range
    min_y, max_y = y_range
    filtered_df = df[(df['x'] >= min_x) & (df['x'] <= max_x) & (df['y'] >= min_y) & (df['y'] <= max_y)]
    return len(filtered_df)

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点间的欧氏距离。"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def task_find_extreme_pairs(df: pd.DataFrame) -> Dict[str, Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """寻找距离最近和最远的点对。"""
    results: Dict[str, Optional[Tuple]] = {'closest_pair': None, 'farthest_pair': None}
    if df is None or df.empty or len(df) < 2: return results

    points = list(df[['x', 'y']].itertuples(index=False, name=None))
    min_dist, max_dist = float('inf'), 0.0
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = euclidean_distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                results['closest_pair'] = (points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                results['farthest_pair'] = (points[i], points[j])
    return results

def task_find_n_closest_pairs(df: pd.DataFrame, n: int) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """寻找距离最近的N个点对。"""
    if df is None or df.empty or len(df) < 2: return []
    points = list(df[['x', 'y']].itertuples(index=False, name=None))
    distances_with_pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = euclidean_distance(points[i], points[j])
            distances_with_pairs.append((dist, (points[i], points[j])))
    
    distances_with_pairs.sort(key=lambda item: item[0])
    return [pair for _, pair in distances_with_pairs[:n]]

# ==============================================================================
# 3. QA FILLING FUNCTIONS (整合自两个文件)
#    - fill_qa_vpr 已更新以使用新的 generation_type
#    - fill_qa_va 已实现
# ==============================================================================

def fill_qa_ctr() -> List[Dict[str, str]]:
    """生成图表类型(CTR)的QA。"""
    return [{"Q": "What type of chart is this?", "A": "This chart is a {scatter} chart."}]

def fill_qa_vec(point_count: int) -> List[Dict[str, str]]:
    """生成数据点数量(VEC)的QA。"""
    return [{"Q": "How many points are in this scatter plot?", "A": f"There are {{{point_count}}} points."}]

def fill_qa_vpr(metadata: Dict[str, str]) -> List[Dict[str, str]]:
    """
    生成视觉模式识别(VPR)的QA。
    优化：直接使用 generation_type 生成描述。
    """
    qa_list = []
    question = "What pattern does the data in the scatter plot show?"
    gen_type = metadata.get('generation_type', '').strip()

    if not gen_type:
        pattern_description = "no clear pattern stated"
    elif gen_type.lower() == 'random':
        pattern_description = "a random distribution with no clear correlation"
    else:
        # e.g., "Positive Linear" -> "a positive linear relationship"
        pattern_description = f"a {gen_type.lower()} relationship"
    
    answer = f"The scatter plot shows {{{pattern_description}}}."
    qa_list.append({"Q": question, "A": answer})
    return qa_list

def fill_qa_evj(min_max_values: Dict[str, float], x_label: str, y_label: str) -> List[Dict[str, str]]:
    """生成极值(EVJ)的QA。"""
    qa_list = []
    x_min, x_max = min_max_values.get('x_min'), min_max_values.get('x_max')
    y_min, y_max = min_max_values.get('y_min'), min_max_values.get('y_max')

    if x_max is not None: qa_list.append({"Q": f"What is the maximum observed value in dimension {x_label} in the scatter plot?", "A": f"The maximum observed value in the {x_label} dimension is {{{x_max:.2f}}}."})
    if x_min is not None: qa_list.append({"Q": f"What is the minimum observed value in dimension {x_label} in the scatter plot?", "A": f"The minimum observed value in the {x_label} dimension is {{{x_min:.2f}}}."})
    if y_max is not None: qa_list.append({"Q": f"What is the maximum observed value in dimension {y_label} in the scatter plot?", "A": f"The maximum observed value in the {y_label} dimension is {{{y_max:.2f}}}."})
    if y_min is not None: qa_list.append({"Q": f"What is the minimum observed value in dimension {y_label} in the scatter plot?", "A": f"The minimum observed value in the {y_label} dimension is {{{y_min:.2f}}}."})
    return qa_list

def fill_qa_sc(average_values: Dict[str, float], x_label: str, y_label: str) -> List[Dict[str, str]]:
    """生成统计计算(SC)的QA。"""
    qa_list = []
    x_avg, y_avg = average_values.get('x_avg'), average_values.get('y_avg')
    if x_avg is not None: qa_list.append({"Q": f"What is the average value of the {x_label} for all points?", "A": f"The average value of {x_label} for all points is {{{x_avg:.2f}}}."})
    if y_avg is not None: qa_list.append({"Q": f"What is the average value of the {y_label} for all points?", "A": f"The average value of {y_label} for all points is {{{y_avg:.2f}}}."})
    return qa_list

def fill_qa_nf(extreme_points_data: Dict[str, List[float]], x_label: str, y_label: str) -> List[Dict[str, str]]:
    """生成数据查找(NF)的QA。"""
    qa_list = []
    def format_list(lst): return ", ".join([f"{v:.2f}" for v in lst]) if lst else "N/A"
    
    if extreme_points_data['top_x_y']: qa_list.append({"Q": f"What are the {y_label} values corresponding to the top 3 {x_label} values in the scatter plot?", "A": f"{{{format_list(extreme_points_data['top_x_y'])}}}."})
    if extreme_points_data['bottom_x_y']: qa_list.append({"Q": f"What are the {y_label} values corresponding to the bottom 3 {x_label} values in the scatter plot?", "A": f"{{{format_list(extreme_points_data['bottom_x_y'])}}}."})
    if extreme_points_data['top_y_x']: qa_list.append({"Q": f"What are the {x_label} values corresponding to the top 3 {y_label} values in the scatter plot?", "A": f"{{{format_list(extreme_points_data['top_y_x'])}}}."})
    if extreme_points_data['bottom_y_x']: qa_list.append({"Q": f"What are the {x_label} values corresponding to the bottom 3 {y_label} values in the scatter plot?", "A": f"{{{format_list(extreme_points_data['bottom_y_x'])}}}."})
    return qa_list

def fill_qa_msr(df: pd.DataFrame) -> List[Dict[str, str]]:
    """生成补充的MSR QA（点数统计，最远/最近点对）。"""
    qa_list = []
    if df is None or df.empty or len(df) < 2: return qa_list

    # --- MSR Q1: 在随机范围内统计点数 ---
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    if pd.notna(x_min) and pd.notna(x_max) and pd.notna(y_min) and pd.notna(y_max) and x_min < x_max and y_min < y_max:
        try:
            x_range_width = random.uniform((x_max - x_min) * 0.1, (x_max - x_min) * 0.3)
            y_range_width = random.uniform((y_max - y_min) * 0.1, (y_max - y_min) * 0.3)
            x_start = random.uniform(x_min, x_max - x_range_width)
            y_start = random.uniform(y_min, y_max - y_range_width)
            x_range_q = (x_start, x_start + x_range_width)
            y_range_q = (y_start, y_start + y_range_width)
            count = task_count_points_in_range(df, x_range_q, y_range_q)
            
            question = f"How many points in the scatter plot have X coordinates between {x_range_q[0]:.2f} and {x_range_q[1]:.2f} and Y coordinates between {y_range_q[0]:.2f} and {y_range_q[1]:.2f}?"
            answer = f"There are {{{count}}} points."
            qa_list.append({"Q": question, "A": answer})
        except Exception as e:
            print(f"Info (MSR): Could not generate random range count QA. Error: {e}")

    # --- MSR Q2 & Q3: 最远和最近的点对 ---
    extreme_pairs = task_find_extreme_pairs(df)
    farthest_pair = extreme_pairs.get('farthest_pair')
    closest_pair = extreme_pairs.get('closest_pair')

    if farthest_pair:
        p1, p2 = farthest_pair
        p1_f, p2_f = f"({p1[0]:.2f}, {p1[1]:.2f})", f"({p2[0]:.2f}, {p2[1]:.2f})"
        question = "What are the coordinates of the pair of points that are farthest from each other in terms of Euclidean distance?"
        answer = f"The pair of points with the largest Euclidean distance has coordinates {{{p1_f} and {p2_f}}}."
        qa_list.append({"Q": question, "A": answer})
        
    if closest_pair:
        p1, p2 = closest_pair
        p1_f, p2_f = f"({p1[0]:.2f}, {p1[1]:.2f})", f"({p2[0]:.2f}, {p2[1]:.2f})"
        question = "What are the coordinates of the pair of points that are closest to each other in terms of Euclidean distance?"
        answer = f"The pair of points with the smallest Euclidean distance has coordinates {{{p1_f} and {p2_f}}}."
        qa_list.append({"Q": question, "A": answer})
        
    return qa_list

def fill_qa_va(df: pd.DataFrame, n: int = 3) -> List[Dict[str, str]]:
    """
    新增：生成视觉关联(VA)的QA，识别最近的点对。
    """
    qa_list = []
    if df is None or df.empty or len(df) < 2: return qa_list

    closest_pairs = task_find_n_closest_pairs(df, n=n)
    if not closest_pairs: return qa_list

    question = f"Identify the coordinates of the {n} pairs of points that are closest to each other in terms of Euclidean distance."
    
    formatted_pairs = []
    for p1, p2 in closest_pairs:
        p1_str = f"({p1[0]:.2f}, {p1[1]:.2f})"
        p2_str = f"({p2[0]:.2f}, {p2[1]:.2f})"
        formatted_pairs.append(f"({p1_str}, {p2_str})")
    
    answer_content = ", ".join(formatted_pairs)
    answer = f"The {n} closest pairs of points are: {{{answer_content}}}."
    qa_list.append({"Q": question, "A": answer})
    return qa_list

# ==============================================================================
# 4. MAIN ORCHESTRATION FUNCTION (整合自两个文件)
# ==============================================================================

def main():
    """主函数，遍历所有CSV文件并生成所有类型的QA。"""
    random.seed(42)  # for reproducibility
    csv_dir = './csv'
    qa_dir = './QA'

    if not os.path.isdir(csv_dir):
        print(f"错误: CSV目录 '{csv_dir}' 不存在。请先运行数据生成脚本。")
        return

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print(f"在 '{csv_dir}' 中未找到任何CSV文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，开始生成QA...")
    
    for csv_path in csv_files:
        print(f"\n--- 正在处理: {os.path.basename(csv_path)} ---")

        # 1. 读取元数据和数据
        metadata = read_metadata(csv_path)
        x_label, _, y_label, _ = read_axis_labels(csv_path)
        df_data = read_scatter_data_df(csv_path)

        # 2. 生成与写入非数据依赖的QA
        write_qa_to_json(csv_path, "CTR", fill_qa_ctr(), qa_dir)
        write_qa_to_json(csv_path, "VPR", fill_qa_vpr(metadata), qa_dir)

        # 3. 如果数据有效，生成并写入数据依赖的QA
        if df_data is not None and not df_data.empty:
            point_count = task_count_points(df_data)
            min_max_vals = task_get_min_max(df_data)
            avg_vals = task_get_averages(df_data)
            extreme_points = task_get_extreme_points(df_data, n=3)

            write_qa_to_json(csv_path, "VEC", fill_qa_vec(point_count), qa_dir)
            
            if x_label and y_label:
                write_qa_to_json(csv_path, "EVJ", fill_qa_evj(min_max_vals, x_label, y_label), qa_dir)
                write_qa_to_json(csv_path, "SC", fill_qa_sc(avg_vals, x_label, y_label), qa_dir)
                write_qa_to_json(csv_path, "NF", fill_qa_nf(extreme_points, x_label, y_label), qa_dir)
            else:
                print("  警告: 坐标轴标签缺失，跳过 EVJ, SC, NF 的QA生成。")

            # 生成补充的 MSR 和 VA 问题
            write_qa_to_json(csv_path, "MSR", fill_qa_msr(df_data), qa_dir)
            write_qa_to_json(csv_path, "VA", fill_qa_va(df_data, n=3), qa_dir)
        else:
            print("  警告: 数据无效或为空，跳过所有数据依赖的QA生成。")
            # 写入一个点数为0的VEC
            write_qa_to_json(csv_path, "VEC", fill_qa_vec(0), qa_dir)

    print(f"\nQA文件生成完毕，共处理 {len(csv_files)} 个文件。输出目录: {qa_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"主程序发生意外错误: {e}")
        traceback.print_exc()

