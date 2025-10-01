import os
import pandas as pd
import json
import random
from io import StringIO
from collections import defaultdict

input_dir = './csv/'
output_dir = './QA/'
os.makedirs(output_dir, exist_ok=True)

random.seed(42)

def get_layer_depths(df):
    child_to_parent = dict(zip(df['child'], df['parent']))
    depths = {}

    def get_depth(node):
        if node == 'Root':
            return 0
        if node in depths:
            return depths[node]
        parent = child_to_parent.get(node, 'Root')
        depth = get_depth(parent) + 1
        depths[node] = depth
        return depth

    for node in set(df['child']).union(set(df['parent'])):
        get_depth(node)
    return depths

for filename in os.listdir(input_dir):
    if not filename.endswith('.csv'):
        continue

    csv_path = os.path.join(input_dir, filename)
    print(f"\n[INFO] Processing file: {filename}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            lines = f.readlines()

        header_parts = [part.strip() for part in first_line.split(',')]
        if len(header_parts) < 3:
            print(f"[WARN] Header too short in: {filename}")
            continue
        big_theme, chart_title, unit_info = header_parts[:3]

        csv_content = ''.join(lines)
        df = pd.read_csv(StringIO(csv_content))

        if not {'parent', 'child', 'value'}.issubset(df.columns):
            print(f"[ERROR] Missing columns in {filename}")
            continue

        depths = get_layer_depths(df)
        max_depth = max(depths.values())

        value_map = dict(zip(df['child'], df['value']))
        layer_segments = defaultdict(list)
        for node, depth in depths.items():
            layer_segments[depth].append(node)

        innermost = min(layer_segments.keys())
        outermost = max(layer_segments.keys())

        top_level = [row['child'] for _, row in df.iterrows() if row['parent'] == 'Root']
        top_level_vals = {c: value_map.get(c, 0) for c in top_level}
        top_max_cat = max(top_level_vals.items(), key=lambda x: x[1])

        sorted_values = sorted(value_map.items(), key=lambda x: x[1], reverse=True)
        max_cat, max_val = sorted_values[0]
        min_cat, min_val = sorted_values[-1]

        # VE 随机选2-3个
        ve_n = random.randint(2, 3)
        ve_items = random.sample(sorted_values, min(ve_n, len(sorted_values)))

        # --- NF: 数据驱动的阈值筛选问题 ---
        import numpy as np

        segment = top_max_cat[0]
        children_of_segment = df[df['parent'] == segment][['child', 'value']].values.tolist()
        total_segment_value = sum(val for _, val in children_of_segment)

        if total_segment_value == 0:
            print(f"[WARN] Segment {segment} has zero total value, skipping NF.")
            above_thresh1, below_thresh2 = [], []
            threshold1, threshold2 = 0.3, 0.1  # fallback
        else:
            # 计算相对比例
            rel_ratios = [val / total_segment_value for _, val in children_of_segment]
            q1 = np.percentile(rel_ratios, 25)
            q3 = np.percentile(rel_ratios, 75)
            mean_ratio = np.mean(rel_ratios)

            threshold1 = round(q3, 2)
            threshold2 = round(q1, 2)

            # 查找对应的类别
            above_thresh1 = [(cat, round(val / total_segment_value * 100, 2)) for cat, val in children_of_segment if
                             val / total_segment_value > threshold1]
            below_thresh2 = [(cat, round(val / total_segment_value * 100, 2)) for cat, val in children_of_segment if
                             val / total_segment_value < threshold2]

            # 如果为空，尝试改为mean
            if not above_thresh1 and threshold1 != mean_ratio:
                threshold1 = round(mean_ratio, 2)
                above_thresh1 = [(cat, round(val / total_segment_value * 100, 2)) for cat, val in children_of_segment if
                                 val / total_segment_value > threshold1]

            if not below_thresh2 and threshold2 != mean_ratio:
                threshold2 = round(mean_ratio, 2)
                below_thresh2 = [(cat, round(val / total_segment_value * 100, 2)) for cat, val in children_of_segment if
                                 val / total_segment_value < threshold2]

        # NC 2~4个随机选择
        nc_qas = []
        nc_items = sorted_values[:5]
        nc_templates = []

        if len(nc_items) >= 2:
            nc_templates.append({
                "Q": f"Which is larger? the value of {nc_items[0][0]} or {nc_items[1][0]}.",
                "A": f"The value of {{{nc_items[0][0]}}} is larger."
            })
        if len(nc_items) >= 3:
            nc_templates.append({
                "Q": f"Which is larger? the value of {nc_items[0][0]}, {nc_items[1][0]}, or {nc_items[2][0]}.",
                "A": f"The value of {{{nc_items[0][0]}}} is larger."
            })
        if len(nc_items) >= 4:
            nc_templates.append({
                "Q": f"Which is smaller? the value of {nc_items[1][0]}, {nc_items[2][0]}, or {nc_items[3][0]}.",
                "A": f"The value of {{{nc_items[3][0]}}} is smaller."
            })
        if len(nc_items) >= 5:
            nc_templates.append({
                "Q": f"Which has the largest value? {nc_items[0][0]}, {nc_items[1][0]}, {nc_items[2][0]}, or {nc_items[4][0]}.",
                "A": f"The value of {{{nc_items[0][0]}}} is the largest."
            })

        nc_qas = random.sample(nc_templates, k=min(random.randint(2, 4), len(nc_templates)))

        # --- SRP: relative position (outside, inside, same level) ---
        srp_qas = []

        available_nodes = list(depths.keys())
        if len(available_nodes) >= 2:
            a, b = random.sample(available_nodes, 2)
            depth_a, depth_b = depths[a], depths[b]
            if depth_a > depth_b:
                relation = "outside"
            elif depth_a < depth_b:
                relation = "inside"
            else:
                relation = "on the same side"

            srp_qas.append({
                "Q": f"Is {a} positioned outside, inside or on the same side as {b}?",
                "A": f"{a} is positioned {{{relation}}} relative to {b}."
            })
        # --- MSR: largest contribution path from innermost leaf ---
        msr_qas = []

        # 构建child到parent映射
        child_to_parent = dict(zip(df['child'], df['parent']))

        # 筛选最深层节点（即层数等于 max_depth）
        deepest_nodes = [node for node, d in depths.items() if d == max_depth and node in value_map]

        if deepest_nodes:
            max_leaf = max(deepest_nodes, key=lambda n: value_map[n])
            path = [max_leaf]
            current = max_leaf
            while current in child_to_parent:
                parent = child_to_parent[current]
                if parent == 'Root':
                    break
                path.append(parent)
                current = parent
            full_path = " -> ".join(reversed(path))
            msr_qas.append({
                "Q": "Which hierarchical path in the sunburst chart makes the largest contribution to the total?",
                "A": f"The path that contributes the most to the total is: {{{full_path}}}."
            })

        qa_data = {
            "CTR": [
                {"Q": "What type of chart is this?", "A": "This chart is a {sunburst} chart."}
            ],
            "VEC": [
                {"Q": f"How many ring layers are in this sunburst chart?", "A": f"There are {{{max_depth}}} ring layers."},
                {"Q": f"How many segments are in this sunburst chart?", "A": f"There are {{{len(df)}}} segments."},
                {"Q": f"How many segments are in the innermost ring of this sunburst chart?", "A": f"There are {{{len(layer_segments[innermost])}}} segments in the innermost ring."},
                {"Q": f"How many segments are in the outermost ring of this sunburst chart?", "A": f"There are {{{len(layer_segments[outermost])}}} segments in the outermost ring."}
            ],
            "SRP":srp_qas,
            "VPR": [
                {"Q": f"Which top-level category has the highest proportion in this sunburst chart?",
                 "A": f"The top-level category with the highest proportion is {{{top_max_cat[0]}}}."}
            ],
            "VE": [
                {"Q": f"What is the value of the {cat} segment in this sunburst chart?",
                 "A": f"The value of the {cat} segment is {{{val}}}."}
                for cat, val in ve_items
            ],
            "EVJ": [
                {"Q": f"What is the maximum value of the innermost layer in this sunburst chart?",
                 "A": f"The maximum value of the innermost layer is {{{max([value_map[n] for n in layer_segments[innermost]])}}}."},
                {"Q": f"What is the minimum value of the innermost layer in this sunburst chart?",
                 "A": f"The minimum value of the innermost layer is {{{min([value_map[n] for n in layer_segments[innermost]])}}}."},
                {"Q": f"What is the maximum value of the outermost layer in this sunburst chart?",
                 "A": f"The maximum value of the outermost layer is {{{max([value_map[n] for n in layer_segments[outermost]])}}}."},
                {"Q": f"What is the minimum value of the outermost layer in this sunburst chart?",
                 "A": f"The minimum value of the outermost layer is {{{min([value_map[n] for n in layer_segments[outermost]])}}}."}
            ],
            "SC": [
                {"Q": f"What is the total value of the second layer?", "A": f"The total value of the second layer is {{{round(sum([value_map[n] for n in layer_segments[2]]), 2)}}}."}
            ],
            "NF": [
                {
                    "Q": f"Which categories in the segment {segment} exceed {int(threshold1 * 100)}%? Please list the category names and corresponding percentages.",
                    "A": " and ".join([f"{{{cat}}} accounts for {{{val}}}%." for cat, val in above_thresh1]) if above_thresh1 else "None."
                },
                {
                    "Q": f"Which categories in the segment {segment} are below {int(threshold2 * 100)}%? Please list the category names and corresponding percentages.",
                    "A": " and ".join([f"{{{cat}}} accounts for {{{val}}}%." for cat, val in below_thresh2]) if below_thresh2 else "None."
                }
            ],
            "NC": nc_qas,
            "MSR": msr_qas,
            "VA":[]
        }

        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)

        print(f"[SUCCESS] QA JSON written to: {json_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
