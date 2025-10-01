import os
import pandas as pd
import json
import random
from io import StringIO
from collections import defaultdict

input_dir = './csv/treemap/'
output_dir = './QA/treemap/'
os.makedirs(output_dir, exist_ok=True)

random.seed(42)

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
            print(f"[WARN] Header too short: {filename}")
            continue
        big_theme, chart_title, unit_info = header_parts[:3]

        csv_content = ''.join(lines)
        df = pd.read_csv(StringIO(csv_content))

        if not {'parent', 'child', 'value'}.issubset(df.columns):
            print(f"[ERROR] Missing required columns.")
            continue

        value_map = dict(zip(df['child'], df['value']))
        parent_map = dict(zip(df['child'], df['parent']))
        parent_to_children = defaultdict(list)
        for _, row in df.iterrows():
            parent_to_children[row['parent']].append((row['child'], row['value']))

        all_nodes = list(value_map.keys())
        sorted_vals = sorted(value_map.items(), key=lambda x: x[1], reverse=True)
        max_node, max_val = sorted_vals[0]
        min_node, min_val = sorted_vals[-1]

        top_level = [row['child'] for _, row in df.iterrows() if pd.isna(row['parent'])]
        top_level_vals = {c: value_map[c] for c in top_level}
        top_max_node = max(top_level_vals.items(), key=lambda x: x[1]) if top_level_vals else (max_node, max_val)

        example_parent = next(iter(parent_to_children.keys() - {None}), None)
        child_count = len(parent_to_children[example_parent]) if example_parent else 0
        children_values = parent_to_children.get(example_parent, [])

        # --- VE 控制在 2-3 个
        ve_items = random.sample(list(value_map.items()), min(random.randint(2, 3), len(value_map)))

        # --- SC 精确生成两个问题：一差值，一加和值，且标签不同
        sc_qas = []
        sc_candidates = sorted_vals.copy()
        random.shuffle(sc_candidates)

        # 选择前四个不重复的标签构造两个问题
        if len(sc_candidates) >= 4:
            a1, b1, a2, b2 = sc_candidates[0], sc_candidates[1], sc_candidates[2], sc_candidates[3]

            sc_qas.append({
                "Q": f"What is the difference between the regions of {a1[0]} and {b1[0]}?",
                "A": f"The difference between {a1[0]} and {b1[0]} is {{{round(abs(a1[1] - b1[1]), 2)}}}."
            })
            sc_qas.append({
                "Q": f"What is the sum of the regions of {a2[0]} and {b2[0]}?",
                "A": f"The sum of {a2[0]} and {b2[0]} is {{{round(a2[1] + b2[1], 2)}}}."
            })

        # --- NF: 强制生成至少一个问题 ---
        import numpy as np

        nf_qas = []

        if children_values:
            child_vals = [v for _, v in children_values]
            q1 = np.percentile(child_vals, 25)
            q3 = np.percentile(child_vals, 75)
            mean_val = np.mean(child_vals)

            # 初始阈值
            threshold1 = round(q3, 2)
            threshold2 = round(q1, 2)

            above_thresh = [(c, v) for c, v in children_values if v > threshold1]
            below_thresh = [(c, v) for c, v in children_values if v < threshold2]

            # 若无结果，尝试均值
            if not above_thresh:
                threshold1 = round(mean_val, 2)
                above_thresh = [(c, v) for c, v in children_values if v > threshold1]

            if not below_thresh:
                threshold2 = round(mean_val, 2)
                below_thresh = [(c, v) for c, v in children_values if v < threshold2]

            # 若仍无结果，强制选1个最大和1个最小值生成
            if not above_thresh and children_values:
                max_item = max(children_values, key=lambda x: x[1])
                threshold1 = max_item[1] - 0.01  # 稍微小于最大值
                above_thresh = [max_item]

            if not below_thresh and children_values:
                min_item = min(children_values, key=lambda x: x[1])
                threshold2 = min_item[1] + 0.01  # 稍微大于最小值
                below_thresh = [min_item]

            # 构造问题
            # 构造问题（保留两位小数）
            if above_thresh:
                nf_qas.append({
                    "Q": f"Under the {example_parent} category in the treemap, which rectangular labels have a value exceeded {threshold1:.2f}? Please list the labels and values.",
                    "A": " and ".join([f"{{{c}}} has {{{v:.2f}}}" for c, v in above_thresh])
                })

            if below_thresh:
                nf_qas.append({
                    "Q": f"Under the {example_parent} category in the treemap, which rectangular labels have a value below {threshold2:.2f}? Please list the labels and values.",
                    "A": " and ".join([f"{{{c}}} has {{{v:.2f}}}" for c, v in below_thresh])
                })

            # ⚠️ 若上述都失败，强制至少出一个问题（取最大值）
            if not nf_qas and children_values:
                fallback_item = max(children_values, key=lambda x: x[1])
                nf_qas.append({
                    "Q": f"In the category {example_parent}, which label has the highest value?",
                    "A": f"The label {{{fallback_item[0]}}} has the highest value of {{{fallback_item[1]:.2f}}}."
                })

        # for label, group, desc in random.sample(nf_templates, nf_sample_count):
        #     nf_qas.append({
        #         "Q": f"Under the {example_parent} category in the treemap, {desc} Please list the labels and values.",
        #         "A": " and ".join([f"{{{c}}} has {{{v}}}" for c, v in group]) if group else "None."
        #     })

        # --- NC 随机 2-4 个问题（真正控制数量）
        nc_items = sorted_vals[:5]
        all_nc_questions = []

        if len(nc_items) >= 2:
            all_nc_questions.append({
                "Q": f"Which is larger? the value of {nc_items[0][0]} or {nc_items[1][0]}.",
                "A": f"The value of {{{nc_items[0][0]}}} is larger."
            })
        if len(nc_items) >= 3:
            all_nc_questions.append({
                "Q": f"Which is larger? the value of {nc_items[0][0]}, {nc_items[1][0]}, or {nc_items[2][0]}.",
                "A": f"The value of {{{nc_items[0][0]}}} is larger."
            })
        if len(nc_items) >= 4:
            all_nc_questions.append({
                "Q": f"Which is smaller? the value of {nc_items[1][0]}, {nc_items[2][0]}, or {nc_items[3][0]}.",
                "A": f"The value of {{{nc_items[3][0]}}} is smaller."
            })
        if len(nc_items) >= 5:
            all_nc_questions.append({
                "Q": f"Which is larger? the value of {nc_items[2][0]} or {nc_items[4][0]}.",
                "A": f"The value of {{{nc_items[2][0]}}} is larger."
            })

        nc_count = min(random.randint(2, 4), len(all_nc_questions))
        nc_qas = random.sample(all_nc_questions, nc_count)

        # --- MSR: leaf nodes below global average ---
        msr_qas = []

        # 找到所有 leaf nodes（不作为 parent 出现的 child）
        all_parents = set(df['parent'].dropna())
        leaf_nodes = [child for child in df['child'] if child not in all_parents and child in value_map]

        leaf_values = [value_map[n] for n in leaf_nodes]
        global_avg = np.mean(list(value_map.values()))
        below_avg_leaves = [n for n in leaf_nodes if value_map[n] < global_avg]

        msr_qas.append({
            "Q": "How many leaf nodes' value below the global average in the treemap?",
            "A": f"There are {{{len(below_avg_leaves)}}} leaf nodes below the global average."
        })

        qa_data = {
            "CTR": [
                {"Q": "What type of chart is this?", "A": "This chart is a {treemap}."}
            ],
            "VEC": [
                {"Q": f"How many child rectangles are under the category {example_parent} in this treemap?",
                 "A": f"There are {{{child_count}}} child rectangles under the category {example_parent}."}
            ],
            "SRP": [],
            "VPR": [
                {"Q": "Which rectangle has the largest area in this treemap?", "A": f"The rectangle with the largest area is {{{max_node}}}."},
                {"Q": "Which rectangle has the smallest area in this treemap?", "A": f"The rectangle with the smallest area is {{{min_node}}}."},
                {"Q": "Which top-level category has the highest proportion in this treemap?",
                 "A": f"The top-level category with the highest proportion is {{{top_max_node[0]}}}."}
            ],
            "VE": [
                {"Q": f"What is the value of the {node} rectangle in this treemap?",
                 "A": f"The value of the {node} rectangle is {{{val}}}."}
                for node, val in ve_items
            ],
            "EVJ": [
                {"Q": "What is the maximum value among the partitions in the treemap?", "A": f"The maximum value is {{{max_val}}}."},
                {"Q": "What is the minimum value among the partitions in the treemap?", "A": f"The minimum value is {{{min_val}}}."}
            ],
            "SC": sc_qas,
            "NF": nf_qas,
            "NC": nc_qas,
            "MSR":msr_qas,
            "VA":[]
        }

        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)

        print(f"[SUCCESS] QA JSON written to: {json_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
