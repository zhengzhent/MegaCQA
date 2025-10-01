import os
import pandas as pd
import json
import random
from io import StringIO

input_dir = './pie/csv/'
output_dir = './pie/QA/'
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

        header_parts = [part.strip() for part in first_line.split(',') if part.strip()]
        if len(header_parts) < 3:
            print(f"[WARN] Header too short: {first_line}")
            continue
        big_theme, chart_title, unit_info = header_parts[:3]

        topic_keywords = chart_title.replace("Share", "").replace("Proportion", "").replace("by", "").replace("(", "").replace(")", "").strip()
        topic_words = topic_keywords.split()
        entity_name = topic_words[0] if topic_words else "category"

        csv_content = ''.join(lines)
        df = pd.read_csv(StringIO(csv_content))

        if 'Category' not in df.columns or 'Proportion' not in df.columns:
            print(f"[ERROR] Missing 'Category' or 'Proportion' columns, skipping.")
            continue

        categories = df['Category'].tolist()
        values = df['Proportion'].tolist()

        if len(categories) < 2:
            print(f"[WARN] Too few categories for QA: {categories}")
            continue

        cat_val_dict = dict(zip(categories, values))
        sorted_items = sorted(cat_val_dict.items(), key=lambda x: x[1], reverse=True)
        max_cat, max_val = sorted_items[0]
        min_cat, min_val = sorted_items[-1]

        # --- VE: randomly pick 2-3
        ve_n = random.randint(2, 3)
        ve_items = random.sample(sorted_items, min(ve_n, len(sorted_items)))
        ve_qas = [
            {
                "Q": f"What is the value of the {cat} slice in this pie chart?",
                "A": f"The value of the {cat} slice is {{{val}}}%."
            }
            for cat, val in ve_items
        ]

        # --- SC: select 2 non-overlapping pairs only if enough categories
        sc_qas = []
        sc_candidates = sorted_items.copy()
        if len(sc_candidates) >= 4:
            pairs = random.sample(sc_candidates, 4)
            (a1, b1, a2, b2) = pairs
            sc_qas.append({
                "Q": f"What is the difference between the values of {a1[0]} and {b1[0]} in this pie chart?",
                "A": f"The difference between {a1[0]} and {b1[0]} is {{{abs(a1[1] - b1[1])}}}%."
            })
            sc_qas.append({
                "Q": f"What is the sum of the values of {a2[0]} and {b2[0]} in this pie chart?",
                "A": f"The sum of {a2[0]} and {b2[0]} is {{{a2[1] + b2[1]}}}%."
            })

        # --- NF: 数据驱动的阈值筛选问题 ---
        import numpy as np

        proportions = np.array(values)
        q1 = np.percentile(proportions, 25)
        q3 = np.percentile(proportions, 75)
        mean_val = np.mean(proportions)

        threshold_1 = round(q3, 2)
        threshold_2 = round(q1, 2)

        filter_above = [{"cat": cat, "val": val} for cat, val in sorted_items if val > threshold_1]
        if not filter_above and threshold_1 != mean_val:
            threshold_1 = round(mean_val, 2)
            filter_above = [{"cat": cat, "val": val} for cat, val in sorted_items if val > threshold_1]

        filter_below = [{"cat": cat, "val": val} for cat, val in sorted_items if val < threshold_2]
        if not filter_below and threshold_2 != mean_val:
            threshold_2 = round(mean_val, 2)
            filter_below = [{"cat": cat, "val": val} for cat, val in sorted_items if val < threshold_2]

        nf_qas = []
        if filter_above:
            nf_qas.append({
                "Q": f"Which categories in the pie chart exceed {threshold_1}%? Please list the category names and corresponding percentages.",
                "A": " and ".join([f"{{{x['cat']}}} accounts for {{{x['val']}}}%" for x in filter_above])
            })

        if filter_below:
            nf_qas.append({
                "Q": f"Which categories in the pie chart are below {threshold_2}%? Please list the category names and corresponding percentages.",
                "A": " and ".join([f"{{{x['cat']}}} accounts for {{{x['val']}}}%" for x in filter_below])
            })

        # --- NC: 2-4 randomly selected comparison questions
        nc_qas = []
        nc_items = sorted_items[:5]
        nc_total = len(nc_items)
        if nc_total >= 2:
            nc_templates = []

            # Binary comparison
            nc_templates.append({
                "Q": f"Which is larger, the value of {nc_items[0][0]} or {nc_items[1][0]}?",
                "A": f"The value of {{{nc_items[0][0]}}} is larger."
            })

            if nc_total >= 3:
                # Three-way max
                nc_templates.append({
                    "Q": f"Which is larger, the value of {nc_items[0][0]}, {nc_items[1][0]}, or {nc_items[2][0]}?",
                    "A": f"The value of {{{nc_items[0][0]}}} is larger."
                })

            if nc_total >= 4:
                # Three-way min
                nc_templates.append({
                    "Q": f"Which is smaller, the value of {nc_items[1][0]}, {nc_items[2][0]}, or {nc_items[3][0]}?",
                    "A": f"The value of {{{nc_items[3][0]}}} is smaller."
                })

            if nc_total >= 5:
                # Four-way max
                nc_templates.append({
                    "Q": f"Which has the largest value: {nc_items[0][0]}, {nc_items[1][0]}, {nc_items[2][0]}, or {nc_items[4][0]}?",
                    "A": f"The value of {{{nc_items[0][0]}}} is the largest."
                })

            # Randomly choose 2 to 4 questions
            nc_count = min(random.randint(2, 4), len(nc_templates))
            nc_qas = random.sample(nc_templates, nc_count)

            # --- SRP: clockwise / counterclockwise directional questions ---
            srp_qas = []
            if len(categories) >= 3:
                idx = random.randint(0, len(categories) - 1)
                label_A = categories[idx]
                clockwise_idx = (idx + 1) % len(categories)
                counter_idx = (idx - 1 + len(categories)) % len(categories)
                label_CW = categories[clockwise_idx]
                label_CCW = categories[counter_idx]

                srp_qas = [
                    {
                        "Q": f"In the pie chart, what is the next slice in the clockwise direction from slice {label_A}?",
                        "A": f"The next slice clockwise from slice {label_A} is {{{label_CCW}}}."
                    },
                    {
                        "Q": f"In the pie chart, what is the next slice in the counterclockwise direction from slice {label_A}?",
                        "A": f"The next slice counterclockwise from slice {label_A} is {{{label_CW}}}."
                    }
                ]
            from itertools import combinations

            from itertools import combinations

            from itertools import combinations

            msr_qas = []

            # ==== 可调参数 ====
            range_lower_bound = random.choice([3, 5, 10])
            range_upper_bound = random.choice([12, 15, 20])
            combo_target_sum = random.choice([40, 45, 50])

            # 范围筛选型问题（如：5%-15%）
            mid_range_items = [cat for cat, val in sorted_items if range_lower_bound <= val <= range_upper_bound]
            if mid_range_items:
                msr_qas.append({
                    "Q": f"How many sectors have a percentage between {range_lower_bound}% and {range_upper_bound}%?",
                    "A": f"There are {{{len(mid_range_items)}}} sectors that have a percentage between {range_lower_bound}% and {range_upper_bound}%."
                })

            # 求和组合问题（所有符合组合）
            valid_combos = []
            triplets = list(combinations(sorted_items, 3))
            for triplet in triplets:
                names, vals = zip(*triplet)
                if sum(vals) < combo_target_sum:
                    valid_combos.append(tuple(names))

            if valid_combos:
                combo_strs = [f"({', '.join(combo)})" for combo in valid_combos]
                msr_qas.append({
                    "Q": f"Which three sectors have a sum of percentage less than {combo_target_sum}%? List all possible combinations.",
                    "A": f"The possible combinations are: {{{', '.join(combo_strs)}}}."
                })

        qa_data = {
            "CTR": [
                {"Q": "What type of chart is this?", "A": "This chart is a {pie chart}."}
            ],
            "VEC": [
                {
                    "Q": "How many slices are there in this pie chart?",
                    "A": f"There are {{{len(df)}}} slices in this pie chart."
                }
            ],
            "SRP":  srp_qas,
            "VPR": [
                {
                    "Q": f"Which {entity_name.lower()} has the largest proportion in this pie chart?",
                    "A": f"The {entity_name.lower()} with the largest proportion is {{{max_cat}}}."
                },
                {
                    "Q": f"Which {entity_name.lower()} has the smallest proportion in this pie chart?",
                    "A": f"The {entity_name.lower()} with the smallest proportion is {{{min_cat}}}."
                }
            ],
            "VE": ve_qas,
            "EVJ": [
                {
                    "Q": "What is the maximum value in this pie chart?",
                    "A": f"The maximum value is {{{max_val}}}%."
                },
                {
                    "Q": "What is the minimum value in this pie chart?",
                    "A": f"The minimum value is {{{min_val}}}%."
                }
            ],
            "SC": sc_qas,
            "NF": nf_qas,
            "NC": nc_qas,
            "MSR": msr_qas,
            "VA": []
        }

        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)

        print(f"[SUCCESS] QA JSON written to: {json_path}")

    except Exception as e:
        print(f"[EXCEPTION] Failed to process {filename}: {e}")
