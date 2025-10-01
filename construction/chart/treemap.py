import os
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import random
import numpy as np

input_folder = './csv/treemap'
output_folder_png = './png/treemap'
output_folder_svg = './svg/treemap'

os.makedirs(output_folder_png, exist_ok=True)
os.makedirs(output_folder_svg, exist_ok=True)

color_scales = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlGnBu', 'YlOrRd']

def build_levels(df, root='Root'):
    child_to_parent = dict(zip(df['child'], df['parent']))
    levels = [[root]]
    visited = set([root])
    while True:
        last_level = levels[-1]
        next_level = [child for child, parent in child_to_parent.items()
                      if parent in last_level and child not in visited]
        if not next_level:
            break
        levels.append(next_level)
        visited.update(next_level)
    return levels


def round_children_to_parent(df):
    df = df.copy()

    # 将 'Root' 标准化为空父节点 ''
    df.loc[df['parent'] == 'Root', 'parent'] = ''

    # 构建 parent -> children 和 child -> value 映射
    parent_map = defaultdict(list)
    for _, row in df.iterrows():
        parent_map[row['parent']].append(row['child'])
    value_map = dict(zip(df['child'], df['value']))

    # 递归函数：将 parent 的所有子节点值调整为 parent 的值，并继续向下调整
    def adjust_node(node):
        children = parent_map.get(node, [])
        if not children:
            return

        parent_value = value_map[node]
        current_children_values = [value_map.get(c, 0.0) for c in children]
        current_sum = sum(current_children_values)

        if round(current_sum, 2) != round(parent_value, 2):
            if current_sum == 0:
                avg = round(parent_value / len(children), 2)
                for c in children[:-1]:
                    df.loc[df['child'] == c, 'value'] = avg
                    value_map[c] = avg
                last = round(parent_value - avg * (len(children) - 1), 2)
                df.loc[df['child'] == children[-1], 'value'] = last
                value_map[children[-1]] = last
            else:
                scaled = [v / current_sum * parent_value for v in current_children_values]
                rounded = [round(v, 2) for v in scaled[:-1]]
                last = round(parent_value - sum(rounded), 2)
                for c, v in zip(children[:-1], rounded):
                    df.loc[df['child'] == c, 'value'] = v
                    value_map[c] = v
                df.loc[df['child'] == children[-1], 'value'] = last
                value_map[children[-1]] = last

        for c in children:
            adjust_node(c)

    # 找出所有非叶子节点进行修正（自上而下）
    non_leaf_nodes = [n for n in set(df['parent']) if n in value_map]
    for node in non_leaf_nodes:
        adjust_node(node)

    # 强制 Root 为 100.00，最后一次全树调整
    if 'Root' in value_map:
        df.loc[df['child'] == 'Root', 'value'] = 100.00
        value_map['Root'] = 100.00
        adjust_node('Root')

        # ⚠️ 确保最终总和为 100.00（浮点误差纠正）
        root_children = parent_map['']
        root_values = [value_map[c] for c in root_children]
        root_rounded = [round(v, 2) for v in root_values[:-1]]
        root_last = round(100.00 - sum(root_rounded), 2)
        for c, v in zip(root_children[:-1], root_rounded):
            df.loc[df['child'] == c, 'value'] = v
            value_map[c] = v
        df.loc[df['child'] == root_children[-1], 'value'] = root_last
        value_map[root_children[-1]] = root_last

    return df







def compute_left_ratios(df):
    parent_children = defaultdict(list)
    value_map = dict(zip(df['child'], df['value']))
    for _, row in df.iterrows():
        parent_children[row['parent']].append(row['child'])

    left_ratio = {}
    for children in parent_children.values():
        total = sum(value_map.get(child, 0) for child in children)
        acc = 0
        for child in children:
            left_ratio[child] = acc / total if total > 0 else 0
            acc += value_map.get(child, 0)
    return left_ratio

def enforce_explicit_root(df, root='Root', total=100.0):
    # 如果缺少 root 的显式顶层父节点记录，则补上
    if not ((df['child'] == root) & ((df['parent'] == '') | (df['parent'].isna()))).any():
        df = pd.concat([
            pd.DataFrame([{'parent': '', 'child': root, 'value': total}]),
            df
        ], ignore_index=True)
    return df


def smart_estimated_label(name, val, parent_val, left_ratio, has_children):
    val_str = f"{val:.2f}"
    score = (val / parent_val) * (1 - left_ratio) if parent_val > 0 else 0
    if has_children:
        return f"{name} {val_str}"
    elif score > 0.12:
        return f"{name}<br>{val_str}" if len(name) > 6 else f"{name} {val_str}"
    else:
        lines = [name[i:i+6] for i in range(0, len(name), 6)][:2]
        return "<br>".join(lines) + f"<br>{val_str}"

file_list = os.listdir(input_folder)
print(f"\U0001F4C1 发现文件: {file_list}")

for filename in file_list:
    if not filename.endswith('.csv'):
        continue
    try:
        filepath = os.path.join(input_folder, filename)
        print(f"\n📄 正在处理文件: {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            title_line = f.readline().strip()
        parts = title_line.split(',')
        theme = parts[1].strip()
        unit = parts[2].strip().replace("(", "").replace(")", "") if len(parts) >= 3 else "Unit"
        chart_title = f"{theme}({unit})"
        print(f"📌 标题: {chart_title}")

        df = pd.read_csv(filepath, skiprows=1)
        df.columns = df.columns.str.strip()
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)

        if not {'parent', 'child', 'value'}.issubset(df.columns):
            print(f"⚠️ 缺少必要列 in {filename}")
            continue

        original_df = pd.read_csv(filepath, skiprows=1)
        original_df.columns = original_df.columns.str.strip()
        original_df['value'] = pd.to_numeric(original_df['value'], errors='coerce')
        original_df.dropna(subset=['value'], inplace=True)

        df = round_children_to_parent(df)
        # 只保留原始结点子集进行比对
        original_nodes = set(original_df['child'])
        df_subset = df[df['child'].isin(original_nodes)].copy()
        df_subset_sorted = df_subset.sort_values(by=['parent', 'child']).reset_index(drop=True)
        original_sorted = original_df.sort_values(by=['parent', 'child']).reset_index(drop=True)

        # 判断是否有数值或格式差异（保留两位小数）
        has_diff = not df_subset_sorted[['parent', 'child', 'value']].round(2).equals(
            original_sorted[['parent', 'child', 'value']].round(2)
        )

        if has_diff:
            # 写回：保留原始第一行 + 原始结构
            print(f"📝 检测到修改，正在写回 {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            corrected = df[df['child'].isin(original_nodes)][['parent', 'child', 'value']]
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(first_line)
                corrected.to_csv(f, index=False)
        else:
            print(f"📂 数据未变化，跳过写回 {filename}")

        left_ratio = compute_left_ratios(df)

        base_name = os.path.splitext(filename)[0]
        chosen_scale = random.choice(color_scales)

        parents = set(df['parent'])
        parent_val_map = dict(zip(df['child'], df['value']))
        labels = []
        for row in df.itertuples():
            has_kids = row.child in parents
            parent_val = parent_val_map.get(row.parent, 100.0)
            ratio = left_ratio.get(row.child, 0)
            label = smart_estimated_label(row.child, row.value, parent_val, ratio, has_kids)
            labels.append(label)

        fig = go.Figure(go.Treemap(
            ids=df['child'],
            labels=labels,
            parents=df['parent'],
            values=df['value'],
            branchvalues="total",
            marker=dict(colors=df['value'], colorscale=chosen_scale, line=dict(color='#000000', width=1)),
            textfont=dict(family='Times New Roman', size=12, color='black')
        ))

        fig.update_layout(
            title=dict(text=chart_title, x=0.5, xanchor='center', font=dict(family='Times New Roman', size=16)),
            margin=dict(t=50, l=10, r=10, b=10),
            paper_bgcolor='white', plot_bgcolor='white'
        )

        if len(df) > 0:
            fig.write_image(os.path.join(output_folder_png, f"{base_name}.png"), width=640, height=480, scale=3)
            fig.write_image(os.path.join(output_folder_svg, f"{base_name}.svg"), width=640, height=480, scale=3)
            print(f"✅ 使用配色 {chosen_scale} 导出: {base_name}.png 和 {base_name}.svg")
        else:
            print(f"⚠️ 图表 {base_name} 无数据，跳过导出。")

    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")
