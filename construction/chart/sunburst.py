import os
import pandas as pd
from collections import defaultdict, deque
import plotly.graph_objects as go
import plotly.io as pio
import random
from tqdm import tqdm
import numpy as np

# ========== 路径配置 ==========
input_folder = './csv/sunburst'
output_folder_png = './png/sunburst'
output_folder_svg = './svg/sunburst'
checkpoint_file = './sunburst_checkpoint.txt'

# ========== 输出目录准备 ==========
os.makedirs(output_folder_png, exist_ok=True)
os.makedirs(output_folder_svg, exist_ok=True)

# ========== 可选颜色方案 ==========
color_scales = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlGnBu', 'YlOrRd']


# ========== ID唯一化修正函数 (绘图前使用) ==========
def make_ids_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    为绘图准备数据：通过附加父节点名使重复ID唯一。
    """
    df = df.copy()
    child_counts = df['child'].value_counts()
    non_unique_children = set(child_counts[child_counts > 1].index)

    if not non_unique_children:
        return df

    id_map = {}
    for _, row in df.iterrows():
        if row['child'] in non_unique_children:
            parent_suffix = row['parent'] if pd.notna(row['parent']) and row['parent'] not in ['', 'Root'] else 'Root'
            unique_id = f"{row['child']} ({parent_suffix})"
        else:
            unique_id = row['child']
        id_map[(row['parent'], row['child'])] = unique_id

    df['id'] = df.apply(lambda r: id_map.get((r['parent'], r['child'])), axis=1)

    parent_id_map = df.set_index('child')['id'].to_dict()
    df['parent_id'] = df['parent'].map(parent_id_map).fillna('')

    df_new = df[['parent_id', 'id', 'value']].copy()
    df_new.columns = ['parent', 'child', 'value']
    return df_new


# ========== [新增/替换] 自顶向下微调函数 ==========
def fine_tune_and_correct_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    采用自顶向下的策略，强制修正数据以满足旭日图要求。
    """
    df_corrected = df.copy()
    value_map = dict(zip(df_corrected['child'], df_corrected['value']))
    parent_map = df_corrected[df_corrected['parent'] != ''].groupby('parent')['child'].apply(list).to_dict()

    # 1. 识别根节点并严格将总和设为100
    all_children_names = set(df_corrected['child'])
    root_nodes = sorted(
        [child for child, parent in df_corrected[['child', 'parent']].values if parent not in all_children_names])

    if not root_nodes: return df  # 如果没有根节点，则无法处理

    root_sum = sum(value_map.get(r, 0) for r in root_nodes)
    if root_sum > 0:
        # 归一化根节点
        scale = 100.0 / root_sum
        temp_sum = 0
        for node in root_nodes[:-1]:
            new_val = round(value_map[node] * scale, 2)
            value_map[node] = new_val
            temp_sum += new_val
        # 最后一个根节点用于凑整，确保总和精确
        value_map[root_nodes[-1]] = round(100.0 - temp_sum, 2)

    # 2. 定义递归函数，自顶向下分配并修正子节点的值
    def distribute_and_fix(parent_name):
        children = parent_map.get(parent_name)
        if not children:
            return

        parent_value = value_map[parent_name]
        current_child_sum = sum(value_map.get(c, 0) for c in children)

        # 如果子节点总和与父节点不匹配，则微调子节点
        if current_child_sum > 0 and not np.isclose(current_child_sum, parent_value):
            child_scale = parent_value / current_child_sum
            child_temp_sum = 0
            sorted_children = sorted(children, key=lambda c: value_map.get(c, 0))  # 按值排序

            for child in sorted_children[:-1]:
                new_val = round(value_map[child] * child_scale, 2)
                value_map[child] = new_val
                child_temp_sum += new_val

            # 最后一个子节点用于凑整
            value_map[sorted_children[-1]] = round(parent_value - child_temp_sum, 2)

        # 对每个子节点递归进行此操作
        for child in children:
            distribute_and_fix(child)

    # 从所有根节点开始执行微调
    for r in root_nodes:
        distribute_and_fix(r)

    df_corrected['value'] = df_corrected['child'].map(value_map)
    return df_corrected


# ========== 断点续跑 (无变化) ==========
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    return set()


def update_checkpoint(filename):
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        f.write(filename + '\n')


# ========== 主程序 ==========
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"❌ 输入目录不存在: {input_folder}")

file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
completed_files = load_checkpoint()

for filename in tqdm(file_list, desc="🌞 生成 Sunburst 图"):
    if filename in completed_files:
        continue

    try:
        filepath = os.path.join(input_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            title_line = f.readline().strip()

        parts = title_line.split(',')
        theme = parts[1].strip() if len(parts) > 1 else os.path.splitext(filename)[0]
        unit = parts[2].strip().replace("(", "").replace(")", "") if len(parts) >= 3 else "Unit"
        chart_title = f"{theme} ({unit})"

        df_original = pd.read_csv(filepath, skiprows=1)
        df = df_original.copy()

        df.columns = df.columns.str.strip()
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        df['parent'] = df['parent'].fillna('')

        if df['value'].sum() < 0.01:
            print(f"⚠️ 文件 {filename} 的数值总和接近于0，已跳过绘图。")
            update_checkpoint(filename)
            continue

        if not {'parent', 'child', 'value'}.issubset(df.columns):
            print(f"⚠️ 文件 {filename} 缺少必要列，已跳过。")
            continue

        # 1. 对原始数据进行微调，生成一份数值上完美的数据
        df_corrected = fine_tune_and_correct_data(df)

        # 2. 检查修正后的数据与原始数据的值是否有差异
        # 使用merge来安全地比较，即使行序不同
        comparison = pd.merge(df_original[['child', 'value']], df_corrected[['child', 'value']], on='child',
                              suffixes=('_orig', '_corr'))
        is_different = not np.allclose(comparison['value_orig'], comparison['value_corr'])

        # 3. 如果数据被微调过，则写回源文件
        if is_different:
            print(f"✍️ 文件 {filename} 的数据不一致，正在微调并写回源文件...")
            df_to_write = df_corrected[['parent', 'child', 'value']]
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(title_line + '\n')
                df_to_write.to_csv(f, index=False)

        # 4. 为绘图准备最终数据（ID唯一化）
        df_for_plot = make_ids_unique(df_corrected)
        df_for_plot['parent'] = df_for_plot['parent'].replace('Root', '')
        df_for_plot = df_for_plot[df_for_plot['child'] != 'Root'].copy()

        df = df_for_plot

        # 为旭日图中心添加一个“核”
        core_value = round(df[df['parent'] == ''].value.sum(), 2)
        if core_value > 0:
            df = pd.concat([
                pd.DataFrame([["", "Core", core_value]], columns=["parent", "child", "value"]),
                df.replace({'parent': {'': 'Core'}})
            ], ignore_index=True)

        # 准备绘图标签和颜色
        labels = []
        node_colors = []
        numeric_colors = []

        for row in df.itertuples():
            if row.child == "Core":
                labels.append("")
                node_colors.append("white")
            else:
                labels.append(f"{row.child}")
                node_colors.append(row.value)
                if row.value > 0:
                    numeric_colors.append(row.value)

        base_name = os.path.splitext(filename)[0]
        chosen_scale = random.choice(color_scales)

        cmin_val = min(numeric_colors) if numeric_colors else 0
        cmax_val = max(numeric_colors) if numeric_colors else 1

        # 创建图表
        fig = go.Figure(go.Sunburst(
            ids=df['child'],
            labels=labels,
            parents=df['parent'],
            values=df['value'],
            branchvalues="total",
            textinfo="label+value",
            textfont=dict(family='Times New Roman', size=12, color='black'),
            marker=dict(
                colors=node_colors,
                colorscale=chosen_scale,
                line=dict(color='#000000', width=1),
                cmin=cmin_val,
                cmax=cmax_val,
                colorbar=dict(
                    title=dict(text=unit, font=dict(family='Times New Roman', size=13)),
                    x=1.05,
                    y=0.5,
                    len=0.7,
                    thickness=15,
                    tickfont=dict(family='Times New Roman', size=12)
                )
            )
        ))

        fig.update_layout(
            title=dict(text=chart_title, x=0.5, xanchor='center', font=dict(family='Times New Roman', size=16)),
            margin=dict(t=50, l=10, r=80, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        fig.write_image(os.path.join(output_folder_png, f"{base_name}.png"), width=640, height=480, scale=3)
        pio.write_image(fig, os.path.join(output_folder_svg, f"{base_name}.svg"), width=640, height=480, scale=3)

        update_checkpoint(filename)

    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")