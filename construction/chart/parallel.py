import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import re

def split_dimension_unit(raw_dim):
    # 修正后的正则表达式：匹配 "Text (unit)"
    match = re.match(r'^(.+?)\s*\((.+?)\)\s*$', raw_dim.strip())
    if match:
        return match.group(1).strip(), f"({match.group(2).strip()})"
    return raw_dim.strip(), None

def read_csv_data(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        
        theme = rows[0][0] if len(rows) > 0 else None
        sub_theme = rows[0][1] if len(rows) > 0 and len(rows[0]) > 1 else None
        
        categories = [row[0] for row in rows[3:]] if len(rows) > 2 else []
        
        data = [row[1:] for row in rows[4:]] if len(rows) > 2 else []
        
        dim_unit = rows[1][1:] if len(rows) > 1 else []
        dimensions, units = zip(*[split_dimension_unit(dim) for dim in dim_unit]) if dim_unit else ([], [])
        return theme, sub_theme, categories, data, dimensions,units
    
def draw(data_path, png_dir, svg_dir):

    theme,sub_theme,categories,data,dimensions,unit=read_csv_data(data_path)
    color_library = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194'
    ]
    categories=np.array(categories)
    data=np.array(data).astype(float)
    dimensions=np.array(dimensions)
    unit=np.array(unit)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    for dim in range(len(dimensions)):
        ax.axvline(x=dim, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        
        ax.text(
            x=dim, y=1.05, 
            s=f"{data[:, dim].max():.2f}{unit[dim]}", 
            ha='center', va='bottom', 
            fontsize=12, color='black'
        )
        
        ax.text(
            x=dim, y=-0.08, 
            s=f"{data[:, dim].min():.2f}", 
            ha='center', va='top', 
            fontsize=12, color='black'
        )
    num_lines = len(data)
    selected_colors = np.random.choice(color_library, size=num_lines, replace=False)

    for i in range(len(data)):
        ax.plot(range(len(dimensions)), normalized[i], 
                color=selected_colors[i],
                label=categories[i], alpha=0.7, linewidth=2)

    ax.set_xticks(range(len(dimensions)))

    ax.set_xticklabels(dimensions, rotation=0)  
    ax.set_ylim(-0.05, 1.05) 
    ax.tick_params(axis='x', which='major', pad=20)

    ax.yaxis.set_ticks([]) 
    ax.yaxis.set_ticklabels([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.title(f"{sub_theme}",pad=20)
    plt.tight_layout()

    base_name=os.path.basename(data_path)

    img_path = os.path.join(png_dir, f"{os.path.splitext(base_name)[0]}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight', transparent=False)

    svg_path = os.path.join(svg_dir, f"{os.path.splitext(base_name)[0]}.svg")
    plt.savefig(svg_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f" {base_name} 已完成")
    plt.close()

if __name__ == "__main__":

    input_dir = "csv/parallel_coordinates"
    png_dir = "png"
    svg_dir = "svg"

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            data_path = os.path.join(input_dir, filename)
            draw(data_path, png_dir, svg_dir)
