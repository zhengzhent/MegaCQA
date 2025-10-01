import csv
import os
import random
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np

def read_csv_file(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        theme = rows[0][0]
        unit = rows[0][1]
        categories = [row[0] for row in rows[2:]]
        values = [[int(item) for item in row[1:]] for row in rows[2:]]
        dimensions = rows[1][1:]
        return theme,unit,categories, values, dimensions
    

def plot_radar_chart(theme,unit,categories, values,dimensions, png_dir,svg_dir,base_name):
    plt.rcParams['font.family'] = 'Times New Roman'

    num_vars = len(dimensions)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(categories)]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        plot_values = values[i] + values[i][:1]  
        ax.plot(angles, plot_values, color=color, linewidth=2, label=category)
        ax.fill(angles, plot_values, color=color, alpha=0.15)  
    

    ax.set_theta_offset(np.pi / 2)  
    ax.set_theta_direction(-1)      

    ax.set_ylim(0, 100)
    ax.set_rgrids([20,40,60,80], angle=0, color='gray') 

    ax.set_thetagrids(np.degrees(angles[:-1]), dimensions)
    

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        label.set_fontsize(18) 

        if angle in [0, np.pi]:
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    
    plt.title(f"{theme}({unit})", size=20, y=1.1)
    plt.legend(loc='upper right',
                bbox_to_anchor=(1.3, 1.1),
                prop={'size': 16})

    
    png_path = os.path.join(png_dir, f"{os.path.splitext(base_name)[0]}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', transparent=False)

    svg_path = os.path.join(svg_dir, f"{os.path.splitext(base_name)[0]}.svg")
    plt.savefig(svg_path, bbox_inches='tight', format='svg')
    print(f"雷达图{base_name}已保存")
    
    plt.close()

if __name__ == "__main__":

    input_dir = "csv/radar_chart"
    png_dir = "png"
    svg_dir = "svg"
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            data_path = os.path.join(input_dir, filename)
            theme,unit,categories, values, dimensions = read_csv_file(data_path)
            plot_radar_chart(theme,unit,categories,values, dimensions, png_dir,svg_dir,os.path.basename(data_path))
