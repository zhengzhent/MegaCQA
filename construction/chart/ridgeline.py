import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import numpy as np
def process_and_plot(data_path, png_dir, svg_dir):
    print(f"\n正在处理文件 {os.path.basename(data_path)}...")
    print("="*50)
    
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    print(f"总行数: {len(lines)}")
    
    header_parts = [p.strip() for p in lines[0].split(',') if p.strip()]
    
    title, sub_title, unit = header_parts[:3]
    
    category_parts = [p.strip() for p in lines[1].split(',')]
    category_names = category_parts[0:]  
    
    data_rows = []
    for i, line in enumerate(lines[3:], start=3):
        parts = [p.strip() for p in line.split(',')]
        values = parts[0:]  
        
        float_values = [float(x.replace(',', '')) if ',' in x else float(x) for x in values]
        data_rows.append(float_values)

    print(f"共找到 {len(data_rows)} 行有效数据")
    
    df = pd.DataFrame(data_rows, columns=category_names)

    plot_title = f"{sub_title}({unit})"
    plot_ridgeline(df, category_names, plot_title, png_dir,svg_dir, os.path.basename(data_path))
        

def plot_ridgeline(data, category_names, title, png_dir, svg_dir, base_name):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    
    color_palettes = [
        "husl", "Set2", "Dark2", "Paired", 
        "tab10", "viridis", "plasma", "coolwarm", "Spectral"
    ]
    selected_palette = random.choice(color_palettes)
    pal = sns.color_palette(selected_palette, len(category_names))
    
    df_melt = data.melt(var_name='Category', value_name='Value')
    

    g = sns.FacetGrid(df_melt, row='Category', hue='Category', 
                     aspect=10, height=1, palette=pal)
    
    g.map(sns.kdeplot, 'Value', 
          bw_adjust=0.7,  
          clip=(data.min().min(), data.max().max()),  
          fill=True, alpha=0.5, linewidth=1)

    g.map(sns.kdeplot, 'Value', 
          bw_adjust=0.7,  
          clip=(data.min().min(), data.max().max()),
          color="w", lw=2)
    
    g.refline(y=0, linewidth=1, linestyle="-", color="#000000", clip_on=False)
    
    def label(x, color, label):
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ax.text(xmin - 0.05 * (xmax - xmin), 0, label,  
               fontsize=12, color='#000000', 
               ha='right', va='center', 
               bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    g.map(label, "Value")
    
    # 透明背景处理
    for ax in g.axes.flat:
        ax.set_facecolor('none')  
        ax.patch.set_alpha(0)    
        ax.set_xlim(data.min().min(), data.max().max())  # 强制统一x轴范围

    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("Value", fontsize=14, color='#000000')  
    
    # 刻度样式
    for ax in g.axes.flat:
        ax.tick_params(axis='x', which='both', labelsize=12, colors='#000000')  

    g.despine(bottom=True, left=True)
    plt.suptitle(title, y=0.95, fontsize=16, color='#000000')
    
    # 输出文件
    img_path = os.path.join(png_dir, f"{os.path.splitext(base_name)[0]}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight', transparent=False)
    
    svg_path = os.path.join(svg_dir, f"{os.path.splitext(base_name)[0]}.svg")
    plt.savefig(svg_path, dpi=300, bbox_inches='tight', transparent=False)
    
    print(f" {img_path} 已完成")
    print(f" {svg_path} 已完成")
    plt.close()


if __name__ == "__main__":

    input_dir = "csv/ridgeline_chart"
    png_dir = "png"
    svg_dir = "svg"

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            data_path = os.path.join(input_dir, filename)
            process_and_plot(data_path, png_dir, svg_dir)
    
    print("所有图表已完成")