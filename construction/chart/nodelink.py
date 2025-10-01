import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
from matplotlib.lines import Line2D
import chardet
import numpy as np
import math


def adjust_colinear_nodes(pos, threshold_degrees=10):
    from math import degrees, acos
    from numpy.linalg import norm

    node_list = list(pos.keys())
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            for k in range(len(node_list)):
                if len({i, j, k}) < 3:
                    continue
                a, b, c = node_list[i], node_list[j], node_list[k]
                va = np.array(pos[a]) - np.array(pos[b])
                vc = np.array(pos[c]) - np.array(pos[b])
                if norm(va) == 0 or norm(vc) == 0:
                    continue
                cosine = np.clip(np.dot(va, vc) / (norm(va) * norm(vc)), -1.0, 1.0)
                angle = degrees(acos(cosine))
                if abs(angle) < threshold_degrees or abs(angle - 180) < threshold_degrees:
                    offset = np.random.uniform(-0.05, 0.05, size=2)
                    pos[b] = tuple(np.array(pos[b]) + offset)


def is_label_position_valid(x, y, label_positions, node_positions, label_width=0.15, label_height=0.03, node_radius=0.03):
    # 标签矩形与已有标签矩形不重叠
    for lx, ly in label_positions:
        if abs(x - lx) < label_width and abs(y - ly) < label_height:
            return False

    # 标签矩形与结点圆不重叠
    for nx_, ny_ in node_positions:
        if abs(x - nx_) < label_width and abs(y - ny_) < label_height:
            return False

    return True


def find_label_position(x, y, node_positions, label_positions, radius=0.2):
    # 优先尝试的偏移方向（上→下→右→左→斜对角）
    directions = [
        (0, radius), (-0.05, radius), (0.05, radius),
        (0, -radius), (-0.05, -radius), (0.05, -radius),
        (radius, 0), (-radius, 0),
        (radius, radius), (-radius, radius), (radius, -radius), (-radius, -radius),
    ]

    for dx, dy in directions:
        nx_, ny_ = x + dx, y + dy
        if is_label_position_valid(nx_, ny_, label_positions, node_positions):
            return nx_, ny_

    # fallback 更强避让
    return x, y + 2 * radius


input_dir = './csv/node_link/'
output_dir_png = './png/node_link/'
output_dir_svg = './svg/node_link/'
os.makedirs(output_dir_png, exist_ok=True)
os.makedirs(output_dir_svg, exist_ok=True)

color_palettes = [
    ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b'],
    ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'],
    ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'],
    ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c'],
    ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'],
    ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
]

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_dir, filename)

        with open(filepath, 'rb') as f:
            raw = f.read(1000)
            detected_encoding = chardet.detect(raw)['encoding']

        with open(filepath, 'r', encoding=detected_encoding) as f:
            title_line = f.readline().strip()
        parts = [s.strip() for s in title_line.split(',')]
        sub_theme = parts[1] if len(parts) > 1 else 'Unknown'
        unit_info = parts[2].replace('(unit:', '').replace(')', '') if len(parts) > 2 else 'units'
        final_title = f'{sub_theme}'

        data = pd.read_csv(filepath, skiprows=1, encoding=detected_encoding)
        data.columns = data.columns.str.strip()

        relation_types = sorted(data['Relation Type'].unique())
        palette = random.choice(color_palettes)
        relation_color_map = {rel: palette[i % len(palette)] for i, rel in enumerate(relation_types)}

        G = nx.MultiGraph()
        for _, row in data.iterrows():
            G.add_node(row['Source Node Name'])
            G.add_node(row['Target Node Name'])
            G.add_edge(row['Source Node Name'], row['Target Node Name'], relation=row['Relation Type'])

        plt.figure(figsize=(9, 6), dpi=300)
        pos = nx.spring_layout(G, seed=42, k=None, iterations=500,scale = 2.0)
        adjust_colinear_nodes(pos)

        plt.rcParams.update({
            "font.family": "Times New Roman",
            "text.color": "#000000",
            "axes.edgecolor": "#000000",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF"
        })

        nx.draw_networkx_nodes(G, pos, node_size=150, node_color='#FFFFFF',
                               edgecolors='#000000', linewidths=1)

        pair_count = {}
        for u, v, key, data_edge in G.edges(keys=True, data=True):
            rel = data_edge.get('relation', '')
            color = relation_color_map.get(rel, '#000000')
            pair = tuple(sorted((u, v)))
            count = pair_count.get(pair, 0)
            rad = 0.2 * (-1) ** count * ((count + 1) // 2)
            pair_count[pair] = count + 1
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color=color,
                width=2,
                connectionstyle=f'arc3,rad={rad}'
            )

        texts = []
        label_positions = []
        node_positions = [pos[n] for n in G.nodes]
        for node, (x, y) in pos.items():
            lx, ly = find_label_position(x, y, node_positions, label_positions, radius=0.3)
            label_positions.append((lx, ly))
            texts.append(
                plt.text(lx, ly, str(node),
                         fontsize=11, fontfamily='Times New Roman',
                         color='#000000', ha='center', va='center',
                         zorder=10)
            )
            # plt.plot([x, lx], [y, ly], color='#555555', lw=0.5, zorder=1)

        legend_elements = [
            Line2D([0], [0], color=color, lw=3, label=rel.replace('_', ' ').title())
            for rel, color in relation_color_map.items()
        ]
        plt.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.8),
            frameon=True,
            fontsize=12,
            title='Relation Type',
            title_fontsize=14
        )

        plt.title(final_title, fontsize=16, pad=20, color='#000000', fontname='Times New Roman')
        plt.axis('off')

        base_name = os.path.splitext(filename)[0]
        plt.savefig(os.path.join(output_dir_png, f'{base_name}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir_svg, f'{base_name}.svg'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {base_name}.png and .svg")
