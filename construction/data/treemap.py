import random
import pandas as pd
import os
import csv
from collections import defaultdict
import numpy as np

random.seed(42)
theme_name = "Cultural Activities"

# 约束参数
MIN_RATIO = 0.05
MAX_CHILDREN = 20
MIN_LEVEL = 3
MAX_LEVEL = 7

# 新增全局值追踪系统
GLOBAL_VALUES = defaultdict(float)
GLOBAL_VALUES['Root'] = 100.0

# 动态结构模板
tree_structure = {
    "children": {
        "Root": {  # 根节点（第1层）
            "children": {
                # ------------------------- 文化活动分支 ------------------------#
                "Cultural Activities": {  # 文化活动（第2层）
                    "children": {
                        "Art Exhibitions": {  # 艺术展览（第3层）
                            "children": {
                                "Gallery Shows": {  # 画廊展览（第4层）
                                    "children": {
                                        "Group Exhibitions": {  # 群展（第5层）
                                            "children": {
                                                "Art Collectives": {  # 艺术团体（第6层）
                                                    "children": {
                                                        "Collective A": {"children": {}},  # 团体A（第7层）
                                                        "Collective B": {"children": {}}  # 团体B（第7层）
                                                    }
                                                },
                                                "Collaborative Projects": {  # 合作项目（第6层）
                                                    "children": {
                                                        "Cross-Disciplinary": {"children": {}},  # 跨学科（第7层）
                                                        "Community Art": {"children": {}}  # 社区艺术（第7层）
                                                    }
                                                }
                                            }
                                        },
                                        "Solo Exhibitions": {  # 个展（第5层）
                                            "children": {
                                                "Modern Art": {"children": {}},  # 现代艺术（第6层）
                                                "Classical Art": {"children": {}}  # 古典艺术（第6层）
                                            }
                                        }
                                    }
                                },
                                "Online Exhibitions": {  # 线上展览（第4层）
                                    "children": {
                                        "Virtual Reality": {"children": {}},  # 虚拟现实（第5层）
                                        "NFT Gallery": {"children": {}}  # NFT画廊（第5层）
                                    }
                                }
                            }
                        },
                        "Public Art": {  # 公共艺术（第3层）
                            "children": {
                                "Urban Sculpture": {"children": {}},  # 城市雕塑（第4层）
                                "Community Projects": {"children": {}}  # 社区项目（第4层）
                            }
                        }
                    }
                },

                # ------------------------- 艺术品交易分支 ------------------------#
                "Artwork Sales": {  # 艺术品交易（第2层）
                    "children": {
                        "Primary Market": {  # 一级市场（第3层）
                            "children": {
                                "Galleries": {  # 画廊（第4层）
                                    "children": {
                                        "Contemporary": {"children": {}},  # 当代艺术（第5层）
                                        "Traditional": {"children": {}}  # 传统艺术（第5层）
                                    }
                                },
                                "Art Fairs": {  # 艺博会（第4层）
                                    "children": {
                                        "International": {"children": {}},  # 国际展会（第5层）
                                        "Regional": {"children": {}}  # 地区展会（第5层）
                                    }
                                }
                            }
                        },
                        "Secondary Market": {  # 二级市场（第3层）
                            "children": {
                                "Auction Houses": {  # 拍卖行（第4层）
                                    "children": {
                                        "Modern Masters": {"children": {}},  # 现代大师（第5层）
                                        "Emerging Artists": {"children": {}}  # 新兴艺术家（第5层）
                                    }
                                },
                                "Art Funds": {  # 艺术基金（第4层）
                                    "children": {
                                        "Investment Grade": {"children": {}},  # 投资级（第5层）
                                        "Experimental": {"children": {}}  # 实验性（第5层）
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
GLOBAL_VALUE_RECORD = {}


def random_proportions(n, parent_global):
    """ 生成符合全局约束的本地比例 """
    min_local = max(0.05, 5.0 / parent_global)  # 关键修正点：动态计算最小本地比例

    # 可行性验证
    if n * min_local > 1.0:
        raise ValueError(f"父节点{parent_global}%无法生成{n}个子节点")

    # # 智能分配算法
    # remaining = 1.0 - n * min_local
    # base = [random.uniform(0, remaining) for _ in range(n)]
    # total = sum(base)
    # return [min_local + (p / total) * remaining for p in base]

    max_retries = 10  # 最大重试次数
    for _ in range(max_retries):
        # 生成基础比例
        remaining = 1.0 - n * min_local
        base = [random.expovariate(1.0) for _ in range(n)]  # 改用指数分布增加差异性
        total = sum(base)
        ratios = [min_local + (p / total) * remaining for p in base]

        # 检查1：禁止完美均分模式（误差<0.1%）
        expected_ratio = 1.0 / n
        if all(abs(r - expected_ratio) < 0.001 for r in ratios):
            if parent_global * expected_ratio >= 5.0:
                continue  # 重新生成

        # 检查2：禁止超过3个相同比例（波动<0.5%）
        rounded_ratios = [round(r, 3) for r in ratios]
        ratio_counts = defaultdict(int)
        for r in rounded_ratios:
            ratio_counts[r] += 1
        if max(ratio_counts.values()) > 3:
            continue  # 重新生成

        return ratios

    # 最终保底方案：使用Dirichlet分布生成
    alpha = [0.5] * n  # 更陡峭的分布参数
    dirichlet_ratios = list(np.random.dirichlet(alpha))
    return [min_local + r * (1 - n * min_local) for r in dirichlet_ratios]


def generate_proportional_data(node, parent_name="Root", current_level=1):
    """ 生成带全局验证的数据 """
    data = []

    # 添加深度过滤
    # 修正2：调整层级过滤条件
    # 修正层级过滤条件
    if current_level > MAX_LEVEL:
        return []

    if isinstance(node, dict) and "children" in node:
        children = list(node["children"].items())[:MAX_CHILDREN]

        # 新增：空子节点检查
        if not children:
            return data
        # 获取父节点全局值
        parent_global = GLOBAL_VALUES[parent_name]

        try:
            ratios = random_proportions(len(children), parent_global)
        except ValueError:
            return data  # 终止无法生成的分支

        for (child_name, child_node), ratio in zip(children, ratios):
            # 计算全局值
            child_global = parent_global * ratio
            GLOBAL_VALUES[child_name] = child_global

            # 验证全局约束
            if child_global < 5.0:
                continue

            # 记录数据（保留原始结构）
            data.append([parent_name, child_name, round(child_global, 2)])

            # 强制展开逻辑
            if current_level < MIN_LEVEL or ratio >= 0.1:
                data += generate_proportional_data(
                    child_node, child_name, current_level + 1
                )

    return data

# 保存路径
output_folder = './csv/treemap'
os.makedirs(output_folder, exist_ok=True)

# 自动生成文件名
def get_next_filename():
    existing_files = [f for f in os.listdir(output_folder) if f.startswith("treemap_")]
    if existing_files:
        max_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
        return f'treemap_{max_num + 1}.csv'
    else:
        return 'treemap_1.csv'

# 保存数据
# 修改后的保存函数
def save_tree_data_to_multiple_csv(num_files):
    for _ in range(num_files):
        GLOBAL_VALUES.clear()
        GLOBAL_VALUES['Root'] = 100.0

        tree_data = generate_proportional_data(tree_structure['children']['Root'])
        df = pd.DataFrame(tree_data, columns=["parent", "child", "value"])

        # 最终全局验证
        invalid = [k for k, v in GLOBAL_VALUES.items() if v < 5 and k != 'Root']
        if invalid:
            print(f"⚠️ 存在违规节点: {invalid}")
            continue

        csv_file = os.path.join(output_folder, get_next_filename())
        title = f'{theme_name} Data(Units:Percentage)'

        with open(csv_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([title])
            writer.writerow(["parent", "child", "value"])
            for row in df.itertuples(index=False, name=None):
                writer.writerow(row)

        print(f"✅ 数据已保存至 {csv_file}")
        print("\n📄 生成的数据样例：")
        print(df.head(10))

# 示例生成 5 份
save_tree_data_to_multiple_csv(1)