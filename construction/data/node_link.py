import random
import pandas as pd
import os
import re  # 导入正则表达式模块

# 配置字典，集中管理所有与主题相关的参数
config = {
    'graph_title': '{theme} Data(Units:{unit})',  # CSV文件标题模板
    'relation_types': ['truck_flow', 'train_flow', 'cargo_flow', 'passenger_flow'],  # 运输流量类型
    'flow_range': (50, 1000),  # 流量值范围（单位：吨/小时）
    'max_connections': 5,  # 每个节点的最大连接数
    'node_prefix': 'Hub_',  # 节点前缀
    'unit': 'tons/hour',  # 单位（例如：吨/小时）
    'theme': 'TransportFlow',  # 主题
    # 更换主题时以下不变化
    'output_dir': './csv/node_link/',  # 输出目录
    'num_files': 1,  # 生成的文件数
}

# 创建文件保存目录
os.makedirs(config['output_dir'], exist_ok=True)


# 随机生成节点
def generate_nodes(num_nodes):
    nodes = []
    for i in range(num_nodes):
        node = {
            'index': i,
            'name': f'{config["node_prefix"]}{i}'  # 节点名称
        }
        nodes.append(node)
    return nodes


# 随机生成节点之间的流量和连接关系
def generate_edges(nodes):
    edges = []

    for node in nodes:
        node_id = node['index']

        # 计算当前节点剩余可连接的节点数量
        available_nodes = [n['index'] for n in nodes if n['index'] != node_id]

        # 确保每个节点的连接数在1到5之间
        num_connections = random.randint(1, min(config['max_connections'], len(available_nodes)))

        connected_nodes = random.sample(available_nodes, num_connections)

        for target_node in connected_nodes:
            # 生成随机类型（如卡车流量、列车流量等）
            relation = random.choice(config['relation_types'])
            # 随机生成值（单位：吨/小时、辆/小时等）
            value = random.uniform(*config['flow_range'])  # 范围（50-1000吨/小时）
            edge = {
                'source': node_id,
                'target': target_node,
                'relation': relation,
                'value': round(value, 2)
            }
            edges.append(edge)

    return edges


# 获取现有的文件编号，以便继续生成新文件
def get_last_file_number(filename_template):
    existing_files = [f for f in os.listdir(config['output_dir']) if f.startswith(filename_template)]

    # 输出调试信息：查看现有的文件
    # print(f"Existing files: {existing_files}")

    if existing_files:
        # 使用正则表达式提取文件名中的数字部分
        numbers = []
        for f in existing_files:
            match = re.search(r'(\d+)', f.split('_')[-1])  # 匹配文件名中的数字部分
            if match:
                numbers.append(int(match.group(1)))

        # 输出调试信息：提取的文件编号
        # print(f"Extracted numbers: {numbers}")

        return max(numbers) if numbers else 0

    return 0


# 生成图数据并保存为CSV文件
def generate_and_save_data(num_nodes, num_files=1, filename_template='node_link_chart'):
    # 获取当前文件序号
    last_file_number = get_last_file_number(filename_template)

    # 输出调试信息：当前文件序号
    # print(f"Last file number: {last_file_number}")

    # 生成指定数量的文件
    for i in range(last_file_number + 1, last_file_number + num_files + 1):
        # 生成节点
        nodes = generate_nodes(num_nodes)
        # 生成边（连接）
        edges = generate_edges(nodes)

        current_filename = f"{filename_template}_{i}"

        # 输出调试信息：正在生成的文件名
        print(
            f"Generating files: {current_filename}.csv")

        # 合并节点和边数据并保存
        data = []
        for edge in edges:
            source_node = next(node for node in nodes if node['index'] == edge['source'])
            target_node = next(node for node in nodes if node['index'] == edge['target'])

            data.append({
                'source_index': edge['source'],
                'source_name': source_node['name'],
                'target_index': edge['target'],
                'target_name': target_node['name'],
                'relation': edge['relation'],
                'value': edge['value']
            })

        # 保存合并数据为CSV文件，并添加标题和属性名称
        combined_df = pd.DataFrame(data)
        combined_filename = os.path.join(config['output_dir'], f'{current_filename}.csv')
        with open(combined_filename, 'w', newline='') as f:
            f.write(config['graph_title'].format(unit=config['unit'], theme=config['theme']) + '\n')  # 标题行
            combined_df.to_csv(f, index=False,
                               header=["Source Node Index", "Source Node Name", "Target Node Index", "Target Node Name",
                                       "Relation Type", "Flow Value"])  # 属性名称行

        # print(f"Transport flow data saved to {config['output_dir']}")


# 主程序执行
if __name__ == "__main__":
    num_nodes = random.randint(5, 15)  # 随机生成节点数
    generate_and_save_data(num_nodes, num_files=config['num_files'])
