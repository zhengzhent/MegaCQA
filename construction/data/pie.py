import os
import random
import pandas as pd

# 主题配置部分
theme_config = {
    "theme_name": "Social Media Market Share",  # 主题名称
    "category": ["Facebook", "Instagram", "Twitter", "TikTok", "Snapchat", "LinkedIn", "YouTube"],  # 社交媒体平台列表
    "output_dir": './csv/pie/',  # 文件保存路径
    "file_prefix": "pie_chart_",  # 文件名前缀
    "min_category": 2,  # 最少选择的分类数
    "max_category": 7  # 最多选择的分类数
}

# 确保文件夹存在
os.makedirs(theme_config["output_dir"], exist_ok=True)

# 获取当前目录下已存在的文件索引，返回最大索引+1
def get_next_file_index():
    existing_files = os.listdir(theme_config["output_dir"])
    max_index = 0  # 默认最大索引为0
    for file in existing_files:
        if file.endswith('.csv'):  # 只处理CSV文件
            try:
                # 获取文件名中的数字部分（例如 pie_chart_1.csv 中的 1）
                index = int(file.split('_')[-1].split('.')[0])
                max_index = max(max_index, index)  # 更新最大索引
            except ValueError:
                continue
    return max_index + 1  # 返回下一个可用的文件索引


# 生成分类的数据
def generate_data(num_files):
    next_index = get_next_file_index()  # 获取下一个文件索引
    for j in range(next_index, next_index + num_files):
        # 随机选择分类数
        num_category = random.choice(range(theme_config["min_category"], theme_config["max_category"] + 1))
        selected_category = random.sample(theme_config["category"], num_category)

        # 生成数据，确保比例总和为100%
        proportions = []
        while len(proportions) < num_category:
            # 随机生成比例，确保每个比例大于10%，避免等分
            proportion = random.randint(10, 30)
            proportions.append(proportion)

        # 生成后进行调整，确保总和为100%
        total = sum(proportions)
        proportions = [int((p / total) * 100) for p in proportions]

        # 确保每个比例不等于其他比例，以避免等分的情况
        while len(set(proportions)) == 1:
            proportions = []
            while len(proportions) < num_category:
                proportion = random.randint(10, 30)
                proportions.append(proportion)

            # 重新调整比例总和为100%
            total = sum(proportions)
            proportions = [int((p / total) * 100) for p in proportions]

        # 创建DataFrame
        data = {"category": selected_category, "proportions (%)": proportions}
        df = pd.DataFrame(data)

        # 生成标题行
        title = f'{theme_config["theme_name"]} Data(Units:Percentage)'

        # 文件路径
        file_path = f"{theme_config['output_dir']}{theme_config['file_prefix']}{j}.csv"

        # 先写入一行标题
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")  # 注意不加逗号

        # 然后将数据追加到文件中
        df.to_csv(file_path, index=False, header=False, mode='a')
        print(f"生成文件: {file_path}")


# 示例用法：
# 生成10个CSV文件
generate_data(1)

