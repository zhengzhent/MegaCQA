import numpy as np
import pandas as pd
import os
import random
from typing import List, Tuple

# =============================================
# 主题和对应的词汇表
# =============================================
topics_vocabs = {
    'Education and Academics': {
        'x_vocabs': ['Physics', 'Chemistry', 'Biology', 'Mathematics',
                     'History', 'Literature', 'Geography'],
        'y_vocabs': ['Freshman', 'Sophomore', 'Junior', 'Senior',
                     'Graduate', 'PhD', 'Postdoc']
    },
    'Transportation and Logistics': {
        'x_vocabs': ['Air', 'Rail', 'Road', 'Maritime', 'Pipeline', 'Intermodal'],
        'y_vocabs': ['Cost', 'Speed', 'Reliability', 'Capacity', 'Safety', 'Sustainability']
    },
    'Tourism and Hospitality': {
        'x_vocabs': ['Hotels', 'Restaurants', 'Attractions', 'Transport', 'Tour Operators', 'Cruises'],
        'y_vocabs': ['Luxury', 'Budget', 'Family', 'Business', 'Adventure', 'Cultural']
    },
    'Business and Finance': {
        'x_vocabs': ['Banking', 'Investments', 'Insurance', 'Accounting', 'Tax', 'Real Estate'],
        'y_vocabs': ['Revenue', 'Profit', 'Growth', 'Risk', 'Liquidity', 'Market Share']
    },
    'Real Estate and Housing Market': {
        'x_vocabs': ['Residential', 'Commercial', 'Industrial', 'Land', 'Rental', 'REITs'],
        'y_vocabs': ['Price', 'Demand', 'Supply', 'Inventory', 'Affordability', 'Interest Rates']
    },
    'Healthcare and Health': {
        'x_vocabs': ['Hospitals', 'Clinics', 'Pharmaceuticals', 'Insurance', 'Wellness', 'Telemedicine'],
        'y_vocabs': ['Cost', 'Quality', 'Access', 'Prevention', 'Outcomes', 'Patient Satisfaction']
    },
    'Retail and E-commerce': {
        'x_vocabs': ['Apparel', 'Electronics', 'Groceries', 'Home Goods', 'Beauty', 'Sports'],
        'y_vocabs': ['Online', 'Brick&Mortar', 'Omnichannel', 'Subscription', 'Marketplace', 'DTC']
    },
    'Human Resources and Employee Management': {
        'x_vocabs': ['Recruitment', 'Training', 'Compensation', 'Benefits', 'Performance', 'Retention'],
        'y_vocabs': ['Satisfaction', 'Productivity', 'Turnover', 'Engagement', 'Diversity', 'Wellbeing']
    },
    'Sports and Entertainment': {
        'x_vocabs': ['Football', 'Basketball', 'Tennis', 'Movies', 'Music', 'Gaming'],
        'y_vocabs': ['Viewership', 'Revenue', 'Attendance', 'Merchandise', 'Sponsorship', 'Social Media']
    },
    'Food and Beverage Industry': {
        'x_vocabs': ['Dairy', 'Meat', 'Bakery', 'Beverages', 'Snacks', 'Frozen'],
        'y_vocabs': ['Production', 'Consumption', 'Imports', 'Exports', 'Prices', 'Innovation']
    },
    'Science and Engineering': {
        'x_vocabs': ['Computer', 'Mechanical', 'Electrical', 'Chemical', 'Civil', 'Biomedical'],
        'y_vocabs': ['Research', 'Development', 'Testing', 'Implementation', 'Maintenance', 'Innovation']
    },
    'Agriculture and Food Production': {
        'x_vocabs': ['Crops', 'Livestock', 'Dairy', 'Poultry', 'Fisheries', 'Forestry'],
        'y_vocabs': ['Yield', 'Quality', 'Sustainability', 'Cost', 'Demand', 'Export']
    },
    'Energy and Utilities': {
        'x_vocabs': ['Oil', 'Gas', 'Coal', 'Nuclear', 'Renewables', 'Electricity'],
        'y_vocabs': ['Production', 'Consumption', 'Prices', 'Efficiency', 'Emissions', 'Investment']
    },
    'Cultural Trends and Influences': {
        'x_vocabs': ['Fashion', 'Art', 'Music', 'Literature', 'Cuisine', 'Language'],
        'y_vocabs': ['Popularity', 'Adoption', 'Influence', 'Diversity', 'Globalization', 'Localization']
    },
    'Social Media and Digital Media and Streaming': {
        'x_vocabs': ['Facebook', 'Instagram', 'Twitter', 'YouTube', 'TikTok', 'Netflix'],
        'y_vocabs': ['Users', 'Engagement', 'Revenue', 'Content', 'Advertising', 'Growth']
    }
}


# =============================================

class DataGenerator:
    def __init__(self, topic: str, model: str):
        """
        初始化数据生成器

        参数:
            topic: 数据主题
            model: 数据生成模式 ('random', 'center', 'diagonal')
        """
        self.topic = topic
        self.model = model
        self.x_vocabs = topics_vocabs[topic]['x_vocabs']
        self.y_vocabs = topics_vocabs[topic]['y_vocabs']

        # 控制区块数量在10-50之间
        while True:
            self.x_blocks_num = random.randint(1, len(self.x_vocabs))
            self.y_blocks_num = random.randint(1, len(self.y_vocabs))
            total_blocks = self.x_blocks_num * self.y_blocks_num
            if 10 <= total_blocks <= 50:
                break

        # 随机选择坐标标签（不重复）
        self.x_labels = random.sample(self.x_vocabs, self.x_blocks_num)
        self.y_labels = random.sample(self.y_vocabs, self.y_blocks_num)

    def _generate_random_data(self) -> np.ndarray:
        """生成随机分布数据"""
        return np.random.rand(self.y_blocks_num, self.x_blocks_num)

    def _generate_center_data(self) -> np.ndarray:
        """生成中心分布数据"""
        x_center = self.x_blocks_num / 2
        y_center = self.y_blocks_num / 2

        x = np.arange(self.x_blocks_num)
        y = np.arange(self.y_blocks_num)
        xx, yy = np.meshgrid(x, y)

        distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)
        max_distance = np.sqrt((self.x_blocks_num / 2) ** 2 + (self.y_blocks_num / 2) ** 2)

        normalized = 1 - distance / max_distance
        noise = np.random.normal(0, 0.1, (self.y_blocks_num, self.x_blocks_num))
        data = np.clip(normalized + noise, 0, 1)

        return data

    def _generate_diagonal_data(self) -> np.ndarray:
        """生成对角分布数据"""
        data = np.zeros((self.y_blocks_num, self.x_blocks_num))

        # 主对角线
        for i in range(min(self.y_blocks_num, self.x_blocks_num)):
            data[i, i] = np.random.uniform(0.7, 1.0)

        # 副对角线
        for i in range(min(self.y_blocks_num, self.x_blocks_num)):
            j = self.x_blocks_num - 1 - i
            if j >= 0:
                data[i, j] = np.random.uniform(0.7, 1.0)

        # 对角线周围的衰减
        for i in range(self.y_blocks_num):
            for j in range(self.x_blocks_num):
                if data[i, j] == 0:
                    dist_main = abs(i - j)
                    dist_anti = abs(i - (self.x_blocks_num - 1 - j))
                    min_dist = min(dist_main, dist_anti)
                    value = max(0.1, 0.9 - min_dist * 0.15 + np.random.uniform(-0.1, 0.1))
                    data[i, j] = value

        return data

    def generate_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """根据指定模式生成数据"""
        if self.model == 'random':
            data = self._generate_random_data()
        elif self.model == 'center':
            data = self._generate_center_data()
        elif self.model == 'diagonal':
            data = self._generate_diagonal_data()
        else:
            raise ValueError(f"未知模式: {self.model}")

        # 创建DataFrame
        rows = []
        for y_idx in range(self.y_blocks_num):
            for x_idx in range(self.x_blocks_num):
                rows.append({
                    'x_block': self.x_labels[x_idx],
                    'y_block': self.y_labels[y_idx],
                    'level': data[y_idx, x_idx]
                })

        df = pd.DataFrame(rows)

        return df, data

    # 修改 save_to_csv 方法，添加 topic_counters 参数
    def save_to_csv(self, df: pd.DataFrame, output_dir: str, topic_counters: dict) -> str:
        """
        保存数据到CSV文件

        参数:
            df: 要保存的DataFrame
            output_dir: 输出目录
            topic_counters: 用于跟踪每个主题序号的字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名（使用 topic_xx.csv 格式，序号按主题计算）
        # 对主题文本进行清理，用于文件名
        sanitized_topic = self.topic.replace(' ', '_').replace('&', 'and').lower()  # 替换空格和&，转小写

        # 获取当前主题的序号并更新计数器
        if sanitized_topic not in topic_counters:
            topic_counters[sanitized_topic] = 0  # 初始化为0，这样第一个文件就是1

        current_count = topic_counters[sanitized_topic] + 1
        topic_counters[sanitized_topic] = current_count  # 更新计数器为下一个序号

        # 构造文件名
        filename_base = f"{sanitized_topic}_{current_count}"
        filename = os.path.join(output_dir, f"{filename_base}.csv")

        # 保存数据（第一行包含主题、区块数量和分布模式）
        with open(filename, 'w', newline='') as f:
            f.write(f"{self.topic}, ")  # 原始主题文本用于头部
            f.write(f"X blocks: {self.x_blocks_num}, Y blocks: {self.y_blocks_num}, ")
            f.write(f"Distribution: {self.model}\n")
            f.write("x_block,y_block,level\n")
            df.to_csv(f, index=False, header=False, lineterminator='\n')

        return filename


def generate_multiple_files(num: int, output_dir: str = 'csv/heatmap'):
    """
    生成多个数据文件

    参数:
        num: 要生成的文件数量
        output_dir: 输出目录
    """
    pattern = ['random', 'center', 'diagonal']
    available_topics = list(topics_vocabs.keys())

    # 初始化主题计数器字典
    topic_counters = {}

    for i in range(num):
        model = random.choice(pattern)
        topic = random.choice(available_topics)
        generator = DataGenerator(topic, model)
        df, data = generator.generate_data()

        print(f"\n文件 {i + 1}/{num}:")
        print(f"主题: {topic}")
        print(f"生成数据区块: {generator.x_blocks_num}×{generator.y_blocks_num}")
        print(f"分布模式: {model}")
        print(f"横坐标标签: {generator.x_labels}")
        print(f"纵坐标标签: {generator.y_labels}")

        # 调用 save_to_csv 时传入 topic_counters
        output_file = generator.save_to_csv(df, output_dir, topic_counters)
        print(f"数据已保存到 {output_file}")


if __name__ == "__main__":
    num = 10  # 指定要生成的文件数量，可以设置多一些来测试不同主题的编号
    generate_multiple_files(num)
