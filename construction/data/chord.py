import random
import numpy as np
import os

# 输出目录
output_dir = "csv/chord/"
os.makedirs(output_dir, exist_ok=True)

def generate_chord_data(num_files, theme_name, data_unit, minimum, maximum, classes, file_prefix):
    for j in range(1, num_files + 1):
        # 1. 选择 3~5 个节点
        num_nodes = random.randint(3, 5)
        nodes = random.sample(classes, num_nodes)

        # 2. 构建所有可能的无向边 (i<j)，并打乱
        all_pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i+1:]]
        random.shuffle(all_pairs)

        # 3. 先确保每个节点至少有一个连接（无向）
        undirected_links = []
        node_covered = {n: False for n in nodes}
        for a, b in all_pairs:
            if not node_covered[a] or not node_covered[b]:
                undirected_links.append((a, b))
                node_covered[a] = True
                node_covered[b] = True
            if all(node_covered.values()):
                break

        # 4. 随机再加一些无向连接，使网络稍微稠密一点
        extras = random.sample(
            [p for p in all_pairs if p not in undirected_links],
            k=min(2, len(all_pairs) - len(undirected_links))
        )
        undirected_links.extend(extras)

        # 5. 为每个无向连接产生两个方向的流量
        num_pairs = len(undirected_links)
        total_directed = num_pairs * 2

        # 6. 选定分布模式和总流量
        distribution_mode = random.choice(["random", "normal", "long_tail", "linear"])
        start_value = random.randint(minimum, maximum)

        # 7. 生成权重并转为整数流量（每条至少 1）
        if distribution_mode == "random":
            weights = np.random.rand(total_directed)
        elif distribution_mode == "linear":
            weights = np.linspace(1, 0.3, total_directed)
        elif distribution_mode == "long_tail":
            weights = 1 / np.linspace(1, total_directed, total_directed)
        elif distribution_mode == "normal":
            mean = total_directed / 2
            std = total_directed / 4
            weights = np.exp(-0.5 * ((np.arange(total_directed) - mean) / std) ** 2)
        else:
            raise ValueError(f"Unknown distribution: {distribution_mode}")

        # 归一化为权重
        weights /= weights.sum()

        # 为了避免不同连接获得相同 value，添加扰动再排序
        noise = np.random.uniform(0.95, 1.05, size=num_pairs * 2)
        weights *= noise

        # 再次归一化并映射为整数
        weights /= weights.sum()
        raw_values = start_value * weights

        # 离散处理 + 避免重复
        values = [int(val) for val in raw_values]

        # 防止多个值完全一样（特别是在数值较小或 link 少时）
        used = set()
        for i in range(len(values)):
            while values[i] in used:
                values[i] += random.randint(1, 3)  # 小幅扰动避免重复
            used.add(values[i])

        # 调整误差到目标总值
        diff = start_value - sum(values)
        values[0] += diff  # 将误差修正到第一个值


        # 8. 构造 CSV 数据：每个无向对写入 (a→b) 和 (b→a)
        chord_data = []
        for idx, (a, b) in enumerate(undirected_links):
            v_ab = values[2 * idx]
            v_ba = values[2 * idx + 1]
            chord_data.append((a, b, v_ab))
            chord_data.append((b, a, v_ba))

        # 9. 写文件
        filename = f"{file_prefix}_{j}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{theme_name}, {data_unit}, {distribution_mode}\n")
            f.write("Source,Target,Value\n")
            for src, tgt, val in chord_data:
                f.write(f"{src},{tgt},{val}\n")

        print(f"Generated file: {filepath} | Nodes: {num_nodes} | Undirected pairs: {num_pairs} | Mode: {distribution_mode}")

"""             Begin Chord Data Classes          """
# 1. **Transportation and Logistics** Passenger Station Transfer Flow
station_classes = [
    "Tokyo",      # 东京站（多条线路交汇）
    "Shinjuku",   # 新宿站（世界最繁忙车站）
    "Osaka",      # 大阪站（关西枢纽）
    "Nagoya",     # 名古屋站（中部枢纽）
    "Ueno",       # 上野站（多条JR和地铁线交汇）
    "Shibuya",    # 涩谷站（重要换乘站）
    "Ikebukuro",  # 池袋站（三大副都心之一）
    "Shinagawa",  # 品川站（新干线停靠站）
    "Yokohama",   # 横滨站（重要枢纽）
    "Kyoto"       # 京都站（旅游枢纽）
]
# 2. **Tourism and Hospitality** Tourism Transit in Global Hubs
city_classes = [
    "Paris",
    "New York",
    "Tokyo",
    "London",
    "Dubai",
    "Sydney",
    "Rome",
    "Bangkok",
    "Istanbul",
    "Barcelona"
]
# 3. **Business and Finance** Global Hub Business Transactions
financial_hubs = [
    "New York",      # Global financial capital
    "London",        # European financial hub
    "Tokyo",         # Asian financial center
    "Hong Kong",     # Gateway to Chinese markets
    "Singapore",     # Southeast Asia financial hub
    "Shanghai",      # Growing financial center
    "Frankfurt",     # Eurozone banking center
    "Zurich",        # Private banking and wealth management
    "Sydney",        # Asia-Pacific financial hub
    "Dubai"          # Middle East financial center
]
# 4. **Real Estate and Housing Market** Real Estate Investment Flow
real_estate_hubs = [
    "New York",       # 高端住宅和商业地产市场活跃
    "Los Angeles",    # 房价高企、海外投资频繁
    "Miami",          # 拉美投资者偏好城市
    "San Francisco",  # 科技行业带动高端住宅需求
    "Chicago",        # 中部大城市，购房成本相对较低
    "Toronto",        # 加拿大主要投资和移民城市
    "Vancouver",      # 亚洲投资者偏好市场
    "London",         # 欧洲地产投资核心
    "Dubai",          # 中东房地产开发热点
    "Singapore"       # 亚洲金融中心及房产交易活跃城市
]

# 5. **Healthcare and Health** Medical Referral Flow
healthcare_institutions = [
    "Tertiary General Hospital",         # 三甲综合医院 (顶级综合医疗机构)
    "Commdata_unity Health Center",          # 社区卫生服务中心 (基层首诊机构) 
    "Specialist Hospital",              # 专科医院 (特定疾病治疗中心)
    "Rehabilitation Center",            # 康复机构 (术后恢复和长期护理)
    "TCM Hospital",                     # 中医院 (传统中医药治疗)
    "Emergency Medical Center",         # 急救中心 (紧急医疗处置)
    "Geriatric Care Hospital",          # 老年病医院 (专治老年疾病)
    "Maternal and Child Health Center", # 妇幼保健院 (妇产儿科专科)
    "Rural Township Clinic",            # 乡镇卫生院 (农村基层医疗)
    "Mental Health Hospital"            # 精神卫生中心 (心理和精神病专科)
]
# 6. **Retail and E-commerce** E-commerce Platform User Migration
retail_ecommerce_hubs = [
    "Amazon",         # 全球最大电商平台，覆盖多国市场
    "Alibaba",        # 中国和亚洲电商巨头，B2B 和 B2C 平台
    "eBay",           # 国际二手交易与拍卖市场平台
    "Walmart",        # 美国零售巨头，线上线下融合
    "JD.com",         # 中国主要自营电商平台，物流强大
    "Shopify",        # 独立站平台，支持全球中小卖家
    "Rakuten",        # 日本领先电商平台
    "MercadoLibre",   # 拉丁美洲最大电商平台
    "Flipkart",       # 印度主要电商平台，被沃尔玛收购
    "Zalando"         # 欧洲时尚类电商平台
]
# 7. **Human Resources and Employee Management** HR Department Flow
department_flow = [
    "Finance Department",          # 财务部
    "Operations Department",       # 运营部
    "Human Resources Department",  # 人力资源部
    "Legal Department",            # 法务部
    "IT Department",               # IT 部门
    "Sales Department",            # 销售部
    "Marketing Department",        # 市场部
    "Admin and Facilities"         # 行政与设施管理部
]
# 8. **Sports and Entertainment** NBA Eastern Team Fans Migration
eastern_teams = [
    "Atlanta Hawks",           # 亚特兰大老鹰
    "Boston Celtics",          # 波士顿凯尔特人
    "Brooklyn Nets",           # 布鲁克林篮网
    "Charlotte Hornets",       # 夏洛特黄蜂
    "Chicago Bulls",           # 芝加哥公牛
    "Cleveland Cavaliers",     # 克利夫兰骑士
    "Detroit Pistons",         # 底特律活塞
    "Indiana Pacers",          # 印第安纳步行者
    "Miami Heat",              # 迈阿密热火
    "Milwaukee Bucks",         # 密尔沃基雄鹿
    "New York Knicks",         # 纽约尼克斯
    "Orlando Magic",           # 奥兰多魔术
    "Philadelphia 76ers",      # 费城76人
    "Toronto Raptors",         # 多伦多猛龙
    "Washington Wizards"       # 华盛顿奇才
]
# 9. **Education and Academics** Student Major Changes
academic_classes = [
    "Computer Science",
    "Business",
    "Engineering",
    "Arts",
    "Natural Sciences",
    "Social Sciences",
    "Health Sciences"
]
# 10. **Food and Beverage Industry** Restaurant Customer Flow
restaurant_customer_flow = [
    "McDonald's",        # 全球最大快餐连锁 (Global fast-food chain)
    "Starbucks",         # 全球咖啡连锁巨头 (Global coffeehouse chain)
    "KFC",               # 炸鸡快餐连锁 (Fried chicken fast-food chain)
    "Subway",            # 三明治连锁 (Sandwich chain)
    "Pizza Hut",         # 披萨连锁 (Pizza chain)
    "Burger King",       # 汉堡快餐连锁 (Burger fast-food chain)
    "Domino's Pizza",    # 披萨外卖连锁 (Pizza delivery chain)
    "Dunkin'",           # 咖啡与甜甜圈连锁 (Coffee and donut chain)
    "Taco Bell",         # 墨西哥风味快餐 (Mexican-inspired fast-food chain)
    "Chick-fil-A"        # 鸡肉三明治连锁 (Chicken sandwich chain)
]
# 11. **Science and Engineering** Research Institution Collaboration Flow
research_institutions = [
    "MIT",               # 麻省理工学院 (Massachusetts Institute of Technology)
    "Stanford",          # 斯坦福大学 (Stanford University)
    "Harvard",           # 哈佛大学 (Harvard University)
    "Caltech",           # 加州理工学院 (California Institute of Technology)
    "Oxford",            # 牛津大学 (University of Oxford)
    "Cambridge",         # 剑桥大学 (University of Cambridge)
    "ETH Zurich",        # 苏黎世联邦理工学院 (Swiss Federal Institute of Technology Zurich)
    "University of Tokyo", # 东京大学 (The University of Tokyo)
    "National University of Singapore", # 新加坡国立大学 (National University of Singapore)
    "Tsinghua University" # 清华大学 (Tsinghua University)
]
# 12. **Agriculture and Food Production** Crops Production Transfer Flow
crops_distribution = [
    "Wheat",            # 小麦 (Wheat)
    "Rice",             # 大米 (Rice)
    "Corn",             # 玉米 (Corn)
    "Soybeans",         # 大豆 (Soybeans)
    "Barley",           # 大麦 (Barley)
    "Oats",             # 燕麦 (Oats)
    "Potatoes",         # 土豆 (Potatoes)
    "Sugarcane"         # 甘蔗 (Sugarcane)
]
# 13. **Energy and Utilities** Energy Production and Consumption Flow
energy_classes = [
    "Coal",             # 煤炭 (Coal)
    "Natural Gas",      # 天然气 (Natural Gas)
    "Oil",              # 石油 (Oil)
    "Nuclear",          # 核能 (Nuclear Energy)
    "Hydropower",       # 水电 (Hydropower)
    "Wind",             # 风能 (Wind Energy)
    "Solar",            # 太阳能 (Solar Energy)
    "Geothermal",       # 地热能 (Geothermal Energy)
    "Biomass",          # 生物质能 (Biomass Energy)
]
# 14. **Cultural Trends and Influences** Fashion Trends Influence
fashion_trends = [
    "Streetwear",            # 街头风格 (Streetwear)
    "Luxury Fashion",        # 奢侈时尚 (Luxury Fashion)
    "Vintage",               # 复古风格 (Vintage)
    "Sustainable Fashion",   # 可持续时尚 (Sustainable Fashion)
    "High Fashion",          # 高端时尚 (High Fashion)
    "Athleisure",            # 运动休闲 (Athleisure)
    "Fast Fashion",          # 快时尚 (Fast Fashion)
]
# 15. **Social Media and Digital Media and Streaming** Digital Platform User Migration
digital_platforms = [
    "TikTok",               # 全球短视频社交平台 (Global short-video platform)
    "Instagram",           # 图文社交网络 (Photo-sharing social network)
    "YouTube",             # 视频流媒体巨头 (Video streaming giant)
    "Twitch",              # 游戏直播平台 (Game live-streaming)
    "Netflix",             # 订阅制流媒体 (Subscription streaming service)
    "Twitter/X",           # 实时信息网络 (Real-time information network)
    "OnlyFans"             # 创作者订阅平台 (Creator subscription platform)
]

"""               End Chord Data Classes          """

chord_data_configs = {
    "station_transfer_flow": {
        "number": 1,
        "theme_name": "Passenger Station Transfer Flow",
        "data_unit": "Passengers",
        "min": 50000,
        "max": 150000,
        "classes": station_classes
    },
    "tourism_transit_flow": {
        "number": 1,
        "theme_name": "Tourism Transit in Global Hubs",
        "data_unit": "Tourists",
        "min": 10000,
        "max": 30000,
        "classes": city_classes
    },
    "financial_transactions_flow": {
        "number": 1,
        "theme_name": "Global Hub Business Transactions",
        "data_unit": "Million $",
        "min": 10000,
        "max": 50000,
        "classes": financial_hubs
    },
    "real_estate_investment_flow": {
        "number": 1,
        "theme_name": "Real Estate Investment Flow",
        "data_unit": "Million $",
        "min": 5000,
        "max": 25000,
        "classes": real_estate_hubs
    },
    "medical_referral_flow": {
        "number": 1,
        "theme_name": "Medical Referral Flow",
        "data_unit": "Patients",
        "min": 1000,
        "max": 3000,
        "classes": healthcare_institutions
    },
    "ecommerce_user_migration": {
        "number": 1,
        "theme_name": "E-commerce Platform User Migration",
        "data_unit": "Million Users",
        "min": 5000,
        "max": 15000,
        "classes": retail_ecommerce_hubs
    },
    "hr_department_flow": {
        "number": 1,
        "theme_name": "HR Department Flow",
        "data_unit": "Employees",
        "min": 1000,
        "max": 5000,
        "classes": department_flow
    },
    "nba_fans_migration": {
        "number": 1,
        "theme_name": "NBA Eastern Team Fans Migration",
        "data_unit": "Thousand Fans",
        "min": 2000,
        "max": 5000,
        "classes": eastern_teams
    },
    "academic_major_changes": {
        "number": 1,
        "theme_name": "Student Major Changes",
        "data_unit": "Students",
        "min": 500,
        "max": 3000,
        "classes": academic_classes
    },
    "restaurant_customer_flow": {
        "number": 1,
        "theme_name": "Restaurant Customer Flow",
        "data_unit": "Customers",
        "min": 5000,
        "max": 20000,
        "classes": restaurant_customer_flow
    },
    "research_institution_collaboration": {
        "number": 1,
        "theme_name": "Research Institution Collaboration Flow",
        "data_unit": "Collaborations",
        "min": 100,
        "max": 500,
        "classes": research_institutions
    },
    "crops_production_transfer_flow": {
        "number": 1,
        "theme_name": "Crops Production Transfer Flow",
        "data_unit": "Tons",
        "min": 100000,
        "max": 500000,
        "classes": crops_distribution
    },
    "energy_production_consumption_flow": {
        "number": 1,
        "theme_name": "Energy Production and Consumption Flow",
        "data_unit": "Gigawatt-hours",
        "min": 50000,
        "max": 200000,
        "classes": energy_classes
    },
    "fashion_trends_influence_flow": {
        "number": 1,
        "theme_name": "Fashion Trends Influence",
        "data_unit": "Trend Influence Score",
        "min": 1000,
        "max": 5000,
        "classes": fashion_trends
    },
    "digital_platform_user_migration": {
        "number": 1,
        "theme_name": "Digital Platform User Migration",
        "data_unit": "Million Users",
        "min": 1000,
        "max": 5000,
        "classes": digital_platforms
    }
}

  
if __name__ == '__main__':
    for prefix, config in chord_data_configs.items():
        generate_chord_data(
            config["number"],
            config["theme_name"],
            config["data_unit"],
            config["min"],
            config["max"],
            config["classes"],
            prefix
        )

