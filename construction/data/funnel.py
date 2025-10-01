import random
import pandas as pd
import numpy as np
import os

# output dir
output_dir = "csv/funnel/"
os.makedirs(output_dir, exist_ok=True)

"""             Begin Funnel Data Classes          """
# 1. **Transportation and Logistics** Freight Logistics Order Processing
transportation_stages = [
    "Search Transport Options",       # 查询运输方式
    "Select Carrier or Service",      # 选择承运商或服务
    "Book Transport Service",         # 预订运输服务
    "Prepare for Pickup",             # 准备接送货/上车
    "Begin Transportation",           # 开始运输
    "Complete Delivery",              # 完成交付或抵达
    "Provide Feedback or Rating"      # 反馈或评分
]

# 2. **Tourism and Hospitality** Tourist Booking and Travel
tourism_stages = [
    "View Destination Info",         # 浏览目的地
    "Search Travel Packages",        # 查询套餐
    "Select Accommodation",          # 选择住宿
    "Book Flight",                   # 预订航班
    "Make Reservation",              # 下单预订
    "Travel to Destination",         # 出发
    "Check-in at Hotel",             # 入住
    "Leave Review"                   # 评价反馈
]

# 3. **Business and Finance** Client Acquisition
business_stages = [
    "Lead Captured",                # 获取潜在客户
    "Initial Contact",              # 首次沟通
    "Needs Analysis",              # 客户需求分析
    "Proposal Sent",                # 发送方案
    "Negotiation",                  # 协商条款
    "Contract Signed",             # 签署合同
    "Client Onboarded"             # 成功转化/客户接入
]

# 4. **Real Estate and Housing Market**  Real Estate Sales Process
real_estate_stages = [
    "Property Listed",               # 1. 房产挂牌
    "Initial Inquiries",             # 2. 初步咨询
    "Property Viewed",               # 3. 看房
    "Negotiation",                   # 4. 协商价格
    "Contract Signed",               # 5. 签订合同
    "Sale Closed",                   # 6. 完成交易
    "Post-sale Survey"               # 7. 售后调查/反馈
]

# 5. **Healthcare and Health** Patient Referral Flow
healthcare_stages = [
    "Initial Consultation",            # 1. 初次就诊
    "Diagnostic Testing",              # 2. 诊断测试
    "Diagnosis Confirmed",             # 3. 诊断确认
    "Treatment Plan Recommended",      # 4. 推荐治疗方案
    "Treatment Initiated",             # 5. 开始治疗
    "Treatment Completed",             # 6. 治疗完成
    "Post-Treatment Assessment"        # 7. 治疗后评估
]

# 6. **Retail and E-commerce** E-commerce User Purchase
ecommerce_stages = [
    "View Advertisement",                 # 接触
    "Visit Product Page",                # 浏览商品详情
    "Add to Wishlist",                   # 初步兴趣
    "Add to Cart",                       # 准备购买
    "Initiate Checkout",                # 进入结账流程
    "Enter Payment Information",        # 提交支付意图
    "Complete Purchase",                # 成交
    "Leave Product Review"              # 售后参与
]

# 7. **Human Resources and Employee Management** Employee Recruitment
hr_stages = [
    "Job Posting Viewed",              # 1. 浏览岗位
    "Application Submitted",          # 2. 提交简历
    "Phone Screen",                   # 3. 电话初筛
    "HR Interview",                   # 4. HR 面试
    "Technical Interview",            # 5. 技术面试
    "Offer Extended",                 # 6. 发放 offer
    "Offer Accepted",                 # 7. 接受 offer / 入职
]

# 8. **Sports and Entertainment** Sports Event Participation
sports_entertainment_stages = [
    "Event Announcement",             # 1. 赛事或活动公告
    "Ticket Purchase",                # 2. 购票
    "Event Attendance",               # 3. 现场参与
    "Fan Engagement",                 # 4. 粉丝互动（社交媒体、活动）
    "Post-Event Feedback",            # 5. 赛后反馈
    "Merchandise Purchase",          # 6. 商品购买（周边商品）
    "Event Reattendance"              # 7. 再次参加（回头客）
]

# 9. **Education and Academics** Online Learning Progression
education_stages = [
    "Browse Course Page",
    "Register for Learning Platform", 
    "Register for Courses",
    "Complete the first lesson", 
    "Complete the assignment", 
    "Pass the final test", 
    "Earn a certificate"
]

# 10. **Food and Beverage Industry** Beverage Sales Process
beverage_sales_funnel = [
    "Product Concept",                # 1. 产品概念
    "Taste Testing",                  # 2. 品鉴
    "Launch in Market",               # 3. 市场上线
    "Retail Placement",               # 4. 上架零售
    "In-Store Promotion",             # 5. 店内促销
    "Customer Purchase",              # 6. 顾客购买
    "Repeat Purchase"                 # 7. 顾客重复购买
]

# 11. **Science and Engineering** Engineering Project Lifecycle
engineering_project_lifecycle = [
    "Project Initiation",             # 1. 项目启动
    "Design Phase",                   # 2. 设计阶段
    "Prototype Development",          # 3. 原型开发
    "Testing and Validation",         # 4. 测试和验证
    "Final Design Approval",          # 5. 最终设计批准
    "Production",                     # 6. 生产
    "Market Release",                 # 7. 市场发布
]

# 12. **Agriculture and Food Production** Agricultural Supply Chain
agriculture_stages = [
    "Total Harvested Crop",
    "After Post-Harvest Loss",
    "After Processing Loss",
    "After Quality Inspection",
    "Packaged for Distribution",
    "Delivered to Retail",
    "Purchased by Consumers"
]

# 13. **Energy and Utilities** Energy Generation and Distribution
energy_stages = [
    "Total Energy Generated",    # 1 发电站输出的原始能量
    "After Generation Losses",   # 2 发电损失
    "After Transmission Losses", # 3 输电损失
    "After Distribution Losses", # 4 配电损失
    "Delivered to End Users",    # 5 送达用户
    "Billed Consumption",        # 6 计费消费
    "Paid Consumption"           # 7 实际支付消费
]

# 14. **Cultural Trends and Influences** Cultural Artwork Exhibitions
cultural_stages = [
    "Artworks Created",           # 1. 所有创作作品
    "Submitted for Review",       # 2. 投稿参展
    "Approved for Exhibition",    # 3. 通过审核
    "Displayed in Exhibition",    # 4. 进入展出
    "Shortlisted for Acquisition",# 5. 入选候选收藏清单
    "Eligible for Acquisition",   # 6. 符合收藏标准（进一步评估确认）
    "Institutionally Acquired"    # 7. 正式收藏入档
]

# 15. **Social Media and Digital Media and Streaming** Social Media Engagement
social_media_stages = [
    "View Content",                    # 1. 浏览内容
    "Like or Comment",                 # 2. 点赞或评论
    "Share Content",                   # 3. 分享内容
    "Follow Account",                  # 4. 关注账号
    "Join Group or Community",         # 5. 加入群组或社区
    "Content Creation or Upload",      # 8. 创建或上传内容
    "Participate in Paid Subscription",# 9. 参与付费订阅
]
"""                End Funnel Data Classes          """

def generate_linear_drop(num_stages, start_value):
    total_drop_ratio = random.uniform(0.7, 0.85)
    avg_drop_ratio_per_stage = total_drop_ratio / (num_stages - 1)
    drop_ratio = random.uniform(avg_drop_ratio_per_stage * 0.9,
                                avg_drop_ratio_per_stage * 1.1)
    return drop_ratio * start_value

def generate_funnel_data(num_files, theme_name, data_unit, min_startVal, max_startVal, classes, file_prefix ):
    for j in range(1, num_files + 1):
        num_stages = random.choice([4, 5, 6, 7])
        start_index = random.randint(0, len(classes) - num_stages)
        selected_stages = classes[start_index : start_index + num_stages]

        attenuation_pattern = random.choice(["linear", "exponential", "logarithmic"])

        # if is percentage data, min_startVal and max_startVal are needless
        start_value = 100 if data_unit == "%" else random.randint(min_startVal, max_startVal)
        min_tail_ratio = random.uniform(0.05, 0.1)
        min_tail_value = int(start_value * min_tail_ratio)
        values = [start_value]

        for i in range(1, num_stages):
            prev = values[-1]

            if attenuation_pattern == "linear":
                drop = generate_linear_drop(num_stages, start_value)
                next_value = int(prev - drop)

            elif attenuation_pattern == "exponential":
                decay_rate = random.uniform(0.4, 0.7)
                expected = int(start_value * np.exp(-decay_rate * i))
                expected = min(expected, prev - 1)
                next_value = max(expected, min_tail_value)

            elif attenuation_pattern == "logarithmic":
                log_base = random.uniform(1.5, 3.0)
                log_i = np.log(i + 1) / np.log(log_base)
                total_log = np.log(num_stages) / np.log(log_base)
                drop_ratio = log_i / total_log
                expected = int(start_value * (1 - drop_ratio))
                expected = min(expected, prev - 1)
                next_value = max(expected, min_tail_value)

            if (next_value >= prev):
                next_value = int(prev * random.uniform(0.9,0.95))

            values.append(next_value)

        # save to csv
        file_path = os.path.join(output_dir, f"{file_prefix}_{j}.csv")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{theme_name}, {data_unit}, {attenuation_pattern}\n")
            f.write("Phase, Value\n")
            for stage, count in zip(selected_stages, values):
                f.write(f"{stage},{count}\n")

        print(f"✅ Generated: {file_path} ({attenuation_pattern})")

funnel_data_configs = {
    "transportation_logistics_process": {
        "number": 1,
        "theme_name": "Transportation and Logistics",
        "data_unit": "Thousand Shipments",
        "min": 2000,
        "max": 5000,
        "classes": transportation_stages
    },
    "tourism_travel_flow": {
        "number": 1,
        "theme_name": "Tourism and Hospitality",
        "data_unit": "Thousand Tourists",
        "min": 1000,
        "max": 3000,
        "classes": tourism_stages
    },
    "business_client_acquisition": {
        "number": 1,
        "theme_name": "Business and Finance",
        "data_unit": "Clients",
        "min": 500,
        "max": 2000,
        "classes": business_stages
    },
    "real_estate_sales": {
        "number": 1,
        "theme_name": "Real Estate and Housing Market",
        "data_unit": "Properties",
        "min": 300,
        "max": 800,
        "classes": real_estate_stages
    },
    "patient_referral_flow": {
        "number": 1,
        "theme_name": "Healthcare and Health",
        "data_unit": "Patients",
        "min": 1000,
        "max": 5000,
        "classes": healthcare_stages
    },
    "ecommerce_user_journey": {
        "number": 1,
        "theme_name": "Retail and E-commerce",
        "data_unit": "Users",
        "min": 5000,
        "max": 20000,
        "classes": ecommerce_stages
    },
    "employee_recruitment": {
        "number": 1,
        "theme_name": "Human Resources and Employee Management",
        "data_unit": "%",
        "min": 1000,
        "max": 3000,
        "classes": hr_stages
    },
    "sports_event_participation": {
        "number": 1,
        "theme_name": "Sports and Entertainment",
        "data_unit": "Participants",
        "min": 500,
        "max": 2000,
        "classes": sports_entertainment_stages
    },
    "education_learning_progression": {
        "number": 1,
        "theme_name": "Education and Academics",
        "data_unit": "%",
        "min": 3000,
        "max": 10000,
        "classes": education_stages
    },
    "beverage_sales_process": {
        "number": 1,
        "theme_name": "Food and Beverage Industry",
        "data_unit": "Thousand Bottles",
        "min": 1000,
        "max": 5000,
        "classes": beverage_sales_funnel
    },
    "engineering_project_lifecycle": {
        "number": 1,
        "theme_name": "Science and Engineering",
        "data_unit": "%",
        "min": 100,
        "max": 500,
        "classes": engineering_project_lifecycle
    },
    "agricultural_supply_chain": {
        "number": 1,
        "theme_name": "Agriculture and Food Production",
        "data_unit": "Tons",
        "min": 10000,
        "max": 30000,
        "classes": agriculture_stages
    },
    "energy_distribution_loss": {
        "number": 1,
        "theme_name": "Energy and Utilities",
        "data_unit": "Megawatt Hours",
        "min": 50000,
        "max": 100000,
        "classes": energy_stages
    },
    "cultural_art_exhibition": {
        "number": 1,
        "theme_name": "Cultural Trends and Influences",
        "data_unit": "%",
        "min": 500,
        "max": 2000,
        "classes": cultural_stages
    },
    "social_media_engagement": {
        "number": 1,
        "theme_name": "Social Media and Digital Media and Streaming",
        "data_unit": "Million Users",
        "min": 100,
        "max": 500,
        "classes": social_media_stages
    }
}



if __name__ == '__main__':
    for prefix, config in funnel_data_configs.items():
        generate_funnel_data(
            config["number"],
            config["theme_name"],
            config["data_unit"],
            config["min"],
            config["max"],
            config["classes"],
            prefix
        )
