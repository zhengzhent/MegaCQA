
import os
import csv
import random
import numpy as np

try:
    from data import THEME_METRIC_PARAMS
except ImportError:
    print("Error: data.py not found or THEME_METRIC_PARAMS not defined in data.py.")
    exit()
except NameError:
    print("Error: THEME_METRIC_PARAMS dictionary not found within data.py.")
    exit()

TREND_TYPES = {
    "stable_rising":
        {"slope_pct": 0.05,  "noise_sd": 0.1},  # 5 %/年,
    "stable_falling":
        {"slope_pct": 0.03,  "noise_sd": 0.1},  # 3 %/年
    "exponential_rising":
        {"factor": 1.10, "noise_sd": 0.1},  # 10 %复合
    "exponential_falling":
        {"factor": 0.90, "noise_sd": 0.1},  # -10 %复合
    "periodic_stable":
        {"ampl_pct": 0.25,   "noise_sd": 0.1, "period": 5},  # 振幅=25 %
    "volatile_rising":
        {"slope_pct": 0.08,  "noise_sd": 0.1},  # 8 %/年 + 大噪声
    "volatile_falling":
        {"slope_pct": 0.06,  "noise_sd": 0.1},  # −6%/year ± noise
    "single_peak":
        {"noise_sd": 0.1},
    "single_valley":
        {"noise_sd": 0.1},
    "bimodal_peak":
        {"noise_sd": 0.1},
    "bimodal_valley":
        {"noise_sd": 0.1},
}

# --- 新增配置参数 ---
# 检查重叠的起始年份数量
NUM_INITIAL_POINTS_CHECK = 5
# 判断重叠的阈值因子：如果起始 NUM_INITIAL_POINTS_CHECK 年的平均绝对差异小于 (该时期平均值 * 此因子)，则认为重叠
OVERLAP_THRESHOLD_FACTOR = 0.1 # 例如，10% 的平均值差异
# Y轴随机上移的范围因子：上移量在 [该时期平均值 * 因子[0], 该时期平均值 * 因子[1]] 之间随机选择
SHIFT_RANGE_FACTOR = [0.05, 0.15] # 例如，上移该时期平均值的 5% 到 15%
# --- 新增配置参数结束 ---


def generate_stable_rising(years, params):
    base = params['base']
    p = TREND_TYPES["stable_rising"]
    slope = base * p["slope_pct"]  # 每年固定增幅
    # Added noise handling
    return [max(0.01 * base, base + slope * (y - years[0]) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"])) for y in years]

def generate_stable_falling(years, params):
    base = params['base']
    p = TREND_TYPES["stable_falling"]
    slope = base * p["slope_pct"]  # 每年固定增幅
    # Added noise handling
    return [max(0.01 * base, base + slope * (y - years[0]) - base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"])) for y in years]

def generate_exponential_rising(years, params):
    base = params['base']
    p = TREND_TYPES["exponential_rising"]
    factor = p["factor"]
    noise_sd = p["noise_sd"]
    start_year = years[0]
    return [
        max(0.01 * base, base * (factor ** (year - start_year)) +
        base * params.get('noise', 0) * np.random.normal(0, noise_sd))
        for year in years
    ]

def generate_exponential_falling(years, params):
    base = params['base']
    p = TREND_TYPES["exponential_falling"]
    factor = p["factor"]
    noise_sd = p["noise_sd"]
    start_year = years[0]
    return [
        max(0.01 * base, base * (factor ** (year - start_year)) -
        base * params.get('noise', 0) * np.random.normal(0, noise_sd))
        for year in years
    ]

def generate_periodic_stable(years, params):
    base = params['base']
    p = TREND_TYPES["periodic_stable"]
    amplitude = params.get('amplitude', base * p["ampl_pct"])
    period = params.get('period', p["period"])
    noise_sd = p["noise_sd"]
    start_year = years[0]
    return [
        max(0.01 * base, base + amplitude * np.sin(2 * np.pi * (year - start_year) / period) +
        base * params.get('noise', 0) * np.random.normal(0, noise_sd))
        for year in years
    ]

def generate_volatile_rising(years, params):
    base = params['base']
    p = TREND_TYPES["volatile_rising"]
    slope = base * p["slope_pct"]
    noise_sd = p["noise_sd"]
    start_year = years[0]
    return [
        max(0.01 * base, base + slope * (year - start_year) +
        base * params.get('noise', 0) * np.random.normal(0, noise_sd))
        for year in years
    ]

def generate_volatile_falling(years, params):
    base = params['base']
    p = TREND_TYPES["volatile_falling"]
    slope = -base * p["slope_pct"]
    noise_sd = p["noise_sd"]
    start_year = years[0]
    return [
        max(0.01 * base, base + slope * (year - start_year) -
        base * params.get('noise', 0) * np.random.normal(0, noise_sd))
        for year in years
    ]

def generate_single_peak(years, params):
    base = params['base']
    p = TREND_TYPES["single_peak"]
    peak_year = (years[0] + years[-1]) // 2
    peak_amplitude = base * 0.75
    return [
        max(0.01 * base, base + peak_amplitude * np.exp(-((year - peak_year) ** 2) / (2 * 5 ** 2)) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"]))
        for year in years
    ]

def generate_single_valley(years, params):
    base = params['base']
    p = TREND_TYPES["single_valley"]
    valley_year = (years[0] + years[-1]) // 2
    valley_depth = base * 0.75
    return [
        max(0.01 * base, base - valley_depth * np.exp(-((year - valley_year) ** 2) / (2 * 5 ** 2)) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"]))
        for year in years
    ]

def generate_bimodal_peak(years, params):
    base = params['base']
    p = TREND_TYPES["bimodal_peak"]
    peak1_year = years[0] + len(years) // 4
    peak2_year = years[0] + len(years) * 3 // 4
    peak_amplitude = base * 0.75
    return [
        max(0.01 * base, base + peak_amplitude * np.exp(-((year - peak1_year) ** 2) / (2 * 5 ** 2)) +
             peak_amplitude * np.exp(-((year - peak2_year) ** 2) / (2 * 5 ** 2)) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"]))
        for year in years
    ]

def generate_bimodal_valley(years, params):
    base = params['base']
    p = TREND_TYPES["bimodal_valley"]
    valley1_year = years[0] + len(years) // 4
    valley2_year = years[0] + len(years) * 3 // 4
    valley_depth = base * 0.75
    return [
        max(0.01 * base, base - valley_depth * np.exp(-((year - valley1_year) ** 2) / (2 * 5 ** 2)) -
             valley_depth * np.exp(-((year - valley2_year) ** 2) / (2 * 5 ** 2)) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"]))
        for year in years
    ]


def generate_data(years, params, trend_type):
    generators = {
        "stable_rising": generate_stable_rising,
        "stable_falling": generate_stable_falling,
        "exponential_rising": generate_exponential_rising,
        "exponential_falling": generate_exponential_falling,
        "periodic_stable": generate_periodic_stable,
        "volatile_rising": generate_volatile_rising,
        "volatile_falling": generate_volatile_falling,
        "single_peak": generate_single_peak,
        "single_valley": generate_single_valley,
        "bimodal_peak": generate_bimodal_peak,
        "bimodal_valley": generate_bimodal_valley
    }
    return generators[trend_type](years, params)

# --- 新增函数：处理曲线重叠并进行Y轴上移 ---
def apply_overlap_shift(all_data, num_initial_points, overlap_threshold_factor, shift_range_factor):
    """
    检测曲线在起始段的重叠度，并对重叠的曲线进行Y轴上移。

    Args:
        all_data: list of lists, 每一项是一个subject的数值列表。
        num_initial_points: 检查重叠的起始点数量。
        overlap_threshold_factor: 判断重叠的阈值因子。
        shift_range_factor: Y轴随机上移的范围因子 [min_factor, max_factor]。

    Returns:
        list of lists, 处理后的数值列表。
    """
    num_lines = len(all_data)
    if num_lines < 2:
        return all_data # 少于两条线无需处理

    # 限制检查点数不超过实际数据长度
    check_len = min(num_initial_points, len(all_data[0]))
    if check_len == 0:
        return all_data # 没有数据点无需处理

    # 标记哪些线已经属于重叠组并已处理
    processed_indices = set()
    modified_data = [list(line) for line in all_data] # 创建数据的副本进行修改

    for i in range(num_lines):
        if i in processed_indices:
            continue

        current_overlap_group_indices = {i}
        # 找出与当前线 i 重叠的所有线
        for j in range(i + 1, num_lines):
            if j in processed_indices:
                continue

            # 比较线 i 和线 j 在起始段的重叠度
            segment_i = np.array(modified_data[i][:check_len])
            segment_j = np.array(modified_data[j][:check_len])

            # 计算起始段的平均值
            avg_i = np.mean(segment_i)
            avg_j = np.mean(segment_j)

            # 防止除以零，如果平均值接近零，使用一个很小的固定阈值
            # 或更简单地，使用两个平均值的平均值作为相对基准
            base_value = (avg_i + avg_j) / 2.0
            # 确保 base_value 不会太小，避免阈值过低
            if abs(base_value) < 1e-6: # 如果平均值接近零
                 # 可以考虑使用一个小的固定值作为阈值，或者跳过这种非常规情况
                 # 在我们的合成数据中，base通常不为零，这里简单处理
                 threshold = overlap_threshold_factor # 或者一个小的固定值 e.g., 1.0
            else:
                threshold = abs(base_value) * overlap_threshold_factor

            # 计算起始段的平均绝对差异
            avg_diff = np.mean(np.abs(segment_i - segment_j))

            # 如果平均差异小于阈值，则认为重叠
            if avg_diff <= threshold:
                current_overlap_group_indices.add(j)

        # 如果找到了重叠的曲线（组大小 > 1）
        if len(current_overlap_group_indices) > 1:
            # 将组内所有线的索引标记为已处理
            processed_indices.update(current_overlap_group_indices)

            # 选择组内的第一条线作为基准，其他线进行上移
            group_list = sorted(list(current_overlap_group_indices)) # 排序保证每次运行结果一致（虽然移位量是随机的）
            base_line_index = group_list[0]

            for line_index in group_list:
                if line_index == base_line_index:
                    continue # 跳过基准线

                # 计算该线起始段的平均值，用于确定上移范围
                avg_initial_shifted = np.mean(np.array(modified_data[line_index][:check_len]))

                # 确定上移的随机范围
                min_shift = avg_initial_shifted * shift_range_factor[0]
                max_shift = avg_initial_shifted * shift_range_factor[1]

                # 确保上移量是正的，且有最小上移量，避免微小移动
                min_shift = max(min_shift, abs(avg_initial_shifted) * 0.01) # 至少上移1%的平均值（或一个固定小值）
                if min_shift >= max_shift: # 防止范围无效
                     max_shift = min_shift + (abs(avg_initial_shifted) * 0.05) # 如果计算出的范围无效，给一个默认最小范围

                # 生成随机上移量
                shift_amount = random.uniform(min_shift, max_shift)

                # 对该线的每个数据点进行上移
                modified_data[line_index] = [value + shift_amount for value in modified_data[line_index]]

    return modified_data
# --- 新增函数结束 ---


# 创建输出目录
os.makedirs('csv', exist_ok=True)

# 遍历每个主题和指标
for theme, metrics in THEME_METRIC_PARAMS.items():
    for metric_idx, metric in enumerate(metrics, 1):
        # 提取指标参数
        metric_name = metric['name']
        unit = metric['unit']
        subjects = metric['subject']
        params = metric['params']

        # 随机选择1-7个subject
        k = random.randint(1, min(7, len(subjects)))
        selected_subjects = random.sample(subjects, k)

        # 生成趋势类型（允许重复）
        trends = random.choices(list(TREND_TYPES.keys()), k=k)

        # 生成年份数据（随机3-30个连续年份）
        num_years = random.randint(3, 30)
        start_year = random.randint(1950, 2025 - num_years)
        years = list(range(start_year, start_year + num_years))

        # 生成每个subject的数据
        all_data = []
        for trend in trends:
            # ---------- 新增开始 ----------
            # 为当前 subject 复制一份 params，并随机调整 base
            subj_params = params.copy()  # 深拷贝，避免污染原字典
            # 你可以按需修改倍率范围：0.5~1.5、0.8~1.2 ……
            subj_params['base'] = params['base'] * random.uniform(0.5, 1.5)  # 3. 加入噪声
            # ---------- 新增结束 ----------

            values = generate_data(years, subj_params, trend)
            # 根据单位类型处理数据 (注意：这里先生成原始数据，后处理单位限制)
            # 这种方式可能导致上移后的数据超出原有的单位限制（如100%），取决于你的需求
            # 如果需要严格限制，可以在上移后再应用max/min，但这可能使曲线不再是简单的Y轴上移
            # 这里我们选择先不应用单位限制，让上移后的数据反映相对位置变化
            all_data.append(values)  # 暂时不应用单位限制

        # --- 调用新增的重叠处理函数 ---
        processed_data = apply_overlap_shift(
            all_data,
            NUM_INITIAL_POINTS_CHECK,
            OVERLAP_THRESHOLD_FACTOR,
            SHIFT_RANGE_FACTOR
        )
        # --- 调用结束 ---

        # 根据单位类型处理数据 (现在应用单位限制，但要注意上移可能导致超出)
        # 如果单位是%，上移可能导致大于100
        # 如果单位是ratio，上移可能导致大于某个预期值
        # 这里的处理方式是简单地限制在0以上，对于%则限制在0-100
        final_data = []
        for line_values in processed_data:
            if unit == '%':
                # 限制在0-100%，上移可能导致大于100
                final_data.append([max(0, min(100, v)) for v in line_values])
            elif unit == 'ratio':
                # 比率不能为负
                final_data.append([max(0, v) for v in line_values])
            else:
                 # 其他单位只保证非负（如果需要）
                 final_data.append([max(0, v) for v in line_values]) # 假设一般数据非负

        # 四舍五入到小数点后两位
        final_data = [[round(v, 2) for v in line] for line in final_data]


        # 转置数据为按年排列
        data_by_year = list(zip(*final_data))

        # 构建文件名
        theme_slug = theme.lower().replace(' ', '_')
        filename = f'csv/line_{theme_slug}_{metric_idx}.csv'

        # 写入CSV文件
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 第一行：主题，指标，单位
            writer.writerow([theme, metric_name, unit])
            # 第二行：年份头
            writer.writerow(['Year'] + selected_subjects)
            # 第三行：趋势类型
            writer.writerow(['trend'] + trends)
            # 数据行
            for year, values in zip(years, data_by_year):
                writer.writerow([year] + list(values))

print("CSV files generated successfully in the 'csv' directory.")

