# line.py (Modified for weighted subject count)

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
    # "exponential_rising":
    #     {"factor": 1.10, "noise_sd": 0.1},  # 10 %复合
    # "exponential_falling":
    #     {"factor": 0.90, "noise_sd": 0.1},  # -10 %复合
    "periodic_stable":
        {"ampl_pct": 0.25,   "noise_sd": 0.1, "period": 5},  # 振幅=25 %
    # "volatile_rising":
    #     {"slope_pct": 0.08,  "noise_sd": 0.1},  # 8 %/年 + 大噪声
    # "volatile_falling":
    #     {"slope_pct": 0.06,  "noise_sd": 0.1},  # −6%/year ± noise
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

# 新增：控制 subject 数量分布的权重
# 权重越高，该数量的 subject 被选中的概率越大
SUBJECT_COUNT_WEIGHTS = {
    1: 4, # 给 1-4 个 subject 更高的权重
    2: 4,
    3: 4,
    4: 4,
    5: 1, # 给 5-7 个 subject 较低的权重
    6: 1,
    7: 1,
}
# --- 新增配置参数结束 ---

# 新增：为双峰/双谷趋势定义默认的随机化参数
BIMODAL_PEAK_RANDOM_PARAMS = {
    'peak_amplitude_range': (1.5, 2.5),  # 峰的幅度是基线的 1.5 到 2.5 倍
    'peak_position_range1': (0.1, 0.4),  # 第一个峰出现在前 10% 到 40% 的位置
    'peak_position_range2': (0.6, 0.9),  # 第二个峰出现在后 60% 到 90% 的位置
    'peak_std_dev_range': (2, 5),        # 峰的宽度 (标准差) 在 2 到 5 之间
    'p_noise_sd': 0.1
}

BIMODAL_VALLEY_RANDOM_PARAMS = {
    'valley_depth_range': (0.8, 1.2),   # 谷的深度是基线的 0.8 到 1.2 倍
    'valley_position_range1': (0.15, 0.45), # 第一个谷的位置范围
    'valley_position_range2': (0.55, 0.85), # 第二个谷的位置范围
    'valley_std_dev_range': (3, 6),         # 谷的宽度范围
    'p_noise_sd': 0.1
}

def generate_stable_rising(years, params):
    base = params['base']
    p = TREND_TYPES["stable_rising"]
    slope = base * p["slope_pct"]  # 每年固定增幅
    # Added noise handling
    return [max(0.01 * base, base + slope * (y - years[0]) + base * params.get('noise', 0) * np.random.normal(0, p["noise_sd"])) for y in years]

def generate_stable_falling(years, params):
    base = params['base']
    p = TREND_TYPES["stable_falling"]

    # 核心修改：将 slope 设置为负值，以实现下降
    # 如果 p["slope_pct"] 是一个正数（表示下降的幅度），则需要乘以 -1
    slope = -base * p["slope_pct"]  # 每年固定降幅

    results = []
    for y in years:
        # 计算没有噪音的趋势值
        trend_value = base + slope * (y - years[0])

        # 处理噪音
        noise_value = 0
        # 只有当 params 中明确提供了 'noise' 并且其值大于0时才应用噪音
        if params.get('noise', 0) > 0:
            # 噪音应该围绕趋势值波动，所以是加或减
            # np.random.normal(0, p["noise_sd"]) 会生成正负随机数
            noise_value = base * params['noise'] * np.random.normal(0, p["noise_sd"])

        current_value = trend_value + noise_value

        # 确保值不会低于基数的 1% (或您定义的最小值)
        results.append(max(0.01 * base, current_value))

    return results

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
    """
    生成具有随机位置和峰值的双峰趋势。

    Args:
        years (list): 年份列表。
        params (dict): 包含以下键的参数字典：
            'base': 基线值。
            'peak_amplitude_range': 一个元组 (min, max)，定义峰值幅度的随机范围。
            'peak_position_range1': 一个元组 (min, max)，定义第一个峰值位置的随机范围 (以比例表示，例如 (0.1, 0.4))。
            'peak_position_range2': 一个元组 (min, max)，定义第二个峰值位置的随机范围 (以比例表示，例如 (0.6, 0.9))。
            'peak_std_dev_range': 一个元组 (min, max)，定义峰宽度的随机范围。
            'noise': 噪声水平。
            'min_value_floor' (optional): 最小值的下限，默认为基线的 1%。
    """
    base = params['base']

    # 从参数中获取随机范围
    min_amp, max_amp = params['peak_amplitude_range']
    min_pos1, max_pos1 = params['peak_position_range1']
    min_pos2, max_pos2 = params['peak_position_range2']
    min_std, max_std = params['peak_std_dev_range']

    # 1. 随机化峰值幅度
    peak1_amplitude = base * random.uniform(min_amp, max_amp)
    peak2_amplitude = base * random.uniform(min_amp, max_amp)

    # 2. 随机化峰值位置
    num_years = len(years)
    peak1_year_pos = years[0] + int(num_years * random.uniform(min_pos1, max_pos1))
    peak2_year_pos = years[0] + int(num_years * random.uniform(min_pos2, max_pos2))

    # 3. 随机化峰值宽度
    peak1_std_dev = random.uniform(min_std, max_std)
    peak2_std_dev = random.uniform(min_std, max_std)

    # 4. 其他参数
    min_value_floor = params.get('min_value_floor', 0.01 * base)
    noise_level = params.get('noise', 0)
    # 假设 p["noise_sd"] 是一个预定义的值，这里为了可运行性设为 0.1
    p_noise_sd = params.get('p_noise_sd', 0.1)

    values = []
    for year in years:
        # 第一个峰的高斯贡献
        gauss1 = peak1_amplitude * np.exp(-((year - peak1_year_pos) ** 2) / (2 * peak1_std_dev ** 2))
        # 第二个峰的高斯贡献
        gauss2 = peak2_amplitude * np.exp(-((year - peak2_year_pos) ** 2) / (2 * peak2_std_dev ** 2))

        # 噪声项
        noise_val = base * noise_level * np.random.normal(0, p_noise_sd)

        # 总和：基线 + 两个峰的贡献 + 噪声
        current_value = base + gauss1 + gauss2 + noise_val

        # 应用最小值限制
        values.append(max(min_value_floor, current_value))

    return values

def generate_bimodal_valley(years, params):
    """
    生成具有随机位置和谷值的双谷趋势。

    Args:
        years (list): 年份列表。
        params (dict): 包含以下键的参数字典：
            'base': 基线值。
            'valley_depth_range': 一个元组 (min, max)，定义谷深的随机范围。
            'valley_position_range1': 一个元组 (min, max)，定义第一个谷位置的随机范围 (以比例表示)。
            'valley_position_range2': 一个元组 (min, max)，定义第二个谷位置的随机范围 (以比例表示)。
            'valley_std_dev_range': 一个元组 (min, max)，定义谷宽度的随机范围。
            'noise': 噪声水平。
            'min_value_floor' (optional): 最小值的下限，默认为基线的 0.1%。
    """
    base = params['base']

    # 从参数中获取随机范围
    min_depth, max_depth = params['valley_depth_range']
    min_pos1, max_pos1 = params['valley_position_range1']
    min_pos2, max_pos2 = params['valley_position_range2']
    min_std, max_std = params['valley_std_dev_range']

    # 1. 随机化谷的深度
    valley1_depth = base * random.uniform(min_depth, max_depth)
    valley2_depth = base * random.uniform(min_depth, max_depth)

    # 2. 随机化谷的位置
    num_years = len(years)
    valley1_year_pos = years[0] + int(num_years * random.uniform(min_pos1, max_pos1))
    valley2_year_pos = years[0] + int(num_years * random.uniform(min_pos2, max_pos2))

    # 3. 随机化谷的宽度
    valley1_std_dev = random.uniform(min_std, max_std)
    valley2_std_dev = random.uniform(min_std, max_std)

    # 4. 其他参数
    min_value_floor = params.get('min_value_floor', 0.001 * base)
    noise_level = params.get('noise', 0.05)
    # 假设 p["noise_sd"] 是一个预定义的值，这里为了可运行性设为 0.1
    p_noise_sd = params.get('p_noise_sd', 0.1)

    values = []
    for year in years:
        # 第一个谷的高斯贡献 (从基线减去)
        gauss1_contribution = valley1_depth * np.exp(-((year - valley1_year_pos) ** 2) / (2 * valley1_std_dev ** 2))
        # 第二个谷的高斯贡献 (从基线减去)
        gauss2_contribution = valley2_depth * np.exp(-((year - valley2_year_pos) ** 2) / (2 * valley2_std_dev ** 2))

        # 噪声项
        noise_val = base * noise_level * np.random.normal(0, p_noise_sd)

        # 总和：基线 - 两个谷的贡献 + 噪声
        current_value = base - gauss1_contribution - gauss2_contribution + noise_val

        # 应用最小值限制
        values.append(max(min_value_floor, current_value))

    return values


def generate_data(years, params, trend_type):
    generators = {
        "stable_rising": generate_stable_rising,
        "stable_falling": generate_stable_falling,
        # "exponential_rising": generate_exponential_rising,
        # "exponential_falling": generate_exponential_falling,
        "periodic_stable": generate_periodic_stable,
        # "volatile_rising": generate_volatile_rising,
        # "volatile_falling": generate_volatile_falling,
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

# --- 准备所有可用的指标配置列表 ---
all_metric_configs = []
for theme, metrics in THEME_METRIC_PARAMS.items():
    for metric_config in metrics:
        all_metric_configs.append((theme, metric_config))

# 检查是否有可用的配置
if not all_metric_configs:
    print("Error: No metric configurations found in THEME_METRIC_PARAMS in data.py.")
    exit()

# --- 主循环，生成指定数量的文件 ---
TARGET_FILE_COUNT = 10000

print(f"Generating {TARGET_FILE_COUNT} CSV files...")

MIN_YEARS_FOR_BIMODAL = 15

# --- 全局状态变量用于逻辑2：优先选取没有的趋势 ---
_GLOBAL_TREND_POOL = []  # 存储所有趋势类型，用于循环选取
_GLOBAL_TREND_INDEX = 0  # 当前全局趋势池的索引


def initialize_global_trend_pool():
    """初始化或重置全局趋势池，并打乱顺序。"""
    global _GLOBAL_TREND_POOL, _GLOBAL_TREND_INDEX
    _GLOBAL_TREND_POOL = list(TREND_TYPES.keys())
    random.shuffle(_GLOBAL_TREND_POOL)
    _GLOBAL_TREND_INDEX = 0


# 在脚本开始时调用一次，初始化全局趋势池
initialize_global_trend_pool()

# --- 新增的后处理函数 ---
def post_process_lowest_initial_value(all_data, trends, years, original_params, num_years, unit, trends_to_choose_from):
    """
    后处理函数：识别初值最低的线，如果其趋势不是上升趋势，则强制其为上升趋势。

    Args:
        all_data (list of list): 包含所有数据线的原始值 (未四舍五入或应用重叠处理)。
        trends (list): 包含每条线对应趋势类型的列表。
        years (list): 年份列表。
        original_params (dict): 原始的 metric 参数字典 (包含 'base', 'noise')。
        num_years (int): 数据跨越的年份数量。
        unit (str): 数据单位 (例如 '%', '$')。
        trends_to_choose_from (list): 当前文件可用的趋势类型列表 (已考虑年份长度)。

    Returns:
        tuple: (modified_all_data, modified_trends)
    """
    if not all_data:
        return [], []

    min_initial_value = float('inf')
    lowest_idx = -1

    # 1. 找出初值最低的线
    for idx, line_values in enumerate(all_data):
        if line_values and line_values[0] < min_initial_value:
            min_initial_value = line_values[0]
            lowest_idx = idx

    if lowest_idx == -1: # 没有找到数据，或者数据为空
        return all_data, trends

    current_trend = trends[lowest_idx]

    # 定义所有可能的上升趋势类型
    rising_trends_pool = ['stable_rising']
    # 过滤出当前文件可用的上升趋势 (已考虑年份长度)
    available_rising_trends = [t for t in rising_trends_pool if t in trends_to_choose_from]

    # 2. 检查其趋势是否为上升趋势
    if current_trend not in available_rising_trends:
        print(f"Correction needed for subject '{selected_subjects[lowest_idx]}': "
              f"Initial value {min_initial_value:.2f}, current trend '{current_trend}'.")

        if not available_rising_trends:
            print("Warning: No available rising trends for correction. Skipping correction.")
            return all_data, trends # 无法纠正，直接返回

        # 3. 如果不是，选择一个新的上升趋势
        new_trend = random.choice(available_rising_trends)
        print(f"Changing trend to '{new_trend}'.")

        # 4. 重新生成这条线的数据
        # 复制原始参数，并调整 base 以便生成的数据初值接近 min_initial_value
        recalc_params = original_params.copy()
        # 尝试将 base 设置为期望的初值，让 generate_data 在此基础上加上噪声和趋势
        recalc_params['base'] = min_initial_value

        # 重新生成噪声因子，与之前的生成方式保持一致
        if 'noise' in original_params:
            recalc_params['noise'] = original_params['noise'] * random.uniform(0.8, 2.5)
        else:
            recalc_params['noise'] = random.uniform(0.05, 0.8)

        if new_trend == 'bimodal_peak':
            recalc_params.update(BIMODAL_PEAK_RANDOM_PARAMS)
        elif new_trend == 'bimodal_valley':
            recalc_params.update(BIMODAL_VALLEY_RANDOM_PARAMS)

        # 生成新的数据
        new_line_values = generate_data(years, recalc_params, new_trend)

        # 5. 覆盖原有数据和趋势
        all_data[lowest_idx] = new_line_values
        trends[lowest_idx] = new_trend
        print(f"Subject '{selected_subjects[lowest_idx]}' data regenerated with trend '{new_trend}'. "
              f"New initial value: {new_line_values[0]:.2f}")

    return all_data, trends

# --- 主文件生成循环 ---
for i in range(TARGET_FILE_COUNT):
    # --- 修改后的逻辑：持续随机选择，直到找到一个单位不是 '%' 的指标 ---
    while True:
        # 从所有配置中随机选择一个
        selected_theme, selected_metric = random.choice(all_metric_configs)
        # 检查其单位是否为 '%'
        if selected_metric.get('unit') != '%':
            # 如果单位不是 '%'，则选择成功，跳出循环
            break
        # 如果是 '%'，循环将继续，重新选择下一个

    # 提取指标参数
    metric_name = selected_metric['name']
    unit = selected_metric['unit']
    subjects = selected_metric['subject']
    params = selected_metric['params']

    # 根据权重随机选择 subject 数量 k (1-7)
    max_k = min(7, len(subjects))
    possible_k_values = list(range(1, max_k + 1))
    actual_weights = [SUBJECT_COUNT_WEIGHTS[val] for val in possible_k_values]
    k = random.choices(possible_k_values, weights=actual_weights, k=1)[0]

    # 随机选择 k 个 subject
    selected_subjects = random.sample(subjects, k)

    # 生成年份数据（随机7-30个连续年份）
    num_years = random.randint(7, 30)
    start_year = random.randint(1950, 2025 - num_years)
    years = list(range(start_year, start_year + num_years))

    # --- 根据规则确定当前文件可用的趋势类型 ---
    # 默认情况下，所有趋势都可用
    trends_to_choose_from = list(TREND_TYPES.keys())

    # 规则1 (新)：如果数据线的数量 k 大于 1，则从可选列表中移除 'bimodal_valley'
    if k > 1:
        # 仅在 k > 1 时应用此过滤
        trends_to_choose_from = [t for t in trends_to_choose_from if t != 'bimodal_valley']
        print(f"Info: k={k} (>1), 'bimodal_valley' is excluded.")  # (可选) 打印信息用于调试

    # 规则2 (旧)：如果年份长度太短，则排除所有双峰/双谷类型
    if num_years < MIN_YEARS_FOR_BIMODAL:
        # 从可能已经过规则1筛选的列表中，进一步筛选
        trends_to_choose_from = [t for t in trends_to_choose_from if t not in ['bimodal_peak', 'bimodal_valley']]

    # 极端情况处理：如果经过层层筛选后，一个可选的趋势都没有了
    if not trends_to_choose_from:
        print(
            f"Warning: No trends available after filtering (k={k}, num_years={num_years}). Falling back to all possible trends for file {i + 1}.")
        # 回退到最初始的所有趋势列表，以避免程序崩溃
        trends_to_choose_from = list(TREND_TYPES.keys())

    # --- 逻辑1：确定哪些 subject 将是“初值偏小”的，并设置其趋势和 base 调整范围 ---
    # 定义上升趋势类型池
    rising_trends_pool = ['stable_rising', 'single_peak', 'bimodal_peak']
    # 实际可用的上升趋势 (根据年份长度过滤)
    rising_trends_actual = [t for t in rising_trends_pool if t in trends_to_choose_from]

    # 随机选择一个 subject 索引来强制其初值最低且为上升趋势
    # 只有当 k > 0 且存在可用的上升趋势时才执行此逻辑
    forced_lowest_rising_idx = -1
    if k > 0 and len(rising_trends_actual) > 0:
        forced_lowest_rising_idx = random.choice(range(k))

    # 为每个 subject 准备配置：包括 base 乘数范围、允许的趋势类型和优先级
    subject_configs = []  # 存储 {'base_multiplier_range': (min, max), 'allowed_trends': list, 'priority': int}
    for idx in range(k):
        config = {'trend': None}  # 预留给最终确定的趋势
        if idx == forced_lowest_rising_idx:
            # 强制初值最低且上升趋势：base 乘数极低，只允许上升趋势，最高优先级
            config['base_multiplier_range'] = (0.4, 0.8)  # 极低的基准值范围
            config['allowed_trends'] = rising_trends_actual
            config['priority'] = 2  # 最高优先级，确保它能获得上升趋势
        else:
            # 正常 subject：base 乘数正常，允许所有趋势，较低优先级
            config['base_multiplier_range'] = (0.9, 1.1)
            config['allowed_trends'] = trends_to_choose_from
            config['priority'] = 1  # 正常优先级
        subject_configs.append(config)


    # --- 逻辑2：优先选取没有的趋势，并确保 k 个 subject 的趋势是不同的 ---
    # 目标：从 _GLOBAL_TREND_POOL 中为这 k 个 subject 选取 k 个不同的趋势，
    # 且满足 subject_configs 中每个 subject 的 'allowed_trends' 约束。

    trends = [None] * k  # 最终为每个 subject 确定的趋势列表
    picked_trends_for_this_file = set()  # 记录当前文件中已选取的趋势，保证不重复

    # 优先处理“初值偏小”的 subject，因为它们的趋势选择范围更窄
    # 将 subject_configs 按照“初值偏小”的优先级排序，先处理限制更多的
    # (subject_idx, config_dict)
    sorted_subject_configs_with_indices = sorted(enumerate(subject_configs),
                                                 key=lambda x: x[1]['allowed_trends'] == rising_trends_actual,
                                                 reverse=True)

    for subj_idx, config in sorted_subject_configs_with_indices:
        chosen_trend = None
        # 尝试从全局趋势池的当前位置开始选取
        temp_global_index = _GLOBAL_TREND_INDEX  # 临时索引，用于当前文件的选取

        # 尝试从全局趋势池中，优先选取允许且未在当前文件中使用的趋势
        found_in_global_pool = False
        # 遍历全局池，从当前索引开始
        for attempt_idx in range(len(_GLOBAL_TREND_POOL)):
            current_idx = (temp_global_index + attempt_idx) % len(_GLOBAL_TREND_POOL)
            potential_trend = _GLOBAL_TREND_POOL[current_idx]

            if potential_trend in config['allowed_trends'] and potential_trend not in picked_trends_for_this_file:
                chosen_trend = potential_trend
                _GLOBAL_TREND_INDEX = (current_idx + 1) % len(_GLOBAL_TREND_POOL)  # 更新全局索引
                found_in_global_pool = True
                break

        if not found_in_global_pool:
            # 如果在全局池中没有找到新的、允许的趋势（可能是因为所有允许的都已在当前文件中被其他 subject 选取）
            # 此时，我们必须允许重复，但仍然优先选择允许的趋势
            # 这种情况在 k <= len(trends_to_choose_from) 的前提下，且所有 rising_trends_actual 都被其他 low-start subjects 占用时发生
            available_allowed_trends = [t for t in config['allowed_trends'] if t not in picked_trends_for_this_file]
            if available_allowed_trends:
                chosen_trend = random.choice(available_allowed_trends)
            else:
                # 理论上，如果 k <= len(trends_to_choose_from)，且 max_low_start 限制了 low-start 数量，
                # 应该总能找到一个不重复的趋势。如果到这里还找不到，说明逻辑有边界问题或需要允许重复。
                # 暂时允许从所有允许的趋势中重复选择一个
                chosen_trend = random.choice(config['allowed_trends'])
                # print(f"Warning: Subject {subj_idx} forced to pick a trend already used in this file for distinctness.")

        trends[subj_idx] = chosen_trend
        picked_trends_for_this_file.add(chosen_trend)
        # 将趋势存入 subject_configs，以便后续使用其 base_multiplier
        subject_configs[subj_idx]['trend'] = chosen_trend

    # 生成每个subject的数据
    all_data = []
    # 遍历为每个 subject 确定的趋势和其配置
    for idx, subj_conf in enumerate(subject_configs):
        trend_type = subj_conf['trend']

        # 为当前 subject 复制一份 params，并根据配置调整 base 和 noise
        subj_params = params.copy()

        # 应用初值偏小逻辑确定的 base 乘数
        base_min, base_max = subj_conf['base_multiplier_range']
        if unit == '%':
            # 对于百分比单位，确保 base 不会过高或过低，留出足够的波动空间
            subj_params['base'] = params['base'] * random.uniform(base_min, base_max)
            # 进一步限制百分比 base 在合理范围内，例如1%到99%
            subj_params['base'] = max(1.0, min(99.0, subj_params['base']))
        else:
            # 其他单位按原逻辑
            subj_params['base'] = params['base'] * random.uniform(base_min, base_max)
            # 确保 base 非负且不为零，避免潜在问题
            subj_params['base'] = max(0.1, subj_params['base'])

        # 随机调整 noise
        if 'noise' in params:
            # 增大 noise 的浮动范围，让趋势更明显
            subj_params['noise'] = params['noise'] * random.uniform(0.8, 2.5)
        else:
            # 如果data.py中没有noise，给一个更大的默认随机范围
            subj_params['noise'] = random.uniform(0.05, 0.8)

        # 为双峰/双谷趋势添加所需的随机范围参数
        if trend_type == 'bimodal_peak':
            subj_params.update(BIMODAL_PEAK_RANDOM_PARAMS)
        elif trend_type == 'bimodal_valley':
            subj_params.update(BIMODAL_VALLEY_RANDOM_PARAMS)

        values = generate_data(years, subj_params, trend_type)

        # 对于百分比数据，如果最大值超过100，则进行归一化
        if unit == '%':
            max_val = max(values)
            if max_val > 0.0 and max_val > 100.0:
                scale_factor = 100.0 / max_val
                scaled_values = [v * scale_factor for v in values]
                all_data.append(scaled_values)
            else:
                all_data.append(values)
        else:
            all_data.append(values)

    # --- 调用后处理函数 ---
    # `trends_to_choose_from` 包含了当前文件所有可用的趋势，包括双峰趋势是否可用。
    # 这对于 `post_process_lowest_initial_value` 函数过滤 `available_rising_trends` 很重要。
    corrected_all_data, corrected_trends = post_process_lowest_initial_value(
        all_data, trends, years, params, num_years, unit, trends_to_choose_from
    )

    # 调用重叠处理函数
    processed_data = apply_overlap_shift(
        all_data,
        NUM_INITIAL_POINTS_CHECK,
        OVERLAP_THRESHOLD_FACTOR,
        SHIFT_RANGE_FACTOR
    )

    # 根据单位类型处理数据并四舍五入
    final_data = []
    for line_values in processed_data:
        if unit == '%':
            # 限制在0-100%
            final_data.append([round(max(0, min(100, v)), 2) for v in line_values])
        elif unit == 'ratio':
            # 比率不能为负
            final_data.append([round(max(0, v), 2) for v in line_values])
        else:
            # 其他单位只保证非负
            final_data.append([round(max(0, v), 2) for v in line_values])

    # 转置数据为按年排列
    data_by_year = list(zip(*final_data))

    # 构建文件名: 类型_主题_序号.csv
    theme_slug = selected_theme.lower().replace(' ', '_')
    filename = f'csv/line_{theme_slug}_{i + 1}.csv'

    # 写入CSV文件
    try:
        os.makedirs('csv', exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 第一行：主题，指标，单位
            writer.writerow([selected_theme, metric_name, unit])
            # 第二行：年份头
            writer.writerow(['Year'] + selected_subjects)
            # 第三行：趋势类型 (使用最终确定的 trends 列表)
            writer.writerow(['trend'] + trends)
            # 数据行
            for year, values in zip(years, data_by_year):
                writer.writerow([year] + list(values))

        if (i + 1) % 100 == 0 or (i + 1) == TARGET_FILE_COUNT:
            print(f"Generated {i + 1}/{TARGET_FILE_COUNT} files...")

    except IOError as e:
        print(f"Error writing file {filename}: {e}")

print(f"Finished generating {TARGET_FILE_COUNT} CSV files in the 'csv' directory.")

# for i in range(TARGET_FILE_COUNT):
#     # 随机选择一个指标配置作为模板
#     selected_theme, selected_metric = random.choice(all_metric_configs)
#
#     # 提取指标参数
#     metric_name = selected_metric['name']
#     unit = selected_metric['unit']
#     subjects = selected_metric['subject']
#     params = selected_metric['params']
#
#     # --- 修改：根据权重随机选择 subject 数量 k (1-7) ---
#     max_k = min(7, len(subjects))
#     possible_k_values = list(range(1, max_k + 1))
#     actual_weights = [SUBJECT_COUNT_WEIGHTS[val] for val in possible_k_values]
#     k = random.choices(possible_k_values, weights=actual_weights, k=1)[0]
#
#     # 随机选择 k 个 subject (使用新的 k 值)
#     selected_subjects = random.sample(subjects, k)
#
#     # 生成年份数据（随机7-30个连续年份）
#     num_years = random.randint(7, 30)  # num_years 在这里确定
#     start_year = random.randint(1950, 2025 - num_years)
#     years = list(range(start_year, start_year + num_years))
#
#     # --- 新增逻辑：为每个 subject 动态选择趋势，考虑年份长度对双峰的影响 ---
#     # 获取所有可用的趋势类型
#     all_possible_trends = list(TREND_TYPES.keys())
#
#     # 检查当前年份长度是否适合双峰类型
#     if num_years < MIN_YEARS_FOR_BIMODAL:
#         # 如果年份太短且双峰是可选趋势，则排除双峰
#         # 创建一个不包含 'bimodal_peak' 、'bimodal_valley'的趋势列表
#         trends_to_choose_from = [t for t in all_possible_trends if t != 'bimodal_peak' and t != 'bimodal_valley']
#
#         # 极端情况处理：如果排除 'bimodal_peak' 后没有其他趋势了（理论上不应该发生）
#         if not trends_to_choose_from:
#             print(
#                 f"Warning: Only 'bimodal_peak' trend available and years ({num_years}) are too short for it. Selecting 'bimodal_peak' anyway for file {i + 1}.")
#             trends_to_choose_from = all_possible_trends  # 退而求其次，还是选它
#     else:
#         # 年份长度足够，或者 'bimodal_peak' 不在可选趋势中，则所有趋势都可选
#         trends_to_choose_from = all_possible_trends
#
#     # --- 修改开始 ---
#     # 为每个 subject 随机选择一个趋势，确保不重复
#     # k (subject数量) 总是小于或等于 trends_to_choose_from 的长度，所以 random.sample 不会报错
#     trends = random.sample(trends_to_choose_from, k)
#     # --- 修改结束 ---
#
#     # 生成每个subject的数据
#     all_data = []
#     # 遍历为每个 subject 确定的趋势
#     for trend_type in trends:  # 使用 trend_type 变量名更清晰
#         # 为当前 subject 复制一份 params，并随机调整 base 和 noise
#         subj_params = params.copy()
#         # 随机调整 base
#         if unit == '%':
#             # 对于百分比单位，确保 base 不会过高，留出足够的波动空间
#             # 例如，让 base 在原始值的 0.3 到 1.0 倍之间，或者直接限制在一个百分比范围内
#             # 假设原始 base 是一个相对合理的值，我们让它有更多向下浮动的空间
#             subj_params['base'] = params['base'] * random.uniform(0.3, 1.0)
#             # 也可以直接设定一个百分比范围，例如 40% 到 90%
#             # subj_params['base'] = random.uniform(40.0, 90.0)
#         else:
#             # 其他单位按原逻辑
#             subj_params['base'] = params['base'] * random.uniform(0.5, 1.5)
#
#         # 随机调整 noise
#         if 'noise' in params:
#             # 增大 noise 的浮动范围，让趋势更明显
#             subj_params['noise'] = params['noise'] * random.uniform(0.8, 2.5)  # 比如从 0.8 倍到 2.5 倍
#         else:
#             # 如果data.py中没有noise，给一个更大的默认随机范围
#             subj_params['noise'] = random.uniform(0.05, 0.8)
#
#         values = generate_data(years, subj_params, trend_type)  # 传入确定的趋势类型
#
#         # --- 新增核心逻辑：对于百分比数据，如果最大值超过100，则进行归一化 ---
#         if unit == '%':
#             max_val = max(values)
#             # 确保 max_val 不为0，避免除以零错误
#             if max_val > 0.0 and max_val > 100.0:  # 如果最大值大于100，则进行缩放
#                 scale_factor = 100.0 / max_val
#                 scaled_values = [v * scale_factor for v in values]
#                 all_data.append(scaled_values)
#             else:
#                 # 如果最大值没有超过100，或者为0，则直接使用原始值
#                 all_data.append(values)
#         else:
#             # 非百分比单位直接添加原始值
#             all_data.append(values)
#
#     # 调用新增的重叠处理函数
#     processed_data = apply_overlap_shift(
#         all_data,
#         NUM_INITIAL_POINTS_CHECK,
#         OVERLAP_THRESHOLD_FACTOR,
#         SHIFT_RANGE_FACTOR
#     )
#
#     # 根据单位类型处理数据并四舍五入
#     final_data = []
#     for line_values in processed_data:
#         if unit == '%':
#             # 限制在0-100%
#             final_data.append([round(max(0, min(100, v)), 2) for v in line_values])
#         elif unit == 'ratio':
#             # 比率不能为负
#             final_data.append([round(max(0, v), 2) for v in line_values])
#         else:
#             # 其他单位只保证非负（如果需要）
#             final_data.append([round(max(0, v), 2) for v in line_values])  # 假设一般数据非负
#
#     # 转置数据为按年排列
#     data_by_year = list(zip(*final_data))
#
#     # 构建文件名: 类型_主题_序号.csv
#     theme_slug = selected_theme.lower().replace(' ', '_')
#     filename = f'csv/line_{theme_slug}_{i + 1}.csv'  # 使用 i+1 作为唯一的序号
#
#     # 写入CSV文件
#     try:
#         # 确保 'csv' 目录存在
#         os.makedirs('csv', exist_ok=True)
#
#         with open(filename, 'w', newline='') as f:
#             writer = csv.writer(f)
#             # 第一行：主题，指标，单位
#             writer.writerow([selected_theme, metric_name, unit])
#             # 第二行：年份头
#             writer.writerow(['Year'] + selected_subjects)
#             # 第三行：趋势类型
#             writer.writerow(['trend'] + trends)  # 使用正确生成的 trends 列表
#             # 数据行
#             for year, values in zip(years, data_by_year):
#                 writer.writerow([year] + list(values))
#
#         # 打印进度
#         if (i + 1) % 100 == 0 or (i + 1) == TARGET_FILE_COUNT:
#             print(f"Generated {i + 1}/{TARGET_FILE_COUNT} files...")
#
#     except IOError as e:
#         print(f"Error writing file {filename}: {e}")
#
# print(f"Finished generating {TARGET_FILE_COUNT} CSV files in the 'csv' directory.")
