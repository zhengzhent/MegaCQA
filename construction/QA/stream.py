# File: stacked_area_QA.py
# Description: Generates QA files for stream chart data based on CSV output.
from pathlib import Path
import collections
import pandas as pd
import os
import json
import numpy as np
from typing import Dict, Any, Tuple, List, Union, Optional
import re
import random # Import random for selections
from typing import List, Dict, Any, Tuple # Import typing hints
import math # Import math for isnan
from sklearn.neighbors import NearestNeighbors # Import for KNN density calculation # Keep imports, even if not all are used in the final version

# --- Utility Functions (Adapted from scatter_QA.py and heatmap_QA.py) ---
# Keep metadata reading as is, it works for identifying series and overall info
def read_line_metadata(filepath: str) -> Dict[str, Any]:
    """
    读取折线图 CSV 的前三行元数据，返回结构示例：
    {
        'topic'        : 'Agriculture and Food Production',
        'little_theme' : 'Crop Yield',
        'y_info'       : {'unit': 'tons/hectare'},
        'x_info'       : {'name': 'Year', 'unit': ''},
        'series_names' : ['Golden Harvest Cooperative', 'Starfall Organics'],
        'series_trends': {               # 与 series_names 一一对应 - NOTE: Trends are ignored for streams
            'Golden Harvest Cooperative': 'stable_falling',
            'Starfall Organics'         : 'periodic_stable'
        }
    }
    """
    try:
        import csv
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [next(reader) for _ in range(3)]
        meta_df = pd.DataFrame(rows)    # 不同长度的行也能放进来

        if len(meta_df) < 3:
            print(f"[WARN] 元数据不足 3 行 → {filepath}")
            return {}

        # 第 1 行：大标题、小标题、Y 轴单位
        line1: List[Any] = meta_df.iloc[0].tolist()
        topic        = (line1[0] or "").strip() if len(line1) > 0 else ""
        little_theme = (line1[1] or "").strip() if len(line1) > 1 else ""
        y_unit       = (line1[2] or "").strip() if len(line1) > 2 else ""

        # 第 3 行：X 轴名称 + 各折线系列名称
        line3: List[Any] = meta_df.iloc[2].tolist()
        x_name       = (line3[0] or "").strip() if len(line3) > 0 else ""
        series_names = [str(c).strip() for c in line3[1:] if pd.notna(c)]

        # 第 2 行：趋势标签；第一个单元格通常是 "trend" - Ignored for stream
        line2: List[Any] = meta_df.iloc[1].tolist()
        trend_values = line2[1:] if len(line2) > 1 else []
        # 若趋势数量不足，用 None 补齐
        trend_values += [None] * (len(series_names) - len(trend_values))
        series_trends = dict(zip(series_names, [str(t).strip() if pd.notna(t) else None
                                                for t in trend_values]))

        # 组装输出字典
        meta: Dict[str, Any] = {
            "topic": topic,
            "little_theme": little_theme,
            "y_info": {"unit": y_unit},
            "x_info": {"name": x_name, "unit": ""},
            "series_names": series_names,
            "series_trends": series_trends, # Keep in metadata, but ignore in QA generation
        }
        return meta

    except Exception as e:
        print(f"[ERROR] 读取元数据失败：{filepath} → {e}")
        return {}

# Keep data reading as is, tidy format is needed for aggregation
def read_line_data_df(filepath: str, metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    读取 CSV 中真正的数据区（元数据 3 行之后），并返回 tidy 格式：
        series | x | y

    兼容两种常见排布：
      ① 列式  Year,SeriesA,SeriesB,...
      ② 行式  SeriesA,1990,1991,1992,...

    解析逻辑：
      - 跳过前 3 行
      - 尝试判断列式 / 行式
      - 统一转换为长表，列名固定为 ['series','x','y']
    """
    try:
        # ---------- 1. 读取原始数据 ----------
        raw = pd.read_csv(filepath, header=None, skiprows=3, encoding="utf-8")
        if raw.empty:
            print(f"[WARN] 文件无数据区 → {filepath}")
            return None

        # ---------- 2. 判断排布方式 ----------
        # 尝试把第一列整体转换成数字，若成功率较高 → 认为是“列式”
        first_col_numeric = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        col_layout = "column" if first_col_numeric.notna().sum() >= len(raw) * 0.8 else "row"

        if col_layout == "column":
            # ====== 3A. 列式处理 ======
            # 给 DataFrame 起列名：X 名称 + 系列列表
            x_name = metadata.get("x_info", {}).get("name", "X")
            series_names = metadata.get("series_names", [])
            # 若列数对不上，补足或截断
            col_names = [x_name] + series_names
            if len(col_names) < raw.shape[1]:
                col_names += [f"series_{i}" for i in range(len(col_names), raw.shape[1])]
            col_names = col_names[: raw.shape[1]]
            raw.columns = col_names

            # melt 成长表
            df_long = raw.melt(id_vars=[x_name], var_name="series", value_name="y")

            # 清洗数值列
            df_long["y"] = pd.to_numeric(df_long["y"], errors="coerce")
            df_long = df_long.dropna(subset=["y"])

            # 统一列名
            df_long = df_long.rename(columns={x_name: "x"})
            return df_long.reset_index(drop=True)

        else:
            # ====== 3B. 行式处理 ======
            # 第 1 行作为“x 轴表头”，其余行每行一个 series
            header = raw.iloc[0].tolist()
            x_vals = header[1:]                                    # e.g. 年份们
            rows = []
            for idx in range(1, len(raw)):
                row = raw.iloc[idx].tolist()
                series_name = str(row[0])
                y_vals = row[1:]
                # 补齐 / 截断
                y_vals += [None] * (len(x_vals) - len(y_vals))
                for x, y in zip(x_vals, y_vals):
                    rows.append({"series": series_name, "x": x, "y": y})

            df_long = pd.DataFrame(rows)

            def _parse_x(val):
                # 纯数字、介于 1000–3000 ⇒ 认为是年份
                try:
                    num = int(float(val))
                    if 1000 <= num <= 3000:
                        return num
                except Exception:
                    pass
                # 其它情况再尝试解析完整日期
                try:
                    return pd.to_datetime(val, format="%Y-%m-%d", errors="raise")
                except Exception:
                    try:
                        # 只有年份的字符串，如 "1990"
                        return pd.to_datetime(val, format="%Y", errors="raise")
                    except Exception:
                        return val  # 保底：原样返回

            df_long["x"] = df_long["x"].apply(_parse_x)
            df_long["y"] = pd.to_numeric(df_long["y"], errors="coerce")
            df_long = df_long.dropna(subset=["y"])
            return df_long.reset_index(drop=True)

    except Exception as e:
        print(f"[ERROR] 读取数据失败：{filepath} → {e}")
        return None


# --- Calculation Functions (Specific to stream Chart) ---

# New function to calculate the total value at each x point
def task_compute_total_value(df_long: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    计算每个 x 值对应的总 y 值 (堆叠面积图的总高度)。
    返回 DataFrame: ['x', 'total_y']
    """
    if df_long is None or df_long.empty:
        return None
    # Group by x and sum y, dropping NaN sums
    df_total = df_long.groupby("x")["y"].sum().reset_index()
    df_total = df_total.rename(columns={"y": "total_y"}).dropna(subset=["total_y"])
    return df_total

# New function to calculate total contribution per series
def task_compute_total_contribution_per_series(df_long: pd.DataFrame) -> Dict[str, float]:
    """
    计算每个 series 的总贡献 (所有 x 值的 y 之和)。
    返回 {series_name: total_contribution}
    """
    if df_long is None or df_long.empty:
        return {}
    # Group by series and sum y, dropping NaN sums
    contribution = df_long.groupby("series")["y"].sum().dropna()
    return contribution.to_dict()


# Keep task_count_points - gives total points and points per series
def task_count_points(df_long: pd.DataFrame, by_series: bool = False):
    """
    统计数据点数量。
    如果 by_series=True，返回 {series: count, ...}
    否则返回整数总数。
    """
    if df_long is None or df_long.empty:
        return 0 if not by_series else {}

    if by_series:
        return df_long.groupby("series").size().to_dict()
    return len(df_long)

# Modify task_get_global_min_max to work on total value or individual series
def task_get_global_min_max(df: pd.DataFrame, value_col: str = 'y', by_series: bool = False) -> Dict[str, Any]:
    """
    计算指定数值列的最小 / 最大值。
    - df: 输入 DataFrame (可以是 df_long 或 df_total)
    - value_col: 要计算极值的列名 ('y' 或 'total_y')
    - by_series=False（默认）：返回整体极值
        {'x_min': ..., 'x_max': ..., 'value_min': ..., 'value_max': ...}
    - by_series=True （仅用于 df_long）：返回分系列极值 (不再用于 EVJ in stream)
        {series1: {'x_min': ..., 'x_max': ..., 'y_min': ..., 'y_max': ...}, ...}
    """
    if df is None or df.empty or value_col not in df.columns:
        return {}

    def _calc(group):
        return {
            "x_min": group["x"].min(),
            "x_max": group["x"].max(),
            f"{value_col}_min": group[value_col].min(),
            f"{value_col}_max": group[value_col].max(),
        }

    if by_series and 'series' in df.columns:
        # Note: This path is kept but won't be used for EVJ in main for stream
        return df.groupby("series", group_keys=False).apply(lambda g: _calc(g.rename(columns={value_col: 'y'})), include_groups=False).to_dict()

    # Overall calculation
    return _calc(df)


# Modify task_get_average_y to work on total value or individual series
def task_get_average_y(df: pd.DataFrame,
                       value_col: str = 'y',
                       by_series: bool = False) -> Optional[Dict[str, float] | float]:
    """
    计算指定数值列平均数。
    - df：输入 DataFrame (可以是 df_long 或 df_total)
    - value_col: 要计算平均值的列名 ('y' 或 'total_y')
    - by_series=False（默认）→ 返回整体平均（float）
    - by_series=True           → 返回 {series: avg, …} (仅用于 df_long)
    """
    if df is None or df.empty or value_col not in df.columns:
        return None if not by_series else {}

    if by_series and 'series' in df.columns:
        return df.groupby("series")[value_col].mean().dropna().to_dict()

    # Overall average
    return df[value_col].mean()


# task_get_extreme_y_points is used by fill_qa_ve, which we keep for individual points.
# It's also used by fill_qa_evj in the line chart version, but we'll remove that call for stream EVJ.
def task_get_extreme_y_points(df_long: pd.DataFrame,
                              n: int = 1,
                              by_series: bool = True) -> List[Dict[str, Any]]:
    """
    找到 y 值最大的前 n 个点和最小的前 n 个点。
    返回列表，每个元素包含：series, type('largest'/'smallest'), x, y
    - by_series=True 时：在每条线内部各取 n 个最大 & n 个最小 (Used for VE)
    - by_series=False 时：在整体数据里取 n 个最大 & n 个最小 (Not used currently)
    """
    results: List[Dict[str, Any]] = []
    if df_long is None or df_long.empty:
        return results

    if by_series:
        grouped = df_long.groupby("series")
        for series_name, g in grouped:
            # 最小 n
            bottom = g.nsmallest(n, "y")
            for _, row in bottom.iterrows():
                results.append({"series": series_name,
                                "type": "smallest",
                                "x": row["x"], "y": row["y"]})
            # 最大 n
            top = g.nlargest(n, "y")
            for _, row in top.iterrows():
                results.append({"series": series_name,
                                "type": "largest",
                                "x": row["x"], "y": row["y"]})
    else:
        # Overall min/max points - Not used in current stream QA
        pass

    return results

# task_get_rate_of_change is not used for individual series in SC anymore,
# and overall ROC on total value is not explicitly requested. Keep for completeness but won't be used.
def task_get_rate_of_change(df: pd.DataFrame,
                            value_col: str = 'y',
                            by_series: bool = False
                            ) -> Optional[Dict[str, float] | float]:
    """
    计算首个有效点 → 末个有效点百分变化率 (%):
        (y_last - y_first) / y_first * 100
    - df: Input DataFrame (df_long or df_total)
    - value_col: Column to calculate ROC on ('y' or 'total_y')
    - by_series=True  → {series: pct_change, …} (Only for df_long)
    - by_series=False → Overall change rate (on df_long or df_total)
    """
    if df is None or df.empty or value_col not in df.columns:
        return {} if by_series else None

    def _roc(df_subset):
        # Sort by x, drop NaNs in the value column
        sorted_df = df_subset.sort_values("x").dropna(subset=[value_col])
        if len(sorted_df) < 2:
            return np.nan
        first = sorted_df.iloc[0][value_col]
        last = sorted_df.iloc[-1][value_col]
        return (last - first) / first * 100 if first != 0 else np.nan

    if by_series and 'series' in df.columns:
        return (df.groupby("series")
                       .apply(_roc, value_col=value_col, include_groups=False) # Pass value_col
                       .dropna()
                       .to_dict())

    # Overall ROC (on df_long or df_total)
    return _roc(df)


# --- QA Filling Functions based on QA整理.txt ---

def fill_qa_ctr() -> List[Dict[str, str]]:
    qa_list: List[Dict[str, str]] = []
    # Modified chart type
    qa_list.append({
        "Q": "What type of chart is this?",
        "A": "This chart is a {stream} chart." # Changed type
    })
    return qa_list


def fill_qa_vec(series_count: int) -> List[Dict[str, str]]:
    qa_list: List[Dict[str, str]] = []
    # Modified wording from lines to series
    question = "How many series are represented in this stream chart?"
    answer = f"There are {{{series_count}}} series." # Added {} and changed wording
    qa_list.append({"Q": question, "A": answer})
    return qa_list

def fill_qa_srp(metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates QA for SRP (Spatial Relationship - Vertical).
    Compares the vertical position (stacking order) of two series.
    In a stream graph, the vertical relationship between two series
    is determined by their stacking order. If series A is stacked below series B,
    then series A is "below" series B in the vertical direction.
    The stacking order is typically determined by the order of series in the data.
    We assume this order is reflected in the 'series_names' list from metadata.

    Q: What is the vertical (above/below) relationship of the area representing {series_1} relative to {series_2}?
    A: {series_1} is {{{answer}}} {series_2} in the vertical direction.
       (where {answer} is 'above' or 'below', enclosed in double curly braces)
    """
    qa_list: List[Dict[str, str]] = []
    series_names = metadata.get("series_names", [])

    if len(series_names) < 2:
        # Need at least two series to compare
        return qa_list

    # Create a list of all unique ordered pairs (s1, s2) where s1 != s2
    # We generate all pairs (s1, s2) and then sample from them.
    all_pairs = [(s1, s2) for s1 in series_names for s2 in series_names if s1 != s2]

    # Determine how many pairs to sample (e.g., up to 3 distinct questions)
    # Limit the number of SRP questions generated.
    max_q = 3
    num_pairs_to_sample = min(max_q, len(all_pairs))

    # Randomly sample pairs
    # Note: If reproducibility is required, set a random seed before this call.
    if num_pairs_to_sample == 0:
        return qa_list

    try:
        sampled_pairs = random.sample(all_pairs, num_pairs_to_sample)
    except ValueError:
         # This can happen if len(all_pairs) < num_pairs_to_sample, though min() should prevent it.
         # Or if all_pairs is empty, which is caught by len(series_names) < 2.
         # As a safeguard, return empty if sampling fails.
         return qa_list


    # Create a mapping from series name to its index in the list
    series_index_map = {name: i for i, name in enumerate(series_names)}

    for s1, s2 in sampled_pairs:
        # Find indices of the two series based on their order in series_names
        idx1 = series_index_map.get(s1)
        idx2 = series_index_map.get(s2)

        # This check is mostly a safeguard; if pairs were generated correctly, indices exist.
        if idx1 is None or idx2 is None or idx1 == idx2:
            continue

        # Determine the vertical relationship based on stacking order (index in series_names list)
        # If s1 appears before s2 (idx1 < idx2) in the series_names list,
        # s1 is typically stacked below s2 in the stream graph.
        # If s1 appears after s2 (idx1 > idx2), s1 is typically stacked above s2.
        relationship = "below" if idx1 < idx2 else "above"

        # Construct the Q and A pair following the specified format.
        # The answer word ('above' or 'below') must be enclosed in double curly braces {{}}.
        # In Python f-strings, {{}} is used to escape literal curly braces.
        # So, to get {{relationship}}, we use {{{{relationship}}}} in the f-string.
        q = f"What is the vertical (above/below) relationship of the area representing {s1} relative to {s2}?"
        a = f"{s1} is {{{relationship}}} {s2} in the vertical direction."

        qa_list.append({"Q": q, "A": a})

    return qa_list


# New function for VPR (Contribution)
def fill_qa_vpr_contribution(series_total_contribution: Dict[str, float],
                             metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    根据每个 series 的总贡献，生成：
      • 贡献最大的 series QA
      • 贡献最小的 series QA
    例：
    Q: Which country contributes the largest portion to total area in this area chart?
    A: The country with the largest area is {China}.
    Q: Which country contributes the smallest portion to total area in this area chart?
    A: The country with the smallest area is {China}.
    """
    qa_list: List[Dict[str, str]] = []
    if not series_total_contribution:
        return qa_list

    little_theme = metadata.get("little_theme", "") # Use little_theme in answer? Or just series name? The example uses series name.

    # Find series with max and min total contribution
    try:
        max_series = max(series_total_contribution, key=series_total_contribution.get)
        min_series = min(series_total_contribution, key=series_total_contribution.get)
    except ValueError: # Handle case where contribution dict might be empty after filtering NaN
        return qa_list


    # Ensure max and min are not the same if there's more than one series
    if len(series_total_contribution) > 1 and max_series == min_series:
         # This can happen if all contributions are zero or NaN, or only one series exists,
         # although the check len() > 1 should prevent the latter. If all are same/zero, skip.
         pass
    else:
        # Largest contribution
        q_max = f"Which series contributes the largest portion to the total {little_theme.lower()}?"
        a_max = f"The series with the largest contribution is {{{max_series}}}."
        qa_list.append({"Q": q_max, "A": a_max})

        # Smallest contribution (only add if different from max, or if only one series)
        if max_series != min_series or len(series_total_contribution) == 1:
            q_min = f"Which series contributes the smallest portion to the total {little_theme.lower()}?"
            a_min = f"The series with the smallest contribution is {{{min_series}}}."
            qa_list.append({"Q": q_min, "A": a_min})

    return qa_list


# Keep fill_qa_ve_values for individual point lookups
import pandas as pd
from typing import Dict, Any, List
import random

# Assuming _to_year and _safe_year functions are defined later in the script
# They are defined below.

def fill_qa_ve_values() -> List[Dict[str, str]]:
    """Generates QA for SRP (SVG related). Currently empty as per request."""
    return []

# def fill_qa_ve_values(df_long: pd.DataFrame,
#                       metadata: Dict[str, Any],
#                       num_single: int = 3,
#                       num_multi: int = 1) -> List[Dict[str, str]]:
#     """
#     生成 VE-类问答（Value Extraction）：关于单个 series 在某个时间点的数值。
#
#     ① 单主体：
#        Q: What is {series}'s {little_theme} in {year}?
#        A: {series}'s ... in {year} is {value} {unit}.
#
#     ② 多主体（默认 3 条线）：
#        Q: What are the {little_theme} of {series1}, {series2}, and {series3} in {year}?
#        A: 列出三个数值。
#
#     参数
#     ----
#     df_long    : tidy 格式 DataFrame（series | x | y）
#     metadata   : read_line_metadata 返回的 dict
#     num_single : 生成单主体问答数量上限（在此函数中，总数优先，此参数作为尝试生成数量）
#     num_multi  : 生成多主体问答数量上限（在此函数中，总数优先，此参数作为尝试生成数量）
#     """
#     qa_list: List[Dict[str, str]] = []
#     if df_long is None or df_long.empty:
#         return qa_list
#
#     little_theme = metadata.get("little_theme", "")
#     unit         = metadata.get("y_info", {}).get("unit", "")
#
#     # ---------- 预处理 ----------
#     # 把 x 列统一成整数年份或字符串
#     df = df_long.copy()
#     df["year"] = _to_year(df["x"]) # Assuming _to_year is defined
#     df = df.dropna(subset=["y"])        # 确保 y 有值
#
#     # ------------ ① 单主体 ----------------
#     # Sample points randomly across all series and years
#     candidates = df.sample(frac=1).reset_index(drop=True) # 打乱顺序
#     taken_single = 0 # Track how many single Qs we've attempted to generate based on num_single
#     # Keep track of (series, year) pairs already used for single questions
#     used_single_points = set()
#
#     for _, row in candidates.iterrows():
#         if taken_single >= num_single: # Limit attempts based on num_single
#             break
#         # Check total limit before generating more
#         if len(qa_list) >= 4: # Limit total VE questions to 4
#             break
#
#         series = row["series"]
#         year   = row["year"]
#         # Ensure we don't ask about the exact same point multiple times in single questions
#         if (series, year) in used_single_points:
#             continue
#         used_single_points.add((series, year))
#
#         value  = row["y"]
#         # Format value: use .0f if it's an integer, otherwise .2f
#         value_fmt = f"{value:.0f}" if value == int(value) else f"{value:.2f}"
#         year_fmt = _safe_year(year) # Assuming _safe_year is defined
#
#         q = f"What is {series}'s {little_theme} in {year_fmt}?"
#         a = f"{series}'s {little_theme.lower()} in {year_fmt} is {{{value_fmt}}} {unit}."
#         qa_list.append({"Q": q, "A": a})
#         taken_single += 1 # Increment attempted single Qs count
#
#         # Check total limit after adding
#         if len(qa_list) >= 4: # Limit total VE questions to 4
#             break
#
#     # ------------ ② 多主体 ----------------
#     # Find years where at least 3 series have data
#     group = df.groupby("year")["series"].nunique()
#     valid_years_for_multi = group[group >= 3].index.tolist()
#     random.shuffle(valid_years_for_multi)
#
#     taken_multi = 0 # Track how many multi Qs we've attempted to generate based on num_multi
#     # Keep track of years already used for multi questions
#     used_multi_years = set()
#
#     for yr in valid_years_for_multi:
#         if taken_multi >= num_multi: # Limit attempts based on num_multi
#             break
#         # Check total limit before generating more
#         if len(qa_list) >= 4: # Limit total VE questions to 4
#             break
#
#         # Ensure we don't ask about the exact same year multiple times in multi questions
#         if yr in used_multi_years:
#             continue
#         used_multi_years.add(yr)
#
#         rows_year = df[(df["year"] == yr) & df["y"].notna()]
#         # Get series names with data in this year and sample 3 randomly
#         available_series = rows_year["series"].unique().tolist()
#
#         # We need at least 3 series for a multi-subject question
#         if len(available_series) < 3:
#              continue
#
#         sample_series = random.sample(available_series, 3) # Randomly sample 3 series
#
#         parts_q, parts_a = [], []
#         for s in sample_series:
#             # Ensure the series 's' actually has data in 'rows_year' before accessing iloc[0]
#             series_data = rows_year.loc[rows_year["series"] == s, "y"]
#             if not series_data.empty:
#                 val = series_data.iloc[0]
#                 # Format value: use .0f if it's an integer, otherwise .2f
#                 val_fmt = f"{val:.0f}" if val == int(val) else f"{val:.2f}"
#                 parts_q.append(s)
#                 parts_a.append(f"{s}'s {little_theme.lower()} is {{{val_fmt}}} {unit}")
#             # If a sampled series somehow doesn't have data here (shouldn't happen with the filter), skip it
#             else:
#                  # This case indicates a potential issue with filtering or sampling,
#                  # but we'll skip this multi-question if we can't get 3 valid parts.
#                  # For simplicity, we'll break and not add this multi-QA if any part is missing.
#                  # A more robust approach might re-sample or handle missing data.
#                  # Given the filtering logic, this 'else' might be unreachable.
#                  parts_q = [] # Clear parts to prevent adding a partial multi-QA
#                  break # Stop building this multi-QA
#
#         # Only add the multi-QA if we successfully got data for 3 series
#         if len(parts_q) == 3:
#             year_fmt = _safe_year(yr) # Assuming _safe_year is defined
#             series_q = ", ".join(parts_q[:-1]) + (f", and {parts_q[-1]}" if len(parts_q) > 1 else parts_q[0])
#             series_a = ", ".join(parts_a) # No 'and' needed in answer list
#
#             q = f"What are the {little_theme.lower()} of {series_q} in {year_fmt}?"
#             a = series_a + "."
#             qa_list.append({"Q": q, "A": a})
#             taken_multi += 1 # Increment attempted multi Qs count
#
#             # Check total limit after adding
#             if len(qa_list) >= 4: # Limit total VE questions to 4
#                 break
#
#     return qa_list



# Modify fill_qa_evj to use total extremes only
def fill_qa_evj(extreme_points: List[Dict[str, Any]],
                     metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    为每条 series 生成两条 QA：
      • 最小值出现在哪一年？
      • 最大值出现在哪一年？

    Q 模板: Which year is {s}'s minimum {title} in the streamgraph?
    A 模板: {s}'s minimum {title} is in {y}.

    其中 {title}=little_theme，{s}=series，{y}=年份
    """
    qa_list: List[Dict[str, str]] = []
    little_theme = metadata.get("little_theme", "")

    # extreme_points 已经是每条线 n=1 的 largest / smallest
    # 用字典归并：{series: {"smallest":year, "largest":year}}
    bins: Dict[str, Dict[str, Any]] = {}
    for p in extreme_points:
        if pd.isna(p["x"]):           # 忽略无效点
            continue
        yr = _safe_year(p["x"])       # 转为纯年份
        bins.setdefault(p["series"], {})[p["type"]] = yr

    for s, info in bins.items():
        if "smallest" in info:
            q = f"Which year is {s}'s minimum {little_theme} in the stream chart?"
            a = f"{s}'s minimum {little_theme} is in {{{info['smallest']}}}."
            qa_list.append({"Q": q, "A": a})

        if "largest" in info:
            q = f"Which year is {s}'s maximum {little_theme} in the stream chart?"
            a = f"{s}'s maximum {little_theme} is in {{{info['largest']}}}."
            qa_list.append({"Q": q, "A": a})

    return random.sample(qa_list, k=2) if len(qa_list) >= 2 else qa_list

# fill_series_extremes is no longer needed as per-series extremes are removed from EVJ.
# def fill_series_extremes(...): pass # Remove or comment out

# Modify fill_qa_sc for stream (average total value)
def fill_qa_sc(overall_total_avg: float | None,
               metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    为【堆叠总平均值】生成问答。
    只生成一个问答。
    """
    qa_list: List[Dict[str, str]] = []
    y_unit = metadata.get("y_info", {}).get("unit", "")
    little_theme = metadata.get("little_theme", "")

    if overall_total_avg is None or pd.isna(overall_total_avg):
        return qa_list

    # ---- ① 堆叠总平均值 (Average Total Value) ----
    # Format value: use .0f if it's an integer, otherwise .2f
    avg_fmt = f"{overall_total_avg:.0f}" if overall_total_avg == int(overall_total_avg) else f"{overall_total_avg:.2f}"
    # Question wording changed to reflect "average total value"
    q_avg = f"What is the average total {little_theme}?"
    a_avg = f"The average total {little_theme.lower()} is {{{avg_fmt}}} {y_unit}."
    qa_list.append({"Q": q_avg, "A": a_avg})

    # We only want one SC question as per the request
    return qa_list


# Utility functions for year/date handling
def _safe_year(val):
    """
    如果 val 本身就是 4 位年份（或可转为 4 位整数），直接返回；
    否则尝试用 pd.to_datetime 解析，失败就返回原值。
    """
    if pd.isna(val): return val
    try:
        # Check for integer year representation (e.g., 2000, 2000.0)
        num = int(float(val))
        if 1000 <= num <= 3000:
            return num
    except (ValueError, TypeError):
        pass # Not a simple number

    # Try parsing as a datetime
    ts = pd.to_datetime(str(val), errors="coerce")
    if pd.notna(ts):
        # Return year if it's a valid date/year
        return ts.year
    else:
        # Fallback: return original value if parsing fails
        return val

def _to_year(col):
    # Ensure this function handles potential non-numeric or non-date values gracefully
    return pd.Series(col).apply(_safe_year)

# Modify _pair_str for NF/NC questions based on total value
def _pair_str_total(val: float, unit: str, year) -> str:
    """格式化 ‘total value is 300 million barrels in {2000} year’ 片段"""
    year_fmt = _safe_year(year)

    # Format value: use .0f if it's an integer, otherwise .2f
    v_fmt = f"{val:.0f}" if val == int(val) else f"{val:.2f}"

    return f"{{{v_fmt}}} {unit} in {{{year_fmt}}}" # Adjusted format


# Modify fill_qa_nf to only ask about total value
def fill_qa_nf_total(df_total: pd.DataFrame,
                     metadata: Dict[str, Any],
                     seed: int | None = None,
                     max_q: int = 4) -> List[Dict[str, str]]:
    """
    生成数值筛选 QA，仅对【堆叠起来的总值】提问。
    随机选取年份，避免重复。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    qa_set: "collections.OrderedDict[str, Dict[str,str]]" = collections.OrderedDict() # Use OrderedDict to store unique QAs
    if df_total is None or df_total.empty or len(df_total) < 2:
        return []

    little_theme = metadata.get("little_theme", "")
    unit = metadata.get("y_info", {}).get("unit", "")

    # Use the df_total DataFrame directly
    df = df_total.copy()
    df["year"] = _to_year(df["x"])
    df = df.dropna(subset=["total_y"]) # Ensure total_y has value

    years_all = df["year"].unique().tolist()

    # Define possible question types (all based on total value over years)
    q_types = ['total_gt', 'total_lt', 'total_between']

    tries = 0
    # Try to generate up to max_q unique questions
    while len(qa_set) < max_q and tries < 30: # Limit tries
        tries += 1
        # Randomly pick a type for THIS question attempt
        q_type = random.choice(q_types)

        try:
            # Need years with total data
            if not years_all: continue
            # Calculate thresholds based on total values
            vals = df["total_y"]
            if len(vals) < 2: continue # Need at least 2 points for percentiles
            # Ensure thresholds are within actual data range and meaningful
            min_val, max_val = vals.min(), vals.max()
            if min_val == max_val: continue # No variation, filtering is trivial

            lo, hi = np.percentile(vals, [30, 70])
            mid_low = (lo + hi) / 2 * 0.95 # Adjust interval slightly
            mid_high = (lo + hi) / 2 * 1.05

            if q_type == 'total_gt' and hi < max_val: # Ensure threshold is meaningful (not above max)
                threshold = hi
                # Find years and total values exceeding threshold
                results_df = df[df["total_y"] > threshold]
                if not results_df.empty:
                     # Sample up to 3 results randomly
                     years_vals = results_df[["year", "total_y"]].sample(min(3, len(results_df))).values
                     # Format threshold: use .0f if integer-like, otherwise .2f
                     threshold_fmt = f"{threshold:.0f}" if threshold == int(threshold) else f"{threshold:.2f}"
                     q = f"Which years did the total {little_theme} exceed {threshold_fmt} {unit}? Please list the years and corresponding total values."
                     # --- 修改此处，使用新的回答句式 ---
                     parts = [_pair_str_total(v, unit, y) for y, v in years_vals]
                     # ----------------------------------
                     a = ", ".join(parts) + "."
                     # Add to set to ensure uniqueness
                     qa_set.setdefault(q, {"Q": q, "A": a})

            elif q_type == 'total_lt' and lo > min_val: # Ensure threshold is meaningful (not below min)
                 threshold = lo
                 # Find years and total values below threshold
                 results_df = df[df["total_y"] < threshold]
                 if not results_df.empty:
                     # Sample up to 3 results randomly
                     years_vals = results_df[["year", "total_y"]].sample(min(3, len(results_df))).values
                     # Format threshold: use .0f if integer-like, otherwise .2f
                     threshold_fmt = f"{threshold:.0f}" if threshold == int(threshold) else f"{threshold:.2f}"
                     q = f"Which years did the total {little_theme} go below {threshold_fmt} {unit}? Please list the years and corresponding total values."
                     # --- 修改此处，使用新的回答句式 ---
                     parts = [_pair_str_total(v, unit, y) for y, v in years_vals]
                     # ----------------------------------
                     a = ", ".join(parts) + "."
                     qa_set.setdefault(q, {"Q": q, "A": a})

            elif q_type == 'total_between' and mid_low < mid_high and mid_low >= min_val and mid_high <= max_val: # Ensure interval is valid and within range
                 # Find years and total values within interval
                 cond = (df["total_y"] >= mid_low) & (df["total_y"] <= mid_high)
                 results_df = df[cond]
                 if not results_df.empty:
                     # Sample up to 3 results randomly
                     years_vals = results_df[["year", "total_y"]].sample(min(3, len(results_df))).values
                     # Format thresholds
                     mid_low_fmt = f"{mid_low:.0f}" if mid_low == int(mid_low) else f"{mid_low:.2f}"
                     mid_high_fmt = f"{mid_high:.0f}" if mid_high == int(mid_high) else f"{mid_high:.2f}"

                     q = (f"Which years did the total {little_theme} fall between "
                          f"{mid_low_fmt} and {mid_high_fmt} {unit}? Please list the years and corresponding total values.")
                     # --- 修改此处，使用新的回答句式 ---
                     parts = [_pair_str_total(v, unit, y) for y, v in years_vals]
                     # ----------------------------------
                     a = ", ".join(parts) + "."
                     qa_set.setdefault(q, {"Q": q, "A": a})

        except Exception as e:
            # Catch potential errors during sampling or percentile calculation on small datasets
            # print(f"Warning: Could not generate NF question of type {q_type} (try {tries}): {e}") # Uncomment for debugging
            continue # Try generating another question

    return list(qa_set.values())



# Modify fill_qa_nc to only compare total value over time
def fill_qa_nc_total(df_total: pd.DataFrame,
                   metadata: Dict[str, Any],
                   seed: int | None = None,
                   max_q: int = 4) -> List[Dict[str, str]]:
    """
    生成 NC（Numerical Comparison）问答，仅对【堆叠起来的总值】提问。
    比较总值在不同年份或区间的差异。
    """
    if df_total is None or df_total.empty or len(df_total) < 2:
        return []

    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    qa_set: "collections.OrderedDict[str, Dict[str,str]]" = collections.OrderedDict() # Use OrderedDict for unique QAs
    little_theme = metadata.get("little_theme", "")
    # unit = metadata.get("y_info", {}).get("unit", "") # Unit is not typically in NC answers

    # Use df_total directly
    df = df_total.copy()
    df["year"] = _to_year(df["x"])
    df = df.dropna(subset=["total_y"])

    years_all = sorted(df["year"].unique().tolist()) # Sort years for consistent interval sampling
    # Need at least 2 years for comparison
    if len(years_all) < 2:
        return []

    # Define possible comparison types for the TOTAL value
    comparison_types = ['total_year_compare', 'total_interval_avg_compare', 'total_interval_change_compare']

    tries = 0
    # Try to generate up to max_q unique questions
    while len(qa_set) < max_q and tries < 30: # Limit tries
        tries += 1
        # Randomly pick a comparison type for THIS question attempt
        comp_type = random.choice(comparison_types)

        try:
            if comp_type == 'total_year_compare':
                # Sample 2 distinct years
                if len(years_all) < 2: continue
                y1, y2 = random.sample(years_all, 2)
                # Ensure y1 < y2 for consistent question phrasing (optional but good practice)
                if y1 > y2: y1, y2 = y2, y1

                v1 = df[df["year"] == y1]["total_y"].iloc[0]
                v2 = df[df["year"] == y2]["total_y"].iloc[0]

                if pd.isna(v1) or pd.isna(v2): continue

                rel = "higher" if v1 > v2 else ("lower" if v1 < v2 else "about the same")
                q = (f"Was the total {little_theme} in {y1} higher or lower than in "
                     f"{y2}?")
                a = (f"The total {little_theme.lower()} in {y1} was {{{rel}}} than in {y2}." ) # Added {} around years and rel
                # Use setdefault to avoid duplicates based on the question string
                qa_set.setdefault(q, {"Q": q, "A": a})

            elif comp_type == 'total_interval_avg_compare':
                 # Sample 2 distinct intervals
                 # Generate all possible intervals of length >= 2
                 possible_intervals = []
                 for i in range(len(years_all)):
                     for j in range(i + 1, len(years_all)): # Interval must have at least 2 years
                         possible_intervals.append((years_all[i], years_all[j]))

                 if len(possible_intervals) < 2: continue # Need at least 2 distinct intervals

                 # Sample two distinct intervals
                 (s1, e1), (s2, e2) = random.sample(possible_intervals, 2)

                 mask1 = (df["year"] >= s1) & (df["year"] <= e1)
                 mask2 = (df["year"] >= s2) & (df["year"] <= e2)

                 avg1 = df[mask1]["total_y"].mean()
                 avg2 = df[mask2]["total_y"].mean()

                 if pd.isna(avg1) or pd.isna(avg2): continue

                 rel = "higher" if avg1 > avg2 else ("lower" if avg1 < avg2 else "about the same")
                 q = (f"Between {s1} and {e1}, was the average total {little_theme} higher or lower "
                      f"than between {s2} and {e2}?")
                 a = (f"The average total {little_theme.lower()} between {s1} and {e1} was "
                      f"{{{rel}}} than between {s2} and {e2}." ) # Added {}

                 qa_set.setdefault(q, {"Q": q, "A": a})

            elif comp_type == 'total_interval_change_compare':
                 # Sample 2 distinct intervals to compare change in total value
                 # Generate all possible intervals of length >= 2
                 possible_intervals = []
                 for i in range(len(years_all)):
                     for j in range(i + 1, len(years_all)): # Interval must have at least 2 years
                         possible_intervals.append((years_all[i], years_all[j]))

                 if len(possible_intervals) < 2: continue # Need at least 2 distinct intervals

                 # Sample two distinct intervals
                 (s1, e1), (s2, e2) = random.sample(possible_intervals, 2)

                 # Get start and end values for each interval
                 # Need to handle cases where s1/e1 or s2/e2 might not be exact x values if x is not purely year
                 # A more robust way is to find the first and last available points within the interval
                 df_interval1 = df[(df["year"] >= s1) & (df["year"] <= e1)].sort_values("year")
                 df_interval2 = df[(df["year"] >= s2) & (df["year"] <= e2)].sort_values("year")

                 if df_interval1.empty or df_interval2.empty: continue

                 v_s1 = df_interval1["total_y"].iloc[0]
                 v_e1 = df_interval1["total_y"].iloc[-1]
                 v_s2 = df_interval2["total_y"].iloc[0]
                 v_e2 = df_interval2["total_y"].iloc[-1]

                 if pd.isna(v_s1) or pd.isna(v_e1) or pd.isna(v_s2) or pd.isna(v_e2): continue

                 change1 = abs(v_e1 - v_s1)
                 change2 = abs(v_e2 - v_s2)

                 if change1 == change2:
                     rel = "about the same"
                 else:
                     rel = "larger" if change1 > change2 else "smaller"

                 q = (f"Between {s1} and {e1}, did the total {little_theme} experience a larger or smaller change "
                      f"than between {s2} and {e2}?")
                 a = (f"The change in total {little_theme.lower()} between {s1} and {e1} was "
                      f"{{{rel}}} than between {s2} and {e2}." ) # Added {}

                 qa_set.setdefault(q, {"Q": q, "A": a})


        except Exception as e:
            # Catch potential errors during sampling or calculations
            # print(f"Warning: Could not generate NC question of type {comp_type} (try {tries}): {e}") # Uncomment for debugging
            continue # Try generating another question

    return list(qa_set.values())


def fill_qa_msr(df_long: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates QA for MSR (Multi-Series Relationship) based on streamgraph properties.
    - Finds the series with the dominant position (largest value) for the longest period.
    - Finds the first time two randomly selected series exchange dominant positions.
    """
    qa_list: List[Dict[str, str]] = []
    if df_long is None or df_long.empty:
        return qa_list

    series_names = metadata.get("series_names", [])
    if len(series_names) < 2:
        # Need at least two series for MSR questions
        return qa_list

    # --- Calculate Dominance at each x ---
    # Find the series with the maximum y value for each x
    # Ensure we handle potential NaNs in y before finding max
    df_cleaned = df_long.dropna(subset=["y"])
    if df_cleaned.empty:
        return qa_list

    # Group by x and find the series with the max y value in each group
    # The result is a Series where index is x, value is the dominant series name
    # If there's a tie, idxmax picks the first one in the group's rows for that x.
    # This is acceptable for "dominant position".
    try:
        # Ensure there are actual groups with data before calling idxmax
        if df_cleaned.groupby('x')['y'].size().min() == 0:
             return qa_list

        dominant_series_per_x = df_cleaned.loc[df_cleaned.groupby('x')['y'].idxmax()]['series']
    except Exception:
        # Handle potential errors during grouping or idxmax on unusual data
        return qa_list


    if dominant_series_per_x.empty:
         return qa_list # No points with dominant series found

    # --- Find series with longest dominance ---
    # Count occurrences of each series being dominant
    dominance_counts = dominant_series_per_x.value_counts()

    if not dominance_counts.empty:
        # Get the series name with the highest count
        longest_dominant_series = dominance_counts.index[0]

        # --- Generate QA1 (Longest Dominance) ---
        q1 = "In this stream chart, which one held the dominant position for the longest period of time?"
        a1 = f"{{{longest_dominant_series}}} held the dominant position for the longest period of time."
        qa_list.append({"Q": q1, "A": a1})


    # --- Find First Exchange Point between two random series ---
    # Select two distinct series randomly
    if len(series_names) >= 2:
        try:
            series_pair = random.sample(series_names, 2)
            s1, s2 = series_pair[0], series_pair[1]

            # Filter data for the selected pair and sort by x
            df_pair = df_long[df_long['series'].isin([s1, s2])].copy()
            # Convert x to year for comparison and sorting
            df_pair["year"] = _to_year(df_pair["x"])
            df_pair = df_pair.dropna(subset=["year", "y"]) # Need valid year and y
            df_pair = df_pair.sort_values("year")

            # Pivot to compare values side-by-side per year
            df_pivot = df_pair.pivot(index='year', columns='series', values='y')
            df_pivot = df_pivot.dropna() # Drop years where either series is missing
            # Ensure both series are still present after dropping NaNs
            if s1 not in df_pivot.columns or s2 not in df_pivot.columns or len(df_pivot) < 2:
                # Not enough years with data for both series or not enough points
                pass # Skip generating the second QA for this pair
            else:
                 # Find the first year where values are not equal to determine initial dominance baseline
                 first_non_equal_year_idx = df_pivot[(df_pivot[s1] != df_pivot[s2])].index
                 if first_non_equal_year_idx.empty:
                      # Values are always equal for all common years with data
                      first_exchange_year: Optional[Any] = None
                 else:
                     first_relevant_year = first_non_equal_year_idx[0]
                     # Get initial dominance status based on the first year where they are not equal
                     initial_s1_dominant = df_pivot.loc[first_relevant_year, s1] > df_pivot.loc[first_relevant_year, s2]

                     # Iterate from the year *after* the first relevant year
                     first_exchange_year: Optional[Any] = None
                     start_idx = df_pivot.index.get_loc(first_relevant_year) + 1
                     for current_year in df_pivot.index[start_idx:]:
                         val1_current = df_pivot.loc[current_year, s1]
                         val2_current = df_pivot.loc[current_year, s2]

                         # Skip if values are equal in the current year
                         if val1_current == val2_current:
                             continue

                         # Check if dominance status has flipped
                         current_s1_dominant = val1_current > val2_current

                         if current_s1_dominant != initial_s1_dominant:
                             first_exchange_year = current_year
                             break # Found the first exchange

                 # --- Generate QA2 (First Exchange) ---
                 q2 = f"In this stream chart, in which year did {s1} and {s2} first exchange dominant positions?"

                 if first_exchange_year is not None:
                     # Format the year safely
                     exchange_year_fmt = _safe_year(first_exchange_year)
                     a2 = f"{s1} and {s2} first exchanged leading positions in {{{exchange_year_fmt}}}."
                     qa_list.append({"Q": q2, "A": a2})
                 else:
                     # No exchange found
                     a2 = f"{s1} and {s2} did not exchange leading positions."
                     qa_list.append({"Q": q2, "A": a2})

        except ValueError:
             # Handle case where random.sample fails (e.g., less than 2 series after filtering)
             pass # Skip generating the second QA
        except Exception as e:
             print(f"Error generating MSR exchange QA: {e}")
             pass # Skip generating the second QA


    return qa_list


def fill_qa_va() -> List[Dict[str, str]]:
    """Generates QA for VA (SVG related). Currently empty as per request."""
    return []


# Keep write_qa_to_json as is, it handles the file structure and merging correctly
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]]):

    json_dir = 'QA'
    os.makedirs(json_dir, exist_ok=True)

    # Construct JSON file full path using the CSV base name
    # Take the basename and remove the .csv suffix
    base_name_with_suffix = os.path.basename(csv_path) # e.g., bubble_Topic_1.csv
    base_name = os.path.splitext(base_name_with_suffix)[0] # e.g., bubble_Topic_1

    # The JSON filename should be the same as the CSV base name
    json_path = os.path.join(json_dir, base_name + '.json')
    # --- END MODIFICATION FOR OUTPUT PATH ---


    # Define the complete template structure (Matching pasted_text_0.txt)
    template_data: Dict[str, List[Dict[str, str]]] = {
        "CTR": [], "VEC": [], "SRP": [], "VPR": [], "VE": [],
        "EVJ": [], "SC": [], "NF": [], "NC": [], "MSR": [], "VA": []
    }

    # Load existing data if file exists
    existing_data: Dict[str, List[Dict[str, str]]] = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            # Ensure loaded data is a dictionary, fallback if not
            if isinstance(loaded_data, dict):
                 existing_data = loaded_data
            else:
                 print(f"Warning: Existing JSON data in {json_path} is not a dictionary. Overwriting with template structure.")

        except (json.JSONDecodeError, FileNotFoundError):
            # File not found is handled by os.path.exists, but keeping it here as a safeguard
            print(f"Warning: Could not load or decode JSON from {json_path}. Starting with template structure.")
        except Exception as e:
             print(f"Warning: Could not read JSON from {json_path}: {e}. Starting with template structure.")


    # Merge existing data into the template structure
    # Start with the template, then copy over the lists from the existing data for any keys that exist and are lists
    data_to_save = template_data.copy() # Start with all keys from the template
    for key in template_data.keys():
         if key in existing_data and isinstance(existing_data[key], list):
             # Copy the existing list for this key
             data_to_save[key] = existing_data[key]


    # Append new QA items to the appropriate list in the merged data
    # Ensure the qa_type exists in the template (which it will now) and is a list
    if qa_type in data_to_save and isinstance(data_to_save[qa_type], list):
         # Avoid adding duplicate QAs if the script is run multiple times on the same CSV
         # This is a simple check - assumes Q and A together are unique within a type
         new_items_to_add = []
         # Create a set of existing Q/A tuples for quick lookup
         existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save[qa_type] if isinstance(item, dict) and 'Q' in item and 'A' in item}

         for item in qa_items:
              # Check if the item is a valid QA dictionary before trying to get Q and A
              if isinstance(item, dict) and 'Q' in item and 'A' in item:
                   if (item.get('Q'), item.get('A')) not in existing_qa_pairs:
                        new_items_to_add.append(item)
                        # Add to set to prevent duplicates within the new list and against existing ones
                        existing_qa_pairs.add((item.get('Q'), item.get('A')))
              else:
                   print(f"Warning: Skipping invalid QA item format for type {qa_type}: {item}")


         data_to_save[qa_type].extend(new_items_to_add)

    else:
         # This case should really not happen with the template initialization,
         # but as a safeguard, print a warning.
         print(f"Error: Attempted to write to invalid QA type '{qa_type}' in {json_path}. This type might be missing from the template.")


    # Write back to file
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        # print(f"Wrote QA to {json_path} under type {qa_type}") # Optional: confirmation print
    except Exception as e:
         print(f"Error writing QA to {json_path} for type {qa_type}: {e}")


def compute_all_tasks(df_long: pd.DataFrame,
                      meta: Dict[str, Any]) -> Dict[str, Any]:
    stats : Dict[str, Any] = {}

    if df_long is None or df_long.empty:
        return stats

    # Calculate the total value at each x point
    df_total = task_compute_total_value(df_long)
    stats["df_total"] = df_total # Store total data for NF/NC

    if df_total is None or df_total.empty:
         # If total data is empty, stop computing stats that depend on it
         return stats

    # ① 数据点数量 (for individual series)
    stats["point_counts_total"] = task_count_points(df_long)
    stats["point_counts_each"] = task_count_points(df_long, by_series=True)

    # ② 全局极值 (based on total value)
    stats["total_extremes"] = task_get_global_min_max(df_total, value_col='total_y')
    # Per-series extremes are not used for EVJ in stream

    # ③ 平均值 (overall total average for SC)
    stats["overall_total_avg"] = task_get_average_y(df_total, value_col='total_y', by_series=False)
    # Per-series average is not used for SC anymore
    # stats["avg_y_each"] = task_get_average_y(df_long, by_series=True) # No longer needed for SC

    # ④ 首年→末年变化率 (not used for QA currently)
    # stats["roc_overall"] = task_get_rate_of_change(df_total, value_col='total_y') # ROC on total
    # stats["roc_each"] = task_get_rate_of_change(df_long, by_series=True) # ROC per series

    # ⑤ 每条线的最高 / 最低点（n=1） (Used by fill_qa_ve)
    stats["extreme_points_1_series"] = task_get_extreme_y_points(df_long, n=1, by_series=True) # Renamed key for clarity

    # ⑥ 总贡献度 (Per series, for VPR)
    stats["series_total_contribution"] = task_compute_total_contribution_per_series(df_long)

    return stats

def main():
    # todo 修改路径和任务类型
    csv_folder = './csv'

    # 检查 CSV 文件夹是否存在
    if not os.path.exists(csv_folder):
        print(f"错误：未找到 CSV 文件夹 {csv_folder}。请先运行生成数据的脚本。") # Adjusted message
        return

    # Identify the target chart type (used in CTR QA)
    chart_type_name = "stream chart"

    for csv_path in Path(csv_folder).glob("*.csv"):
        print(f"\n正在处理文件：{csv_path} ...")

        # ---------- 读取 ----------
        meta = read_line_metadata(csv_path)
        df_long = read_line_data_df(csv_path, meta)

        if df_long is None or df_long.empty:
            print(f"跳过 {csv_path.name} —— 无有效数据")
            continue

        # ---------- 统计 ----------
        stats = compute_all_tasks(df_long, meta)

        # Get required data from stats
        df_total = stats.get("df_total")
        # Check if df_total is valid before proceeding with stats dependent on it
        if df_total is None or df_total.empty:
             print(f"跳过 {csv_path.name} —— 无法计算总堆叠值")
             # Still write basic QAs if possible
             write_qa_to_json(csv_path, "CTR", fill_qa_ctr())
             write_qa_to_json(csv_path, "VEC", fill_qa_vec(len(meta.get("series_names", []))))
             write_qa_to_json(csv_path, "SRP", [])
             write_qa_to_json(csv_path, "MSR", [])
             write_qa_to_json(csv_path, "VA", [])
             continue # Skip generating other QAs if total data is missing


        total_extremes = stats.get("total_extremes", {})
        overall_total_avg = stats.get("overall_total_avg") # Get the new overall total average stat
        series_total_contribution = stats.get("series_total_contribution", {})
        # extreme_points_1_series is used by fill_qa_ve but it's called internally there.

        # ---------- 生成 QA ----------
        # CTR: Chart Type
        qa_ctr = fill_qa_ctr() # This function now hardcodes "stream chart"

        # VEC: Series Count
        # Use the count from metadata for consistency, though stats also has it
        series_count = len(meta.get("series_names", []))
        qa_vec = fill_qa_vec(series_count)

        # VPR: Contribution (replaces trend)
        qa_vpr = fill_qa_vpr_contribution(series_total_contribution, meta)

        # VE: Value Extraction (individual points)
        # fill_qa_ve_values works on df_long
        # qa_ve = fill_qa_ve_values(df_long, meta, num_single=3, num_multi=1) # Generate 3 single + 1 multi VE questions

        # EVJ: Extremes (global total only)
        # fill_qa_evj now uses total_extremes
        extreme_points = stats.get("extreme_points_1_series", [])
        qa_evj = fill_qa_evj(extreme_points, meta)

        # SC: Statistical Comparison (average total height)
        # fill_qa_sc now uses overall_total_avg and generates only ONE question
        qa_sc = fill_qa_sc(overall_total_avg, meta) # Pass the overall average

        # NF: Numerical Filtering (total value only)
        # fill_qa_nf_total works on df_total
        qa_nf = fill_qa_nf_total(df_total, meta, max_q=4) # Generate up to 4 random NF questions on total

        # NC: Numerical Comparison (total value over time)
        # fill_qa_nc_total works on df_total
        qa_nc = fill_qa_nc_total(df_total, meta, max_q=4) # Generate up to 4 random NC questions on total
        qa_msr = fill_qa_msr(df_long, meta)
        qa_srp = fill_qa_srp(meta)

        # ---------- 写入 JSON ----------
        write_qa_to_json(csv_path, "CTR", qa_ctr)
        write_qa_to_json(csv_path, "VEC", qa_vec)
        write_qa_to_json(csv_path, "VPR", qa_vpr)  # Use VPR key for contribution

        write_qa_to_json(csv_path, "VE", [])

        # write_qa_to_json(csv_path, "VE", qa_ve)
        write_qa_to_json(csv_path, "EVJ", qa_evj)
        # write_qa_to_json(csv_path, "SC", qa_sc)
        write_qa_to_json(csv_path, "SC", [])
        # write_qa_to_json(csv_path, "NF", qa_nf)
        write_qa_to_json(csv_path, "NF", [])
        write_qa_to_json(csv_path, "NC", qa_nc)
        write_qa_to_json(csv_path, "MSR", qa_msr)
        write_qa_to_json(csv_path, "SRP", qa_srp)

        # 占位（保持原键，避免前端报缺）


        write_qa_to_json(csv_path, "VA", [])


    print(f"\n{chart_type_name.capitalize()} QA 文件生成完毕。")

if __name__ == "__main__":
    main()
