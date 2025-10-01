# File: line_QA.py
# Description: Generates QA files for bubble chart data based on bubble.py CSV output and 气泡图QA整理.txt template.
from pathlib import Path
import collections
import pandas as pd
import os
import json
import numpy as np
from typing import Dict, Any, Tuple, List, Union, Optional
import re
import random  # Import random for selections
from typing import List, Dict, Any, Tuple  # Import typing hints
import math  # Import math for isnan
from sklearn.neighbors import NearestNeighbors  # Import for KNN density calculation
# Import polyfit and poly1d specifically from numpy
from numpy import polyfit, poly1d  # Use numpy's standard polynomial functions


# --- Utility Functions (Adapted from scatter_QA.py and heatmap_QA.py) ---
def read_line_metadata(filepath: str) -> Dict[str, Any]:
    """
    读取折线图 CSV 的前三行元数据，返回结构示例：
    {
        'topic'        : 'Agriculture and Food Production',
        'little_theme' : 'Crop Yield',
        'y_info'       : {'unit': 'tons/hectare'},
        'x_info'       : {'name': 'Year', 'unit': ''},
        'series_names' : ['Golden Harvest Cooperative', 'Starfall Organics'],
        'series_trends': {               # 与 series_names 一一对应
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
        meta_df = pd.DataFrame(rows)  # 不同长度的行也能放进来

        if len(meta_df) < 3:
            print(f"[WARN] 元数据不足 3 行 → {filepath}")
            return {}

        # 第 1 行：大标题、小标题、Y 轴单位
        line1: List[Any] = meta_df.iloc[0].tolist()
        topic = (line1[0] or "").strip() if len(line1) > 0 else ""
        little_theme = (line1[1] or "").strip() if len(line1) > 1 else ""
        y_unit = (line1[2] or "").strip() if len(line1) > 2 else ""


        # 第 3 行：X 轴名称 + 各折线系列名称
        line3: List[Any] = meta_df.iloc[2].tolist()
        x_name = (line3[0] or "").strip() if len(line3) > 0 else ""
        series_names = [str(c).strip() for c in line3[1:] if pd.notna(c)]

        # 第 2 行：趋势标签；第一个单元格通常是 "trend"
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
            "series_trends": series_trends,
        }
        return meta

    except Exception as e:
        print(f"[ERROR] 读取折线图元数据失败：{filepath} → {e}")
        return {}


def read_line_data_df(filepath: str, metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    读取折线图 CSV 中真正的数据区（元数据 3 行之后），并返回 tidy 格式：
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
            x_vals = header[1:]  # e.g. 年份们
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
        print(f"[ERROR] 读取折线图数据失败：{filepath} → {e}")
        return None


# --- Calculation Functions (Specific to Bubble Chart) ---
# These functions calculate the data needed for different QA types.

def task_count_points(df_long: pd.DataFrame, by_series: bool = False):
    """
    统计折线图的数据点数量。
    如果 by_series=True，返回 {series: count, ...}
    否则返回整数总数。
    """
    if df_long is None or df_long.empty:
        return 0 if not by_series else {}

    if by_series:
        return df_long.groupby("series").size().to_dict()
    return len(df_long)


def task_get_global_min_max(df_long: pd.DataFrame, by_series: bool = False) -> Dict[str, Any]:
    """
    计算 X、Y 的最小 / 最大值。
    - by_series=False（默认）：返回整体极值
        {'x_min': ..., 'x_max': ..., 'y_min': ..., 'y_max': ...}
    - by_series=True ：返回分系列极值
        {series1: {'x_min': ..., 'x_max': ..., 'y_min': ..., 'y_max': ...}, ...}
    """
    if df_long is None or df_long.empty:
        return {} if not by_series else {}

    def _calc(group):
        # Need to handle potential non-numeric x values before min/max
        # Convert x to numeric for calculation if possible, otherwise handle appropriately
        # For now, assuming x is comparable (like years or dates)
        return {
            "x_min": group["x"].min(),
            "x_max": group["x"].max(),
            "y_min": group["y"].min(),
            "y_max": group["y"].max(),
        }

    if by_series:
        # Use group_keys=False to avoid group keys in the output if pandas version supports it
        # If not, the .to_dict() will handle it
        return df_long.groupby("series", group_keys=False).apply(_calc).to_dict()

    return _calc(df_long)


def task_get_average_y(df_long: pd.DataFrame,
                       by_series: bool = False) -> Optional[Dict[str, float] | float]:
    """
    计算 y 值平均数。
    - df_long：必须是 ['series','x','y'] 三列的长表
    - by_series=False（默认）→ 返回整体平均（float）
    - by_series=True           → 返回 {series: avg, …}
    """
    if df_long is None or df_long.empty:
        return None if not by_series else {}

    if by_series:
        return df_long.groupby("series")["y"].mean().to_dict()

    return df_long["y"].mean()


def task_get_extreme_y_points(df_long: pd.DataFrame,
                              n: int = 1,
                              by_series: bool = True) -> List[Dict[str, Any]]:
    """
    找到 y 值最大的前 n 个点和最小的前 n 个点。
    返回列表，每个元素包含：series, type('largest'/'smallest'), x, y
    - by_series=True 时：在每条线内部各取 n 个最大 & n 个最小
    - by_series=False 时：在整体数据里取 n 个最大 & n 个最小
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
        # 整体最小 n
        bottom = df_long.nsmallest(n, "y")
        for _, row in bottom.iterrows():
            results.append({"series": row["series"],
                            "type": "smallest",
                            "x": row["x"], "y": row["y"]})
        # 整体最大 n
        top = df_long.nlargest(n, "y")
        for _, row in top.iterrows():
            results.append({"series": row["series"],
                            "type": "largest",
                            "x": row["x"], "y": row["y"]})

    return results


def _pick_two_series(series_list: List[str]) -> Tuple[str, str]:
    """随机选取两个不同的主体"""
    if len(series_list) < 2:
        raise ValueError("主体数量不足 2 个，无法比较")
    return tuple(random.sample(series_list, 2))


def _random_interval(common_x_vals: pd.Series) -> Tuple[Any, Any]:
    """在共同的 X 值范围内随机选取一个非空区间（start < end）"""
    xs_sorted = np.sort(common_x_vals.unique())
    if len(xs_sorted) < 2:
        raise ValueError("共同的时间点不足 2 个，无法构造区间")
    start_idx, end_idx = sorted(random.sample(range(len(xs_sorted)), 2))
    return xs_sorted[start_idx], xs_sorted[end_idx]


def _compute_slope(df_sub: pd.DataFrame) -> float:
    """最简单的斜率：y 对 x 做一次线性拟合，返回系数。"""
    # 将 x 转成可运算的数字
    x_vals = pd.to_numeric(df_sub["x"], errors="coerce")
    mask = x_vals.notna() & df_sub["y"].notna()
    if mask.sum() < 2:
        return np.nan
    # Use numpy.polyfit for simplicity
    coeffs = polyfit(x_vals[mask], df_sub.loc[mask, "y"], 1)
    return coeffs[0]


# This function is not used in the main QA generation flow but kept for completeness
def task_compare_subjects(df_long: pd.DataFrame,
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    一次性完成 4 种比较：
      1. 同一年两个主体谁更大
      2. 随机区间内平均值谁更大
      3. 随机区间内斜率谁更大
      4. 一个主体两年对比（更大 / 更小 / 相等）
    返回统一字典，便于 fill_qa_* 调用。
    """
    results: Dict[str, Any] = {}

    # ---------- 基本健壮性检查 ----------
    if df_long is None or df_long.empty or len(df_long["series"].unique()) < 2:
        return results

    series_names = metadata.get("series_names") or df_long["series"].unique().tolist()
    # Ensure series_names is not empty before picking
    if not series_names:
        return results

    # 1️⃣ 两主体同一年比较 -----------------------------------------
    # Pick series pair and year within the loop that uses this task, not here
    pass  # This logic is now handled in fill_qa_nc

    # 2️⃣ 区间平均值比较 ------------------------------------------
    pass  # This logic is now handled in fill_qa_nc

    # 3️⃣ 区间斜率比较 -------------------------------------------
    pass  # This logic is now handled in fill_qa_nc

    # 4️⃣ 单主体两年对比 ------------------------------------------
    pass  # This logic is now handled in fill_qa_nc

    return results  # Will be empty as logic is moved


def task_get_rate_of_change(df_long: pd.DataFrame,
                            by_series: bool = False
                            ) -> Optional[Dict[str, float] | float]:
    """
    计算首年 → 末年百分变化率 (%):
        (y_last - y_first) / y_first * 100
    - by_series=True  → {series: pct_change, …}
    - by_series=False → 整体变化率
    """
    if df_long is None or df_long.empty:
        return {} if by_series else None

    def _roc(df):
        if df.empty:
            return np.nan
        # Sort by x to get first and last points correctly
        sorted_df = df.sort_values("x").dropna(subset=["y"])
        if len(sorted_df) < 2:
            return np.nan
        first = sorted_df.iloc[0]["y"]
        last = sorted_df.iloc[-1]["y"]
        # Handle division by zero
        return (last - first) / first * 100 if first != 0 else np.nan

    if by_series:
        # Use group_keys=False to avoid group keys in the output if pandas version supports it
        # If not, the .to_dict() will handle it
        return (df_long.groupby("series", group_keys=False)
                .apply(_roc)
                .dropna()  # Drop series where ROC could not be calculated (e.g., < 2 points)
                .to_dict())
    # For overall rate of change, consider all points across all series?
    # Or average of individual series rates? Let's do the latter for now.
    # If the user wants overall trend of the *sum* or *average* of all series,
    # the data needs to be aggregated first.
    series_rocs = task_get_rate_of_change(df_long, by_series=True)
    if not series_rocs:
        return None
    # Return the average of the individual series rates of change
    return np.mean(list(series_rocs.values()))


# --- QA Filling Functions based on QA整理.txt ---
# These functions format the calculated data into the Q&A structure.
# Leave functions empty or return empty lists for QA types not specified in the text file
# or designated as placeholder.

def fill_qa_ctr() -> List[Dict[str, str]]:
    qa_list: List[Dict[str, str]] = []
    qa_list.append({
        "Q": "What type of chart is this?",
        "A": "This chart is a {line} chart."  # Corrected type and added {}
    })
    return qa_list


def fill_qa_vec(line_count: int) -> List[Dict[str, str]]:
    qa_list: List[Dict[str, str]] = []
    question = "How many lines are in this lines chart?"
    answer = f"There are {{{line_count}}} lines."  # Added {}
    qa_list.append({"Q": question, "A": answer})

    return qa_list


def fill_qa_srp(df_long: pd.DataFrame, seed: int | None = None, max_q: int = 2) -> List[Dict[str, str]]:
    """
    Generates QA for SRP (Spatial Relationship Point and Line Vertical Relationship).

    Generates at most one question of each type:
    1. Point Relationship: Vertical (above/below) and Horizontal (left/right) between two points on the same series.
       Q: On {series}, what is the spatial relationship of the data point in {year_1} relative to that in {year_2} in terms of vertical (above/below) and horizontal (left/right) directions?
       A: On {series}, the data point in {year_1} is {{{vertical_rel}}} and to the {{{horizontal_rel}}} of the data point in {year_2}.

    2. Line Vertical Relationship: Vertical (above/below) between two series in a specific year.
       Q: In {year}, what is the vertical (above/below) relationship of the line representing {series_1} relative to {series_2}?
       A: In {year}, {series_1} is {{{vertical_rel}}} {series_2} in the vertical direction.

    Args:
        df_long: DataFrame in long format with columns 'series', 'x', 'y'.
        seed: Optional random seed for reproducibility.
        max_q: Maximum number of questions to generate. (Note: The implementation
               now prioritizes generating one of each type, effectively limiting
               the total to at most 2 if both types can be generated).

    Returns:
        A list of dictionaries, where each dictionary contains 'Q' and 'A' keys
        for a generated question-answer pair.
    """
    # Use OrderedDict to maintain insertion order and ensure uniqueness based on the question string
    qa_set: "OrderedDict[str, Dict[str, str]]" = collections.OrderedDict()

    if df_long is None or df_long.empty:
        return []

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df = df_long.copy()
    # Ensure x is sortable and y is numeric
    # Assuming _to_year converts x to a comparable format (like int or string)
    df["year"] = df["x"] # Changed from _to_year to use x directly as year, assuming x is already year or comparable
    # If x needs conversion, uncomment and use the correct _to_year function:
    # df["year"] = _to_year(df["x"])

    df = df.dropna(subset=["year", "y"])  # Need valid year and y for comparisons

    series_names = df["series"].unique().tolist()
    # years_all = sorted(df["year"].unique().tolist()) # Not used in the modified logic

    # Define possible SRP question types
    srp_types = ['point_relationship', 'line_vertical_relationship']
    # Keep track of types for which we have successfully generated at least one question
    generated_types: set[str] = set()

    tries = 0
    # Try to generate one question of each type until both are generated or tries run out
    # The loop aims to fill generated_types with all srp_types
    while len(generated_types) < len(srp_types) and tries < 30:  # Limit tries to prevent infinite loops
        tries += 1

        # Get the list of types that still need to be generated
        types_to_generate = [t for t in srp_types if t not in generated_types]
        if not types_to_generate:
            # This break condition should ideally be caught by the while loop,
            # but it's a safeguard if generated_types somehow gets full early.
            break

        # Randomly pick a type that hasn't been successfully generated yet
        type_to_try = random.choice(types_to_generate)

        try:
            if type_to_try == 'point_relationship':
                # Attempt to generate a Point Relationship question
                # Need a series with at least two distinct years with data
                series_with_enough_years = [s for s in series_names if df[df["series"] == s]["year"].nunique() >= 2]
                if not series_with_enough_years:
                    continue # Cannot generate this type with current data, try again or pick another type

                selected_series = random.choice(series_with_enough_years)
                # Filter data for the selected series and ensure unique points per year
                df_s = df[df["series"] == selected_series].drop_duplicates(subset=["year", "series"])
                # Sort by year to ensure consistent year handling, although not strictly needed for sampling pair
                df_s = df_s.sort_values("year")

                # Get years for this series that have data
                series_years = sorted(df_s["year"].unique().tolist())
                if len(series_years) < 2:
                    continue # Safeguard, should be caught by series_with_enough_years check

                # Sample two distinct years WITHOUT sorting, so year_1 can be before or after year_2
                year_pair = random.sample(series_years, 2)
                year_1, year_2 = year_pair # Use the years as sampled

                # Get y values for these years from the filtered/deduplicated df_s
                y1_row = df_s[(df_s["year"] == year_1)]
                y2_row = df_s[(df_s["year"] == year_2)]

                # Ensure data exists for both sampled years in the selected series
                if y1_row.empty or y2_row.empty:
                    continue

                y1 = y1_row["y"].iloc[0]
                y2 = y2_row["y"].iloc[0]

                # Only generate QA if both y values are valid and distinct
                if pd.notna(y1) and pd.notna(y2) and y1 != y2:
                    vertical_rel = "above" if y1 > y2 else "below"
                    # Determine horizontal relationship based on the order of sampled years
                    horizontal_rel = "right" if year_1 > year_2 else "left"

                    q = f"On {selected_series}, what is the spatial relationship of the data point in {year_1} relative to that in {year_2} in terms of vertical (above/below) and horizontal (left/right) directions?"
                    # Construct the answer with both vertical and horizontal relationships
                    a = f"On {selected_series}, the data point in {year_1} is {{{vertical_rel}}} and to the {{{horizontal_rel}}} of the data point in {year_2}."

                    # Add this QA to the set if the question is unique and we haven't successfully generated this type yet
                    # The check `type_to_try not in generated_types` is done by selecting `type_to_try` from `types_to_generate`.
                    # We just need to check question uniqueness here.
                    if q not in qa_set:
                        qa_set[q] = {"Q": q, "A": a}
                        # Mark 'point_relationship' as successfully generated
                        generated_types.add('point_relationship')


            elif type_to_try == 'line_vertical_relationship':
                # Attempt to generate a Line Vertical Relationship question
                # Need at least two distinct series
                if len(series_names) < 2:
                    continue # Cannot generate this type

                # Need a year where at least two series have data
                # Group by year and filter years where the number of unique series is >= 2
                years_with_multiple_series = df.groupby("year").filter(lambda x: x["series"].nunique() >= 2)["year"].unique().tolist()
                if not years_with_multiple_series:
                    continue # Cannot generate this type

                # Pick a random year from those with multiple series
                selected_year = random.choice(years_with_multiple_series)

                # Get series that have data in this specific year
                series_in_year = df[df["year"] == selected_year]["series"].unique().tolist()
                if len(series_in_year) < 2:
                     # This should be caught by years_with_multiple_series check, but as a safeguard
                     continue

                # Pick two random distinct series that have data in this year
                s1, s2 = random.sample(series_in_year, 2)

                # Get y values for these series in this year from the main df
                y1_row = df[(df["series"] == s1) & (df["year"] == selected_year)]
                y2_row = df[(df["series"] == s2) & (df["year"] == selected_year)]

                # Ensure data exists for both selected series in the selected year
                if y1_row.empty or y2_row.empty:
                     continue

                y1 = y1_row["y"].iloc[0]
                y2 = y2_row["y"].iloc[0]

                # Only generate QA if both y values are valid and distinct
                if pd.notna(y1) and pd.notna(y2) and y1 != y2:
                    vertical_rel = "above" if y1 > y2 else "below"
                    q = f"In {selected_year}, what is the vertical (above/below) relationship of the line representing {s1} relative to {s2}?"
                    a = f"In {selected_year}, {s1} is {{{vertical_rel}}} {s2} in the vertical direction."

                    # Add this QA to the set if the question is unique and we haven't successfully generated this type yet
                    if q not in qa_set:
                        qa_set[q] = {"Q": q, "A": a}
                        # Mark 'line_vertical_relationship' as successfully generated
                        generated_types.add('line_vertical_relationship')

        except Exception as e:
            # Catch potential errors during sampling, data access, or calculations for this specific try.
            # This allows the loop to continue trying other types or data combinations.
            # print(f"Warning: Could not generate SRP question of type {type_to_try} (try {tries}): {e}") # Uncomment for debugging
            continue # Skip this try and attempt another in the next loop iteration

    # Return a list of the unique QAs generated.
    # The loop attempts to generate one of each type. The final list will contain at most one of each type.
    # The max_q parameter is effectively superseded by the goal of generating one of each type.
    return list(qa_set.values())



def _pretty_trend(label: str) -> str:
    """
    把 snake_case / camelCase 的趋势标签，转成人能读的短语：
      'stable_falling'  → 'stable falling'
      'periodicStable'  → 'periodic stable'
    """
    if not label:
        return ""
    if "_" in label:
        return label.replace("_", " ")
    # camelCase → 加空格并小写
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", label).lower()


def fill_qa_trend(metadata: Dict[str, Dict]) -> List[Dict[str, str]]:
    """
    根据 metadata['series_trends'] 生成两类 QA：
      ① 每条线自己的趋势问答
         Q: What is the trend of {series}'s {little_theme} levels?
         A: {series}'s ... levels show a {trend_phrase} trend.
      ② 给定某一种趋势，问“哪条线呈现这种趋势？”
         Q: Which line shows a {trend_phrase} trend in {little_theme} levels?
         A: {series}.
    """
    qa_list: List[Dict[str, str]] = []

    little_theme = metadata.get("little_theme", "")
    series_trends: Dict[str, str] = metadata.get("series_trends", {})

    # ------ ① 每个主体自身的趋势 ------
    for series, trend in series_trends.items():
        if not trend:
            continue
        trend_phrase = _pretty_trend(trend)
        q = f"What is the trend of {series}'s {little_theme} levels?"
        a = f"{series}'s {little_theme.lower()} levels show a {{{trend_phrase}}} trend."
        qa_list.append({"Q": q, "A": a})

    # ------ ② “哪条线是某趋势？” ------
    # 先把 trend → 列表[series] 的映射聚合
    trend_to_series: Dict[str, List[str]] = {}
    for series, trend in series_trends.items():
        if not trend:
            continue
        trend_to_series.setdefault(trend, []).append(series)

    for trend, s_list in trend_to_series.items():
        trend_phrase = _pretty_trend(trend)
        # If multiple series share the same trend, list all of them in the answer
        series_answer = ", ".join([f"{{{s}}}" for s in s_list])
        q = f"Which line shows a {trend_phrase} trend in {little_theme} levels?"
        a = f"{series_answer}."
        qa_list.append({"Q": q, "A": a})

    return qa_list


def fill_qa_ve_values(df_long: pd.DataFrame,
                      metadata: Dict[str, Any],
                      num_single: int = 3,
                      num_multi: int = 2) -> List[Dict[str, str]]:
    """
    生成 VE-类问答（Value Extraction）：

    ① 单主体：
       Q: What is {series}'s {little_theme} in {year}?
       A: {series}'s ... in {year} is {value} {unit}.

    ② 多主体（默认 3 条线）：
       Q: What are the {little_theme} of {series1}, {series2}, and {series3} in {year}?
       A: 列出三个数值。

    参数
    ----
    df_long    : tidy 格式 DataFrame（series | x | y）
    metadata   : read_line_metadata 返回的 dict
    num_single : 生成单主体问答数量上限
    num_multi  : 生成多主体问答数量上限
    """
    qa_list: List[Dict[str, str]] = []
    if df_long is None or df_long.empty:
        return qa_list

    little_theme = metadata.get("little_theme", "")
    unit = metadata.get("y_info", {}).get("unit", "")

    # ---------- 预处理 ----------
    # 把 x 列统一成整数年份或字符串
    df = df_long.copy()
    df["year"] = _to_year(df["x"])
    df = df.dropna(subset=["y"])  # 确保 y 有值

    # ------------ ① 单主体 ----------------
    # Sample points randomly across all series and years
    candidates = df.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    taken = 0
    # Keep track of (series, year) pairs already used for single questions
    used_single_points = set()

    for _, row in candidates.iterrows():
        if taken >= num_single:
            break
        series = row["series"]
        year = row["year"]
        # Ensure we don't ask about the exact same point multiple times in single questions
        if (series, year) in used_single_points:
            continue
        used_single_points.add((series, year))

        value = row["y"]
        q = f"What is {series}'s {little_theme} in {year}?"
        # Format value: use .0f if it's an integer, otherwise .2f
        value_fmt = f"{value:.0f}" if value == int(value) else f"{value:.2f}"
        a = f"{series}'s {little_theme.lower()} in {year} is {{{value_fmt}}} {unit}."
        qa_list.append({"Q": q, "A": a})
        taken += 1

    # ------------ ② 多主体 ----------------
    # Find years where at least 3 series have data
    group = df.groupby("year")["series"].nunique()
    valid_years_for_multi = group[group >= 3].index.tolist()
    random.shuffle(valid_years_for_multi)

    taken = 0
    # Keep track of years already used for multi questions
    used_multi_years = set()

    for yr in valid_years_for_multi:
        if taken >= num_multi:
            break
        # Ensure we don't ask about the exact same year multiple times in multi questions
        if yr in used_multi_years:
            continue
        used_multi_years.add(yr)

        rows_year = df[(df["year"] == yr) & df["y"].notna()]
        # Get series names with data in this year and sample 3 randomly
        available_series = rows_year["series"].unique().tolist()
        if len(available_series) < 3:
            continue  # Should not happen due to valid_years_for_multi filter, but safeguard

        sample_series = random.sample(available_series, 3)  # Randomly sample 3 series

        parts_q, parts_a = [], []
        for s in sample_series:
            val = rows_year.loc[rows_year["series"] == s, "y"].iloc[0]
            val_fmt = f"{val:.0f}" if val == int(val) else f"{val:.2f}"
            parts_q.append(s)
            parts_a.append(f"{s}'s {little_theme.lower()} is {{{val_fmt}}} {unit}")
        series_q = ", ".join(parts_q[:-1]) + f", and {parts_q[-1]}"
        series_a = ", ".join(parts_a[:-1]) + f", {parts_a[-1]}"
        q = f"What are the {little_theme.lower()} of {series_q} in {yr}?"
        a = series_a + "."
        qa_list.append({"Q": q, "A": a})
        taken += 1

    return qa_list


def fill_qa_ve(extreme_points_n1: List[Dict[str, Any]],
               metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    根据 task_get_extreme_y_points(n=1, by_series=True) 的结果，
    为每条折线生成两类 QA：
      ① 最高值  Q: What is the highest {little_theme} level recorded for {series}?
      ② 最低值  Q: What is the lowest  {little_theme} level recorded for {series}?
    """
    qa_list: List[Dict[str, str]] = []
    if not extreme_points_n1:
        return qa_list

    little_theme = metadata.get("little_theme", "")
    y_unit = metadata.get("y_info", {}).get("unit", "")

    # 先聚合出各 series 的最大 / 最小点
    series_extremes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in extreme_points_n1:
        series = item.get("series")
        typ = item.get("type")  # 'largest' / 'smallest'
        if series and typ in ("largest", "smallest"):
            series_extremes.setdefault(series, {})[typ] = item

    for series, ext_dict in series_extremes.items():
        # ---------- ① 最高 ----------
        if "largest" in ext_dict:
            pt = ext_dict["largest"]
            y_val = pt["y"]
            x_val = pt["x"]
            y_fmt = f"{y_val:.2f}" if isinstance(y_val, (float, int)) else y_val
            x_fmt = _safe_year(x_val)
            q = f"What is the highest {little_theme} level recorded for {series}?"
            a = f"{series} reached approximately {{{y_fmt}}} {y_unit} in {{{x_fmt}}}."
            qa_list.append({"Q": q, "A": a})

        # ---------- ② 最低 ----------
        if "smallest" in ext_dict:
            pt = ext_dict["smallest"]
            y_val = pt["y"]
            x_val = pt["x"]
            y_fmt = f"{y_val:.2f}" if isinstance(y_val, (float, int)) else y_val
            x_fmt = _safe_year(x_val)
            q = f"What is the lowest {little_theme} level recorded for {series}?"
            a = f"{series}'s lowest recorded level is approximately {{{y_fmt}}} {y_unit} in {{{x_fmt}}}."
            qa_list.append({"Q": q, "A": a})

    return qa_list


def fill_series_extremes(extreme_points_n1, metadata):
    """把每条线的最高 & 最低点问答归入 EVJ."""
    return fill_qa_ve(extreme_points_n1, metadata)  # 复用原实现


# --- 折线图 Statistical Comparison（SC） ---
def fill_qa_sc(avg_each: Dict[str, float] | None,
               roc_each: Dict[str, float] | None,
               metadata: Dict[str, Any],
               max_q: int = 4) -> List[Dict[str, str]]:
    """
    生成【平均值 + 变化率】两类问题的完整池，然后从中随机选取 max_q 个 QA。
    这样确保选出的 max_q 个问题是完全随机组合，不固定 pairing。
    """
    all_possible_qa: List[Dict[str, str]] = []
    y_unit = metadata.get("y_info", {}).get("unit", "")
    little_theme = metadata.get("little_theme", "")

    # ---- ① 生成所有可能的平均值问题 ----
    if avg_each:
        for name, value in avg_each.items():
            # 检查值是否存在且不是 NaN
            if pd.notna(value):
                avg_fmt = f"{value:.2f}"
                q_avg = f"What is the average value of {name}'s {little_theme}?"
                a_avg = f"The average value of {name}'s {little_theme.lower()} is {{{avg_fmt}}} {y_unit}."
                all_possible_qa.append({"Q": q_avg, "A": a_avg})

    # ---- ② 生成所有可能的变化率问题 ----
    if roc_each:
        for name, value in roc_each.items():
            # 检查值是否存在且不是 NaN
            if pd.notna(value):
                roc_fmt = f"{abs(value):.2f}"
                q_roc = (f"What is the rate of change in {name}'s {little_theme} "
                         f"from the starting year to the final year?")
                # Determine if it's a growth or decline for better phrasing
                trend_desc = "growth" if value > 0 else ("decline" if value < 0 else "no change")
                a_roc = (f"The rate of {trend_desc} in {name}'s {little_theme.lower()} is "
                         f"{{{roc_fmt}}}%."
                         )
                all_possible_qa.append({"Q": q_roc, "A": a_roc})

    # ---- ③ 随机打乱所有可能的问题列表 ----
    # Now all_possible_qa contains all average and ROC questions for labels with data
    random.shuffle(all_possible_qa)

    # ---- ④ Return the first max_q questions ----
    # If the total number of possible questions is less than max_q, return all of them.
    return all_possible_qa[:max_q]


def _safe_year(val):
    """
    如果 val 本身就是 4 位年份（或可转为 4 位整数），直接返回；
    否则尝试用 pd.to_datetime 解析，失败就返回原值。
    """
    try:
        # Check for integer year representation (e.g., 2000, 2000.0)
        num = int(float(val))
        if 1000 <= num <= 3000:
            return num
    except (ValueError, TypeError):
        pass  # Not a simple number

    # Try parsing as a datetime
    ts = pd.to_datetime(str(val), errors="coerce")
    if pd.notna(ts):
        # Return year if it's a valid date/year
        return ts.year
    else:
        # Fallback: return original value if parsing fails
        return val


def _ensure_year_col(df: pd.DataFrame) -> pd.Series:
    return df["x"].apply(_safe_year)


def _to_year(col):
    # Ensure this function handles potential non-numeric or non-date values gracefully
    # Using _safe_year which already does this
    return pd.Series(col).apply(_safe_year)


def _pair_str(val: float, unit: str, year) -> str:
    """格式化 ‘300 million barrels in {2000} year’ 片段"""
    # 年份若为 2000.0 → 2000
    year_fmt = year
    if isinstance(year, float) and year.is_integer():
        year_fmt = int(year)
    elif isinstance(year, pd.Timestamp):
        year_fmt = year.year  # Extract year from Timestamp
    else:
        year_fmt = str(year)  # Keep as string if cannot format as year

    # Format value: use .0f if it's an integer, otherwise .2f
    v_fmt = f"{val:.0f}" if val == int(val) else f"{val:.2f}"

    # Include unit only if it's not empty
    unit_str = f" {unit}" if unit else ""

    return f"{{{v_fmt}}}{unit_str} in {{{year_fmt}}}"


def fill_qa_nf(df_long: pd.DataFrame,
               metadata: Dict[str, Any],
               seed: int | None = None,
               max_q: int = 4) -> List[Dict[str, str]]:
    """
    生成数值筛选 QA，随机选取主体或年份，避免重复。
    每类随机生成 1 题（若数据不足则跳过）。返回 [{'Q':..., 'A':...}, ...]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    qa_set: "OrderedDict[str, Dict[str,str]]" = collections.OrderedDict()  # Use OrderedDict to store unique QAs
    if df_long is None or df_long.empty:
        return []

    little_theme = metadata.get("little_theme", "")
    unit = metadata.get("y_info", {}).get("unit", "")

    # 预处理：加 year 列
    df = df_long.copy()
    df["year"] = _ensure_year_col(df)
    df = df.dropna(subset=["y"])  # 确保 y 有值

    series_names = df["series"].unique().tolist()
    years_all = df["year"].unique().tolist()

    # Define possible question types
    q_types = ['series_gt', 'series_lt', 'series_between',
               'year_gt', 'year_lt', 'year_between']

    tries = 0
    # Try to generate up to max_q unique questions
    while len(qa_set) < max_q and tries < 30:  # Limit tries to prevent infinite loops on sparse data
        tries += 1
        # Randomly pick a type for THIS question attempt
        q_type = random.choice(q_types)

        try:
            if q_type in ['series_gt', 'series_lt', 'series_between']:
                # Need series and years with data
                if not series_names: continue
                # Pick a random series for THIS question
                series = random.choice(series_names)
                df_s = df[df["series"] == series].dropna(subset=["y"])
                if df_s.empty or len(df_s) < 2: continue  # Need at least 2 points for percentiles

                vals = df_s["y"]
                # Calculate thresholds based on this series' data
                # Use 25th and 75th percentiles for a more robust range
                lo, hi = np.percentile(vals, [25, 75])
                # Adjust interval slightly to avoid edge cases
                mid_low = lo + (hi - lo) * 0.1
                mid_high = hi - (hi - lo) * 0.1

                # Ensure thresholds are within the data range and distinct
                if lo >= hi: continue  # Cannot form a valid interval/comparison

                if q_type == 'series_gt':
                    # Ensure threshold is meaningful (not above max)
                    threshold = min(hi, vals.max() * 0.9)  # Use 75th percentile or slightly below max
                    if threshold <= vals.min(): continue  # Threshold must be above min
                    # Find years and values exceeding threshold for this series
                    results_df = df_s[df_s["y"] > threshold]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        years_vals = results_df[["year", "y"]].values
                        q = f"Which years did {series}'s {little_theme} exceed {threshold:.0f} {unit}? Please list the years and corresponding values."
                        parts = [_pair_str(v, unit, y) for y, v in years_vals]
                        a = ", ".join(parts) + "."
                        # Add to set to ensure uniqueness
                        qa_set.setdefault(q, {"Q": q, "A": a})

                elif q_type == 'series_lt':
                    # Ensure threshold is meaningful (not below min)
                    threshold = max(lo, vals.min() * 1.1)  # Use 25th percentile or slightly above min
                    if threshold >= vals.max(): continue  # Threshold must be below max
                    # Find years and values below threshold for this series
                    results_df = df_s[df_s["y"] < threshold]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        years_vals = results_df[["year", "y"]].values
                        q = f"Which years did {series}'s {little_theme} below {threshold:.0f} {unit}? Please list the years and corresponding values."
                        parts = [_pair_str(v, unit, y) for y, v in years_vals]
                        a = ", ".join(parts) + "."
                        qa_set.setdefault(q, {"Q": q, "A": a})

                elif q_type == 'series_between' and mid_low < mid_high:  # Ensure interval is valid
                    # Find years and values within interval for this series
                    cond = (df_s["y"] >= mid_low) & (df_s["y"] <= mid_high)
                    results_df = df_s[cond]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        years_vals = results_df[["year", "y"]].values
                        q = (f"Which years did {series}'s {little_theme} between "
                             f"{mid_low:.0f} and {mid_high:.0f} {unit}? Please list the years and corresponding values.")
                        parts = [_pair_str(v, unit, y) for y, v in years_vals]
                        a = ", ".join(parts) + "."
                        qa_set.setdefault(q, {"Q": q, "A": a})


            elif q_type in ['year_gt', 'year_lt', 'year_between']:
                # Need years with data
                if not years_all: continue
                # Pick a random year for THIS question
                year = random.choice(years_all)
                df_y = df[df["year"] == year].dropna(subset=["y"])
                if df_y.empty or len(
                    df_y) < 2: continue  # Need at least 2 series with data in this year for percentiles

                vals = df_y["y"]
                # Calculate thresholds based on this year's data
                # Use 25th and 75th percentiles for a more robust range
                lo, hi = np.percentile(vals, [25, 75])
                # Adjust interval slightly
                mid_low = lo + (hi - lo) * 0.1
                mid_high = hi - (hi - lo) * 0.1

                # Ensure thresholds are within the data range and distinct
                if lo >= hi: continue  # Cannot form a valid interval/comparison

                if q_type == 'year_gt':  # Ensure threshold is meaningful
                    threshold = min(hi, vals.max() * 0.9)  # Use 75th percentile or slightly below max
                    if threshold <= vals.min(): continue  # Threshold must be above min
                    # Find series and values exceeding threshold in this year
                    results_df = df_y[df_y["y"] > threshold]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        series_vals = results_df[["series", "y"]].sample(min(3, len(results_df))).values
                        q = (f"In the {year}, which line had {little_theme} exceed "
                             f"Please list the lines and corresponding values.")
                        parts = [f"{{{s}}} had {{{v:.2f}}} {unit}" for s, v in series_vals]
                        a = ", ".join(parts) + "."
                        qa_set.setdefault(q, {"Q": q, "A": a})

                elif q_type == 'year_lt':  # Ensure threshold is meaningful
                    threshold = max(lo, vals.min() * 1.1)  # Use 25th percentile or slightly above min
                    if threshold >= vals.max(): continue  # Threshold must be below max
                    # Find series and values below threshold in this year
                    results_df = df_y[df_y["y"] < threshold]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        series_vals = results_df[["series", "y"]].sample(min(3, len(results_df))).values
                        q = (f"In the {year}, which lines had {little_theme} below {threshold:.0f} {unit}? "
                             f"Please list the lines and corresponding values.")
                        parts = [f"{{{s}}} had {{{v:.2f}}} {unit}" for s, v in series_vals]
                        a = ", ".join(parts) + "."
                        qa_set.setdefault(q, {"Q": q, "A": a})

                elif q_type == 'year_between' and mid_low < mid_high:  # Ensure interval is valid
                    # Find series and values within interval in this year
                    cond = (df_y["y"] >= mid_low) & (df_y["y"] <= mid_high)
                    results_df = df_y[cond]
                    if not results_df.empty:
                        # Sample up to 3 results randomly
                        series_vals = results_df[["series", "y"]].sample(min(3, len(results_df))).values
                        q = (f"In the {year}, which lines had {little_theme} between "
                             f"{mid_low:.0f} and {mid_high:.0f} {unit}? Please list the lines and corresponding values.")
                        parts = [f"{{{s}}} had {{{v:.2f}}} {unit}" for s, v in series_vals]
                        a = ", ".join(parts) + "."
                        qa_set.setdefault(q, {"Q": q, "A": a})

        except Exception as e:
            # Catch potential errors during sampling or percentile calculation on small datasets
            # print(f"Warning: Could not generate NF question of type {q_type} (try {tries}): {e}") # Uncomment for debugging
            continue  # Try generating another question

    # Return list of unique QAs, up to max_q. Order is preserved by OrderedDict.
    # Optionally shuffle the final list if desired, but OrderedDict ensures variety of types first.
    # qa_list = list(qa_set.values())
    # random.shuffle(qa_list)
    return list(qa_set.values())


def fill_qa_nc(df_long: pd.DataFrame,
               metadata: Dict[str, Any],
               seed: int | None = None,
               max_q: int = 4) -> List[Dict[str, str]]:
    """
    生成 NC（Numerical Comparison）问答，随机选取年份、区间或主体，避免重复。
    """
    if df_long is None or df_long.empty:
        return []

    if seed is not None:
        random.seed(seed);
        np.random.seed(seed)

    qa_set: "OrderedDict[str, Dict[str,str]]" = collections.OrderedDict()  # Use OrderedDict for unique QAs
    little_theme = metadata.get("little_theme", "")
    # unit = metadata.get("y_info", {}).get("unit", "") # Unit is not typically in NC answers

    df = df_long.copy()
    df["year"] = _to_year(df["x"])
    df = df.dropna(subset=["y"])

    series_names = df["series"].unique().tolist()
    years_all = df["year"].unique().tolist()

    # Define possible comparison types
    comparison_types = ['same_year', 'interval_avg', 'interval_change', 'subject_year_compare']

    tries = 0
    # Try to generate up to max_q unique questions
    while len(qa_set) < max_q and tries < 30:  # Limit tries
        tries += 1
        # Randomly pick a comparison type for THIS question attempt
        comp_type = random.choice(comparison_types)

        try:
            if comp_type == 'same_year':
                # Pick a random year for THIS Same Year question
                valid_years = df.groupby("year").filter(lambda x: x["series"].nunique() >= 2)["year"].unique().tolist()
                if not valid_years: continue
                year = random.choice(valid_years)
                rows_year = df[df["year"] == year].dropna(subset=["y"])
                series_vals = rows_year.groupby("series")["y"].first().dropna()  # Ensure values are not NaN
                if len(series_vals) < 2: continue  # Should be >= 2 due to filter, but safeguard

                # Sample 2-4 series randomly for comparison
                k = min(len(series_vals), random.choice([2, 3, 4]))
                # Ensure sampled series have valid data in this year before sampling
                sampled_series_names = random.sample(series_vals.index.tolist(), k)
                sampled_vals = series_vals.loc[sampled_series_names]

                winner = sampled_vals.idxmax()
                s_list = sampled_vals.index.tolist()

                if k == 2:
                    s1, s2 = s_list
                    q = (f"In {year}, which line had a higher {little_theme}? "
                         f"{{{s1}}} or {{{s2}}}.")
                    a = f"In {year}, {{{winner}}} had a higher {little_theme.lower()}."
                else:
                    others = ", ".join([f"{s}" for s in s_list[:-1]]) + f", or {s_list[-1]}"
                    q = (f"In {year}, which line had the highest {little_theme}? "
                         f"{others}.")
                    a = f"In {year}, {{{winner}}} had the highest {little_theme.lower()}."
                # Add to set to ensure uniqueness
                qa_set.setdefault(q, {"Q": q, "A": a})

            elif comp_type == 'interval_avg':
                # Pick a random interval for THIS Interval Average question
                valid_years = df.groupby("year").filter(lambda x: x["series"].nunique() >= 2)["year"].unique().tolist()
                if len(valid_years) < 2: continue
                # Sample 2 distinct years to form an interval
                start_year, end_year = sorted(random.sample(valid_years, 2))
                mask = (df["year"] >= start_year) & (df["year"] <= end_year)
                # Calculate means for series within the interval that have data
                means = df[mask].groupby("series")["y"].mean().dropna()
                if len(means) < 2: continue  # Need at least 2 series with average data in this interval

                # Sample 2-4 series randomly for comparison
                k = min(len(means), random.choice([2, 3, 4]))
                sampled_series_names = random.sample(means.index.tolist(), k)
                sampled_means = means.loc[sampled_series_names]

                winner = sampled_means.idxmax()
                s_list = sampled_means.index.tolist()

                if k == 2:
                    s1, s2 = s_list
                    q = (f"Between {start_year} and {end_year}, which line had a higher "
                         f"average {little_theme}? {s1} or {s2}.")
                    a = f"{{{winner}}} had a higher average {little_theme.lower()}."
                else:
                    others = ", ".join([f"{s}" for s in s_list[:-1]]) + f", or {s_list[-1]}"
                    q = (f"Between {start_year} and {end_year}, which line had the highest "
                         f"average {little_theme}? {others}.")
                    a = f"{{{winner}}} had the highest average {little_theme.lower()}."
                qa_set.setdefault(q, {"Q": q, "A": a})

            elif comp_type == 'interval_change':
                # Pick a random interval for THIS Interval Change question
                # Need years where at least 2 series have data in BOTH start and end year
                # Find years with at least 2 series
                years_with_min_series = df.groupby("year").filter(lambda x: x["series"].nunique() >= 2)[
                    "year"].unique().tolist()
                if len(years_with_min_series) < 2: continue

                # Try sampling pairs of years until we find one with common series
                found_interval = False
                interval_tries = 0
                while not found_interval and interval_tries < 10:
                    interval_tries += 1
                    start_year, end_year = sorted(random.sample(years_with_min_series, 2))
                    df_start = df[df["year"] == start_year].set_index("series")["y"]
                    df_end = df[df["year"] == end_year].set_index("series")["y"]
                    # Only consider series present in BOTH start and end year and have non-NaN values
                    common = df_start.dropna().index.intersection(df_end.dropna().index)
                    if len(common) >= 2:
                        found_interval = True
                        # Calculate absolute change
                        # Ensure we only calculate for series present in both start and end AND have non-NaN values
                        valid_series_in_interval = df_start.loc[common].dropna().index.intersection(
                            df_end.loc[common].dropna().index)
                        if len(
                            valid_series_in_interval) < 2: continue  # Need at least 2 series with valid data in both years

                        deltas = (df_end.loc[valid_series_in_interval] - df_start.loc[
                            valid_series_in_interval]).abs().dropna()
                        if len(deltas) < 2: continue  # Need at least 2 series with valid change
                        # Sample 2-4 series randomly for comparison of change
                        k = min(len(deltas), random.choice([2, 3, 4]))
                        sampled_series_names = random.sample(deltas.index.tolist(), k)
                        sampled_deltas = deltas.loc[sampled_series_names]

                        winner = sampled_deltas.idxmax()  # Series with largest absolute change
                        s_list = sampled_deltas.index.tolist()

                        if k == 2:
                            s1, s2 = s_list
                            q = (f"Between {start_year} and {end_year}, which line experienced a "
                                 f"larger change in {little_theme}? {s1} or {s2}.")
                            a = f"{{{winner}}} experienced a larger change in {little_theme.lower()}."
                        else:
                            others = ", ".join([f"{s}" for s in s_list[:-1]]) + f", or {s_list[-1]}"
                            q = (f"Between {start_year} and {end_year}, which line experienced the "
                                 f"largest change in {little_theme}? {others}.")
                            a = f"{{{winner}}} experienced the largest change in {little_theme.lower()}."
                        qa_set.setdefault(q, {"Q": q, "A": a})
                if not found_interval: continue  # If we couldn't find a valid interval after tries

            elif comp_type == 'subject_year_compare':
                # Pick a random subject for THIS Subject Year Compare question
                valid_series = df.groupby("series").filter(lambda x: x["year"].nunique() >= 2)[
                    "series"].unique().tolist()
                if not valid_series: continue
                series = random.choice(valid_series)
                df_s = df[df["series"] == series].dropna(subset=["y"])
                yrs_s = df_s["year"].unique().tolist()
                if len(yrs_s) < 2: continue  # Should be >= 2 due to filter, but safeguard

                # Sample 2 distinct years for this subject
                y1, y2 = sorted(random.sample(yrs_s, 2))
                v1 = df_s[df_s["year"] == y1]["y"].iloc[0]
                v2 = df_s[df_s["year"] == y2]["y"].iloc[0]
                rel = "higher" if v1 > v2 else ("lower" if v1 < v2 else "the same")
                q = (f"Was {series}'s {little_theme} in {y1} higher or lower than in "
                     f"{y2}?")
                a = (f"The {little_theme.lower()} in {y1} was {{{rel}}} than in {y2}.")
                # Use setdefault to avoid duplicates based on the question string
                qa_set.setdefault(q, {"Q": q, "A": a})

        except Exception as e:
            # Catch potential errors during sampling or calculations
            # print(f"Warning: Could not generate NC question of type {comp_type} (try {tries}): {e}") # Uncomment for debugging
            continue  # Try generating another question

    # Return list of unique QAs, up to max_q. Order is preserved by OrderedDict.
    # Optionally shuffle the final list if desired
    # qa_list = list(qa_set.values())
    # random.shuffle(qa_list)
    return list(qa_set.values())


def fill_qa_evj(global_extremes: Dict[str, Any],
                metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    根据 task_get_global_min_max(df_long) 的结果，生成：
      • 全局最小值 QA
      • 全局最大值 QA
    例：
      Q: What is the global minimum Crop Yield in the line chart?
      A: The global minimum crop yield is {2.45} tons/hectare.
    """
    qa_list: List[Dict[str, str]] = []
    little_theme = metadata.get("little_theme", "")
    y_unit = metadata.get("y_info", {}).get("unit", "")

    # ---------- 全局最小 ----------
    if "y_min" in global_extremes and pd.notna(global_extremes["y_min"]):
        y_min = global_extremes["y_min"]
        y_min_fmt = f"{y_min:.2f}" if isinstance(y_min, (int, float)) else y_min
        q = f"What is the global minimum {little_theme} in the line chart?"
        a = f"The global minimum {little_theme.lower()} is {{{y_min_fmt}}} {y_unit}."
        qa_list.append({"Q": q, "A": a})

    # ---------- 全局最大 ----------
    if "y_max" in global_extremes and pd.notna(global_extremes["y_max"]):
        y_max = global_extremes["y_max"]
        y_max_fmt = f"{y_max:.2f}" if isinstance(y_max, (int, float)) else y_max
        q = f"What is the global maximum {little_theme} in the line chart?"
        a = f"The global maximum {little_theme.lower()} is {{{y_max_fmt}}} {y_unit}."
        qa_list.append({"Q": q, "A": a})

    return qa_list


def fill_qa_msr(df_long: pd.DataFrame, metadata: Dict[str, Any], max_q: int = 4) -> List[Dict[str, str]]:
    """
    Generates QA for MSR (Multiple Series Comparison) based on specific templates.

    Q: Which label shows the largest fluctuation in the line chart?
    A: The label with the largest fluctuation is {{{answer}}}.

    Q: Which {label_type} shows the fastest average growth rate between {year_start} and {year_end}?
    A: {{{answer}}} has the fastest average growth rate between {year_start} and {year_end}.

    Q: In which year did {series_1} first surpass {series_2}?
    A: {series_1} first surpassed {series_2} in the {{{answer}}}.

    Q: Which label had the leading {metric} since {year}?
    A: Since {year}, {{{answer}}} had the leading {metric}.
    """
    qa_set: "OrderedDict[str, Dict[str, str]]" = collections.OrderedDict()  # Use OrderedDict for unique QAs

    if df_long is None or df_long.empty:
        return []

    df = df_long.copy()
    df["year"] = _to_year(df["x"])
    df = df.dropna(subset=["y"])  # Ensure y has value and x can be converted to year

    series_names = df["series"].unique().tolist()
    years_all = sorted(df["year"].unique().tolist())

    if len(series_names) < 2:
        # print("Not enough series for MSR questions")
        return []  # Need at least two series for most MSR questions
    if len(years_all) < 2:
        # print("Not enough years for MSR questions")
        return []  # Need at least two years for time-based questions

    # Define possible MSR question types
    msr_types = ['fluctuation', 'growth_rate', 'first_surpass', 'leading_since']

    # --- MODIFICATION START ---
    # Iterate through each question type and attempt to generate ONE question of that type
    for msr_type in msr_types:
        # We will attempt to generate one question for the current msr_type.
        # If data doesn't support it, the 'continue' inside the type's logic will skip adding a QA for this type.
        # We don't need a separate 'tries' counter here, as we try each type only once in this outer loop.

        try:
            if msr_type == 'fluctuation':
                # Calculate standard deviation of y for each series as fluctuation measure
                # Only consider series with at least 2 points to calculate std dev
                fluctuations = df.groupby("series").filter(lambda x: len(x) >= 2).groupby("series")["y"].std().dropna()
                if fluctuations.empty: continue  # Skip if no fluctuation can be calculated

                # Find series with largest fluctuation
                largest_fluctuation_series = fluctuations.idxmax()

                q = "Which label shows the largest fluctuation in the line chart?"
                a = f"The label with the largest fluctuation is {{{largest_fluctuation_series}}}."
                qa_set.setdefault(q, {"Q": q, "A": a})  # Add if not already exists (shouldn't for this type)

            elif msr_type == 'growth_rate':
                # Pick a random interval (at least 2 years apart)
                if len(years_all) < 2: continue  # Need at least 2 years

                # Pick a random start year, ensuring there's at least one year after it
                possible_start_years = years_all[:-1]
                if not possible_start_years: continue
                year_start = random.choice(possible_start_years)

                # Pick a random end year after the start year
                possible_end_years = [y for y in years_all if y > year_start]
                if not possible_end_years: continue
                year_end = random.choice(possible_end_years)

                # Calculate growth rate for each series in the interval
                growth_rates: Dict[str, float] = {}
                # Filter series to only include those with data in both start_year and end_year
                # Use a temporary dataframe for filtering to avoid modifying original df state in loop
                df_interval = df[(df["year"] == year_start) | (df["year"] == year_end)]
                series_in_interval_df = df_interval.groupby("series").filter(
                    lambda x: len(x) == 2)  # Series with exactly 2 points (start and end)

                series_in_interval = series_in_interval_df["series"].unique().tolist()

                if len(series_in_interval) < 2: continue  # Need at least 2 series with data in both years

                for series in series_in_interval:
                    df_s = series_in_interval_df[series_in_interval_df["series"] == series].set_index("year")
                    y_start_val = df_s.loc[year_start, "y"] if year_start in df_s.index else np.nan
                    y_end_val = df_s.loc[year_end, "y"] if year_end in df_s.index else np.nan

                    if pd.notna(y_start_val) and pd.notna(y_end_val):
                        # Calculate rate of change (not percentage growth)
                        if year_end > year_start:
                            rate = (y_end_val - y_start_val) / (year_end - year_start)
                            growth_rates[series] = rate

                if len(growth_rates) < 2: continue  # Need at least 2 series with calculated growth rates

                # Find series with fastest average growth rate (highest rate)
                # If all are negative, find the one with the smallest decrease (rate closest to 0 from below)
                if growth_rates:
                    fastest_series = max(growth_rates, key=growth_rates.get)
                    label_type = "line"  # Or "series"
                    q = f"Which {label_type} shows the fastest average growth rate between {year_start} and {year_end}?"
                    a = f"{{{fastest_series}}} has the fastest average growth rate between {year_start} and {year_end}."
                    qa_set.setdefault(q, {"Q": q, "A": a})
                else:
                    continue  # No growth rates could be calculated


            elif msr_type == 'first_surpass':
                # Pick two random series. Need at least two series.
                if len(series_names) < 2: continue
                # Try a few times to find a pair that might surpass each other if needed
                surpass_pair_found = False
                for _ in range(10):  # Limit internal tries for a suitable pair
                    s1, s2 = random.sample(series_names, 2)

                    # Get data for these two series and filter for common years with data
                    df_s1 = df[df["series"] == s1].set_index("year").dropna(subset=["y"])
                    df_s2 = df[df["series"] == s2].set_index("year").dropna(subset=["y"])

                    # Find common years with data for both series
                    common_years = sorted(list(df_s1.index.intersection(df_s2.index)))
                    if len(common_years) < 2: continue  # Need at least two common points to check for surpassing

                    first_surpass_year_s1_over_s2 = None
                    first_surpass_year_s2_over_s1 = None

                    # Check for s1 surpassing s2
                    for i in range(1, len(common_years)):
                        year = common_years[i]
                        prev_year = common_years[i - 1]

                        # Ensure both series have data in current and previous year
                        if (prev_year in df_s1.index and prev_year in df_s2.index and
                                year in df_s1.index and year in df_s2.index):

                            y1_prev = df_s1.loc[prev_year, "y"]
                            y2_prev = df_s2.loc[prev_year, "y"]
                            y1_curr = df_s1.loc[year, "y"]
                            y2_curr = df_s2.loc[year, "y"]

                            # Check if s1 was NOT strictly greater than s2 in the previous point AND s1 IS strictly greater in the current point
                            if (y1_prev <= y2_prev) and (y1_curr > y2_curr):
                                first_surpass_year_s1_over_s2 = year
                                break  # Found the first time s1 surpassed s2

                    # Check for s2 surpassing s1 (using the same common years)
                    for i in range(1, len(common_years)):
                        year = common_years[i]
                        prev_year = common_years[i - 1]
                        if (prev_year in df_s1.index and prev_year in df_s2.index and
                                year in df_s1.index and year in df_s2.index):
                            y1_prev = df_s1.loc[prev_year, "y"]
                            y2_prev = df_s2.loc[prev_year, "y"]
                            y1_curr = df_s1.loc[year, "y"]
                            y2_curr = df_s2.loc[year, "y"]
                            if (y2_prev <= y1_prev) and (y2_curr > y1_curr):
                                first_surpass_year_s2_over_s1 = year
                                break  # Found the first time s2 surpassed s1

                    # If at least one surpassing event found, add the question(s) and break the internal loop
                    if first_surpass_year_s1_over_s2:
                        q = f"In which year did {s1} first surpass {s2}?"
                        a = f"{s1} first surpassed {s2} in {{{first_surpass_year_s1_over_s2}}}."
                        qa_set.setdefault(q, {"Q": q, "A": a})
                        surpass_pair_found = True  # Mark that we found a suitable pair and added questions

                    if first_surpass_year_s2_over_s1:
                        q = f"In which year did {s2} first surpass {s1}?"
                        a = f"{s2} first surpassed {s1} in {{{first_surpass_year_s2_over_s1}}}."
                        qa_set.setdefault(q, {"Q": q, "A": a})
                        surpass_pair_found = True  # Mark that we found a suitable pair and added questions

                    if surpass_pair_found:
                        break  # Exit the internal loop once a surpassing pair is found and QAs are added

                if not surpass_pair_found:
                    continue  # If no surpassing pair was found after tries, skip this type.


            elif msr_type == 'leading_since':
                # Pick a random start year from the middle/later part of the range
                if len(years_all) < 3: continue  # Need at least 3 years to have a meaningful "since" year
                # Exclude the last year and potentially the first year
                possible_start_years = years_all[1:-1]  # Years excluding first and last
                if not possible_start_years: continue
                start_year = random.choice(possible_start_years)

                # Filter data from start_year onwards
                df_since = df[df["year"] >= start_year].dropna(subset=["y"])

                # Need at least 2 series with data points *since* the start year
                series_since = df_since["series"].unique().tolist()
                if len(series_since) < 2: continue

                # Calculate average y for each series since start_year
                avg_since = df_since.groupby("series")["y"].mean().dropna()

                if len(avg_since) < 2: continue

                # Find series with the highest average since start_year
                leading_series = avg_since.idxmax()
                metric_name = metadata.get("little_theme", "value").lower()  # Use little_theme as the metric name

                q = f"Which label had the leading {metric_name} since {start_year}?"
                a = f"Since {start_year}, {{{leading_series}}} had the leading {metric_name}."
                qa_set.setdefault(q, {"Q": q, "A": a})

        except Exception as e:
            # Catch potential errors during sampling or calculations
            # print(f"Warning: Could not generate MSR question of type {msr_type} (try {tries}): {e}") # Uncomment for debugging
            continue  # Try generating another question

    return list(qa_set.values())

# --- Douglas-Peucker Helper Functions ---
def _perpendicular_distance(point, line_start, line_end):
    """Calculates the perpendicular distance from a point to a line segment."""
    # point: (px, py)
    # line_start: (x0, y0)
    # line_end: (x1, y1)
    px, py = point
    x0, y0 = line_start
    x1, y1 = line_end

    # Calculate the line vector
    line_vec = (x1 - x0, y1 - y0)
    # Calculate the vector from the start point to the point
    point_vec = (px - x0, py - y0)

    # Calculate the squared length of the line segment
    line_len_sq = line_vec[0]**2 + line_vec[1]**2

    if line_len_sq == 0:
        # The start and end points are the same, distance is the distance between point and the single point
        return math.dist(point, line_start)

    # Calculate the projection of the point vector onto the line vector
    # t = dot_product(point_vec, line_vec) / line_len_sq
    t = (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq

    # Clamp t to [0, 1] to find the closest point on the *segment*
    t = max(0.0, min(1.0, t)) # Ensure t is float literals

    # Calculate the closest point on the line segment
    closest_point_on_line = (x0 + t * line_vec[0], y0 + t * line_vec[1])

    # Calculate the distance from the point to the closest point on the segment
    distance = math.dist(point, closest_point_on_line)
    return distance

def _douglas_peucker_recursive(points, index_start, index_end, epsilon):
    """Recursive helper for Douglas-Peucker."""
    # Base case: Segment is 0 or 1 points
    if index_end <= index_start:
        # Return the start index if it's a valid point, otherwise empty list.
        # This handles segments of length 0 or 1.
        return [index_start] if 0 <= index_start < len(points) else []


    # Find the point with the maximum distance
    max_dist = 0
    index_max_dist = -1 # Initialize to -1
    line_start_point = points[index_start]
    line_end_point = points[index_end]

    # Iterate over intermediate points (excluding start and end)
    for i in range(index_start + 1, index_end):
        dist = _perpendicular_distance(points[i], line_start_point, line_end_point)
        if dist > max_dist:
            max_dist = dist
            index_max_dist = i

    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Simplify the two segments divided by the point of maximum distance
        # The recursive calls return indices relative to the *original* points list
        rec_results_1 = _douglas_peucker_recursive(points, index_start, index_max_dist, epsilon)
        rec_results_2 = _douglas_peucker_recursive(points, index_max_dist, index_end, epsilon)

        # Combine the results and remove duplicates (the split point index_max_dist is the end of seg1 and start of seg2)
        # The split point index_max_dist is handled correctly by the recursive calls' range and base case.
        # We just need to combine the lists. The split point will appear in one or both lists if preserved.
        # A simple concatenation and unique/sort should work.
        combined_indices = sorted(list(set(rec_results_1 + rec_results_2)))
        return combined_indices
    else:
        # Max distance is less than or equal to epsilon, discard intermediate points
        return sorted(list(set([index_start, index_end]))) # Always include start and end, handle potential start==end


def douglas_peucker(points, epsilon):
    """Main Douglas-Peucker simplification function."""
    if len(points) < 2:
        # Handle segments with less than 2 points directly.
        return list(range(len(points)))

    if epsilon <= 0:
        # No simplification with epsilon <= 0
        return list(range(len(points)))

    # Ensure points are sorted by x-coordinate before applying DP.
    # The points list is assumed to be sorted by x based on how it's created in fill_qa_va.

    # Get indices of preserved points
    preserved_indices = _douglas_peucker_recursive(points, 0, len(points) - 1, epsilon)

    # The recursive function returns indices relative to the input 'points' list
    return preserved_indices


def fill_qa_va(df_long: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates QA for Visual Analysis (VA), specifically using the Douglas-Peucker algorithm
    to simplify a line and list the preserved x-coordinates.

    exampleQA：
    Q: Perform the Douglas-Peucker algorithm to simplify the line representing the China. List the x-coordinates of data points to be preserved. (5 points need to be provided)
    A: The x-coordinates of the data points to be preserved are: 2021, 2022，2023, 2025.

    template:
    Q: Perform the Douglas-Peucker algorithm to simplify the line representing the {s}. List the x-coordinates of data points to be preserved.({num_points} points need to be provided)
    A: The x-coordinates of the data points to be preserved are: {y1}, {y2}，{y3}, {y4}.
    """
    qa_list: List[Dict[str, str]] = []
    if df_long is None or df_long.empty:
        return qa_list

    # Ensure x is sortable and y is numeric
    df = df_long.copy()
    # Use _safe_year for x to handle potential date/string formats consistently
    df["x_sorted"] = df["x"].apply(_safe_year) # Create a sortable version of x
    # Drop rows where x_sorted or y is NaN before sorting and processing
    df = df.dropna(subset=["x_sorted", "y"])
    # Ensure x_sorted is numeric for range calculation, coerce errors will turn invalid entries to NaN
    df["x_sorted_numeric"] = pd.to_numeric(df["x_sorted"], errors='coerce')
    df = df.dropna(subset=["x_sorted_numeric"]) # Drop rows where x_sorted couldn't be made numeric

    df = df.sort_values(by=["series", "x_sorted_numeric"]).reset_index(drop=True) # Sort by series and the sortable numeric x

    series_names = df["series"].unique().tolist()

    # Need at least one series with enough points for simplification
    # Let's require at least 5 *valid* points (with non-NaN x_sorted_numeric and y)
    series_with_enough_points = [s for s in series_names if len(df[df["series"] == s]) >= 5]
    if not series_with_enough_points:
        # print("Not enough series with sufficient data points for Douglas-Peucker simplification.")
        return qa_list

    # Randomly pick one series to simplify
    selected_series = random.choice(series_with_enough_points)
    df_s = df[df["series"] == selected_series].reset_index(drop=True) # Reset index for DP

    # Prepare points for Douglas-Peucker: list of (x_sorted_numeric, y) tuples
    # Use x_sorted_numeric for the DP calculation as it's numeric and ordered correctly
    # We will retrieve the original x value later for the answer formatting.
    points = list(zip(df_s["x_sorted_numeric"].tolist(), df_s["y"].tolist()))
    original_x_values = df_s["x"].tolist() # Store original x values corresponding to sorted points

    # Determine epsilon dynamically based on the data range
    y_min = df_s["y"].min()
    y_max = df_s["y"].max()
    x_min_sortable = df_s["x_sorted_numeric"].min()
    x_max_sortable = df_s["x_sorted_numeric"].max()

    y_range = y_max - y_min if pd.notna(y_max) and pd.notna(y_min) else 0
    x_range_sortable = x_max_sortable - x_min_sortable if pd.notna(x_max_sortable) and pd.notna(x_min_sortable) else 0

    # Choose epsilon. A value relative to the scale of the data.
    # Let's try a small fraction of the maximum range of either axis.
    # Add a small floor to epsilon to ensure some simplification happens even on flat data.
    max_range = max(y_range, x_range_sortable)
    # Use a slightly larger floor or relative value to ensure some simplification occurs
    epsilon = max(0.5, max_range * 0.03) # Epsilon is at least 0.5 and at most 3% of the max range

    # Add a safeguard for cases where max_range is zero or very small
    if max_range <= 1e-6:
         epsilon = 0.5 # Use a default small epsilon if range is negligible

    # Perform simplification
    # The douglas_peucker function returns indices relative to the 'points' list
    try:
        # Need to pass points using x_sorted_numeric for DP calculation
        preserved_indices = douglas_peucker(points, epsilon)
    except Exception as e:
        print(f"Error during Douglas-Peucker simplification for series {selected_series}: {e}")
        return qa_list # Return empty if DP fails

    # Get the original x-coordinates of the preserved points using the stored list
    preserved_original_x_coords = [original_x_values[i] for i in preserved_indices]

    # --- MODIFICATION START ---
    # Calculate the number of preserved points
    num_preserved_points = len(preserved_original_x_coords)
    # --- MODIFICATION END ---

    # Format the x-coordinates for the answer.
    # Format using _safe_year to get consistent year representation if possible.
    formatted_x_coords = []
    for x_val in preserved_original_x_coords:
        formatted_x_coords.append(str(_safe_year(x_val)))

    # Join the formatted x-coordinates
    x_coords_str = ", ".join(formatted_x_coords)

    # Construct the Q&A pair
    # --- MODIFICATION START ---
    # Include the number of preserved points in the question string
    q = f"Perform the Douglas-Peucker algorithm to simplify the line representing the {selected_series}. List the x-coordinates of data points to be preserved.({num_preserved_points} points need to be provided)"
    # --- MODIFICATION END ---

    # Construct the answer string based on the example format
    a = f"The x-coordinates of the data points to be preserved are: {{{x_coords_str}}}."

    qa_list.append({"Q": q, "A": a})

    return qa_list



# 写入json，使用新的模板初始化结构并合并现有数据 (Adapted from heatmap_QA.py)
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]]):
    json_dir = 'QA'
    os.makedirs(json_dir, exist_ok=True)

    # Construct JSON file full path using the CSV base name
    # Take the basename and remove the .csv suffix
    base_name_with_suffix = os.path.basename(csv_path)  # e.g., bubble_Topic_1.csv
    base_name = os.path.splitext(base_name_with_suffix)[0]  # e.g., bubble_Topic_1

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
                print(
                    f"Warning: Existing JSON data in {json_path} is not a dictionary. Overwriting with template structure.")

        except (json.JSONDecodeError, FileNotFoundError):
            # File not found is handled by os.path.exists, but keeping it here as a safeguard
            print(f"Warning: Could not load or decode JSON from {json_path}. Starting with template structure.")
        except Exception as e:
            print(f"Warning: Could not read JSON from {json_path}: {e}. Starting with template structure.")

    # Merge existing data into the template structure
    # Start with the template, then copy over the lists from the existing data for any keys that exist and are lists
    data_to_save = template_data.copy()  # Start with all keys from the template
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
        existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save[qa_type] if
                             isinstance(item, dict) and 'Q' in item and 'A' in item}

        for item in qa_items:
            # Check if the item is a valid QA dictionary before trying to get Q and A
            if isinstance(item, dict) and 'Q' in item and 'A' in item:
                # Check if the item is already in the existing list based on the Q/A tuple
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
        print(
            f"Error: Attempted to write to invalid QA type '{qa_type}' in {json_path}. This type might be missing from the template.")

    # Write back to file
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        # print(f"Wrote QA to {json_path} under type {qa_type}") # Optional: confirmation print
    except Exception as e:
        print(f"Error writing QA to {json_path} for type {qa_type}: {e}")


def compute_all_tasks(df_long: pd.DataFrame,
                      meta: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    stats["extreme_points_1"] = task_get_extreme_y_points(df_long, n=1, by_series=True)
    stats["extreme_points_3"] = task_get_extreme_y_points(df_long, n=3, by_series=True)

    # ① 数据点数量
    stats["point_counts_total"] = task_count_points(df_long)
    stats["point_counts_each"] = task_count_points(df_long, by_series=True)

    # ② 全局 / 分系列极值
    stats["global_extremes"] = task_get_global_min_max(df_long)
    stats["series_extremes"] = task_get_global_min_max(df_long, by_series=True)

    # ③ 平均值
    stats["avg_y_overall"] = task_get_average_y(df_long)
    stats["avg_y_each"] = task_get_average_y(df_long, by_series=True)

    # ④ 首年→末年变化率
    stats["roc_overall"] = task_get_rate_of_change(df_long)
    stats["roc_each"] = task_get_rate_of_change(df_long, by_series=True)

    # ⑤ 每条线的最高 / 最低点（n=1）
    stats["extreme_points"] = task_get_extreme_y_points(df_long, n=1, by_series=True)

    # ⑥ 随机比较（同年、区间平均、区间斜率、单线两年对比） - Logic moved to fill_qa_nc
    # stats["compare"] = task_compare_subjects(df_long, meta)

    return stats


def main():
    # todo 修改路径和任务类型
    csv_folder = './csv'

    # 检查 CSV 文件夹是否存在
    if not os.path.exists(csv_folder):
        print(f"错误：未找到 CSV 文件夹 {csv_folder}。请先运行 line.py 生成数据。")
        return

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
        # avg_y_overall = stats["avg_y_overall"]  # compute_all_tasks 已经算好
        # roc_overall = stats["roc_overall"]

        # ---------- 生成 QA ----------
        qa_ctr = fill_qa_ctr()  # 图表类型
        qa_vec = fill_qa_vec(len(meta.get("series_names", [])))  # 线数 - Use .get with default for safety

        # 1. 趋势题：只保留前 4 条
        qa_trd_all = fill_qa_trend(meta)
        qa_trd = qa_trd_all[:4]  # Take up to 4 trend questions

        # 2. VE questions (Value Extraction - specific points by year/series)
        # fill_qa_ve_values generates random single/multi point lookups
        qa_ve = fill_qa_ve_values(df_long, meta, num_single=3, num_multi=1)  # Generate 3 single + 1 multi VE questions

        # 3. EVJ questions (Extremes - global min/max, series min/max)
        # fill_qa_evj gets global min/max
        # fill_series_extremes gets per-series min/max (re-uses fill_qa_ve logic)
        qa_evj = (fill_qa_evj(stats["global_extremes"], meta)
                  + fill_series_extremes(stats["extreme_points_1"], meta))[:4]  # Take up to 4 combined EVJ questions

        # 4. SC questions (Statistical Comparison - average, ROC per series)
        qa_sc = fill_qa_sc(stats.get("avg_y_each"), stats.get("roc_each"), meta, max_q=4)  # Take up to 4 SC questions

        # 5. NF questions (Numerical Filtering) - MODIFIED TO BE RANDOM PER Q
        qa_nf = fill_qa_nf(df_long, meta, max_q=4)  # Generate up to 4 random NF questions

        # 6. NC questions (Numerical Comparison) - MODIFIED TO BE RANDOM PER Q
        qa_nc = fill_qa_nc(df_long, meta, max_q=4)  # Generate up to 4 random NC questions

        # 7. MSR questions (Multiple Series Relationship) - NEWLY IMPLEMENTED
        qa_msr = fill_qa_msr(df_long, meta, max_q=4)  # Generate up to 4 random MSR questions

        # 8. VA questions (Visual Analysis - outlier, sampling) - NEWLY IMPLEMENTED
        qa_va = fill_qa_va(df_long, meta)  # Generate VA questions

        # 9. SRP questions (Spatial Relationship Point and Line Vertical Relationship) - NEWLY IMPLEMENTED
        qa_srp = fill_qa_srp(df_long, max_q=2)  # Generate up to 4 random SRP questions

        # ---------- 写入 JSON ----------
        write_qa_to_json(csv_path, "CTR", qa_ctr)
        write_qa_to_json(csv_path, "VEC", qa_vec)
        write_qa_to_json(csv_path, "VPR", qa_trd)  # 用 VPR 键存“趋势”
        write_qa_to_json(csv_path, "VE", qa_ve)
        write_qa_to_json(csv_path, "EVJ", qa_evj)
        write_qa_to_json(csv_path, "SC", qa_sc)
        write_qa_to_json(csv_path, "NF", qa_nf)
        write_qa_to_json(csv_path, "NC", qa_nc)
        write_qa_to_json(csv_path, "MSR", qa_msr)  # Write MSR questions
        write_qa_to_json(csv_path, "VA", qa_va)  # Write VA questions
        write_qa_to_json(csv_path, "SRP", qa_srp)  # Write SRP questions

    print("\n折线图 QA 文件生成完毕。")


if __name__ == "__main__":
    main()

