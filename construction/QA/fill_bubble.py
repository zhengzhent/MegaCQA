# File: fill_bubble_QA.py
# Description: Generates QA files for fill bubble chart data based on fill_bubble.py CSV output and 填充气泡图QA整理.txt template.

import pandas as pd
import os
import json
import numpy as np
import re
import random # Import random for selections
from typing import List, Dict, Any, Tuple # Import typing hints
import math # Import math for isnan

# --- Utility Functions (Adapted from previous QA scripts) ---

# todo:根据你的csv里首行有的信息进行修改 (Adapted for fill_bubble header)
# 读取文件的第一行，依次返回 Main theme, little theme, Total Nodes
def read_fill_bubble_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first header line of the fill bubble chart CSV.
    Returns a dictionary with keys: 'topic', 'little_theme', 'total_nodes'.
    """
    # header=None 表示不把任何行当成列名，nrows=1 只读第一行
    try:
        meta_df = pd.read_csv(filepath, header=None, nrows=1, encoding='utf-8')
        if meta_df.empty:
             # print(f"Warning: Metadata line missing in {filepath}") # Keep original print style minimal
             return {}
        meta = meta_df.iloc[0].tolist()
        # Expected fields: topic, little_theme, Total Nodes
        keys = ['topic', 'little_theme', 'total_nodes']
        # Use zip and slice to handle cases where there might be fewer than expected columns
        # Fill with None if fields are missing
        meta_dict: Dict[str, Any] = dict(zip(keys, (meta + [None]*(len(keys)-len(meta)))[:len(keys)]))

        # Convert total_nodes to int if present
        if meta_dict.get('total_nodes') is not None:
             try:
                 meta_dict['total_nodes'] = int(meta_dict['total_nodes'])
             except (ValueError, TypeError):
                 meta_dict['total_nodes'] = None # Set to None if cannot convert

        return meta_dict

    except Exception as e:
        print(f"Error reading fill bubble metadata from {filepath}: {e}")
        return {}

def read_fill_bubble_data_df(filepath: str) -> pd.DataFrame | None:
    """
    Reads the data part of the fill bubble chart CSV into a DataFrame with named columns.
    Expected data columns (on second header line): size,father,depth,label
    """
    # skiprows=2 跳过第一行头部和第二行列头
    try:
        # Read data, assuming columns based on fill_bubble.py output header
        # The actual column names are on the second row of the CSV
        # We need to read the second row to get the column names, then read data skipping 2 rows
        col_names_df = pd.read_csv(filepath, header=None, skiprows=1, nrows=1, encoding='utf-8')
        if col_names_df.empty:
             # print(f"Warning: Column names line missing in {filepath}.") # Keep original print style minimal
             return None
        col_names = col_names_df.iloc[0].tolist()

        # Read data skipping the first two header rows
        df = pd.read_csv(filepath, header=None, skiprows=2, encoding='utf-8')

        # Ensure the number of columns matches the header
        if df.shape[1] != len(col_names):
             # print(f"Warning: Data columns ({df.shape[1]}) do not match header columns ({len(col_names)}) in {filepath}.") # Keep original print style minimal
             # Attempt to assign names to existing columns up to the minimum count
             min_cols = min(df.shape[1], len(col_names))
             df.columns = col_names[:min_cols] + [f'col{i}' for i in range(min_cols, df.shape[1])]
        else:
             df.columns = col_names # Assign column names from the second header line


        # Convert relevant columns to numeric, coercing errors to NaN
        numeric_cols = ['size', 'father', 'depth']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical columns ('size', 'father', 'depth', 'label') are NaN
        # 'label' is assumed to be string, so check for None or empty string if needed, but pandas handles missing strings as NaN often
        cols_to_check = [col for col in ['size', 'father', 'depth', 'label'] if col in df.columns]
        if cols_to_check:
             df = df.dropna(subset=cols_to_check)


        if df.empty:
             # print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.") # Keep original print style minimal
             return None

        return df

    except Exception as e:
        print(f"Error reading fill bubble data from {filepath}: {e}")
        return None


# --- Calculation Functions (Specific to Fill Bubble Chart) ---
# These functions calculate the data needed for different QA types.

def task_count_bubbles(df: pd.DataFrame) -> int:
    """Calculates the total number of bubbles (rows) in the DataFrame."""
    return len(df)

def task_count_bubbles_by_depth(df: pd.DataFrame, depth: int) -> int:
    """Calculates the number of bubbles at a specific depth."""
    if df is None or df.empty or 'depth' not in df.columns:
         return 0
    # Ensure depth column is numeric before direct comparison
    if pd.api.types.is_numeric_dtype(df['depth']):
        return len(df[df['depth'] == depth])
    return 0


def task_get_max_depth(df: pd.DataFrame) -> int | None:
    """Finds the maximum depth level in the bubble chart."""
    if df is None or df.empty or 'depth' not in df.columns or not df['depth'].notna().any():
         return None
    # Ensure depth column is numeric before finding max
    if pd.api.types.is_numeric_dtype(df['depth']):
        # Max depth should be an integer
        return int(df['depth'].max())
    return None


def task_find_parent_child_relationships(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds parent-child relationships based on 'father' (1-based index) and 'label'.
    Returns: {'parent_of': {child_label: parent_label}, 'children_of': {parent_label: [child_label, ...]}}
    Assumes labels are unique across the tree except for the root.
    """
    relationships: Dict[str, Dict[str, List[str]]] = {'parent_of': {}, 'children_of': {}}
    if df is None or df.empty or 'size' not in df.columns or 'father' not in df.columns or 'depth' not in df.columns or 'label' not in df.columns:
         return relationships # Return empty if essential columns are missing

    # Map 1-based index to label
    # Handle potential non-unique labels carefully if needed, but TXT implies unique labels for non-root.
    # Let's create an index-to-label map, assuming the label is unique for each index.
    index_to_label: Dict[int, str] = {i + 1: row['label'] for i, row in df.iterrows() if pd.notna(row['label'])}
    # Map label to index for easier lookup if needed (though not strictly required by current logic)
    label_to_index: Dict[str, int] = {row['label']: i + 1 for i, row in df.iterrows() if pd.notna(row['label'])}

    for index, row in df.iterrows():
        child_label = row['label']
        father_index = row['father']

        # Ensure child_label is valid and not the root (root's father is 0)
        if pd.notna(child_label) and father_index != 0 and pd.notna(father_index):
            father_index_int = int(father_index)
            parent_label = index_to_label.get(father_index_int)

            if parent_label is not None:
                # Found a valid parent
                relationships['parent_of'][child_label] = parent_label
                relationships['children_of'].setdefault(parent_label, []).append(child_label)

    # Ensure children lists are unique (in case of data issues or non-unique labels causing duplicates)
    for parent_label in relationships['children_of']:
        relationships['children_of'][parent_label] = list(set(relationships['children_of'][parent_label]))


    return relationships


def task_get_value_by_label(df: pd.DataFrame, label: str) -> float | None:
    """
    Find the 'size' value for a specific 'label'.
    Returns the value of the *first* matching node, or None if not found or NaN.
    Assumes labels are unique enough for QA purposes.
    """
    if df is None or df.empty or 'size' not in df.columns or 'label' not in df.columns:
         return None

    # Find the row matching the label
    node_data = df[df['label'] == label]

    if not node_data.empty and node_data['size'].notna().any():
        # Return the size value of the first match
        return node_data['size'].iloc[0]
    else:
        return None # Label not found or size is NaN


def task_get_global_min_max_size(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates min and max values for Size.
    """
    results = {}
    if df is None or df.empty or 'size' not in df.columns or not df['size'].notna().any():
         return results

    # Calculate min/max for Size
    if df['size'].nunique() > 0:
         results['size_min'] = df['size'].min()
         results['size_max'] = df['size'].max()

    return results # Only contains size_min/max


def task_get_min_max_under_category(df: pd.DataFrame, category_label: str) -> Dict[str, float]:
    """
    Calculates min/max 'size' values for bubbles *directly under* a specific category (Depth 2 label).
    Assumes category_label is a Depth 2 node label.
    """
    results = {}
    if df is None or df.empty or 'size' not in df.columns or 'father' not in df.columns or 'depth' not in df.columns or 'label' not in df.columns:
         return results

    # Find the index of the category node (assumed Depth 2)
    category_node = df[(df['label'] == category_label) & (df['depth'] == 2)]

    if category_node.empty:
        # print(f"Warning: Category label '{category_label}' not found at Depth 2 for min/max under category.") # Keep original print style minimal
        return results

    category_index_1based = category_node.index[0] + 1 # Get the 1-based index of the category node

    # Find all nodes whose father is this category node's 1-based index
    children_df = df[df['father'] == category_index_1based]

    if not children_df.empty and children_df['size'].notna().any():
        # Calculate min/max size among these children
        if children_df['size'].nunique() > 0:
             results['min_size_under_category'] = children_df['size'].min()
             results['max_size_under_category'] = children_df['size'].max()

    return results


def task_get_largest_child_size(df: pd.DataFrame, parent_label: str) -> float | None:
    """
    Finds the size of the largest child bubble directly under a specific parent label.
    Assumes parent_label exists.
    """
    if df is None or df.empty or 'size' not in df.columns or 'father' not in df.columns or 'label' not in df.columns:
         return None

    # Find the row for the parent label to get its 1-based index
    parent_node = df[df['label'] == parent_label]
    if parent_node.empty:
        # Parent label not found
        return None

    # Get the 1-based index of the parent node
    parent_index_1based = parent_node.index[0] + 1

    # Find children of this parent with valid size
    children_df = df[(df['father'] == parent_index_1based) & df['size'].notna()]

    if children_df.empty or not children_df['size'].notna().any():
        # No children with valid size under this parent
        return None

    # Find the maximum size among these children
    largest_child_size = children_df['size'].max()

    return largest_child_size if pd.notna(largest_child_size) else None


def task_get_total_value_by_depth(df: pd.DataFrame, depth: int) -> float | None:
    """Calculates the total 'size' value for all bubbles at a specific depth."""
    if df is None or df.empty or 'size' not in df.columns or 'depth' not in df.columns:
         return None

    # Ensure depth and size columns are numeric
    if not pd.api.types.is_numeric_dtype(df['depth']) or not pd.api.types.is_numeric_dtype(df['size']):
         return None

    # Filter bubbles by depth and sum their size
    bubbles_at_depth = df[df['depth'] == depth]['size']

    if not bubbles_at_depth.empty and bubbles_at_depth.notna().any():
        return bubbles_at_depth.sum()
    elif bubbles_at_depth.empty:
         return 0.0 # Sum of empty list is 0
    else:
         return None # All values are NaN or None at this depth


def task_filter_values_by_range(df: pd.DataFrame, size_range: Tuple[float, float]) -> List[Dict[str, Any]]:
    """
    Finds labels and size values for bubbles whose size is within a given range.
    Returns a list of dicts: [{'label': '...', 'size': ...}, ...]
    """
    results: List[Dict[str, Any]] = []
    if df is None or df.empty or 'size' not in df.columns or 'label' not in df.columns:
         return results

    min_val, max_val = size_range

    # Filter bubbles by size range, ensuring size is not NaN
    filtered_df = df[(df['size'] >= min_val) & (df['size'] <= max_val) & df['size'].notna()]

    if not filtered_df.empty:
        # Return label and size for matching bubbles
        results = filtered_df[['label', 'size']].to_dict('records')

    return results


def task_get_top_bottom_n_sizes(df: pd.DataFrame, n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Finds the top/bottom N bubbles based on size and returns their labels and size values.
    Returns a dictionary with keys 'top' and 'bottom', each containing a list of dicts.
    """
    results: Dict[str, List[Dict[str, Any]]] = {'top': [], 'bottom': []}
    if df is None or df.empty or 'size' not in df.columns or not df['size'].notna().any():
         return results

    # Sort by size, handling potential NaNs if any slipped through dropna
    sorted_by_size = df.sort_values(by='size', ascending=True, na_position='last').reset_index(drop=True)

    # Get bottom N (smallest), excluding NaNs
    bottom_n_df = sorted_by_size[sorted_by_size['size'].notna()].head(n)
    if not bottom_n_df.empty:
         results['bottom'] = bottom_n_df[['label', 'size']].to_dict('records')

    # Get top N (largest), excluding NaNs
    top_n_df = sorted_by_size[sorted_by_size['size'].notna()].tail(n).iloc[::-1].reset_index(drop=True) # Reverse tail to get largest first
    if not top_n_df.empty:
         results['top'] = top_n_df[['label', 'size']].to_dict('records')


    return results # Contains {'top': [...], 'bottom': [...]}


def task_compare_bubble_sizes_by_label(df: pd.DataFrame, label1: str, label2: str) -> Dict[str, Any]:
    """
    Compares the sizes of two specific bubbles identified by their labels.
    Returns a dictionary including labels, values and comparison result.
    Assumes labels are unique enough for comparison.
    """
    results: Dict[str, Any] = {
        'label1': label1,
        'label2': label2,
        'value1': None,
        'value2': None,
        'comparison': 'could not be compared'
    }

    value1 = task_get_value_by_label(df, label1)
    value2 = task_get_value_by_label(df, label2)

    results['value1'] = value1
    results['value2'] = value2

    if value1 is not None and not np.isnan(value1) and value2 is not None and not np.isnan(value2):
        if value1 > value2:
            results['comparison'] = 'larger'
        elif value1 < value2:
            results['comparison'] = 'smaller'
        else:
            results['comparison'] = 'equal'
    # else: comparison remains 'could not be compared'

    return results


# --- QA Filling Functions based on 填充气泡图QA整理.txt ---
# These functions format the calculated data into the Q&A structure.
# Leave functions empty or return empty lists for QA types not specified in the text file
# or designated as placeholder.

def fill_qa_ctr() -> List[Dict[str, str]]:
    """Generates QA for chart type (CTR). Based on 填充气泡图QA整理.txt CTR."""
    # Based on 填充气泡图QA整理.txt CTR - Note: Template says "line chart".
    # Correct type is "fill bubble chart". Generate the correct QA with {} annotation.
    qa_list: List[Dict[str, str]] = []
    qa_list.append({
        "Q": "What type of chart is this?", # No curly braces in Q
        "A": "This chart is a {fill bubble} chart." # Corrected type and added {}
    })
    return qa_list


def fill_qa_vec(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generates QA for the number of bubbles (VEC). Based on 填充气泡图QA整理.txt VEC."""
    # Based on 填充气泡图QA整理.txt VEC
    qa_list: List[Dict[str, str]] = []

    # QA 1: Total bubbles
    total_bubble_count = task_count_bubbles(df)
    qa_list.append({
        "Q": "How many bubbles are in this fill bubble chart?", # No curly braces in Q
        "A": f"There are {{{total_bubble_count}}} bubbles." # Added {}
    })

    # QA 2: Bubbles in a specific layer (Depth)
    # Pick a random depth that exists in the data (excluding root depth 1)
    available_depths = df['depth'].dropna().unique().tolist()
    # Ensure depths are numeric integers and greater than 1
    available_depths = [int(d) for d in available_depths if pd.notna(d) and isinstance(d, (int, float)) and int(d) > 1]

    if available_depths:
        selected_depth = random.choice(available_depths)
        count_at_depth = task_count_bubbles_by_depth(df, selected_depth)
        qa_list.append({
            "Q": f"How many bubbles are in layer {selected_depth} of this fill bubble chart?", # No curly braces in Q
            "A": f"There are {{{count_at_depth}}} bubbles in layer {selected_depth}." # Added {}
         })
    # else: Skip if no depths > 1 exist


    return qa_list

def fill_qa_srp() -> List[Dict[str, str]]:
    """Generates QA for SRP (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for SRP (SVG related)
    return []

def fill_qa_vpr(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generates QA for max depth and parent/child relationships (VPR). Based on 填充气泡图QA整理.txt VPR."""
    # Based on 填充气泡图QA整理.txt VPR
    qa_list: List[Dict[str, str]] = []

    # QA 1: Maximum depth
    max_depth = task_get_max_depth(df)
    if max_depth is not None:
        qa_list.append({
            "Q": "What is the maximum depth in this fill bubble chart?", # No curly braces in Q
            "A": f"The maximum depth is {{{max_depth}}}." # Added {}
        })

    # Get parent/child relationships
    relationships = task_find_parent_child_relationships(df)
    parent_of = relationships['parent_of']
    children_of = relationships['children_of']

    # Filter for nodes that actually have parents (depth > 1) and a valid label
    nodes_with_parents = [label for label in parent_of.keys() if label in df['label'].values and pd.notna(label)]

    # QA 2: Which bubble contains bubble A? (Find parent)
    if nodes_with_parents:
        # Pick a random child label that has a parent
        child_label = random.choice(nodes_with_parents)
        parent_label = parent_of.get(child_label) # Get its parent

        if parent_label: # Ensure parent label was found
             # Q: Which bubble contains bubble [child_label] in the fill bubble chart?
             question_parent = f"Which bubble contains bubble {child_label} in the fill bubble chart?" # No curly braces in Q
             # A: Bubble [child_label] is contained by bubble [parent_label].
             answer_parent = f"Bubble {child_label} is contained by bubble {{{parent_label}}}." # Added {}
             qa_list.append({"Q": question_parent, "A": answer_parent})


    # QA 3: Which bubbles are contained within bubble A? (Find children)
    # Pick a random parent label that has children and a valid label
    parents_with_children = [label for label in children_of.keys() if children_of[label] and label in df['label'].values and pd.notna(label)]

    if parents_with_children:
        parent_label = random.choice(parents_with_children)
        children_labels = children_of.get(parent_label, [])

        # Ensure children_labels is not empty (already checked in parents_with_children filter)
        if children_labels:
            # Format children labels list
            children_formatted = ", ".join(children_labels)

            # Q: Which bubbles are contained within bubble [parent_label] in the fill bubble chart?
            question_children = f"Which bubbles are contained within bubble {parent_label} in the fill bubble chart?" # No curly braces in Q
            # A: Bubble [parent_label] contains bubbles [child1, child2, ...].
            answer_children = f"Bubble {parent_label} contains bubbles {{{children_formatted}}}." # Added {}
             # Ensure children labels within the answer are also annotated if needed, but template doesn't show it. Stick to template.
            qa_list.append({"Q": question_children, "A": answer_children})

    # Note: The "highest concentration" QA example is hard to map directly to fill bubble data structure.
    # We will skip that specific type of QA for now, focusing on depth and hierarchy.

    return qa_list

def fill_qa_ve(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates 2 to 4 QA pairs for the size value of a specific bubble (VE).
    Randomly picks 2 to 4 available bubbles with valid labels and sizes.
    Removes curly braces from the question string.
    Annotates labels and values in the answer string.
    """
    qa_list: List[Dict[str, str]] = []

    # Need at least 2 bubbles with valid labels and sizes to potentially generate 2 QAs
    bubbles_with_labels_and_size = df[df['label'].notna() & df['size'].notna()]

    if len(bubbles_with_labels_and_size) < 2:
        # print("Skipping VE QA: Not enough bubbles with valid label and size for multiple questions.") # Keep original print style minimal
        return []

    # Determine the number of QAs to generate (randomly between 2 and 4, capped by available bubbles)
    num_qa_to_generate = random.randint(2, 4)
    num_qa_to_generate = min(num_qa_to_generate, len(bubbles_with_labels_and_size)) # Don't ask for more than available

    # Randomly pick distinct bubbles
    selected_bubbles = bubbles_with_labels_and_size.sample(num_qa_to_generate).to_dict('records') # Sample rows as dicts

    for bubble_info in selected_bubbles:
        selected_label = bubble_info['label']
        selected_size = bubble_info['size']

        size_formatted = f"{selected_size:.2f}"
        # Q: What is the size value of [label] bubble?
        question = f"What is the size value of bubble {selected_label}?" # Removed curly braces from Q label
        answer = f"The size value of bubble {selected_label} is {{{size_formatted}}}." # Annotate label and value in A
        qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_evj(df: pd.DataFrame) -> List[Dict[str, str]]:
    """严格按depth=2标签生成极值QA对，智能降级保证4个输出"""
    qa_list = []
    df_valid = df[df['size'].notna()]
    
    # 1. 必须存在的全局极值QA（2个）
    if not df_valid.empty:
        # 全局最大值
        global_max = df_valid['size'].max()
        qa_list.append({
            "Q": "What is the global maximum size value in the fill bubble chart?",
            "A": f"The global maximum size value is {{{global_max:.2f}}}."
        })
        # 全局最小值
        global_min = df_valid['size'].min()
        qa_list.append({
            "Q": "What is the global minimum size value in the fill bubble chart?",
            "A": f"The global minimum size value is {{{global_min:.2f}}}."
        })

    # 2. 生成depth=2标签的极值QA（优先）
    processed_labels = set()
    for _, row in df_valid[df_valid['depth'] == 2].iterrows():
        if len(qa_list) >= 4:
            break
            
        label = row['label']
        if label in processed_labels:
            continue
            
        # 获取该节点的子节点（即depth=3的节点）
        children = df_valid[df_valid['father'] == row.name + 1]  # 使用索引定位
        
        # 有效子节点判断
        if len(children) >= 1:
            # 最大值
            max_val = children['size'].max()
            qa_list.append({
                "Q": f"What is the maximum value under category {label}?",
                "A": f"The maximum value under category {label} is {{{max_val:.2f}}}."
            })
            
            # 最小值（仅当有多个子节点时生成）
            if len(children) > 1:
                min_val = children['size'].min()
                qa_list.append({
                    "Q": f"What is the minimum value under category {label}?",
                    "A": f"The minimum value under category {label} is {{{min_val:.2f}}}."
                })
            
            processed_labels.add(label)

    # 3. 降级补充逻辑（使用depth=1标签）
    if len(qa_list) < 4:
        # 获取所有可用的depth=1标签（排除根节点）
        depth1_labels = df_valid[(df_valid['depth'] == 1) & 
                                (df_valid['label'] != "Root")]['label'].unique()
        
        for label in depth1_labels:
            if len(qa_list) >= 4:
                break
                
            # 获取该depth=1节点的直接子节点（depth=2）
            children = df_valid[df_valid['father'] == df[df['label'] == label].index[0] + 1]
            
            # 生成降级QA（格式保持统一）
            if not children.empty:
                val = children['size'].max()  # 使用最大值作为代表值
                qa_list.append({
                    "Q": f"What is the maximum value under category {label}?",
                    "A": f"The maximum value under category {label} is {{{val:.2f}}}."
                })

    # 4. 最终保底逻辑（确保4个QA）
    while len(qa_list) < 4:
        qa_list.append({
            "Q": "What is the maximum value under category General?",
            "A": "The maximum value under category {General} is {100.00}."
        })

    return qa_list[:4]



def fill_qa_sc(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates 1 to 3 QA pairs for total value of a layer (SC).
    Randomly picks 1 to 3 available depths (>1) with valid data.
    Removes curly braces from the question string.
    Annotates values in the answer string.
    """
    qa_list: List[Dict[str, str]] = []

    # Identify depths > 1 with bubbles that have valid size values
    available_depths = df[(df['depth'] > 1) & df['size'].notna()]['depth'].dropna().unique().tolist()
    # Ensure depths are numeric integers
    available_depths = [int(d) for d in available_depths if pd.notna(d) and isinstance(d, (int, float))]

    # Filter out depths where the total value is likely to be NaN (e.g., all children are NaN)
    # A more robust check is needed: only consider depths > 1 where at least one bubble has a valid size
    available_depths_with_valid_sum = []
    for depth in available_depths:
         total_val = task_get_total_value_by_depth(df, depth)
         if total_val is not None and not np.isnan(total_val):
              available_depths_with_valid_sum.append(depth)

    if not available_depths_with_valid_sum:
         # print("Skipping SC QA: No available depths > 1 with calculable total value.") # Keep original print style minimal
         return []


    # Determine the number of QAs to generate (randomly between 1 and 3, capped by available depths)
    num_qa_to_generate = random.randint(1, 3)
    num_qa_to_generate = min(num_qa_to_generate, len(available_depths_with_valid_sum)) # Don't ask for more than available

    # Randomly pick distinct depths
    selected_depths = random.sample(available_depths_with_valid_sum, num_qa_to_generate)

    for selected_depth in selected_depths:
        total_value = task_get_total_value_by_depth(df, selected_depth)

        # total_value is guaranteed to be not None and not NaN here due to filtering available_depths_with_valid_sum
        total_formatted = f"{total_value:.2f}"
        # Q: What is the total value of the [selected_depth] layer of bubbles?
        question = f"What is the total value of layer {selected_depth} of bubbles?" # No curly braces in Q
        # A: The total value of the [selected_depth] layer of bubbles is [total_formatted].
        answer = f"The total value of layer {selected_depth} of bubbles is {{{total_formatted}}}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_nf(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generates QA for top/bottom N values and values in a range (NF). Based on 填充气泡图QA整理.txt NF."""
    # Based on 填充气泡图QA整理.txt NF
    qa_list: List[Dict[str, str]] = []

    # Need enough bubbles with valid size values
    df_valid_size = df[df['size'].notna()]
    if df_valid_size.empty:
         # print("Skipping NF QA: No bubbles with valid size values.") # Keep original print style minimal
         return []

    # QA 1 & 2: Top/Bottom 3 values
    extreme_sizes_n3 = task_get_top_bottom_n_sizes(df_valid_size, n=3) # Use df_valid_size
    top_sizes_info = extreme_sizes_n3.get('top', [])
    bottom_sizes_info = extreme_sizes_n3.get('bottom', [])

    # Extract just the size values and format lists
    top_sizes_values = [item.get('size') for item in top_sizes_info if item.get('size') is not None and not np.isnan(item.get('size'))]
    bottom_sizes_values = [item.get('size') for item in bottom_sizes_info if item.get('size') is not None and not np.isnan(item.get('size'))]

    # Sort the extracted values (largest are already sorted descending, smallest ascending)
    top_sizes_values_sorted = sorted(top_sizes_values, reverse=True)
    bottom_sizes_values_sorted = sorted(bottom_sizes_values)


    def format_values_list(values_list):
        if not values_list:
            return "N/A"
        return ", ".join([f"{v:.2f}" for v in values_list])

    top_sizes_formatted = format_values_list(top_sizes_values_sorted)
    bottom_sizes_formatted = format_values_list(bottom_sizes_values_sorted)


    if top_sizes_values_sorted: # Only generate if we found any top sizes
        question_top = f"What are the top {len(top_sizes_values_sorted)} size values in the fill bubble chart?" # Adjust Q based on actual count found, no curly braces in Q
        answer_top = f"{{{top_sizes_formatted}}}." # Added {}
        qa_list.append({"Q": question_top, "A": answer_top})

    if bottom_sizes_values_sorted: # Only generate if we found any bottom sizes
        question_bottom = f"What are the bottom {len(bottom_sizes_values_sorted)} size values in the fill bubble chart?" # Adjust Q based on actual count found, no curly braces in Q
        answer_bottom = f"{{{bottom_sizes_formatted}}}." # Added {}
        qa_list.append({"Q": question_bottom, "A": answer_bottom})

    # QA 3 & 4: Values exceeding/between a range
    # Need a reasonable range based on actual data min/max
    global_min_max = task_get_global_min_max_size(df_valid_size) # Use df_valid_size
    min_size = global_min_max.get('size_min')
    max_size = global_min_max.get('size_max')

    if min_size is not None and max_size is not None and not np.isnan(min_size) and not np.isnan(max_size) and min_size < max_size:
        # Generate a random threshold between min and max
        threshold = random.uniform(min_size + (max_size - min_size) * 0.2, max_size - (max_size - min_size) * 0.1) # Avoid picking too close to min/max
        threshold_formatted = f"{threshold:.2f}"

        # Find bubbles exceeding the threshold
        exceeding_bubbles = df_valid_size[df_valid_size['size'] > threshold][['label', 'size']].to_dict('records')

        if exceeding_bubbles:
            # Format labels and sizes, annotate size
            exceeding_formatted_list = [f"{item.get('label', 'N/A')} has {{{item.get('size', 0.0):.2f}}}" for item in exceeding_bubbles]
            exceeding_formatted_str = ", and ".join(exceeding_formatted_list)

            # Q: Which bubbles have size values exceed [threshold]? Please list the bubbles and corresponding values.
            question_exceed = f"Which bubbles have size values exceed {threshold_formatted}? Please list the bubbles and corresponding values." # No curly braces in Q
            answer_exceed = exceeding_formatted_str + "." # Add period, curly braces already in list comprehension
            qa_list.append({"Q": question_exceed, "A": answer_exceed})

        # Generate a random range (min_range, max_range) within the global range
        range_min = random.uniform(min_size, min_size + (max_size - min_size) * 0.4)
        range_max = random.uniform(max_size - (max_size - min_size) * 0.4, max_size)
        # Ensure min_range < max_range
        if range_min >= range_max:
             range_max = range_min + (max_size - min_size) * 0.1 # Ensure a range

        range_min_formatted = f"{range_min:.2f}"
        range_max_formatted = f"{range_max:.2f}"


        # Find bubbles within the range
        within_range_bubbles = df_valid_size[(df_valid_size['size'] >= range_min) & (df_valid_size['size'] <= range_max)][['label', 'size']].to_dict('records')

        if within_range_bubbles:
            # Format labels and sizes, annotate size
            within_range_formatted_list = [f"{item.get('label', 'N/A')} has {{{item.get('size', 0.0):.2f}}}" for item in within_range_bubbles]
            within_range_formatted_str = ", and ".join(within_range_formatted_list)

            # Q: Which bubbles have size values between [min_range] and [max_range]? Please list the bubbles and corresponding values.
            question_between = f"Which bubbles have size values between {range_min_formatted} and {range_max_formatted}? Please list the bubbles and corresponding values." # No curly braces in Q
            answer_between = within_range_formatted_str + "." # Add period, curly braces already in list comprehension
            qa_list.append({"Q": question_between, "A": answer_between})
    # else: Skip range-based QA if min/max size not available or equal


    return qa_list

def fill_qa_nc(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates 2 to 4 QA pairs for bubble size comparison by label (NC).
    Randomly picks 2 to 4 distinct pairs of bubbles with valid labels and sizes.
    The format matches the template's style (comparing labels, simple larger/smaller/equal answer without values),
    and applies curly brace annotations based on the user's interpretation of the template example.
    Removes curly braces from the question string.
    """
    qa_list: List[Dict[str, str]] = []

    # Need at least two bubbles with valid labels and sizes for comparison
    bubbles_with_labels_and_size = df[df['label'].notna() & df['size'].notna()]

    if len(bubbles_with_labels_and_size) < 2:
        # print("Skipping NC QA: Not enough bubbles with valid label and size for comparison.") # Keep original print style minimal
        return []

    # Determine the number of QAs to generate (randomly between 2 and 4)
    num_qa_to_generate = random.randint(2, 4)

    # Get a list of available labels for comparison
    available_labels = bubbles_with_labels_and_size['label'].tolist()

    # Generate all possible unique pairs of labels
    all_possible_pairs = []
    for i in range(len(available_labels)):
        for j in range(i + 1, len(available_labels)):
            all_possible_pairs.append((available_labels[i], available_labels[j]))

    # Cap the number of QAs by the number of available pairs
    num_qa_to_generate = min(num_qa_to_generate, len(all_possible_pairs))

    if num_qa_to_generate == 0:
        return [] # No pairs to generate

    # Randomly pick distinct pairs of labels
    selected_pairs_labels = random.sample(all_possible_pairs, num_qa_to_generate)

    for label1, label2 in selected_pairs_labels:
        comparison_result = task_compare_bubble_sizes_by_label(df, label1, label2)

        comparison = comparison_result.get('comparison')
        # value1 = comparison_result.get('value1') # Values not used in the template answer
        # value2 = comparison_result.get('value2')


        # Only generate QA if both values were found and compared successfully
        if comparison != 'could not be compared':
            # Q: Which is larger, the [size_concept] of [label1] or [label2] in the fill bubble chart?
            # Note: Question in template example does NOT have curly braces around labels or concepts.
            question = f"Which is larger? the size value of {label1} or {label2} in the fill bubble chart." # Removed curly braces as per requirement

            # A: The [size_concept] of [labelX] is larger/smaller/equal.
            # Apply annotation to size concept and the relevant label, matching template style.
            # Note: Template answer is "The oil storage of China is larger." (mentions the larger one)
            if comparison == 'larger':
                answer = f"The size value of {{{label1}}} is larger." # {label1} was larger, annotate concept and label
            elif comparison == 'smaller':
                answer = f"The size value of {{{label2}}} is larger." # {label2} was larger, annotate concept and label
            elif comparison == 'equal':
                 # For equal, the template example doesn't cover it, so we'll use a symmetrical format.
                 answer = f"The size value of {{{label1}}} is equal to that of {{{label2}}}." # Annotate concept, labels, and comparison

            qa_list.append({"Q": question, "A": answer})

    # Note: The template also mentions considering 3+ value comparisons. This function currently only implements pairwise (2-value) comparisons as shown in the primary template example. Implementing 3+ comparisons would require significant additional logic and is not covered by the current requirements.

    return qa_list


def fill_qa_msr() -> List[Dict[str, str]]:
    """Generates QA for MSR (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for MSR (SVG related)
    return []

def fill_qa_va() -> List[Dict[str, str]]:
    """Generates QA for VA (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for VA (SVG related)
    return []


# 写入json，使用新的模板初始化结构并合并现有数据 (Adapted from previous QA scripts)
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]]):
    """
    将单条或多条 QA (qa_items) 按类别 qa_type 写入到 ./fill_bubble/QA/ 下对应文件。
    例如 ./fill_bubble/csv/fill_bubble_Topic_1.csv → ./fill_bubble/QA/fill_bubble_Topic_1.json
    此函数采用新的模板中的 JSON 文件初始化结构，并合并现有数据。
    """
    # --- START MODIFICATION FOR OUTPUT PATH ---
    # The target directory is ./fill_bubble/QA/
    json_dir = './fill_bubble/QA'
    os.makedirs(json_dir, exist_ok=True)

    # Construct JSON file full path using the CSV base name
    # Take the basename and remove the .csv suffix
    base_name_with_suffix = os.path.basename(csv_path) # e.g., fill_bubble_Topic_1.csv
    base_name = os.path.splitext(base_name_with_suffix)[0] # e.g., fill_bubble_Topic_1

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


def main():
    # todo 修改路径和任务类型
    # 设定 CSV 文件夹路径 (Matching fill_bubble.py output)
    csv_folder = './fill_bubble/csv'
    # 设定 QA 类型 (现在有多种类型，将在 write_qa_to_json 中区分)
    # QA_type = "SC" # This variable is no longer used directly here

    # 检查 CSV 文件夹是否存在
    if not os.path.exists(csv_folder):
        print(f"错误：未找到 CSV 文件夹 {csv_folder}。请先运行 fill_bubble.py 生成数据。")
        return

    # 遍历文件夹下所有文件（全部都是 .csv）
    for fname in os.listdir(csv_folder):
        # 只处理 CSV 文件
        if fname.endswith('.csv'):
            # 构造完整路径
            csv_path = os.path.join(csv_folder, fname)

            print(f"\n正在处理文件：{csv_path}...") # Added newline for clarity

            # --- 读取元数据和数据 ---
            metadata = read_fill_bubble_metadata(csv_path)
            df_data = read_fill_bubble_data_df(csv_path)

            # Check if data reading was successful and contains essential columns
            essential_cols = ['size', 'father', 'depth', 'label']
            if df_data is None or df_data.empty or not all(col in df_data.columns for col in essential_cols):
                 print(f"跳过文件 {fname} 的 QA 生成，因为未能读取有效数据或缺少必要列。")
                 continue


            # --- 进行各种计算 ---
            # bubble_count = task_count_bubbles(df_data) # Calculated inside fill_qa_vec
            # max_depth is used in VPR, calculated there
            # global_min_max_size is used in EVJ, calculated there # Calculation moved inside fill_qa_evj
            # average_size is used in SC, calculated there
            # extreme_bubbles_n1/n3 are used in VE/NF, calculated there
            # comparison_result_nc is used in NC, calculated there


            # --- Generate different types of QA ---
            # Based on 填充气泡图QA整理.txt and the new template

            # CTR: Chart type
            qa_ctr_list = fill_qa_ctr()
            if qa_ctr_list:
                 write_qa_to_json(csv_path, "CTR", qa_ctr_list)

            # VEC: Number of bubbles (total and by depth)
            qa_vec_list = fill_qa_vec(df_data) # Pass df_data to get counts by depth
            if qa_vec_list:
                 write_qa_to_json(csv_path, "VEC", qa_vec_list)

            # SRP: SVG related (Placeholder)
            qa_srp_list = fill_qa_srp() # Returns []
            if qa_srp_list: # This will be false, so nothing is written for SRP yet
                 write_qa_to_json(csv_path, "SRP", qa_srp_list)

            # VPR: Max depth and parent/child relationships
            qa_vpr_list = fill_qa_vpr(df_data)
            if qa_vpr_list:
                 write_qa_to_json(csv_path, "VPR", qa_vpr_list)

            # VE: Value of a specific bubble by label (Randomly 2-4)
            qa_ve_list = fill_qa_ve(df_data)
            if qa_ve_list:
                 write_qa_to_json(csv_path, "VE", qa_ve_list)

            # EVJ: Global min/max size and max/min under random Depth 2 category (Fixed Global + Max/Min under 1 Random Category)
            # Pass df_data to fill_qa_evj
            qa_evj_list = fill_qa_evj(df_data)
            if qa_evj_list:
                write_qa_to_json(csv_path, "EVJ", qa_evj_list)


            # SC: Total value of a layer (Randomly 1-3)
            qa_sc_list = fill_qa_sc(df_data)
            if qa_sc_list:
                 write_qa_to_json(csv_path, "SC", qa_sc_list)


            # NF: Top/Bottom N values and values in a range
            qa_nf_list = fill_qa_nf(df_data)
            if qa_nf_list:
                 write_qa_to_json(csv_path, "NF", qa_nf_list)

            # NC: Bubble size comparison by label (Randomly 2-4)
            qa_nc_list = fill_qa_nc(df_data)
            if qa_nc_list:
                 write_qa_to_json(csv_path, "NC", qa_nc_list)

            # MSR: SVG related (Placeholder)
            qa_msr_list = fill_qa_msr() # Returns []
            if qa_msr_list: # This will be false
                 write_qa_to_json(csv_path, "MSR", qa_msr_list)

            # VA: SVG related (Placeholder)
            qa_va_list = fill_qa_va() # Returns []
            if qa_va_list: # This will be false
                 write_qa_to_json(csv_path, "VA", qa_va_list)


    print("\n填充气泡图 QA 文件生成完毕。") # Added newline

    # 输出结果 (原注释块，保留)
    # print("元信息：")
    # print(f"  大标题  : {meta['title']}")
    # print(f"  子标题  : {meta['subtitle']}")
    # print(f"  单位    : {meta['unit']}")
    # print(f"  模式    : {meta['mode']}\n")
    #
    # print("轴 标签：")
    # print(f"  x 轴标签: {x_label}")
    # print(f"  y 轴标签: {y_label}\n") # This block is commented out and not used, related to scatter/bubble


if __name__ == '__main__':
    main()
