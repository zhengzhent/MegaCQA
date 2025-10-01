# File: heatmap_QA.py
# Description: Generates QA files for heatmap data based on heatmap.py CSV output and 热力QA整理.txt template.

import pandas as pd
import os
import json
import numpy as np
import re
from typing import List, Dict, Any, Tuple # Import List, Dict, Any, and Tuple for type hinting 
import random

# --- Utility Functions (Adapted from scatter_QA.py and heatmap.py) ---

# todo:根据你的csv里首行有的信息进行修改 (Adapted for heatmap header)
# 读取文件的第一行，依次返回 Main theme, little theme, dimension, pattern
def read_heatmap_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first header line of the heatmap CSV.
    Returns a dictionary with keys: 'topic', 'little_theme', 'dimension', 'pattern'.
    """
    # header=None 表示不把任何行当成列名，nrows=1 只读第一行
    try:
        # Use converters to ensure dimension is read as string before splitting
        meta_df = pd.read_csv(filepath, header=None, nrows=1, encoding='utf-8', converters={2: str})
        if meta_df.empty:
             # print(f"Warning: Metadata line missing in {filepath}") # Keep original print style minimal
             return {}
        meta = meta_df.iloc[0].tolist()
        # Expected fields: topic, little_theme, dimension, pattern
        keys = ['topic', 'little_theme', 'dimension', 'pattern']
        # Use zip and slice to handle cases where there might be fewer than expected columns
        # Fill with None if fields are missing
        meta_dict = dict(zip(keys, (meta + [None]*(len(keys)-len(meta)))[:len(keys)]))

        # Parse dimension (e.g., "10x10")
        if meta_dict.get('dimension') is not None:
            dim_match = re.match(r'(\d+)x(\d+)', str(meta_dict['dimension']))
            if dim_match:
                meta_dict['cols'] = int(dim_match.group(1))
                meta_dict['rows'] = int(dim_match.group(2))
            else:
                 meta_dict['cols'] = None
                 meta_dict['rows'] = None

        return meta_dict

    except Exception as e:
        print(f"Error reading heatmap metadata from {filepath}: {e}")
        return {}

# Heatmap CSV does NOT have a second header row for axis labels.
# Labels are in the data columns. We need a function to read the data DataFrame.
# read_axis_labels from scatter_QA.py is NOT applicable here.

def read_heatmap_data_df(filepath: str) -> pd.DataFrame | None:
    """
    Reads the data part of the heatmap CSV into a DataFrame with named columns.
    Expected columns in data part: x_block, y_block, level
    """
    # skiprows=1 跳过第一行头部，header=None 保留原始数据
    try:
        # Read data, assuming 3 columns: x_block, y_block, level
        df = pd.read_csv(filepath, header=None, skiprows=1, encoding='utf-8')

        # Ensure there are at least 3 columns
        if df.shape[1] < 3:
            # print(f"Warning: Not enough data columns ({df.shape[1]}) in {filepath}.") # Keep original print style minimal
            return None

        # Rename columns for clarity
        df = df.rename(columns={0: 'x_block', 1: 'y_block', 2: 'level'})

        # Convert 'level' column to numeric, coercing errors to NaN
        df['level'] = pd.to_numeric(df['level'], errors='coerce')

        # Drop rows where 'level' is NaN, as these are invalid data points for calculations
        df = df.dropna(subset=['level'])

        if df.empty:
             # print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.") # Keep original print style minimal
             return None

        return df

    except Exception as e:
        print(f"Error reading heatmap data from {filepath}: {e}")
        return None

# --- Calculation Functions (Specific to Heatmap) ---
# These functions calculate the data needed for different QA types.

def task_count_heatmap_cells(df: pd.DataFrame) -> int:
    """Calculates the number of data cells (rows) in the DataFrame."""
    return len(df) # Already dropped NaNs in read_heatmap_data_df

def task_get_unique_labels(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Gets unique x_block and y_block labels."""
    results: Dict[str, List[str]] = {'x_labels': [], 'y_labels': []}
    if df is not None and not df.empty:
        if 'x_block' in df.columns:
             results['x_labels'] = df['x_block'].unique().tolist()
        if 'y_block' in df.columns:
             results['y_labels'] = df['y_block'].unique().tolist()
    return results


def task_get_global_min_max_level(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates min and max 'level' values globally."""
    results = {}
    if df is not None and not df.empty and 'level' in df.columns:
        # Check if there is variance before calculating min/max
        if df['level'].nunique() > 0:
             results['min_level'] = df['level'].min()
             results['max_level'] = df['level'].max()
        # else: min/max are undefined or all values are the same (0 or NaN handled by dropna)
    return results

def task_get_label_min_max_level(df: pd.DataFrame, label: str, axis: str) -> Dict[str, float]:
    """
    Calculates min and max 'level' values for cells associated with a specific label on an axis.
    axis can be 'x' or 'y'.
    """
    results = {}
    if df is None or df.empty or 'level' not in df.columns:
         return results

    if axis == 'x' and 'x_block' in df.columns:
        subset_df = df[df['x_block'] == label]
    elif axis == 'y' and 'y_block' in df.columns:
        subset_df = df[df['y_block'] == label]
    else:
        return results # Invalid axis or column missing

    if not subset_df.empty and subset_df['level'].nunique() > 0:
        results[f'{axis}_label_min_level'] = subset_df['level'].min()
        results[f'{axis}_label_max_level'] = subset_df['level'].max()

    return results

def task_get_label_average_level(df: pd.DataFrame, label: str, axis: str) -> float | None:
    """
    Calculates the average 'level' value for cells associated with a specific label on an axis.
    axis can be 'x' or 'y'.
    """
    if df is None or df.empty or 'level' not in df.columns:
         return None

    if axis == 'x' and 'x_block' in df.columns:
        subset_df = df[df['x_block'] == label]
    elif axis == 'y' and 'y_block' in df.columns:
        subset_df = df[df['y_block'] == label]
    else:
        return None # Invalid axis or column missing

    if not subset_df.empty:
        # .mean() returns NaN for empty Series, which is handled by the return type None | float
        return subset_df['level'].mean()
    else:
        return None

def task_get_cell_value(df: pd.DataFrame, x_label: str, y_label: str) -> float | None:
    """
    Gets the 'level' value for a specific x_block and y_block combination.
    Returns the value or None if not found.
    """
    if df is None or df.empty or 'x_block' not in df.columns or 'y_block' not in df.columns or 'level' not in df.columns:
         return None

    # Find the row matching both x_label and y_label
    cell_data = df[(df['x_block'] == x_label) & (df['y_block'] == y_label)]

    if not cell_data.empty:
        # Return the level value of the first match (should be unique in a well-formed heatmap data)
        return cell_data['level'].iloc[0]
    else:
        return None # Cell not found


def task_compare_cell_values(df: pd.DataFrame, cell1_labels: Tuple[str, str], cell2_labels: Tuple[str, str]) -> Dict[str, Any]:
    """
    Compares the 'level' values of two specific cells identified by their labels.
    Returns a dictionary including values and comparison result.
    """
    results: Dict[str, Any] = {
        'cell1_labels': cell1_labels,
        'cell2_labels': cell2_labels,
        'cell1_value': None,
        'cell2_value': None,
        'comparison': 'could not be compared'
    }

    value1 = task_get_cell_value(df, cell1_labels[0], cell1_labels[1])
    value2 = task_get_cell_value(df, cell2_labels[0], cell2_labels[1])

    results['cell1_value'] = value1
    results['cell2_value'] = value2

    if value1 is not None and not np.isnan(value1) and value2 is not None and not np.isnan(value2):
        if value1 > value2:
            results['comparison'] = 'higher'
        elif value1 < value2:
            results['comparison'] = 'lower'
        else:
            results['comparison'] = 'equal'
    # else: comparison remains 'could not be compared'

    return results


# --- QA Filling Functions based on 热力QA整理.txt ---
# These functions format the calculated data into the Q&A structure.
# Leave functions empty or return empty lists for QA types not specified in the text file
# or designated as placeholder.

def fill_qa_ctr() -> List[Dict[str, str]]:
    """Generates QA for chart type (CTR). Based on 热力QA整理.txt CTR."""
    # Based on 热力QA整理.txt CTR - Note: The template says "line chart".
    # For a heatmap, the correct type is "heatmap".
    # Generate the correct QA for a heatmap with {} annotation.
    qa_list: List[Dict[str, str]] = []
    qa_list.append({
        "Q": "What type of chart is this?",
        "A": "This chart is a {heatmap} chart." # Corrected type and added {}
    })
    return qa_list


def fill_qa_vec(metadata: Dict[str, Any], cell_count: int) -> List[Dict[str, str]]:
    """Generates QA for the number of cells, rows, and columns (VEC). Based on 热力QA整理.txt VEC."""
    # Based on 热力QA整理.txt VEC
    qa_list: List[Dict[str, str]] = []

    # QA 1: How many cells?
    qa_list.append({
        "Q": "How many cells are in this heatmap?",
        "A": f"There are {{{cell_count}}} cells." # Added {}
    })

    # QA 2: How many rows?
    rows = metadata.get('rows')
    if rows is not None:
         qa_list.append({
            "Q": "How many rows are in this heatmap?",
            "A": f"There are {{{rows}}} rows." # Added {}
         })
    else:
         # Fallback if rows couldn't be parsed from metadata
         qa_list.append({
            "Q": "How many rows are in this heatmap?",
            "A": f"The number of rows could not be determined."
         })


    # QA 3: How many columns?
    cols = metadata.get('cols')
    if cols is not None:
         qa_list.append({
            "Q": "How many columns are in this heatmap?",
            "A": f"There are {{{cols}}} columns." # Added {}
         })
    else:
         # Fallback if cols couldn't be parsed from metadata
         qa_list.append({
            "Q": "How many columns are in this heatmap?",
            "A": f"The number of columns could not be determined."
         })


    return qa_list

def fill_qa_srp() -> List[Dict[str, str]]:
    """Generates QA for SRP (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for SRP (SVG related)
    return []

def fill_qa_vpr(metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for the overall spatial distribution pattern (VPR). Based on 热力QA整理.txt VPR."""
    # Based on 热力QA整理.txt VPR
    qa_list: List[Dict[str, str]] = []
    question = "What is the overall spatial distribution pattern shown in the heatmap?"
    # No curly braces in Q

    # Get pattern from metadata if available
    pattern = metadata.get('pattern')

    if pattern:
        # Map internal pattern key to a more user-friendly description if needed,
        # but for now, let's use the internal key and add {}
        pattern_description = pattern.replace('_', ' ') # Simple formatting
        answer = f"The heatmap shows a {{{pattern_description}}} pattern." # Added {}
    else:
        answer = "The overall spatial distribution pattern could not be determined."


    qa_list.append({"Q": question, "A": answer})
    return qa_list

def fill_qa_ve(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates 2 to 4 QA pairs for the heat value of specific cells (VE).
    Randomly picks 2 to 4 available cells.
    Removes curly braces from the question string.
    """
    qa_list: List[Dict[str, str]] = []

    # Need at least 2 unique cells to potentially generate 2 QAs
    available_cells = df[['x_block', 'y_block']].drop_duplicates().values.tolist()

    if len(available_cells) < 2:
        # print("Skipping VE QA: Not enough unique cells for multiple questions.") # Keep original print style minimal
        return []

    # Determine the number of QAs to generate (randomly between 2 and 4, capped by available cells)
    num_qa_to_generate = random.randint(2, 4)
    num_qa_to_generate = min(num_qa_to_generate, len(available_cells)) # Don't ask for more than available

    # Randomly pick distinct cells
    selected_cells = random.sample(available_cells, num_qa_to_generate)

    for x_label, y_label in selected_cells:
        cell_value = task_get_cell_value(df, x_label, y_label)

        if cell_value is not None and not np.isnan(cell_value):
            value_formatted = f"{cell_value:.2f}" # Format value
            # Removed curly braces from Q
            question = f"What is the heat value of the cell corresponding to {x_label} and {y_label} in the heatmap?"
            answer = f"The heat value is {{{value_formatted}}}." # Retained curly braces in A
            qa_list.append({"Q": question, "A": answer})
        # else: If value is None or NaN, skip this cell and try to generate another if needed (handled by sampling)

    return qa_list


def fill_qa_evj(df: pd.DataFrame, global_min_max: Dict[str, float], unique_labels: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Generates QA for global and label-specific min/max values (EVJ).
    Includes fixed global min/max and 2 random label max value QAs.
    Removes curly braces from the question string.
    Annotates labels and values in the answer string.
    """
    qa_list: List[Dict[str, str]] = []

    # QA 1: Global maximum heat value (Fixed)
    max_level = global_min_max.get('max_level')
    if max_level is not None and not np.isnan(max_level):
        max_formatted = f"{max_level:.2f}"
        question = "What is the global maximum heat value in the heatmap?" # No curly braces in Q
        answer = f"The global maximum heat value is {{{max_formatted}}}." # Retained curly braces in A
        qa_list.append({"Q": question, "A": answer})

    # QA 2: Global minimum heat value (Fixed)
    min_level = global_min_max.get('min_level')
    if min_level is not None and not np.isnan(min_level):
        min_formatted = f"{min_level:.2f}"
        question = "What is the global minimum heat value in the heatmap?" # No curly braces in Q
        answer = f"The global minimum heat value is {{{min_formatted}}}." # Retained curly braces in A
        qa_list.append({"Q": question, "A": answer})

    # QA 3 & 4: Max for a specific label (Randomly pick 2 distinct labels)
    x_labels = unique_labels.get('x_labels', [])
    y_labels = unique_labels.get('y_labels', [])
    all_labels_with_axis = [(label, 'x') for label in x_labels] + [(label, 'y') for label in y_labels]

    # Need at least 2 distinct labels to pick from for the random QAs
    if len(all_labels_with_axis) >= 2:
        # Randomly pick 2 distinct labels (can be mix of X and Y)
        selected_labels_with_axis = random.sample(all_labels_with_axis, 2)

        for selected_label, selected_axis in selected_labels_with_axis:
             label_min_max = task_get_label_min_max_level(df, selected_label, selected_axis)
             label_max = label_min_max.get(f'{selected_axis}_label_max_level')

             if label_max is not None and not np.isnan(label_max):
                  max_formatted = f"{label_max:.2f}"
                  # Adjust Q/A based on the axis type
                  if selected_axis == 'x':
                       # Removed curly braces from Q label
                       question = f"What is the maximum heat value of label {selected_label} on the x-axis?"
                       # Annotate label and value in A
                       answer = f"The maximum heat value of {selected_label} on the x-axis is {{{max_formatted}}}."
                  else: # selected_axis == 'y'
                       # Removed curly braces from Q label
                       question = f"What is the maximum heat value for label {selected_label} on the y-axis?"
                       # Annotate label and value in A
                       answer = f"The maximum heat value for {selected_label} on the y-axis is {{{max_formatted}}}."

                  qa_list.append({"Q": question, "A": answer})
             # else: Skip if max not available for this label (will result in fewer than 4 EVJ QAs)

    # The function should return the qa_list which will contain 2 global QAs
    # plus up to 2 random label-specific max QAs, for a total of up to 4.
    return qa_list


def fill_qa_sc(df: pd.DataFrame, unique_labels: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Generates 1 to 3 QA pairs for average values for labels (SC).
    Randomly picks 1 to 3 available labels.
    Removes curly braces from the question string.
    Annotates labels and values in the answer string.
    """
    qa_list: List[Dict[str, str]] = []

    x_labels = unique_labels.get('x_labels', [])
    y_labels = unique_labels.get('y_labels', [])
    all_labels = [(label, 'x') for label in x_labels] + [(label, 'y') for label in y_labels]

    if not all_labels:
         # print("Skipping SC QA: No unique labels available.") # Keep original print style minimal
         return []

    # Determine the number of QAs to generate (randomly between 1 and 3, capped by available labels)
    num_qa_to_generate = random.randint(1, 3)
    num_qa_to_generate = min(num_qa_to_generate, len(all_labels)) # Don't ask for more than available

    # Randomly pick distinct labels (can mix X and Y)
    selected_labels_with_axis = random.sample(all_labels, num_qa_to_generate)

    for selected_label, selected_axis in selected_labels_with_axis:
        average_value = task_get_label_average_level(df, selected_label, selected_axis)

        if average_value is not None and not np.isnan(average_value):
            avg_formatted = f"{average_value:.2f}"
            # Adjust Q/A based on the axis type
            if selected_axis == 'x':
                 # Removed curly braces from Q label
                 question = f"What is the average heat value for label {selected_label} on the x-axis?"
                 # Annotate label and value in A
                 answer = f"The average heat value for {selected_label} on the x-axis is {{{avg_formatted}}}."
            else: # selected_axis == 'y'
                 # Removed curly braces from Q label
                 question = f"What is the average heat value for label {selected_label} on the y-axis?"
                 # Annotate label and value in A
                 answer = f"The average heat value for {selected_label} on the y-axis is {{{avg_formatted}}}."

            qa_list.append({"Q": question, "A": answer})
        # else: Skip if average not available for this label

    return qa_list

def fill_qa_nf() -> List[Dict[str, str]]:
    """Generates QA for NF. Currently empty as per template/request."""
    # TODO: Implement QA generation for NF
    return []

def fill_qa_nc(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates 2 to 4 QA pairs for cell value comparison (NC).
    Randomly picks 2 to 4 distinct pairs of cells.
    The answer format is concise, matching the template style, and labels are annotated.
    Removes curly braces from the question string.
    """
    qa_list: List[Dict[str, str]] = []

    # Need at least two unique cells to compare
    available_cells = df[['x_block', 'y_block']].drop_duplicates().values.tolist()

    if len(available_cells) < 2:
        # print("Skipping NC QA: Not enough unique cells for comparison.") # Keep original print style minimal
        return []

    # Determine the number of QAs to generate (randomly between 2 and 4)
    num_qa_to_generate = random.randint(2, 4)

    # Calculate the maximum number of unique pairs available: n*(n-1)/2
    max_available_pairs = len(available_cells) * (len(available_cells) - 1) // 2
    num_qa_to_generate = min(num_qa_to_generate, max_available_pairs) # Cap by available pairs

    if num_qa_to_generate == 0:
        return [] # No pairs to generate

    # Generate all possible unique pairs of cells
    all_possible_pairs = []
    for i in range(len(available_cells)):
        for j in range(i + 1, len(available_cells)):
            all_possible_pairs.append((available_cells[i], available_cells[j]))

    # Randomly pick distinct pairs
    selected_pairs = random.sample(all_possible_pairs, num_qa_to_generate)

    for cell1_labels, cell2_labels in selected_pairs:
        comparison_result = task_compare_cell_values(df, tuple(cell1_labels), tuple(cell2_labels))

        value1 = comparison_result['cell1_value']
        value2 = comparison_result['cell2_value']
        comparison = comparison_result['comparison']

        # Only generate QA if both values were found and compared
        if comparison != 'could not be compared':
            # Q: Which has the higher heat value: the cell corresponding to [x1, y1] or the cell corresponding to [x2, y2]?
            # Removed curly braces from Q labels
            question = f"Which has the higher heat value? the cell corresponding to {cell1_labels[0]} and {cell1_labels[1]} or the cell corresponding to {cell2_labels[0]} and {cell2_labels[1]}."

            # A: The cell corresponding to [x, y] has a [higher/lower/equal] heat value. (Concise template style)
            if comparison == 'higher':
                # Annotate labels and comparison in A
                answer = f"The cell corresponding to {{{cell1_labels[0]}}} and {{{cell1_labels[1]}}} has a higher heat value."
            elif comparison == 'lower':
                # Template style implies stating the cell with the higher value
                # Annotate labels and comparison in A
                answer = f"The cell corresponding to {{{cell2_labels[0]}}} and {{{cell2_labels[1]}}} has a higher heat value."
            elif comparison == 'equal':
                 # Template doesn't show equal, use a concise symmetrical format
                 # Annotate comparison in A
                 answer = f"The heat values are {{equal}}."
            # else: comparison == 'could not be compared' - this case is now handled by the outer if

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


# 写入json，使用新的模板初始化结构并合并现有数据 (Adapted from scatter_QA.py)
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]]):
    """
    将单条或多条 QA (qa_items) 按类别 qa_type 写入到 ./heatmap/QA/ 下对应文件。
    例如 ./heatmap/csv/heatmap_Topic_1.csv → ./heatmap/QA/heatmap_Topic_1.json
    此函数采用新的模板中的 JSON 文件初始化结构，并合并现有数据。
    """
    # --- START MODIFICATION FOR OUTPUT PATH ---
    # The target directory is simply ./heatmap/QA/
    json_dir = './heatmap/QA'
    os.makedirs(json_dir, exist_ok=True)

    # Construct JSON file full path using the CSV base name
    # Take the basename and remove the .csv suffix
    base_name_with_suffix = os.path.basename(csv_path) # e.g., heatmap_Topic_1.csv
    base_name = os.path.splitext(base_name_with_suffix)[0] # e.g., heatmap_Topic_1

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
         existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save[qa_type]}

         for item in qa_items:
              # Check if the item is a valid QA dictionary before trying to get Q and A
              if isinstance(item, dict) and 'Q' in item and 'A' in item:
                   if (item.get('Q'), item.get('A')) not in existing_qa_pairs:
                        new_items_to_add.append(item)
                        existing_qa_pairs.add((item.get('Q'), item.get('A'))) # Add to set to prevent duplicates within the new list
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
    # 设定 CSV 文件夹路径 (Matching heatmap.py output)
    csv_folder = './heatmap/csv'
    # 设定 QA 类型 (现在有多种类型，将在 write_qa_to_json 中区分)
    # QA_type = "SC" # This variable is no longer used directly here

    # 检查 CSV 文件夹是否存在
    if not os.path.exists(csv_folder):
        print(f"错误：未找到 CSV 文件夹 {csv_folder}。请先运行 heatmap.py 生成数据。")
        return

    # 遍历文件夹下所有文件（全部都是 .csv）
    for fname in os.listdir(csv_folder):
        # 只处理 CSV 文件
        if fname.endswith('.csv'):
            # 构造完整路径
            csv_path = os.path.join(csv_folder, fname)

            print(f"\n正在处理文件：{csv_path}...") # Added newline for clarity

            # --- 读取元数据和数据 ---
            metadata = read_heatmap_metadata(csv_path)
            df_data = read_heatmap_data_df(csv_path)

            # Check if data reading was successful
            if df_data is None or df_data.empty:
                 print(f"跳过文件 {fname} 的 QA 生成，因为未能读取有效数据。")
                 continue

            # Get unique labels for dynamic QA generation
            unique_labels = task_get_unique_labels(df_data)

            # --- 生成不同类型的 QA ---
            # 根据 热力QA整理.txt 和新的原始模板，生成已指定或需要保留的 QA 类型

            # CTR: Chart type
            qa_ctr_list = fill_qa_ctr()
            if qa_ctr_list:
                 write_qa_to_json(csv_path, "CTR", qa_ctr_list)

            # VEC: Number of cells, rows, columns
            cell_count = task_count_heatmap_cells(df_data)
            qa_vec_list = fill_qa_vec(metadata, cell_count) # Pass metadata for dimensions
            if qa_vec_list:
                 write_qa_to_json(csv_path, "VEC", qa_vec_list)

            # SRP: SVG related (Placeholder)
            qa_srp_list = fill_qa_srp() # Returns []
            if qa_srp_list: # This will be false, so nothing is written for SRP yet
                 write_qa_to_json(csv_path, "SRP", qa_srp_list)

            # VPR: Overall spatial distribution pattern
            qa_vpr_list = fill_qa_vpr(metadata) # Pass metadata for pattern
            if qa_vpr_list:
                 write_qa_to_json(csv_path, "VPR", qa_vpr_list)

            # VE: Specific cell value (Randomly 2-4)
            # Pass only df_data, as fill_qa_ve now handles label selection internally
            qa_ve_list = fill_qa_ve(df_data)
            if qa_ve_list:
                 write_qa_to_json(csv_path, "VE", qa_ve_list)

            # EVJ: Global and label min/max values (Fixed Global + 2 Random Label Max)
            global_min_max = task_get_global_min_max_level(df_data)
            qa_evj_list = fill_qa_evj(df_data, global_min_max, unique_labels)
            if qa_evj_list:
                write_qa_to_json(csv_path, "EVJ", qa_evj_list)

            # SC: Average value for a label (Randomly 1-3)
            qa_sc_list = fill_qa_sc(df_data, unique_labels)
            if qa_sc_list:
                 write_qa_to_json(csv_path, "SC", qa_sc_list)

            # NF: Placeholder from 热力QA整理.txt
            qa_nf_list = fill_qa_nf() # Returns []
            if qa_nf_list: # This will be false
                 write_qa_to_json(csv_path, "NF", qa_nf_list)

            # NC: Cell value comparison (Randomly 2-4)
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


    print("\n热力图 QA 文件生成完毕。") # Added newline

    # 输出结果 (原注释块，保留)
    # print("元信息：")
    # print(f"  大标题  : {meta['title']}")
    # print(f"  子标题  : {meta['subtitle']}")
    # print(f"  单位    : {meta['unit']}")
    # print(f"  模式    : {meta['mode']}\n")
    #
    # print("轴 标签：")
    # print(f"  x 轴标签: {x_label}")
    # print(f"  y 轴标签: {y_label}\n") # This block is commented out and not used


if __name__ == '__main__':
    main()
