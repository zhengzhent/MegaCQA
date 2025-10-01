# File: bubble_QA.py
# Description: Generates QA files for bubble chart data based on bubble.py CSV output and 气泡图QA整理.txt template.

import traceback
import glob
import pandas as pd
import os
import json
import numpy as np
import re
import random # Import random for selections
from typing import List, Dict, Any, Tuple # Import typing hints
import math # Import math for isnan
from sklearn.neighbors import NearestNeighbors # Import for KNN density calculation

# --- Utility Functions (Adapted from scatter_QA.py and heatmap_QA.py) ---

# todo:根据你的csv里首行有的信息进行修改 (Adapted for bubble header)
# 读取文件的第一行和第二行，返回元数据
# --- MODIFIED read_bubble_metadata for two header lines ---
def read_bubble_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first two header lines of the bubble chart CSV (new format).
    Line 1: topic, little_theme, pattern
    Line 2: {x_col_name} ({x_unit}), {y_col_name} ({y_unit}), {size_meaning} ({size_unit})

    Returns a dictionary with keys: 'topic', 'little_theme', 'pattern',
    'x_info': {'name': str, 'unit': str},
    'y_info': {'name': str, 'unit': str},
    'size_info': {'name': str, 'unit': str}.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            header_line1_content = f.readline().strip()
            header_line2_content = f.readline().strip()

        if not header_line1_content or not header_line2_content:
             print(f"Warning: Header lines missing or incomplete in {filepath}")
             return {}

        # Parse the first header line (Topic, Little Theme, Size Distribution)
        parts1 = [part.strip() for part in header_line1_content.split(',', 2)] # Split only into 3 parts max
        expected_parts1 = 3
        if len(parts1) < expected_parts1:
             # Pad with None if fewer parts than expected
             parts1.extend([None] * (expected_parts1 - len(parts1)))
             # print(f"Warning: Less than {expected_parts1} parts on header line 1 in {filepath}.")

        metadata: Dict[str, Any] = {
            'topic': parts1[0] if len(parts1) > 0 else None,
            'little_theme': parts1[1] if len(parts1) > 1 else None,
            'pattern': parts1[2] if len(parts1) > 2 else None # Pattern is the 3rd part
        }


        # Parse the second header line (X, Y, Size descriptions with units)
        parts2 = [part.strip() for part in header_line2_content.split(',', 2)] # Split only into 3 parts max
        expected_parts2 = 3
        if len(parts2) < expected_parts2:
             # Pad with None if fewer parts than expected
             parts2.extend([None] * (expected_parts2 - len(parts2)))
             # print(f"Warning: Less than {expected_parts2} parts on header line 2 in {filepath}.")

        # Function to parse "Label (Unit)" string - already exists, reuse
        def parse_label_unit(label_str):
            if isinstance(label_str, str):
                 match = re.match(r'(.+)\s*\((.+)\)', label_str)
                 if match:
                     return {'name': match.group(1).strip(), 'unit': match.group(2).strip()}
                 return {'name': label_str.strip(), 'unit': ''} # Return label and empty unit if parsing fails
            return {'name': str(label_str).strip() if label_str is not None else '', 'unit': ''} # Handle non-string or None

        # Parse the info strings from line 2
        metadata['x_info'] = parse_label_unit(parts2[0] if len(parts2) > 0 else None)
        metadata['y_info'] = parse_label_unit(parts2[1] if len(parts2) > 1 else None)
        metadata['size_info'] = parse_label_unit(parts2[2] if len(parts2) > 2 else None)


        return metadata

    except Exception as e:
        print(f"Error reading bubble metadata from {filepath}: {e}")
        return {}

# --- MODIFIED read_bubble_data_df for two header lines ---
def read_bubble_data_df(filepath: str, metadata: Dict[str, Any]) -> pd.DataFrame | None:
    """
    Reads the data part of the bubble chart CSV into a DataFrame with named columns.
    Uses column names extracted from metadata.
    Expected data starts from the THIRD line (skiprows=2).
    Expected data columns: X_value, Y_value, Size_value (order matters)
    """
    try:
        # Read data, assuming 3 columns based on bubble chart structure
        # skiprows=2 to skip the two header lines
        df = pd.read_csv(filepath, header=None, skiprows=2, encoding='utf-8')

        # Ensure there are at least 3 columns
        if df.shape[1] < 3:
            # print(f"Warning: Not enough data columns ({df.shape[1]}) in {filepath}.") # Keep original print style minimal
            return None

        # Get column names from metadata. Provide default names if metadata is missing or incomplete.
        x_col_name = metadata.get('x_info', {}).get('name', 'X')
        y_col_name = metadata.get('y_info', {}).get('name', 'Y')
        size_col_name = metadata.get('size_info', {}).get('name', 'Size')

        # Use the actual column names from metadata for the first 3 columns
        # Handle potential extra columns by giving them generic names
        new_column_names = [x_col_name, y_col_name, size_col_name] + [f'col{i}' for i in range(df.shape[1] - 3)]
        df.columns = new_column_names

        # Convert relevant columns to numeric, coercing errors to NaN
        # Use the names obtained from metadata for conversion
        cols_to_convert = [x_col_name, y_col_name, size_col_name]
        for col in cols_to_convert:
            # Check if the column name actually exists in the DataFrame after renaming
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 # This case should ideally not happen if metadata parsing and column assignment are correct
                 print(f"Warning: Column name '{col}' derived from metadata not found in DataFrame columns {list(df.columns)}. Skipping numeric conversion for this column.")


        # Drop rows where X, Y, or Size is NaN, as these are invalid data points for calculations
        # Ensure columns exist before dropping
        cols_to_check = [col for col in [x_col_name, y_col_name, size_col_name] if col in df.columns]
        if cols_to_check:
             df = df.dropna(subset=cols_to_check)


        if df.empty:
             # print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.") # Keep original print style minimal
             return None

        return df

    except pd.errors.EmptyDataError:
        # Handle case where file is empty after skipping headers
        # print(f"Warning: File {filepath} is empty after skipping headers.") # Keep original print style minimal
        return None
    except Exception as e:
        print(f"Error reading bubble data from {filepath}: {e}")
        return None


# --- Calculation Functions (Specific to Bubble Chart) - UNMODIFIED ---
# These functions calculate the data needed for different QA types.

def task_count_bubbles(df: pd.DataFrame) -> int:
    """Calculates the number of bubbles (rows) in the DataFrame."""
    return len(df) # Already dropped NaNs in read_bubble_data_df

def task_get_global_min_max_xyz(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, float]:
    """Calculates min and max values for X, Y, and Size globally."""
    results = {}
    if df is None or df.empty:
         return results

    x_col = metadata.get('x_info', {}).get('name')
    y_col = metadata.get('y_info', {}).get('name')
    size_col = metadata.get('size_info', {}).get('name')

    # Calculate min/max for X
    if x_col and x_col in df.columns and df[x_col].notna().any():
         results['x_min'] = df[x_col].min()
         results['x_max'] = df[x_col].max()

    # Calculate min/max for Y
    if y_col and y_col in df.columns and df[y_col].notna().any():
         results['y_min'] = df[y_col].min()
         results['y_max'] = df[y_col].max()

    # Calculate min/max for Size (Z)
    if size_col and size_col in df.columns and df[size_col].notna().any():
         results['size_min'] = df[size_col].min()
         results['size_max'] = df[size_col].max()

    return results

def task_get_averages_xyz(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, float]:
    """Calculates average values for X, Y, and Size globally."""
    results = {}
    if df is None or df.empty:
         return results

    x_col = metadata.get('x_info', {}).get('name')
    y_col = metadata.get('y_info', {}).get('name')
    size_col = metadata.get('size_info', {}).get('name')

    # Calculate average for X
    if x_col and x_col in df.columns and df[x_col].notna().any():
         results['x_avg'] = df[x_col].mean()

    # Calculate average for Y
    if y_col and y_col in df.columns and df[y_col].notna().any():
         results['y_avg'] = df[y_col].mean()

    # Calculate average for Size (Z)
    if size_col and size_col in df.columns and df[size_col].notna().any():
         results['size_avg'] = df[size_col].mean()

    return results

def task_get_extreme_size_bubbles(df: pd.DataFrame, metadata: Dict[str, Any], n: int = 1) -> List[Dict[str, Any]]:
    """
    Finds the top/bottom N bubbles based on size and returns their X, Y, and Size values.
    Returns a list of dictionaries, one for each extreme bubble found.
    """
    results: List[Dict[str, Any]] = []
    if df is None or df.empty:
         return results

    size_col = metadata.get('size_info', {}).get('name')
    x_col = metadata.get('x_info', {}).get('name')
    y_col = metadata.get('y_info', {}).get('name')

    if not size_col or size_col not in df.columns or not df[size_col].notna().any():
         return results # Cannot determine extreme sizes

    # Sort by size
    sorted_by_size = df.sort_values(by=size_col, ascending=True).reset_index(drop=True)

    # Get bottom N (smallest)
    bottom_n_df = sorted_by_size.head(n)
    for index, row in bottom_n_df.iterrows():
        bubble_info: Dict[str, Any] = {'type': 'smallest'}
        if x_col and x_col in row: bubble_info['x_value'] = row[x_col]
        if y_col and y_col in row: bubble_info['y_value'] = row[y_col] # Corrected row[y[col]
        if size_col and size_col in row: bubble_info['size_value'] = row[size_col]
        results.append(bubble_info)

    # Get top N (largest)
    top_n_df = sorted_by_size.tail(n).iloc[::-1].reset_index(drop=True) # Reverse tail to get largest first
    for index, row in top_n_df.iterrows():
        bubble_info: Dict[str, Any] = {'type': 'largest'}
        if x_col and x_col in row: bubble_info['x_value'] = row[x_col]
        if y_col and y_col in row: bubble_info['y_value'] = row[y_col] # Corrected row[y[col]
        if size_col and size_col in row: bubble_info['size_value'] = row[size_col]
        results.append(bubble_info)

    return results # Contains a list of dicts for smallest and largest bubbles


def task_compare_bubble_sizes(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares the sizes of two randomly selected bubbles.
    Returns a dictionary including values and comparison result.
    """
    results: Dict[str, Any] = {
        'bubble1_index': None,
        'bubble2_index': None,
        'bubble1_size': None,
        'bubble2_size': None,
        'comparison': 'could not be compared'
    }

    if df is None or df.empty or len(df) < 2:
         return results # Need at least 2 bubbles

    size_col = metadata.get('size_info', {}).get('name')
    if not size_col or size_col not in df.columns or not df[size_col].notna().any():
         return results # Cannot compare sizes

    # Select two random indices
    indices = df.index.tolist()
    if len(indices) < 2: return results # Should be caught by len(df) < 2 but double check
    idx1, idx2 = random.sample(indices, 2)

    size1 = df.loc[idx1, size_col]
    size2 = df.loc[idx2, size_col]

    results['bubble1_index'] = idx1
    results['bubble2_index'] = idx2
    results['bubble1_size'] = size1
    results['bubble2_size'] = size2

    if pd.notna(size1) and pd.notna(size2):
        if size1 > size2:
            results['comparison'] = 'larger'
        elif size1 < size2:
            results['comparison'] = 'smaller'
        else:
            results['comparison'] = 'equal'
    # else: comparison remains 'could not be compared'

    return results

def task_find_densest_point(df: pd.DataFrame, metadata: Dict[str, Any], k: int = 5) -> Tuple[float | None, float | None]:
    """
    Finds the coordinates of the most dense point using k-Nearest Neighbors.
    Density is inversely proportional to the average distance to the k nearest neighbors.
    The point with the minimum average k-distance is considered the most dense.
    Returns a tuple (x_value, y_value) of the densest point, or (None, None) if not enough data.
    """
    x_col = metadata.get('x_info', {}).get('name')
    y_col = metadata.get('y_info', {}).get('name')

    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        # print("Warning: X or Y column not found for density calculation.") # Keep original print style minimal
        return (None, None)

    # Ensure data is numeric and drop NaNs for these columns
    coords_df = df[[x_col, y_col]].dropna()

    if len(coords_df) < k + 1: # Need at least k+1 points for k neighbors + the point itself
        # print(f"Warning: Not enough data points ({len(coords_df)}) for KNN density calculation with k={k}.") # Keep original print style minimal
        return (None, None)

    X = coords_df[[x_col, y_col]].values

    # Use k+1 neighbors because the point itself is included with distance 0
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)

    # Calculate distances to k+1 neighbors
    # The first column is distance to itself (0), subsequent k columns are distances to neighbors
    distances = nn.kneighbors(X, return_distance=True)[0]

    # Calculate average distance to the k actual neighbors (excluding the point itself)
    # We take the mean of columns 1 to k (inclusive)
    avg_k_distances = np.mean(distances[:, 1:], axis=1)

    # Find the index of the point with the minimum average k-distance
    densest_point_index_in_X = np.argmin(avg_k_distances)

    # Get the coordinates of the densest point from the filtered DataFrame
    densest_point_coords = coords_df.iloc[densest_point_index_in_X]

    return (densest_point_coords[x_col], densest_point_coords[y_col])


# --- QA Filling Functions based on 气泡图QA整理.txt - UNMODIFIED ---
# These functions format the calculated data into the Q&A structure.
# Leave functions empty or return empty lists for QA types not specified in the text file
# or designated as placeholder.

def fill_qa_ctr() -> List[Dict[str, str]]:
    """Generates QA for chart type (CTR). Based on 气泡图QA整理.txt CTR."""
    # Based on 气泡图QA整理.txt CTR - Note: Template says "line chart".
    # Correct type is "bubble chart". Generate the correct QA with {} annotation.
    qa_list: List[Dict[str, str]] = []
    qa_list.append({
        "Q": "What type of chart is this?",
        "A": "This chart is a {bubble} chart." # Corrected type and added {}
    })
    return qa_list


def fill_qa_vec(bubble_count: int) -> List[Dict[str, str]]:
    """Generates QA for the number of bubbles (VEC). Based on 气泡图QA整理.txt VEC."""
    # Based on 气泡图QA整理.txt VEC
    qa_list: List[Dict[str, str]] = []
    question = "How many bubbles are in this bubble chart?"
    answer = f"There are {{{bubble_count}}} bubbles." # Added {}
    qa_list.append({"Q": question, "A": answer})

    # TXT also asks for rows/columns, which isn't standard for bubbles in this context.
    # Let's generate the bubble count QA only, as per the first example Q/A.
    # If row/column count were needed, we'd get it from the metadata's original dimension string
    # or by counting unique X/Y labels, but cell/row/col count is more heatmap specific.
    # Sticking strictly to the first VEC example Q/A for bubble count.

    return qa_list

def fill_qa_srp() -> List[Dict[str, str]]:
    """Generates QA for SRP (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for SRP (SVG related)
    return []

def fill_qa_vpr(densest_point_coords: Tuple[float | None, float | None], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for the location of the most dense point (VPR) using KNN result."""
    # Based on 气泡图QA整理.txt VPR template
    qa_list: List[Dict[str, str]] = []

    densest_x, densest_y = densest_point_coords

    # Check if valid coordinates were returned
    if densest_x is None or densest_y is None or np.isnan(densest_x) or np.isnan(densest_y):
         return qa_list # Cannot generate QA if no dense point found or not enough data

    x_col_name = metadata.get('x_info', {}).get('name', 'X')
    y_col_name = metadata.get('y_info', {}).get('name', 'Y')
    x_unit = metadata.get('x_info', {}).get('unit', '')
    y_unit = metadata.get('y_info', {}).get('unit', '')

    # Construct Question including axis labels as requested
    # Example template Q: "Where is the highest concentration of bubbles located in terms of tourist numbers and satisfaction levels?"
    # Adapting: "Where is the highest concentration of bubbles located in terms of [X axis name] and [Y axis name]?"
    question = f"Where is the highest concentration of bubbles located in terms of {x_col_name} and {y_col_name}?"

    # Construct Answer using coordinates and units, following template format
    # Example template A: "The highest concentration of bubbles corresponds to approximately 5000 visitors and 4.2 satisfaction."
    # Adapting: "The highest concentration of bubbles corresponds to approximately [X value] [X unit] and [Y value] [Y unit]."
    x_formatted = f"{densest_x:.2f}"
    y_formatted = f"{densest_y:.2f}"

    answer = f"The highest concentration of bubbles corresponds to approximately {{{x_formatted}}} {x_unit} and {{{y_formatted}}} {y_unit}." # Added {}

    qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_ve(extreme_bubbles_n1: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for values of the largest/smallest bubble (VE). Based on 气泡图QA整理.txt VE."""
    # Based on 气泡图QA整理.txt VE
    qa_list: List[Dict[str, str]] = []

    # Find largest and smallest bubble info
    largest_bubble = next((b for b in extreme_bubbles_n1 if b.get('type') == 'largest'), None)
    smallest_bubble = next((b for b in extreme_bubbles_n1 if b.get('type') == 'smallest'), None)

    size_col_name = metadata.get('size_info', {}).get('name', 'Size')
    size_unit = metadata.get('size_info', {}).get('unit', '')
    x_col_name = metadata.get('x_info', {}).get('name', 'X')
    y_col_name = metadata.get('y_info', {}).get('name', 'Y')
    x_unit = metadata.get('x_info', {}).get('unit', '')
    y_unit = metadata.get('y_info', {}).get('unit', '')


    # QA 1: Size of the largest bubble
    if largest_bubble and 'size_value' in largest_bubble and pd.notna(largest_bubble['size_value']):
        size_value = largest_bubble['size_value']
        size_formatted = f"{size_value:.2f}"
        question = f"What is the {size_col_name} of the largest bubble?"
        answer = f"The {size_col_name} of the largest bubble is {{{size_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 2: Size of the smallest bubble
    if smallest_bubble and 'size_value' in smallest_bubble and pd.notna(smallest_bubble['size_value']):
        size_value = smallest_bubble['size_value']
        size_formatted = f"{size_value:.2f}"
        question = f"What is the {size_col_name} of the smallest bubble?"
        answer = f"The {size_col_name} of the smallest bubble is {{{size_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 3: X and Y values of the largest bubble - MODIFIED AS REQUESTED
    if largest_bubble and 'x_value' in largest_bubble and 'y_value' in largest_bubble and \
       pd.notna(largest_bubble['x_value']) and pd.notna(largest_bubble['y_value']):
        x_value = largest_bubble['x_value']
        y_value = largest_bubble['y_value']
        x_formatted = f"{x_value:.2f}"
        y_formatted = f"{y_value:.2f}"

        # Changed question and answer to ask for X and Y coordinates
        question = f"What are the {x_col_name} and {y_col_name} values of the largest bubble?"
        answer = f"The {x_col_name} of the largest bubble is {{{x_formatted}}} {x_unit}, and {{{y_col_name}}} is {{{y_formatted}}} {y_unit}." # Added {}

        qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_evj(global_min_max: Dict[str, float], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for global min/max values for X, Y, and Size (EVJ). Based on 气泡图QA整理.txt EVJ."""
    # Based on 气泡图QA整理.txt EVJ
    qa_list: List[Dict[str, str]] = []

    x_col_name = metadata.get('x_info', {}).get('name', 'X')
    y_col_name = metadata.get('y_info', {}).get('name', 'Y')
    size_col_name = metadata.get('size_info', {}).get('name', 'Size')

    # QA 1 & 2: Global max/min size
    max_size = global_min_max.get('size_max')
    min_size = global_min_max.get('size_min')
    size_unit = metadata.get('size_info', {}).get('unit', '')

    if max_size is not None and not np.isnan(max_size):
        max_formatted = f"{max_size:.2f}"
        question = f"What is the global maximum {size_col_name} in the bubble chart?"
        answer = f"The global maximum {size_col_name} is {{{max_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    if min_size is not None and not np.isnan(min_size):
        min_formatted = f"{min_size:.2f}"
        question = f"What is the global minimum {size_col_name} in the bubble chart?"
        answer = f"The global minimum {size_col_name} is {{{min_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 3 & 4: Global max/min X
    max_x = global_min_max.get('x_max')
    min_x = global_min_max.get('x_min')
    x_unit = metadata.get('x_info', {}).get('unit', '')

    if max_x is not None and not np.isnan(max_x):
        max_formatted = f"{max_x:.2f}"
        question = f"What is the maximum observed value in dimension {x_col_name} in the bubble chart?"
        answer = f"The maximum observed value in the {x_col_name} dimension is {{{max_formatted}}} {x_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    if min_x is not None and not np.isnan(min_x):
        min_formatted = f"{min_x:.2f}"
        question = f"What is the minimum observed value in dimension {x_col_name} in the bubble chart?"
        answer = f"The minimum observed value in the {x_col_name} dimension is {{{min_formatted}}} {x_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})


    # QA 5 & 6: Global max/min Y
    max_y = global_min_max.get('y_max')
    min_y = global_min_max.get('y_min')
    y_unit = metadata.get('y_info', {}).get('unit', '')

    if max_y is not None and not np.isnan(max_y):
        max_formatted = f"{max_y:.2f}"
        question = f"What is the maximum observed value in dimension {y_col_name} in the bubble chart?"
        answer = f"The maximum observed value in the {y_col_name} dimension is {{{max_formatted}}} {y_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    if min_y is not None and not np.isnan(min_y):
        min_formatted = f"{min_y:.2f}"
        question = f"What is the minimum observed value in dimension {y_col_name} in the bubble chart?"
        answer = f"The minimum observed value in the {y_col_name} dimension is {{{min_formatted}}} {y_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_sc(average_values: Dict[str, float], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for average values for X, Y, and Size (SC). Based on 气泡图QA整理.txt SC."""
    # Based on 气泡图QA整理.txt SC (both sections)
    qa_list: List[Dict[str, str]] = []

    x_col_name = metadata.get('x_info', {}).get('name', 'X')
    y_col_name = metadata.get('y_info', {}).get('name', 'Y')
    size_col_name = metadata.get('size_info', {}).get('name', 'Size')

    x_avg = average_values.get('x_avg')
    y_avg = average_values.get('y_avg')
    size_avg = average_values.get('size_avg')

    x_unit = metadata.get('x_info', {}).get('unit', '')
    y_unit = metadata.get('y_info', {}).get('unit', '')
    size_unit = metadata.get('size_info', {}).get('unit', '')


    # QA 1: Average size (Z-axis)
    if size_avg is not None and not np.isnan(size_avg):
        avg_formatted = f"{size_avg:.2f}"
        question = f"What is the average value of the {size_col_name} for all bubbles?"
        answer = f"The average value of the {size_col_name} for all bubbles is {{{avg_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 2: Average X-axis value
    if x_avg is not None and not np.isnan(x_avg):
        avg_formatted = f"{x_avg:.2f}"
        question = f"What is the average value of the {x_col_name} for all bubbles?" # Adjusted Q to be consistent
        answer = f"The average value of {x_col_name} for all bubbles is {{{avg_formatted}}} {x_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 3: Average Y-axis value
    if y_avg is not None and not np.isnan(y_avg):
        avg_formatted = f"{y_avg:.2f}"
        question = f"What is the average value of the {y_col_name} for all bubbles?" # Adjusted Q to be consistent
        answer = f"The average value of {y_col_name} for all bubbles is {{{avg_formatted}}} {y_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})


    return qa_list

def fill_qa_nf(extreme_bubbles_n3: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for top/bottom 3 size values (NF). Based on 气泡图QA整理.txt NF."""
    # Based on 气泡图QA整理.txt NF
    qa_list: List[Dict[str, str]] = []

    size_col_name = metadata.get('size_info', {}).get('name', 'Size')
    size_unit = metadata.get('size_info', {}).get('unit', '')

    # Extract size values from the list of bubble dicts
    top_3_sizes = [b.get('size_value') for b in extreme_bubbles_n3 if b.get('type') == 'largest' and b.get('size_value') is not None and not np.isnan(b.get('size_value'))]
    bottom_3_sizes = [b.get('size_value') for b in extreme_bubbles_n3 if b.get('type') == 'smallest' and b.get('size_value') is not None and not np.isnan(b.get('size_value'))]

    # Sort the extracted values (largest are already sorted descending, smallest ascending)
    top_3_sizes_sorted = sorted(top_3_sizes, reverse=True)
    bottom_3_sizes_sorted = sorted(bottom_3_sizes)

    # Helper to format a list of numbers to a comma-separated string (2 decimal places)
    def format_values_list(values_list):
        if not values_list:
            return "N/A"
        return ", ".join([f"{v:.2f}" for v in values_list])

    top_3_formatted = format_values_list(top_3_sizes_sorted)
    bottom_3_formatted = format_values_list(bottom_3_sizes_sorted)


    # QA 1: Top 3 size values
    if top_3_sizes_sorted: # Only generate if we found any top 3 sizes
        question = f"What are the top {len(top_3_sizes_sorted)} {size_col_name} values in the bubble chart?" # Adjust Q based on actual count found
        answer = f"{{{top_3_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    # QA 2: Bottom 3 size values
    if bottom_3_sizes_sorted: # Only generate if we found any bottom 3 sizes
        question = f"What are the bottom {len(bottom_3_sizes_sorted)} {size_col_name} values in the bubble chart?" # Adjust Q based on actual count found
        answer = f"{{{bottom_3_formatted}}} {size_unit}." # Added {}
        qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_nc(comparison_result: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generates QA for bubble size comparison (NC). Based on 热力QA整理.txt NC example (adapted for bubbles)."""
    # Based on adapted 热力QA整理.txt NC for bubble size comparison
    # Deleted NC QA generation function as requested, leaving placeholder.
    return [] # This function is intentionally empty as per the original request's implied scope


def fill_qa_msr() -> List[Dict[str, str]]:
    """Generates QA for MSR (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for MSR (SVG related)
    return []

def fill_qa_va() -> List[Dict[str, str]]:
    """Generates QA for VA (SVG related). Currently empty as per request."""
    # TODO: Implement QA generation for VA (SVG related)
    return []


# 写入json，使用新的模板初始化结构并合并现有数据 (Adapted from heatmap_QA.py)
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]]):
    """
    将单条或多条 QA (qa_items) 按类别 qa_type 写入到 ./bubble/QA/ 下对应文件。
    例如 ./bubble/csv/bubble_Topic_1.csv → ./bubble/QA/bubble_Topic_1.json
    此函数采用新的模板中的 JSON 文件初始化结构，并合并现有数据。
    """
    # --- START MODIFICATION FOR OUTPUT PATH ---
    # The target directory is simply ./bubble/QA/
    json_dir = './QA'
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
            # print(f"Warning: Could not load or decode JSON from {json_path}. Starting with template structure.") # Keep original print style minimal
            pass # Suppress warning for file not found/decode error, let it start fresh
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
    # 设定 CSV 文件夹路径 (Matching bubble.py output)
    csv_folder = './csv'
    # 设定 QA 类型 (现在有多种类型，将在 write_qa_to_json 中区分)
    # QA_type = "SC" # This variable is no longer used directly here

    # 检查 CSV 文件夹是否存在
    if not os.path.exists(csv_folder):
        print(f"错误：未找到 CSV 文件夹 {csv_folder}。请先运行 bubble.py 生成数据。")
        return

    # 遍历文件夹下所有文件（全部都是 .csv）
    # Use glob for a more robust way to find files matching a pattern
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    if not csv_files:
        print(f"未找到 '*.csv' 文件在目录中: {csv_folder}")
        return

    # Optional: Sort files numerically by the sequence number in the filename
    try:
        def sort_key(f):
            base = os.path.basename(f)
            match = re.search(r'_(\d+)\.csv$', base)
            return int(match.group(1)) if match else base # Use sequence number if found, else use base name
        csv_files.sort(key=sort_key)
    except Exception:
         # Fallback to alphabetical sort if numerical sort fails
         csv_files.sort()
         print("Warning: Could not sort files numerically. Sorting alphabetically.")


    processed_count = 0
    for csv_path in csv_files:
        # 构造完整路径
        # csv_path = os.path.join(csv_folder, fname) # Already have full path from glob

        print(f"\n正在处理文件：{csv_path}...") # Added newline for clarity

        # --- 读取元数据和数据 ---
        # The metadata reading function is modified to handle two lines
        metadata = read_bubble_metadata(csv_path)
        # The data reading function is modified to skip two header lines
        df_data = read_bubble_data_df(csv_path, metadata)

        # Check if data reading was successful and metadata is usable
        if df_data is None or df_data.empty or not metadata or not metadata.get('x_info') or not metadata.get('y_info') or not metadata.get('size_info'):
             print(f"跳过文件 {os.path.basename(csv_path)} 的 QA 生成，因为未能读取有效数据或元数据不完整。")
             continue

        print("数据读取成功。进行计算和 QA 生成...") # Added success message

        # --- 进行各种计算 - UNMODIFIED ---
        bubble_count = task_count_bubbles(df_data)
        global_min_max = task_get_global_min_max_xyz(df_data, metadata)
        average_values = task_get_averages_xyz(df_data, metadata)
        # For VE, VPR we need info about the single largest/smallest bubble
        extreme_bubbles_n1 = task_get_extreme_size_bubbles(df_data, metadata, n=1)
         # For NF, we need info about the top/bottom 3 bubbles by size
        extreme_bubbles_n3 = task_get_extreme_size_bubbles(df_data, metadata, n=3)
        # For NC, we need a comparison of two random bubbles (Calculation function remains but QA generation is empty)
        comparison_result_nc = task_compare_bubble_sizes(df_data, metadata)
        # For VPR, we need the coordinates of the most dense point using KNN
        densest_point_coords = task_find_densest_point(df_data, metadata)


        # --- 生成不同类型的 QA - UNMODIFIED ---
        # 根据 气泡图QA整理.txt 和新的原始模板，生成已指定或需要保留的 QA 类型

        # CTR: Chart type
        qa_ctr_list = fill_qa_ctr()
        if qa_ctr_list:
             write_qa_to_json(csv_path, "CTR", qa_ctr_list)

        # VEC: Number of bubbles
        qa_vec_list = fill_qa_vec(bubble_count)
        if qa_vec_list:
             write_qa_to_json(csv_path, "VEC", qa_vec_list)

        # SRP: SVG related (Placeholder)
        qa_srp_list = fill_qa_srp() # Returns []
        if qa_srp_list: # This will be false, so nothing is written for SRP yet
             write_qa_to_json(csv_path, "SRP", qa_srp_list)

        # VPR: Overall spatial distribution pattern (based on densest point)
        qa_vpr_list = fill_qa_vpr(densest_point_coords, metadata)
        if qa_vpr_list:
             write_qa_to_json(csv_path, "VPR", qa_vpr_list)

        # VE: Values of largest/smallest bubbles
        qa_ve_list = fill_qa_ve(extreme_bubbles_n1, metadata)
        if qa_ve_list:
             write_qa_to_json(csv_path, "VE", qa_ve_list)

        # EVJ: Global min/max values for X, Y, Size
        qa_evj_list = fill_qa_evj(global_min_max, metadata)
        if qa_evj_list:
            write_qa_to_json(csv_path, "EVJ", qa_evj_list)

        # SC: Average values for X, Y, Size
        qa_sc_list = fill_qa_sc(average_values, metadata)
        if qa_sc_list:
             write_qa_to_json(csv_path, "SC", qa_sc_list)

        # NF: Top/Bottom N size values
        qa_nf_list = fill_qa_nf(extreme_bubbles_n3, metadata) # Pass top/bottom 3 info
        if qa_nf_list:
             write_qa_to_json(csv_path, "NF", qa_nf_list)

        # NC: Bubble size comparison (Generation is empty, but placeholder key in JSON is kept)
        qa_nc_list = fill_qa_nc(comparison_result_nc, metadata) # This will return []
        # The write_qa_to_json function ensures the "NC" key exists with an empty list
        # if fill_qa_nc returns [], so we don't need an 'if qa_nc_list:' check here
        write_qa_to_json(csv_path, "NC", qa_nc_list)


        # MSR: SVG related (Placeholder)
        qa_msr_list = fill_qa_msr() # Returns []
        if qa_msr_list: # This will be false
             write_qa_to_json(csv_path, "MSR", qa_msr_list)

        # VA: SVG related (Placeholder)
        qa_va_list = fill_qa_va() # Returns []
        if qa_va_list: # This will be false
             write_qa_to_json(csv_path, "VA", qa_va_list)

        print(f"已为文件 {os.path.basename(csv_path)} 生成 QA。") # Added confirmation message
        processed_count += 1


    print(f"\n气泡图 QA 文件生成完毕。总共处理了 {processed_count} 个文件。") # Added newline and count

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
    # Added a try-except block for the main execution loop
    try:
        main()
    except Exception as e_main_loop:
        print(f"\n一个未预期的错误发生在主执行过程中: {e_main_loop}")
        traceback.print_exc()

