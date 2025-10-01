# File: heatmap_QA_addition.py
# Description: Supplements existing heatmap QA JSON files with new MSR questions based on the provided 补充问题.txt template.
# This script focuses ONLY on the supplemental QA types defined in the input 补充问题.txt.
# Corrected: Included the missing write_qa_to_json function definition.
# Modified: Restricted the width of the random range for MSR QA to [0.01, 0.3].
# Added: VA (Value Area / Connected Hotspot Regions) QA generation.

import traceback
import glob
import pandas as pd
import os
import json
import numpy as np
import re
import random
from typing import List, Dict, Any, Tuple
import math
import collections # Added for deque
import decimal # Added for precise rounding for comparison if needed, though simple round() might suffice for this case

# --- Utility Functions (Adopted from provided heatmap_QA.py) ---

# Read the first header line (metadata)
def read_heatmap_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first header line of the heatmap CSV.
    Returns a dictionary with keys: 'topic', 'little_theme', 'dimension', 'pattern'.
    """
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

# --- Calculation Functions (Adopted from provided heatmap_QA.py) ---
# Only include the ones potentially useful for supplemental QAs from the new TXT.
# task_get_global_min_max_level is needed for MSR range.
# task_filter_values_by_range is needed for MSR range.

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

def task_filter_values_by_range(df: pd.DataFrame, level_range: Tuple[float, float]) -> List[Dict[str, Any]]:
    """
    Finds x_block and y_block labels for cells whose level is within a given range.
    Returns a list of dicts: [{'x_block': '...', 'y_block': '...'}, ...]
    """
    results: List[Dict[str, Any]] = []
    if df is None or df.empty or 'level' not in df.columns or 'x_block' not in df.columns or 'y_block' not in df.columns:
         return results

    min_val, max_val = level_range

    # Filter cells by level range, ensuring level is not NaN (already done in read_heatmap_data_df)
    filtered_df = df[(df['level'] >= min_val) & (df['level'] <= max_val)]

    if not filtered_df.empty:
        # Return x_block and y_block for matching cells
        results = filtered_df[['x_block', 'y_block']].to_dict('records')

    return results

# --- JSON Writing Function (Adopted from provided heatmap_QA.py context and corrected) ---
# This function was missing in the previous response.
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]], qa_dir: str = './heatmap/QA'):
    """
    将单条或多条 QA (qa_items) 按类别 qa_type 写入到 ./heatmap/QA/ 下对应文件。
    例如 ./heatmap/csv/heatmap_Topic_1.csv → ./heatmap/QA/heatmap_Topic_1.json
    此函数采用新的模板中的 JSON 文件初始化结构，并合并现有数据。
    """
    # --- START MODIFICATION FOR OUTPUT PATH ---
    # The target directory is simply ./heatmap/QA/
    json_dir = qa_dir # Use the passed qa_dir argument
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


# --- Supplemental QA Filling Functions based strictly on 补充问题.txt ---
# Only implement fill_qa_msr_supplemental_heatmap as per the new TXT.

def fill_qa_msr_supplemental_heatmap(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates supplemental MSR QA for heatmaps.
    - Finds cells whose heat values fall within a random range with limited width.
    Based strictly on the 补充问题.txt MSR example.
    """
    qa_list: List[Dict[str, str]] = []

    # Need valid level data with at least some variance to determine a range
    if df is None or df.empty or 'level' not in df.columns or df['level'].nunique() < 2:
        print("Info (MSR-HM): Not enough distinct heat values to generate range QA.")
        return qa_list

    global_min_max = task_get_global_min_max_level(df)
    min_level = global_min_max.get('min_level')
    max_level = global_min_max.get('max_level')

    if min_level is None or max_level is None or np.isnan(min_level) or np.isnan(max_level) or min_level >= max_level:
        print("Info (MSR-HM): Could not determine a valid global min/max heat range for QA.")
        return qa_list

    # Define desired range width bounds
    min_range_width_req = 0.01
    max_range_width_req = 0.30

    # Determine the maximum possible width we can generate within the data's global range
    max_possible_width_in_data = max_level - min_level

    # Check if the data's range is large enough to accommodate the minimum required width
    if max_possible_width_in_data < min_range_width_req:
         print(f"Info (MSR-HM): Data range ({max_possible_width_in_data:.2f}) is smaller than minimum required range width ({min_range_width_req}). Skipping range QA.")
         return qa_list

    # Determine the effective maximum width we can generate (capped by requirement and data)
    effective_max_width = min(max_range_width_req, max_possible_width_in_data)

    # Generate a random width within the allowed bounds
    try:
        # Ensure min_range_width_req is not greater than effective_max_width, which could happen if data range is very small.
        # This check might be redundant given the max_possible_width_in_data < min_range_width_req check above,
        # but as a safeguard for random.uniform if effective_max_width somehow becomes less than min_range_width_req.
        if min_range_width_req > effective_max_width:
             print(f"Info (MSR-HM): Adjusted effective_max_width ({effective_max_width:.2f}) is less than min_range_width_req ({min_range_width_req:.2f}). Skipping range QA.")
             return qa_list

        range_width = random.uniform(min_range_width_req, effective_max_width)

        # Now generate a random starting point (range_min) such that range_min + range_width <= max_level
        # The maximum possible value for range_min is max_level - range_width
        max_possible_range_min = max_level - range_width
        
        # Ensure min_level is not greater than max_possible_range_min before calling random.uniform
        if min_level > max_possible_range_min:
            # This can happen if range_width is very close to (max_level - min_level)
            # In this scenario, range_min effectively must be min_level
            if math.isclose(min_level, max_possible_range_min):
                range_min = min_level
            else: # Should not really happen if logic is correct, but as a fallback
                print(f"Warning (MSR-HM): min_level ({min_level:.2f}) > max_possible_range_min ({max_possible_range_min:.2f}). Skipping range QA.")
                return qa_list
        else:
             range_min = random.uniform(min_level, max_possible_range_min)


        # Calculate range_max
        range_max = range_min + range_width

        # Final check just in case of float precision issues (should not happen with this approach)
        if range_min >= range_max:
             print(f"Warning (MSR-HM): Generated invalid range {range_min:.2f} to {range_max:.2f}. Skipping range QA.")
             return qa_list


    except ValueError as e: # Handle potential errors from random.uniform if bounds are inverted or equal
        print(f"Warning (MSR-HM): Error generating random range: {e}. min_level={min_level:.2f}, max_level={max_level:.2f}, min_range_width_req={min_range_width_req}, effective_max_width={effective_max_width}, max_possible_range_min={max_possible_range_min if 'max_possible_range_min' in locals() else 'N/A'}. Skipping range QA.")
        return qa_list


    # Find cells within the generated range
    cells_in_range = task_filter_values_by_range(df, (range_min, range_max))

    # Format the list of cells as (X, Y) tuples
    # Ensure x_block and y_block are strings for formatting
    formatted_cells_list = [f"({str(cell.get('x_block', 'N/A'))}, {str(cell.get('y_block', 'N/A'))})" for cell in cells_in_range]

    # Format the range for the question
    range_min_formatted = f"{range_min:.2f}"
    range_max_formatted = f"{range_max:.2f}"

    # Q template: In the heatmap, what are the row and column labels for all cells whose heat values fall within the range of 0.3 to 0.5?
    question = (f"In the heatmap, what are the row and column labels for all cells whose heat values "
                f"fall within the range of {range_min_formatted} to {range_max_formatted}?") # No {}

    # A template: The cells are: (A, B), (C, D), (E, F).
    # Handle the case where no cells are found in the range
    if formatted_cells_list:
         answer = f"The cells are: {{{', '.join(formatted_cells_list)}}}."
    else:
         answer = f"There are {{no cells}} with heat values within the range of {range_min_formatted} to {range_max_formatted}." # Provide a meaningful answer


    qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_va_supplemental_heatmap(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generates supplemental VA (Value Area / Connected Hotspot Regions) QA for heatmaps.
    - Identifies 'hotspots' (cells with heat value > average, where 'average' is rounded for the question's threshold).
    - Counts distinct connected regions of these hotspots using 4-way adjacency, EXCLUDING regions of size 1.
    """
    qa_list: List[Dict[str, str]] = []

    if df is None or df.empty or 'level' not in df.columns or \
       'x_block' not in df.columns or 'y_block' not in df.columns:
        print("Info (VA-HM): DataFrame is unsuitable for VA QA (None, empty, or missing required columns 'level', 'x_block', 'y_block').")
        return qa_list

    if df['level'].dtype not in [np.float64, np.int64, float, int] or df['level'].isnull().all():
        print("Info (VA-HM): 'level' column has no valid numeric data.")
        return qa_list
    
    avg_level_precise = df['level'].mean()
    if pd.isna(avg_level_precise):
        print("Info (VA-HM): Average level could not be computed.")
        return qa_list

    # Determine the threshold for 'hotspot' based on the rounded average value used in the question.
    # We'll round the average to 2 decimal places for comparison, consistent with display.
    avg_level_display_precision = 2 
    threshold_value = round(avg_level_precise, avg_level_display_precision)
    
    # Format the average value for display in the question (e.g., "0.50")
    avg_level_formatted_for_q = f"{avg_level_precise:.{avg_level_display_precision}f}"

    # Hotspots are cells with 'level' strictly greater than the 'threshold_value'
    hotspot_df = df[df['level'] > threshold_value].copy() # Use .copy() to avoid SettingWithCopyWarning later if modifying

    if hotspot_df.empty:
        question = (f"Considering cells with heat values strictly greater than the average heat value "
                    f"{avg_level_formatted_for_q} as 'hotspots', how many distinct connected regions "
                    f"(using 4-way adjacency) of these hotspots are present in the heatmap, "
                    f"excluding regions that consist of only a single cell?")
        answer = (f"There are no hotspot regions with heat values strictly greater than the average "
                  f"({avg_level_formatted_for_q}).")
        qa_list.append({"Q": question, "A": answer})
        return qa_list

    # Map x_block and y_block to grid indices
    try:
        unique_y_labels = sorted(list(pd.unique(df['y_block'].astype(str)))) # Rows
        unique_x_labels = sorted(list(pd.unique(df['x_block'].astype(str)))) # Columns
    except TypeError:
        print("Info (VA-HM): x_block or y_block labels are of mixed types and cannot be sorted reliably. Skipping VA QA.")
        return qa_list


    num_rows = len(unique_y_labels)
    num_cols = len(unique_x_labels)

    y_map = {label: i for i, label in enumerate(unique_y_labels)}
    x_map = {label: i for i, label in enumerate(unique_x_labels)}

    # Create a grid representing the heatmap, marking hotspots
    grid = np.zeros((num_rows, num_cols), dtype=int)
    # Store original labels for mapping back if needed, though not strictly needed for counting
    # label_grid = np.full((num_rows, num_cols), '', dtype=object) 

    # Populate the grid with 1s for hotspots
    for _, row_data in hotspot_df.iterrows():
        try:
            # Ensure labels are treated as strings for mapping
            y_label_str = str(row_data['y_block'])
            x_label_str = str(row_data['x_block'])
            
            if y_label_str in y_map and x_label_str in x_map:
                r_idx = y_map[y_label_str]
                c_idx = x_map[x_label_str]
                grid[r_idx, c_idx] = 1
                # label_grid[r_idx, c_idx] = f"({x_label_str}, {y_label_str})" # Store labels
            else:
                 # This case should ideally not happen if unique_y_labels/x_labels are from the full df,
                 # but as a safeguard:
                 print(f"Warning (VA-HM): Hotspot cell labels ({y_label_str}, {x_label_str}) not found in grid mapping. Skipping cell.")

        except KeyError:
            # This catch is less likely now with the explicit map lookups, but kept as a safeguard.
            print(f"Warning (VA-HM): Could not map hotspot cell ({row_data.get('y_block', 'N/A')}, {row_data.get('x_block', 'N/A')}) to grid indices. Skipping cell.")
            continue


    visited = np.zeros_like(grid, dtype=bool)
    num_connected_regions = 0 # This will count regions with size > 1

    for r in range(num_rows):
        for c in range(num_cols):
            # If it's a hotspot cell and hasn't been visited yet
            if grid[r, c] == 1 and not visited[r, c]:
                
                # Start BFS for a new potential component
                q = collections.deque([(r, c)])
                visited[r, c] = True
                current_component_size = 0
                
                # List to store cells in the current component (optional, for debugging/future use)
                # current_component_cells = [] 

                while q:
                    curr_r, curr_c = q.popleft()
                    current_component_size += 1
                    # if label_grid[curr_r, curr_c]:
                    #     current_component_cells.append(label_grid[curr_r, curr_c])

                    # Explore 4-way neighbors
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc

                        # Check bounds, if neighbor is a hotspot, and if it's not visited
                        if 0 <= nr < num_rows and 0 <= nc < num_cols and \
                           grid[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

                # --- OPTIMIZATION: Check component size AFTER BFS ---
                # If the component size is greater than 1, count it as a connected region
                if current_component_size > 1:
                    num_connected_regions += 1
                    # print(f"Found component of size {current_component_size} at start ({r},{c}). Cells: {current_component_cells}") # Debugging

    # --- END MODIFICATION ---
    
    question = (f"Considering cells with heat values strictly greater than the average heat value "
                f"{avg_level_formatted_for_q} as 'hotspots', how many distinct connected regions "
                f"of these hotspots are present in the heatmap, "
                f"excluding regions that consist of only a single cell?")

    if num_connected_regions == 0:
         answer = (f"There are {{no connected hotspot regions}} with more than one cell, "
                   f"with heat values strictly greater than the average ({avg_level_formatted_for_q}).")
    elif num_connected_regions == 1:
        answer = "There is {1} distinct connected hotspot region (excluding single-cell regions)."
    else:
        answer = f"There are {{{num_connected_regions}}} distinct connected hotspot regions (excluding single-cell regions)."

    qa_list.append({"Q": question, "A": answer})
    return qa_list


# --- Main Processing Logic ---
def main():
    random.seed(42) # for reproducibility
    np.random.seed(42) # for numpy related randomness if any

    # Assuming heatmap CSVs are in ./heatmap/csv and QAs in ./heatmap/QA
    csv_dir = './heatmap/csv'
    qa_dir = './heatmap/QA'

    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory not found at {csv_dir}")
        return

    # Find heatmap CSV files (adjust pattern if needed)
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv")) # Using *.csv for flexibility
    # Optional: Filter specifically for 'heatmap' in filename if needed
    # csv_files = [f for f in csv_files if 'heatmap' in os.path.basename(f).lower()]


    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return

    print(f"Found {len(csv_files)} CSV files to process for QA supplementation.")
    processed_count = 0

    for csv_path in csv_files:
        print(f"\nProcessing file: {csv_path}")

        # --- Read data ---
        # Metadata is not strictly needed for supplemental QAs here, but read it for consistency context
        file_metadata = read_heatmap_metadata(csv_path)
        df_data = read_heatmap_data_df(csv_path) 

        if df_data is None:
             print(f"Skipping file {os.path.basename(csv_path)} due to read error or no valid data rows.")
             continue

        essential_cols = ['x_block', 'y_block', 'level']
        if df_data.empty or not all(col in df_data.columns for col in essential_cols):
             print(f"Skipping file {os.path.basename(csv_path)} QA generation due to no valid data or missing essential columns.")
             continue
        
        print(f"  Read {len(df_data)} data rows.")

        # --- Generate SUPPLEMENTAL MSR QA ---
        msr_qas = fill_qa_msr_supplemental_heatmap(df_data, file_metadata) # Pass metadata
        if msr_qas:
            write_qa_to_json(csv_path, "MSR", msr_qas, qa_dir)
            print(f"  Added {len(msr_qas)} MSR QA(s).")
        else:
            print(f"  No supplemental MSR QA generated for {os.path.basename(csv_path)}.")

        # --- Generate SUPPLEMENTAL VA QA ---
        va_qas = fill_qa_va_supplemental_heatmap(df_data, file_metadata) # Pass metadata
        if va_qas:
            write_qa_to_json(csv_path, "VA", va_qas, qa_dir)
            print(f"  Added {len(va_qas)} VA QA(s).")
        else:
            print(f"  No supplemental VA QA generated for {os.path.basename(csv_path)}.")


        processed_count += 1

    print(f"\nHeatmap SUPPLEMENTAL QA supplementation complete. Processed {processed_count} CSV files.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
        traceback.print_exc()

