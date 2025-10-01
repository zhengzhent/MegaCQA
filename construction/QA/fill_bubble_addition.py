# File: fill_bubble_QA_addition.py
# Description: Supplements existing fill bubble chart QA JSON files with new SRP, MSR, and VA questions.
# Uses corrected data reading functions based on user's fill_bubble_QA.py.
# Implements QA types strictly based on the provided 补充问题.txt template for SRP, MSR, VA.
# Modified MSR logic for min/max subcategory size to randomly select level 2 or 3 with fallback.
# Modified SRP logic to compare any two random bubbles based ONLY on depth.

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

# --- Utility Functions (Adopted from user's fill_bubble_QA.py) ---

# Read the first header line (metadata)
def read_fill_bubble_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first header line of the fill bubble chart CSV.
    Returns a dictionary with keys: 'topic', 'little_theme', 'total_nodes'.
    """
    try:
        # header=None 表示不把任何行当成列名，nrows=1 只读第一行
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

# Read the data part and column names (from the second line)
def read_fill_bubble_data_df(filepath: str) -> pd.DataFrame | None:
    """
    Reads the data part of the fill bubble chart CSV into a DataFrame with named columns.
    Expected data columns (on second header line): size,father,depth,label
    """
    try:
        # Read data, assuming columns based on fill_bubble.py output header
        # The actual column names are on the second row of the CSV
        # We need to read the second row to get the column names, then read data skipping 2 rows
        col_names_df = pd.read_csv(filepath, header=None, skiprows=1, nrows=1, encoding='utf-8')
        if col_names_df.empty:
             # print(f"Warning: Column names line missing in {filepath}.") # Keep original print style minimal
             return None, {}
        col_names = col_names_df.iloc[0].tolist()

        # Read data skipping the first two header rows
        df = pd.read_csv(filepath, header=None, skiprows=2, encoding='utf-8')

        # Ensure the number of columns matches the header
        if df.shape[1] != len(col_names):
             print(f"Warning: Data columns ({df.shape[1]}) do not match header columns ({len(col_names)}) in {filepath}. Attempting partial assignment.") # Keep original print style minimal
             # Attempt to assign names to existing columns up to the minimum count
             min_cols = min(df.shape[1], len(col_names))
             df.columns = col_names[:min_cols] + [f'col{i}' for i in range(min_cols, df.shape[1])]
        else:
             df.columns = col_names # Assign column names from the second header line


        # Convert relevant columns to numeric, coercing errors to NaN
        # Based on expected columns: size, father, depth
        numeric_cols_to_check = ['size', 'father', 'depth']
        for col in numeric_cols_to_check:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical columns ('size', 'father', 'depth', 'label') are NaN
        # 'label' is assumed to be string, so check for None or empty string if needed, but pandas handles missing strings as NaN often
        critical_cols = [col for col in ['size', 'father', 'depth', 'label'] if col in df.columns]
        if critical_cols:
             df = df.dropna(subset=critical_cols)


        if df.empty:
             print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.") # Keep original print style minimal
             return None, {}

        # Store assumed column names in metadata-like structure for generation functions
        # Although metadata is read separately, we also infer key column roles here
        inferred_metadata = {
            'size_col': 'size', # Based on expected column name
            'father_col': 'father',
            'depth_col': 'depth',
            'label_col': 'label'
        }
        # Return both the dataframe and the inferred column names
        return df, inferred_metadata

    except Exception as e:
        print(f"Error reading fill bubble data from {filepath}: {e}")
        return None, {} # Return None, {} on error


# --- JSON Writing Function (Kept from previous version, adapted for fill_bubble paths) ---
def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]], qa_dir: str = './fill_bubble/QA'):
    """
    Writes QA items to the JSON file corresponding to the CSV path for fill bubbles.
    Initializes JSON with a template structure if it doesn't exist.
    """
    os.makedirs(qa_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    json_path = os.path.join(qa_dir, base_name + '.json')

    # Define the expected QA types for fill bubbles based on the common template
    template_data: Dict[str, List[Dict[str, str]]] = {
        "CTR": [], "VEC": [], "SRP": [], "VPR": [], "VE": [],
        "EVJ": [], "SC": [], "NF": [], "NC": [], "MSR": [], "VA": []
    }

    existing_data: Dict[str, List[Dict[str, str]]] = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                existing_data = loaded_data
            else:
                 print(f"Warning: Existing JSON data in {json_path} is not a dictionary. Overwriting with template structure.")
        except json.JSONDecodeError:
            print(f"Warning: JSONDecodeError for {json_path}. Starting with template.")
        except Exception as e:
            print(f"Warning: Could not read or parse JSON from {json_path}: {e}. Starting with template.")

    data_to_save = template_data.copy()
    for key in template_data.keys():
        if key in existing_data and isinstance(existing_data.get(key), list):
            data_to_save[key] = existing_data[key]

    if qa_type in data_to_save:
        existing_qa_pairs = set()
        if isinstance(data_to_save[qa_type], list):
             # Use tuple of Q and A for uniqueness check
            existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save[qa_type]
                                 if isinstance(item, dict) and 'Q' in item and 'A' in item}

        new_items_to_add = []
        for item in qa_items:
            if isinstance(item, dict) and 'Q' in item and 'A' in item:
                if (item.get('Q'), item.get('A')) not in existing_qa_pairs:
                    new_items_to_add.append(item)
                    existing_qa_pairs.add((item.get('Q'), item.get('A'))) # Add to set to avoid adding duplicates within the new items
            else:
                print(f"Warning: Skipping invalid QA item format for type {qa_type}: {item}")

        if isinstance(data_to_save[qa_type], list):
            data_to_save[qa_type].extend(new_items_to_add)
        else:
            # This case should ideally not happen if template_data is correct, but added for robustness
            data_to_save[qa_type] = new_items_to_add
    else:
        print(f"Error: QA type '{qa_type}' not in template. Items not added.")
        return

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing QA to {json_path} for type {qa_type}: {e}")

# --- Helper function to find min/max label at a specific depth ---
def get_min_max_label_at_depth(df: pd.DataFrame, depth: int, value_col: str, label_col: str, depth_col: str) -> Dict[str, Any] | None:
    """
    Finds the label(s) with the min/max size value for bubbles at a specific depth.
    Returns a dictionary {'max_val': ..., 'max_labels': [...], 'min_val': ..., 'min_labels': [...]}
    or None if no valid data at that depth.
    """
    df_at_depth = df[(df[depth_col] == depth) & df[value_col].notna() & df[label_col].notna()].copy()

    if df_at_depth.empty or df_at_depth[value_col].nunique() == 0:
        # No valid bubbles at this depth or all sizes are the same (nunique=0 after dropna is unlikely but safe check)
        return None

    max_val = df_at_depth[value_col].max()
    min_val = df_at_depth[value_col].min()

    max_labels = df_at_depth[df_at_depth[value_col] == max_val][label_col].tolist()
    min_labels = df_at_depth[df_at_depth[value_col] == min_val][label_col].tolist()

    return {
        'max_val': max_val,
        'max_labels': max_labels,
        'min_val': min_val,
        'min_labels': min_labels
    }


# --- Supplemental QA Generation Functions for Fill Bubble (Based strictly on 补充问题.txt) ---

def fill_qa_srp_supplemental_fill_bubble(df: pd.DataFrame, metadata: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Generates supplemental SRP QA for fill bubbles by comparing the depth of two random bubbles.
    Answers whether one is 'deeper', 'shallower', or 'on the same depth as' the other.
    Based strictly on the user's requested template.
    """
    qa_list: List[Dict[str, str]] = []

    label_col = metadata.get('label_col')
    depth_col = metadata.get('depth_col')

    if not all([label_col, depth_col]) or df.empty:
        print("Warning (SRP-FB): Missing column names in metadata or DataFrame is empty.")
        return qa_list

    # Filter for bubbles that have a valid label and depth
    df_valid_depth_label = df[df[label_col].notna() & df[depth_col].notna() & pd.notna(df[depth_col])].reset_index(drop=True)

    if len(df_valid_depth_label) < 2:
        print("Info (SRP-FB): Not enough bubbles (need at least 2) with valid labels and depths to compare for SRP QA.")
        return qa_list

    # Randomly select two distinct 0-based indices from the valid DataFrame
    idx_a, idx_b = random.sample(range(len(df_valid_depth_label)), 2)

    bubble_a_row = df_valid_depth_label.iloc[idx_a]
    bubble_b_row = df_valid_depth_label.iloc[idx_b]

    label_a = bubble_a_row[label_col]
    label_b = bubble_b_row[label_col]
    depth_a = bubble_a_row[depth_col]
    depth_b = bubble_b_row[depth_col]

    # Ensure labels and depths are valid for comparison
    if not isinstance(label_a, str) or not isinstance(label_b, str) or not pd.notna(depth_a) or not pd.notna(depth_b):
        print(f"Warning (SRP-FB): Skipping comparison due to invalid labels or depths: '{label_a}', '{label_b}', depths {depth_a}, {depth_b}.")
        return qa_list # Return empty list if labels/depths are invalid


    # Determine the relationship based ONLY on depth
    relationship_english = ""
    relationship_chinese = "" # Store the Chinese term for the answer

    if depth_a > depth_b:
        relationship_english = "deeper than"
        relationship_chinese = "更深"
    elif depth_a < depth_b:
        relationship_english = "shallower than"
        relationship_chinese = "更浅"
    else: # depth_a == depth_b
        relationship_english = "on the same depth as"
        relationship_chinese = "一样深"


    # Construct the question and answer following the user's template
    # Q template: Is Bubble A positioned deeper, shallower or on the same depth as Bubble B?
    # The template uses Chinese characters directly in the question.
    question = (f"Is Bubble {label_a} positioned deeper, shallower or on the same depth "
                f"as Bubble {label_b}?") # No {} around labels in Q

    # A template: Bubble A is positioned 更深 Bubble B. / 一样深 Bubble B. / 更浅 Bubble B.
    # Based on previous SRP answer format, use {{}} around the relationship term.
    answer = f"Bubble {label_a} is positioned {{{relationship_english}}} Bubble {label_b}." # {{}} around the Chinese term


    qa_list.append({"Q": question, "A": answer})

    return qa_list

def fill_qa_msr_supplemental_fill_bubble(df: pd.DataFrame, metadata: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Generates supplemental MSR QA for fill bubbles.
    - Max/Min bubble area (size) for subcategories at a randomly chosen level (2 or 3), with fallback to 2.
    - Count bubbles above/below average value (individual rows).
    Based strictly on the 补充问题.txt MSR examples.
    """
    qa_list: List[Dict[str, str]] = []

    label_col = metadata.get('label_col')
    value_col = metadata.get('size_col') # 'size' column is the value
    depth_col = metadata.get('depth_col')
    # father_col = metadata.get('father_col') # Not directly needed for this logic

    if not all([label_col, value_col, depth_col]) or df.empty:
        print("Warning (MSR-FB): Missing column names in metadata or DataFrame is empty.")
        return qa_list

    df_valid = df[df[value_col].notna() & df[label_col].notna() & df[depth_col].notna()]
    if df_valid.empty:
        print("Warning (MSR-FB): No valid data points with size, label, and depth.")
        # Still generate count QAs if possible, but min/max will be skipped
        pass # Continue to count QAs

    # --- MSR Q1 & Q2: Max/Min bubble area (size) for subcategories ---

    # Determine viable depths for random selection (Level 2 and/or Level 3)
    viable_depths = []
    # Check if Level 2 exists and has valid size data
    if not df_valid[df_valid[depth_col] == 2].empty and df_valid[df_valid[depth_col] == 2][value_col].notna().any():
         viable_depths.append(2)
    # Check if Level 3 exists and has valid size data
    if not df_valid[df_valid[depth_col] == 3].empty and df_valid[df_valid[depth_col] == 3][value_col].notna().any():
         viable_depths.append(3)

    if not viable_depths:
        print("Info (MSR-FB): No viable depths (2 or 3) with valid size data for Max/Min QA.")
        # Skip Max/Min QAs
    else:
        # Generate QA for Largest Bubble Area
        # Randomly choose a target depth from viable depths
        target_depth_max = random.choice(viable_depths)
        min_max_data_max = get_min_max_label_at_depth(df_valid, target_depth_max, value_col, label_col, depth_col)

        # Fallback logic: If chosen depth yielded no data (shouldn't happen if in viable_depths, but as safeguard) or if we want a different fallback
        # A simpler fallback if the randomly chosen depth has no valid data is to just skip that specific QA,
        # or always fallback to Level 2 if Level 2 is viable. Let's implement the latter: If random choice gives Depth 3 and it fails, try Depth 2.
        if min_max_data_max is None and target_depth_max == 3 and 2 in viable_depths:
             print(f"Info (MSR-FB): Fallback for Max Size QA: Depth {target_depth_max} had no valid data, trying Depth 2.")
             target_depth_max = 2 # Fallback to Level 2
             min_max_data_max = get_min_max_label_at_depth(df_valid, target_depth_max, value_col, label_col, depth_col)
        elif min_max_data_max is None and target_depth_max == 2 and 3 in viable_depths:
             # If Level 2 failed but Level 3 exists, maybe we should try Level 3?
             # The request implies fallback *to* Level 2. Let's stick to that.
             # If chosen depth (2) failed, and fallback (3) is requested in future, add here.
             pass # No fallback if Level 2 fails

        if min_max_data_max:
            max_labels = min_max_data_max['max_labels']
            if max_labels:
                 # Template answer format is 'China' (singular, quoted)
                 answer_subcat_max = f"{max_labels[0]}"
                 if len(max_labels) > 1:
                      print(f"Info (MSR-FB): Multiple subcategories at Depth {target_depth_max} have the max size ({min_max_data_max['max_val']:.2f}). Answering with '{max_labels[0]}'.")

                 # Q template: In the fill bubble chart, which subcategory in the second level has the largest bubble area?
                 # Modify Q to mention the correct level
                 question1 = f"In the fill bubble chart, which subcategory in the level {target_depth_max} has the largest bubble area?" # No {}
                 # A template: The subcategory with the largest bubble area in the second level is China.
                 # Modify A to mention the correct level
                 answer1 = f"The subcategory with the largest bubble area in the level {target_depth_max} is {{{answer_subcat_max}}}." # {{}} around subcat name
                 qa_list.append({"Q": question1, "A": answer1})
            else:
                 print(f"Info (MSR-FB): No labels found for max size at Depth {target_depth_max}.")
        else:
            print(f"Info (MSR-FB): Could not find max size data at Depth {target_depth_max} (and fallback if applicable). Skipping Max Size QA.")


        # Generate QA for Smallest Bubble Area
        # Randomly choose a target depth from viable depths (independent of max)
        target_depth_min = random.choice(viable_depths)
        min_max_data_min = get_min_max_label_at_depth(df_valid, target_depth_min, value_col, label_col, depth_col)

        # Fallback logic for min size
        if min_max_data_min is None and target_depth_min == 3 and 2 in viable_depths:
             print(f"Info (MSR-FB): Fallback for Min Size QA: Depth {target_depth_min} had no valid data, trying Depth 2.")
             target_depth_min = 2 # Fallback to Level 2
             min_max_data_min = get_min_max_label_at_depth(df_valid, target_depth_min, value_col, label_col, depth_col)
        elif min_max_data_min is None and target_depth_min == 2 and 3 in viable_depths:
             pass # No fallback if Level 2 fails


        if min_max_data_min:
            min_labels = min_max_data_min['min_labels']
            if min_labels:
                 # Template answer format is 'China' (singular, quoted)
                 answer_subcat_min = f"{min_labels[0]}"
                 if len(min_labels) > 1:
                      print(f"Info (MSR-FB): Multiple subcategories at Depth {target_depth_min} have the min size ({min_max_data_min['min_val']:.2f}). Answering with '{min_labels[0]}'.")

                 # Q template: In the fill bubble chart, which subcategory in the second level has the smallest bubble area?
                 # Modify Q to mention the correct level
                 question2 = f"In the fill bubble chart, which subcategory in the level {target_depth_min} has the smallest bubble area?" # No {}
                 # A template: The subcategory with the smallest bubble area in the second level is China.
                 # Modify A to mention the correct level
                 answer2 = f"The subcategory with the smallest bubble area in the level {target_depth_min} is {{{answer_subcat_min}}}." # {{}} around subcat name
                 qa_list.append({"Q": question2, "A": answer2})
            else:
                 print(f"Info (MSR-FB): No labels found for min size at Depth {target_depth_min}.")
        else:
             print(f"Info (MSR-FB): Could not find min size data at Depth {target_depth_min} (and fallback if applicable). Skipping Min Size QA.")


    # --- MSR Q3 & Q4: Count bubbles above/below overall average value (individual rows) ---
    # Use all valid rows with size values
    if not df_valid[value_col].empty:
        average_value = df_valid[value_col].mean()

        # Check if average_value is NaN or Inf before comparison
        if pd.notna(average_value) and np.isfinite(average_value):
            # Count rows where value > average
            count_above_average = df_valid[df_valid[value_col] > average_value].shape[0]

            # Count rows where value < average (strictly less than for clarity match example)
            count_below_average = df_valid[df_valid[value_col] < average_value].shape[0]
            # Note: The example answers sum to 14, but the sample CSV only has 7 data rows.
            # The counts should be based on the actual data rows in the input file.

            # Q template: How many bubbles have values above the average oil storage across all categories?
            question3 = f"How many bubbles have values above the average {value_col} across all categories?" # No {}
            # A template: There are 7 bubbles with values above the average oil storage.
            answer3 = f"There are {{{count_above_average}}} bubbles with values above the average {value_col}." # {} around count ONLY
            qa_list.append({"Q": question3, "A": answer3})

            # Q template: How many bubbles have values below the average oil storage across all categories?
            question4 = f"How many bubbles have values below the average size across all categories?" # No {}
            # A template: There are 7 bubbles with values below the average oil storage.
            answer4 = f"There are {{{count_below_average}}} bubbles with values below the average {value_col}." # {} around count ONLY
            qa_list.append({"Q": question4, "A": answer4})
        else:
             print("Info (MSR-FB): Average value could not be calculated (e.g., all sizes are same or NaN). Skipping count QA.")

    else:
         print("Info (MSR-FB): No valid values found to calculate average for count QA.")

    return qa_list

def fill_qa_va_supplemental_fill_bubble(df: pd.DataFrame, metadata: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Generates supplemental VA QA for fill bubbles.
    Provides a placeholder QA for label overlap as actual detection is not possible from data alone.
    Based strictly on the 补充问题.txt VA example (no {} in Q or A).
    """
    qa_list: List[Dict[str, str]] = []

    # This QA type is based on visualization layout, not just data values.
    # We generate a placeholder based on the example.
    # To make it slightly dynamic, we could pick two random labels if available.
    label_col = metadata.get('label_col')

    if label_col and label_col in df.columns and not df[label_col].empty:
        # Get some sample labels, ensure uniqueness and not NaN
        valid_labels = df[label_col].dropna().unique().tolist()
        # Exclude the Root label if present
        valid_labels = [label for label in valid_labels if label != "Root"]

        if len(valid_labels) >= 2:
            # Pick two random distinct labels
            sample_labels = random.sample(valid_labels, 2)
            label_a = sample_labels[0]
            label_b = sample_labels[1]
        elif len(valid_labels) == 1:
             label_a = valid_labels[0]
             label_b = "Another Label" # Placeholder if only one non-root label exists
        else:
             label_a = "Label A" # Default placeholders if no valid non-root labels
             label_b = "Label B"
    else:
        label_a = "Label A" # Default placeholders if no label column or empty
        label_b = "Label B"


    # Q template: To reduce visual clutter ..., sample the labels and list the top 2 labels recommended for removal.
    question = ("To reduce visual clutter caused by overlapping labels in the bubble chart, "
                "sample the labels and list the top 2 labels recommended for removal.") # No {}
    # A template: The top 2 labels recommended for removal ... are: Label A, Label B.
    answer = f"The top 2 labels recommended for removal to reduce label overlap are: {{{label_a}}}, {{{label_b}}}." # {{}} around each label

    qa_list.append({"Q": question, "A": answer})

    return qa_list


# --- Main Processing Logic ---
def main():
    random.seed(42) # for reproducibility

    # Assuming fill bubble CSVs are in ./fill_bubble/csv and QAs in ./fill_bubble/QA
    csv_dir = './fill_bubble/csv'
    qa_dir = './fill_bubble/QA'

    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory not found at {csv_dir}")
        return

    # Find fill bubble CSV files (adjust pattern if needed, e.g., fill_bubble_*.csv)
    # Use glob to get full paths directly
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv")) # Using *.csv for flexibility
    # Optional: Filter specifically for 'fill_bubble' in filename if needed
    # csv_files = [f for f in csv_files if 'fill_bubble' in os.path.basename(f).lower()]


    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return

    print(f"Found {len(csv_files)} CSV files to process for QA supplementation.")
    processed_count = 0

    for csv_path in csv_files:
        print(f"\nProcessing file: {csv_path}")

        # --- Read data and get assumed column metadata using corrected functions ---
        # metadata from first line is not directly used by supplemental QA functions, but we read it anyway
        # file_metadata = read_fill_bubble_metadata(csv_path) # Not needed for supplemental QA logic
        df_data, inferred_metadata = read_fill_bubble_data_df(csv_path) # Use the corrected reader

        # Check if data reading was successful and contains essential columns inferred
        essential_keys = ['size_col', 'father_col', 'depth_col', 'label_col']
        # Check if all inferred column names exist in the DataFrame
        if df_data is None or df_data.empty or not all(inferred_metadata.get(key) in df_data.columns for key in essential_keys):
             # As a fallback, check if the standard essential column names exist
             standard_essential_cols = ['size', 'father', 'depth', 'label']
             if df_data is None or df_data.empty or not all(col in df_data.columns for col in standard_essential_cols):
                 print(f"Skipping file {os.path.basename(csv_path)} QA generation due to no valid data or missing essential columns ('size', 'father', 'depth', 'label').")
                 continue
             else:
                 # If data is valid but inferred_metadata is somehow incomplete or mismatched,
                 # populate inferred_metadata from df.columns assuming standard names
                 print(f"Warning: Inferred metadata potentially incomplete or mismatched for {os.path.basename(csv_path)}. Using default essential column names.")
                 inferred_metadata = {
                     'size_col': 'size',
                     'father_col': 'father',
                     'depth_col': 'depth',
                     'label_col': 'label'
                 }


        print(f"  Read {len(df_data)} data rows.")
        # print(f"  Inferred columns for QA: Size='{inferred_metadata.get('size_col')}', Father='{inferred_metadata.get('father_col')}', Depth='{inferred_metadata.get('depth_col')}', Label='{inferred_metadata.get('label_col')}'")


        # --- Generate different types of SUPPLEMENTAL QA ---
        # These are the QA types from the first 补充问题.txt

        # SRP: Spatial Relationship (Containment) - MODIFIED to compare depth
        srp_qas = fill_qa_srp_supplemental_fill_bubble(df_data, inferred_metadata)
        if srp_qas:
            write_qa_to_json(csv_path, "SRP", srp_qas, qa_dir)
            print(f"  Added {len(srp_qas)} SRP QA(s).")
        else:
            print(f"  No supplemental SRP QA generated for {os.path.basename(csv_path)}.")

        # MSR: Min/Max/Sum/Range (Level2/3 size, Above/Below Avg Count)
        msr_qas = fill_qa_msr_supplemental_fill_bubble(df_data, inferred_metadata)
        if msr_qas:
            write_qa_to_json(csv_path, "MSR", msr_qas, qa_dir)
            print(f"  Added {len(msr_qas)} MSR QA(s).")
        else:
            print(f"  No supplemental MSR QA generated for {os.path.basename(csv_path)}.")

        # VA: Visual Anomaly (Label Overlap Placeholder)
        va_qas = fill_qa_va_supplemental_fill_bubble(df_data, inferred_metadata)
        if va_qas:
            write_qa_to_json(csv_path, "VA", va_qas, qa_dir)
            print(f"  Added {len(va_qas)} VA QA(s).")
        else:
            print(f"  No supplemental VA QA generated for {os.path.basename(csv_path)}.")


        processed_count += 1

    print(f"\nFill bubble chart SUPPLEMENTAL QA supplementation complete. Processed {processed_count} CSV files.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
        traceback.print_exc()
