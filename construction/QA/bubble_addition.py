# File: supplement_bubble_QA.py
# Description: Supplements existing bubble chart QA JSON files with new MSR, NC, and SRP questions.
# Final refinements for MSR question quotes (using single quotes for groups).
# NC smaller comparison question removed. MSR max size/min size coordinates replaced with max Y size and max X size questions.
# Added SRP QA based on relative position of bubbles with unique Size values.
# MSR threshold for sum difference question is now dynamically selected based on Y-axis data range.
# Fixed pandas.sample random_state issue by using numpy.random.Generator.
# Fixed SyntaxError in write_qa_to_json.
# Added formatting to retain 2 decimal places for numerical values in SRP (both Q and A) and MSR answers.
# MODIFIED: Take absolute value for the difference in the last MSR question.

import traceback
import glob
import pandas as pd
import os
import json
import numpy as np # Import numpy
import re
import random
from typing import List, Dict, Any, Tuple

# --- Utility Functions (Adapted from bubble_QA.py) ---

def parse_label_unit(label_str: str | None) -> Dict[str, str]:
    """Helper to parse 'Label (Unit)' strings."""
    if isinstance(label_str, str):
        match = re.match(r'(.+)\s*\((.+)\)', label_str)
        if match:
            return {'name': match.group(1).strip(), 'unit': match.group(2).strip()}
        return {'name': label_str.strip(), 'unit': ''}
    return {'name': str(label_str).strip() if label_str is not None else '', 'unit': ''}

def read_bubble_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the first two header lines of the bubble chart CSV.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: CSV file not found at {filepath}")
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            header_line1_content = f.readline().strip()
            header_line2_content = f.readline().strip()

        if not header_line1_content or not header_line2_content:
            print(f"Warning: Header lines missing or incomplete in {filepath}")
            return {}

        parts1 = [part.strip() for part in header_line1_content.split(',', 2)]
        metadata: Dict[str, Any] = {
            'topic': parts1[0] if len(parts1) > 0 else None,
            'little_theme': parts1[1] if len(parts1) > 1 else None,
            'pattern': parts1[2] if len(parts1) > 2 else None
        }

        parts2 = [part.strip() for part in header_line2_content.split(',', 2)]
        metadata['x_info'] = parse_label_unit(parts2[0] if len(parts2) > 0 else None)
        metadata['y_info'] = parse_label_unit(parts2[1] if len(parts2) > 1 else None)
        metadata['size_info'] = parse_label_unit(parts2[2] if len(parts2) > 2 else None)
        
        if not metadata.get('x_info', {}).get('name'): metadata['x_info']['name'] = 'X-axis'
        if not metadata.get('y_info', {}).get('name'): metadata['y_info']['name'] = 'Y-axis'
        if not metadata.get('size_info', {}).get('name'): metadata['size_info']['name'] = 'Size'
        
        return metadata
    except Exception as e:
        print(f"Error reading bubble metadata from {filepath}: {e}")
        return {}

def read_bubble_data_df(filepath: str, metadata: Dict[str, Any]) -> pd.DataFrame | None:
    """
    Reads the data part of the bubble chart CSV into a DataFrame.
    """
    try:
        df = pd.read_csv(filepath, header=None, skiprows=2, encoding='utf-8')
        if df.shape[1] < 3:
            print(f"Warning: Not enough data columns ({df.shape[1]}) in {filepath}.")
            return None

        x_col_name = metadata.get('x_info', {}).get('name', 'X')
        y_col_name = metadata.get('y_info', {}).get('name', 'Y')
        size_col_name = metadata.get('size_info', {}).get('name', 'Size')
        
        new_column_names = [x_col_name, y_col_name, size_col_name] + [f'col{i}' for i in range(df.shape[1] - 3)]
        df.columns = new_column_names

        cols_to_convert = [x_col_name, y_col_name, size_col_name]
        for col in cols_to_convert:
            if col in df.columns:
                # Use errors='coerce' to turn non-numeric values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Column name '{col}' derived from metadata not found in DataFrame columns {list(df.columns)}. Skipping numeric conversion for this column.")
        
        # Keep rows where X, Y, and Size are not NaN, as these are needed for most QA types
        cols_to_check_na = [col for col in [x_col_name, y_col_name, size_col_name] if col in df.columns]
        if cols_to_check_na:
            df = df.dropna(subset=cols_to_check_na).reset_index(drop=True) 

        if df.empty:
            print(f"Warning: No valid data rows found in {filepath} after dropping NaNs.")
            return None
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: File {filepath} is empty after skipping headers.")
        return None
    except Exception as e:
        print(f"Error reading bubble data from {filepath}: {e}")
        return None

def write_qa_to_json(csv_path: str, qa_type: str, qa_items: List[Dict[str, str]], qa_dir: str = './bubble/QA'):
    """
    Writes QA items to the JSON file corresponding to the CSV path.
    Assumes qa_items are dictionaries with both 'Q' and 'A' keys for types generated here (MSR, NC, SRP).
    """
    os.makedirs(qa_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    json_path = os.path.join(qa_dir, base_name + '.json')

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
        # For types generated by this script (MSR, NC, SRP), we expect Q and A
        if isinstance(data_to_save[qa_type], list):
             existing_qa_pairs = {(item.get('Q'), item.get('A')) for item in data_to_save[qa_type]
                                 if isinstance(item, dict) and 'Q' in item and 'A' in item is not None} # A can be empty string
        
        new_items_to_add = []
        for item in qa_items:
            # Require both Q and A for any item being added
            if isinstance(item, dict) and 'Q' in item and 'A' in item is not None: # A can be empty string
                item_q = item.get('Q')
                item_a = item.get('A')

                # Check if a QA pair with the same Q and A already exists
                if (item_q, item_a) not in existing_qa_pairs:
                    new_items_to_add.append({"Q": item_q, "A": item_a})
                    existing_qa_pairs.add((item_q, item_a))
            else:
                # The warning message is useful for debugging potential issues
                print(f"Warning: Skipping invalid QA item format (missing Q or A is None) for type {qa_type}: {item}")
        
        if isinstance(data_to_save[qa_type], list):
            # Corrected the syntax error here
            data_to_save[qa_type].extend(new_items_to_add)
        else: 
            # If it wasn't a list, replace it with the new list (shouldn't happen with template_data)
            data_to_save[qa_type] = new_items_to_add 
    else:
        print(f"Error: QA type '{qa_type}' not in template. Items not added.")
        return

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing QA to {json_path} for type {qa_type}: {e}")


# --- Supplemental QA Generation Functions ---

def fill_qa_srp_supplemental(df: pd.DataFrame, metadata: Dict[str, Any], rng: np.random.Generator) -> List[Dict[str, str]]:
    """
    Generates supplemental SRP QA by comparing the relative position of two randomly selected bubbles
    whose Size value is unique within the dataset.
    Returns an empty list if fewer than 2 bubbles with unique Size values are found.
    Questions and answers are in English.
    Takes a NumPy random Generator object for reproducible sampling.
    Formats Size values in BOTH the question and answer to 2 decimal places.
    """
    qa_list: List[Dict[str, str]] = []

    size_info = metadata.get('size_info', {})
    size_col_name = size_info.get('name', 'Size')
    x_col_name = metadata.get('x_info', {}).get('name', 'X-axis')
    y_col_name = metadata.get('y_info', {}).get('name', 'Y-axis')

    # Ensure required columns exist and are numeric (df is already filtered for NaNs in these cols)
    required_cols = [x_col_name, y_col_name, size_col_name]
    if not all(col in df.columns for col in required_cols):
         print(f"Warning (SRP): DataFrame does not contain required columns: {required_cols}")
         return qa_list
         
    if df.empty:
        print("Info (SRP): No valid bubbles (with non-NaN X, Y, Size) for SRP QA.")
        return qa_list

    try:
        # Find unique size values that are not NaN
        # Use value_counts on the column, filtering for non-NaN values
        size_counts = df[size_col_name].dropna().value_counts()
        unique_sizes = size_counts[size_counts == 1].index.tolist()

        if len(unique_sizes) < 2:
            # Not enough bubbles with unique size values to form a pair
            print(f"Info (SRP): Found only {len(unique_sizes)} bubble(s) with unique {size_col_name} values. Need at least 2 for SRP QA.")
            # As requested, return empty list if pair cannot be found, no Q generated.
            return qa_list

        # Randomly select two distinct unique size values using the provided generator
        # rng.choice requires the input list/array to be non-empty, which is guaranteed by the check above
        try:
             size_val1, size_val2 = rng.choice(unique_sizes, 2, replace=False)
        except ValueError:
             # This might happen if unique_sizes has exactly 1 element, but the check above should prevent it.
             # Add defensive check just in case.
             print(f"Error (SRP): Could not select 2 unique sizes from list of size {len(unique_sizes)}. Skipping QA.")
             return qa_list


        # Find the bubbles corresponding to these unique sizes
        # Ensure we get the row from the original filtered df
        bubble1 = df[df[size_col_name] == size_val1].iloc[0]
        bubble2 = df[df[size_col_name] == size_val2].iloc[0]

        # Get coordinates (guaranteed non-NaN by df filter)
        x1, y1 = bubble1[x_col_name], bubble1[y_col_name]
        x2, y2 = bubble2[x_col_name], bubble2[y_col_name]

        # Determine relative vertical position
        vertical_pos = ""
        if y1 > y2:
            vertical_pos = "{above}"
        elif y1 < y2:
            vertical_pos = "{below}"
        else:
            vertical_pos = "at {the same vertical level}"

        # Determine relative horizontal position
        horizontal_pos = ""
        if x1 > x2:
            horizontal_pos = "to the {right}"
        elif x1 < x2:
            horizontal_pos = "to the {left}"
        else:
            horizontal_pos = "at {the same horizontal level}"

        # Combine positions
        relative_position = ""
        if vertical_pos == "at {the same vertical level}" and horizontal_pos == "at {the same horizontal level}":
            # This case should not happen if X, Y, Size are all unique, but handle defensively
             relative_position = "at {the same location}"
        elif vertical_pos == "at {the same vertical level}":
            relative_position = horizontal_pos
        elif horizontal_pos == "at {the same horizontal level}":
             relative_position = vertical_pos
        else:
            relative_position = f"{vertical_pos} and {horizontal_pos}"

        # Construct Question and Answer
        # Format size values in the question to 2 decimal places
        question = (
            f"What is the relative position of the bubble with a {size_col_name} of {size_val1:.2f} "
            f"relative to the bubble with a {size_col_name} of {size_val2:.2f} "
            f"in terms of vertical (above/below) and horizontal (left/right) directions?"
        )

        # Bracing only the final combined relative position phrase in the answer
        # Format size values in the answer to 2 decimal places
        answer = (
            f"The bubble with a {size_col_name} of {size_val1:.2f} "
            f"is {relative_position} of the bubble with a {size_col_name} of {size_val2:.2f}."
        )

        # If for some reason relative_position calculation failed or is empty, set Answer to empty string as requested
        if not relative_position:
             answer = "" # This should ideally not happen with the current logic, but good failsafe

        qa_list.append({"Q": question, "A": answer})

    except Exception as e:
         print(f"Error generating SRP QA: {e}")
         traceback.print_exc()

    return qa_list


def fill_qa_msr_supplemental(df: pd.DataFrame, metadata: Dict[str, Any], rng: np.random.Generator) -> List[Dict[str, str]]:
    """
    Generates supplemental MSR QA:
    1. Size of bubble with max Y value.
    2. Size of bubble with max X value.
    3. Difference in sum of Size for groups based on a dynamic Y threshold.
    Questions and answers are in English.
    Takes a NumPy random Generator object for reproducible sampling.
    Formats numerical values in the answer to 2 decimal places.
    Formats the threshold in the question to 0 decimal places (as it's conceptually a splitting point).
    MODIFIED: Take absolute value for the difference calculation in QA 3.
    """
    qa_list: List[Dict[str, str]] = []

    size_info = metadata.get('size_info', {})
    size_col_name = size_info.get('name', 'Size')
    x_info = metadata.get('x_info', {})
    x_col_name = x_info.get('name', 'X-axis')
    y_info = metadata.get('y_info', {})
    y_col_name = y_info.get('name', 'Y-axis')

    # Ensure required columns exist and are numeric (df is already filtered for NaNs in these cols)
    required_cols = [x_col_name, y_col_name, size_col_name]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning (MSR): DataFrame does not contain required columns: {required_cols}")
        return qa_list
    
    if df.empty:
        print("Info (MSR): No valid bubbles (with non-NaN X, Y, Size) for MSR QA.")
        return qa_list

    try:
        # --- QA 1: Size of bubble with max Y value ---
        # df is already filtered for NaNs in required_cols
        if df[y_col_name].notna().any(): # Double check just in case df was empty before this check
            max_y_val = df[y_col_name].max()
            max_y_bubbles = df[df[y_col_name] == max_y_val] # These are already guaranteed to have non-NaN Size

            if not max_y_bubbles.empty:
                # Pick one bubble to get the size value from using the provided generator
                sample_bubble_max_y = max_y_bubbles.sample(1, random_state=rng).iloc[0]
                size_at_max_y = sample_bubble_max_y[size_col_name]
                count_max_y = len(max_y_bubbles)

                question1 = f"What is the {size_col_name} value of the bubble with the largest {y_col_name} value?"
                
                # Format size value in the answer to 2 decimal places
                if count_max_y == 1:
                    answer1 = f"The {size_col_name} value of the bubble with the largest {y_col_name} is {{{size_at_max_y:.2f}}}."
                else:
                     # If multiple bubbles have the same max Y, mention their count and report the size of one.
                    answer1 = (f"There are {count_max_y} bubbles with the largest {y_col_name}. "
                               f"One of them has a {size_col_name} value of {{{size_at_max_y:.2f}}}."
                               ) # Reporting one size value
                qa_list.append({"Q": question1, "A": answer1})
            else:
                print(f"Warning (MSR): Max {y_col_name} value found, but no valid bubbles with that value.")

        # --- QA 2: Size of bubble with max X value ---
        # df is already filtered for NaNs in required_cols
        if df[x_col_name].notna().any(): # Double check just in case df was empty before this check
            max_x_val = df[x_col_name].max()
            max_x_bubbles = df[df[x_col_name] == max_x_val] # These are already guaranteed to have non-NaN Size

            if not max_x_bubbles.empty:
                # Pick one bubble to get the size value from using the provided generator
                sample_bubble_max_x = max_x_bubbles.sample(1, random_state=rng).iloc[0]
                size_at_max_x = sample_bubble_max_x[size_col_name]
                count_max_x = len(max_x_bubbles)

                question2 = f"What is the {size_col_name} value of the bubble with the largest {x_col_name} value?"

                # Format size value in the answer to 2 decimal places
                if count_max_x == 1:
                    answer2 = f"The {size_col_name} value of the bubble with the largest {x_col_name} is {{{size_at_max_x:.2f}}}."
                else:
                     # If multiple bubbles have the same max X, mention their count and report the size of one.
                     # Corrected variable name from count_x to count_max_x
                    answer2 = (f"There are {count_max_x} bubbles with the largest {x_col_name}. "
                               f"One of them has a {size_col_name} value of {{{size_at_max_x:.2f}}}."
                               ) # Reporting one size value
                qa_list.append({"Q": question2, "A": answer2})
            else:
                 print(f"Warning (MSR): Max {x_col_name} value found, but no valid bubbles with that value.")


        # --- QA 3: Dynamic Threshold Sum QA ---
        # This part uses the df which is already filtered for NaNs in required_cols including Y and Size.
        if not df.empty and df[y_col_name].notna().any() and df[size_col_name].notna().any():
            try:
                min_y = df[y_col_name].min()
                max_y = df[y_col_name].max()

                # Check if there's a sufficient range to split
                # Use a small tolerance for float comparison and check for valid numbers
                if pd.isna(min_y) or pd.isna(max_y) or (max_y - min_y) < 1e-6: # Increased tolerance slightly
                     print(f"Info (MSR): Y-axis data range is too small or NaN ({min_y} to {max_y}) for threshold split in {os.path.basename(df.name) if hasattr(df, 'name') else 'a file'}. Skipping threshold QA.")
                else:
                    # Choose a random threshold within a range slightly excluding the min/max
                    # This increases the chance that both groups will be non-empty
                    # Define a buffer percentage, e.g., 10%
                    buffer_percent = 0.10
                    range_y = max_y - min_y
                    lower_bound = min_y + range_y * buffer_percent
                    upper_bound = max_y - range_y * buffer_percent

                    # Ensure bounds haven't crossed if range was very small or data is clustered
                    # If buffer made bounds cross/meet, just pick anywhere between min and max
                    if lower_bound >= upper_bound:
                         y_threshold_float = random.uniform(min_y, max_y)
                    else:
                         # Normal case: pick a threshold within the buffered range
                         y_threshold_float = random.uniform(lower_bound, upper_bound)

                    # Perform the split using the random float threshold
                    high_value_group_df = df[df[y_col_name] > y_threshold_float]
                    low_value_group_df = df[df[y_col_name] <= y_threshold_float]

                    # Check if both resulting groups are non-empty
                    if high_value_group_df.empty or low_value_group_df.empty:
                         # If the random threshold didn't work well, try up to N times
                         max_retries = 5
                         retries = 0
                         split_successful = False
                         while retries < max_retries:
                              print(f"Info (MSR): Random Y threshold ({y_threshold_float:.1f}) resulted in one or both groups being empty. Retrying ({retries+1}/{max_retries})...")
                              # Recalculate a new random threshold
                              if max_y - min_y < 1e-6: break # Should not happen due to earlier check, but safe
                              
                              # Try picking anywhere in the full range if buffering failed
                              y_threshold_float = random.uniform(min_y, max_y)
                              
                              high_value_group_df = df[df[y_col_name] > y_threshold_float]
                              low_value_group_df = df[df[y_col_name] <= y_threshold_float]

                              if not high_value_group_df.empty and not low_value_group_df.empty:
                                   split_successful = True
                                   break
                              retries += 1
                         
                         if not split_successful:
                              print(f"Info (MSR): Failed to find a suitable random Y threshold after {max_retries} retries in {os.path.basename(df.name) if hasattr(df, 'name') else 'a file'}. Skipping threshold QA.")
                              # Skip generating this QA if split wasn't successful
                              return qa_list # Return early for this specific QA type

                    # If we reach here, the split was successful or we are using the initial split
                    # Calculate sums if groups are valid (guaranteed by the checks above)
                    sum_high = high_value_group_df[size_col_name].sum()
                    sum_low = low_value_group_df[size_col_name].sum()
                    
                    # Calculate the difference and take the absolute value
                    absolute_difference = abs(sum_high - sum_low)
                    
                    # Format the threshold for the question (integer as per example)
                    # Using round for display, but comparison used the float value
                    y_threshold_formatted = round(y_threshold_float)

                    # Using single quotes ' for group names in the Python string
                    # Format threshold in question to 0 decimal places as it's a conceptual split point
                    question3 = (f"Classify bubbles with a {y_col_name}-coordinate greater than {y_threshold_formatted:.0f} as 'high value group' "
                                f"and the rest as 'low value group'. What is the difference between the sum of {size_col_name} "
                                f"for the high value group and the low value group?")
                                
                    # Format the absolute difference value in the answer to 2 decimal places
                    # The phrasing "the difference is" implies the magnitude, so using absolute value is fine here.
                    answer3 = (f"The difference between the total {size_col_name} of the high-value and low-value groups "
                            f"is {{{absolute_difference:.2f}}}.")
                    qa_list.append({"Q": question3, "A": answer3})

            except Exception as e:
                 print(f"Error during MSR threshold sum QA generation: {e}")
                 traceback.print_exc()

    except Exception as e:
         print(f"Error during MSR QA generation: {e}")
         traceback.print_exc()


    return qa_list

def fill_qa_nc_supplemental(df: pd.DataFrame, metadata: Dict[str, Any], rng: np.random.Generator) -> List[Dict[str, str]]:
    """
    Generates supplemental NC QA by comparing the Size value of two randomly selected bubbles
    based on their X and Y coordinates (Only generates "larger" comparison question).
    Questions and answers are in English.
    Takes a NumPy random Generator object for reproducible sampling.
    (NC answers already have formatting, not changed here based on request for SRP/MSR answer values)
    """
    qa_list: List[Dict[str, str]] = []

    size_info = metadata.get('size_info', {})
    size_col_name = size_info.get('name', 'Size')
    size_unit = size_info.get('unit', '')
    x_info = metadata.get('x_info', {})
    x_col_name = x_info.get('name', 'X-axis')
    x_unit = x_info.get('unit', '')
    y_info = metadata.get('y_info', {})
    y_col_name = y_info.get('name', 'Y-axis')
    y_unit = y_info.get('unit', '')

    # Check for required columns and enough data points (df is already filtered for NaNs in these cols)
    required_cols = [x_col_name, y_col_name, size_col_name]
    if not all(col in df.columns for col in required_cols):
         print(f"Warning (NC): DataFrame does not contain required columns: {required_cols}")
         return qa_list

    # Need at least two distinct valid bubbles for comparison (df is already filtered for NaNs in these cols)
    valid_indices = df.index.tolist()
    if len(valid_indices) < 2:
        print("Info (NC): Not enough valid bubbles (need at least 2) after dropping NaNs for NC QA.")
        return qa_list

    try:
        # Randomly select two distinct bubble indices from valid indices using the provided generator
        bubble1_idx, bubble2_idx = rng.choice(valid_indices, 2, replace=False)
        bubble1 = df.loc[bubble1_idx]
        bubble2 = df.loc[bubble2_idx]

        # Get values - already checked for NaN via valid_indices
        x1, y1, size1 = bubble1[x_col_name], bubble1[y_col_name], bubble1[size_col_name]
        x2, y2, size2 = bubble2[x_col_name], bubble2[y_col_name], bubble2[size_col_name]

        # Format values for the question and answer (using .1f for consistency with previous NC)
        x1_str = f"{x1:.1f}"
        y1_str = f"{y1:.1f}"
        x2_str = f"{x2:.1f}"
        y2_str = f"{y2:.1f}"

        x_unit_str = f" {x_unit}" if x_unit else ""
        y_unit_str = f" {y_unit}" if y_unit else ""

        # Construct the "larger" question
        question_larger = (
            f"Which has a larger {size_col_name} value, "
            f"the bubble with a {x_col_name} of {x1_str}{x_unit_str} and a {y_col_name} of {y1_str}{y_unit_str}, "
            f"or the bubble with a {x_col_name} of {x2_str}{x_unit_str} and a {y_col_name} of {y2_str}{y_unit_str}?"
        )

        # Construct the answer for the "larger" question
        answer_larger = ""
        # NC answers brace the coordinates, not the size value itself
        if size1 > size2:
            answer_larger = (
                f"The bubble with a {x_col_name} of {{{x1_str}}} and a {y_col_name} of {{{y1_str}}} "
                f"is larger."
            )
        elif size1 < size2:
            answer_larger = (
                f"The bubble with a {x_col_name} of {{{x2_str}}} and a {y_col_name} of {{{y2_str}}} "
                f"is larger."
            )
        else: # size1 == size2
             # Example didn't show bracing for same value, follow that.
             answer_larger = (
                f"The bubble with a {x_col_name} of {x1_str} and a {y_col_name} of {y1_str} "
                f"and the bubble with a {x_col_name} of {x2_str} and a {y_col_name} of {y2_str} "
                f"have the same {size_col_name} value."
            )

        qa_list.append({"Q": question_larger, "A": answer_larger})

        # Removed the generation of the "smaller" question and answer as requested.
        # if size1 != size2:
        #      question_smaller = (...)
        #      answer_smaller = (...)
        #      qa_list.append({"Q": question_smaller, "A": answer_smaller})

    except Exception as e:
         print(f"Error generating NC QA: {e}")
         traceback.print_exc()


    return qa_list


# --- Main Processing Logic ---
def main():
    # Use a consistent seed for reproducibility across runs
    seed = 42
    random.seed(seed) # Seed standard random module
    rng = np.random.default_rng(seed) # Create and seed NumPy generator

    csv_dir = './csv'
    qa_dir = './QA'

    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory not found at {csv_dir}")
        return

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return

    print(f"Found {len(csv_files)} CSV files to process for QA supplementation.")
    processed_count = 0

    for csv_path in csv_files:
        print(f"\nProcessing file: {csv_path}")
        
        metadata = read_bubble_metadata(csv_path)
        if not metadata or not metadata.get('x_info') or not metadata.get('y_info') or not metadata.get('size_info'):
            print(f"Skipping {csv_path} due to missing or incomplete metadata or column names.")
            continue
            
        df_data = read_bubble_data_df(csv_path, metadata)
        if df_data is None or df_data.empty:
            print(f"Skipping {csv_path} due to no valid data after cleaning.")
            continue
            
        # Add file path to dataframe object for logging in helper functions
        df_data.name = os.path.basename(csv_path)

        # --- Add SRP QA Generation ---
        # Pass the NumPy generator to SRP
        srp_qas = fill_qa_srp_supplemental(df_data, metadata, rng)
        if srp_qas:
            write_qa_to_json(csv_path, "SRP", srp_qas, qa_dir)
            print(f"  Added {len(srp_qas)} SRP QA(s).")
        else:
            print(f"  No supplemental SRP QA generated for {os.path.basename(csv_path)} (e.g., not enough bubbles with unique size).")
        # --- End Add SRP QA Generation ---

        # --- Add MSR QA Generation ---
        # msr_y_threshold is now determined dynamically inside the function
        # Pass the NumPy generator to MSR
        msr_qas = fill_qa_msr_supplemental(df_data, metadata, rng)
        if msr_qas:
            write_qa_to_json(csv_path, "MSR", msr_qas, qa_dir)
            print(f"  Added {len(msr_qas)} MSR QA(s).")
        else:
            print(f"  No supplemental MSR QA generated for {os.path.basename(csv_path)}.")
        # --- End Add MSR QA Generation ---
            
        # --- Add NC QA Generation ---
        # Pass the NumPy generator to NC
        nc_qas = fill_qa_nc_supplemental(df_data, metadata, rng)
        if nc_qas:
            write_qa_to_json(csv_path, "NC", nc_qas, qa_dir)
            print(f"  Added {len(nc_qas)} NC QA(s).")
        else:
            print(f"  No supplemental NC QA generated for {os.path.basename(csv_path)}.")
        # --- End Add NC QA Generation ---

        processed_count += 1

    print(f"\nBubble chart QA supplementation complete. Processed {processed_count} CSV files.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
        traceback.print_exc()
