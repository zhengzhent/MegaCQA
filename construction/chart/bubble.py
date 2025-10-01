# File: bubble_vis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re
import glob
import random
import traceback
from typing import Dict, Any, List, Tuple
import textwrap # Added import for text wrapping

import adjustText as adjust_text

# --- Configuration and Formatting ---

def apply_chart_formatting():
    """Adds 'Times New Roman' to the font list and applies the specified formatting."""
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.labelcolor'] = '#000000'
    plt.rcParams['ytick.labelcolor'] = '#000000'
    plt.rcParams['legend.labelcolor'] = '#000000'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['axes.edgecolor'] = '#000000'
    plt.rcParams['axes.titlesize'] = 16 # Adjusted for potentially longer titles
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.0
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


# --- Data Reading and Parsing ---

def parse_label_unit(label_str: str) -> Tuple[str, str]:
    """
    Parses a string like 'Label (Unit)' or 'Label' into label and unit.
    """
    label_str = label_str.strip()
    match = re.match(r'^(.*?)\s*\((.*?)\)$', label_str)
    if match:
        label_part = match.group(1).strip()
        unit = match.group(2).strip()
        return label_part, unit
    else:
        return label_str, ""


# --- parse_bubble_csv (Retained modifications for two header lines) ---
def parse_bubble_csv(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads the bubble chart CSV file (new format with two header lines) and parses its metadata.
    Expected header format:
    Line 1: Main theme, little theme, Size Distribution
    Line 2: {x_col_name} ({x_unit}), {y_col_name} ({y_unit}), {size_meaning} ({size_unit})
    Expected data starts from the third line WITHOUT a header row.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A tuple containing:
            - pd.DataFrame: The data from the CSV with columns correctly named.
            - Dict[str, Any]: A dictionary containing the parsed metadata.
    Raises:
         FileNotFoundError, ValueError, Exception for errors.
    """
    metadata: Dict[str, Any] = {}
    df = pd.DataFrame()

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")

        # Read the first two header lines separately
        with open(filepath, 'r', encoding='utf-8') as f:
            header_line1_content = f.readline().strip()
            header_line2_content = f.readline().strip()

        # Parse the first header line (Topic, Little Theme, Size Distribution)
        parts1 = [part.strip() for part in header_line1_content.split(',')]
        expected_parts1 = 3
        if len(parts1) != expected_parts1:
            raise ValueError(
                f"Incorrect format on header line 1 in {filepath}. Expected {expected_parts1} parts separated by commas, "
                f"but found {len(parts1)}. Content: '{header_line1_content}'"
            )
        metadata['Topic'] = parts1[0]
        metadata['Little_Theme'] = parts1[1]
        metadata['Size Distribution'] = parts1[2]

        # Parse the second header line (X, Y, Size descriptions with units)
        parts2 = [part.strip() for part in header_line2_content.split(',')]
        expected_parts2 = 3
        if len(parts2) != expected_parts2:
             raise ValueError(
                 f"Incorrect format on header line 2 in {filepath}. Expected {expected_parts2} parts separated by commas, "
                 f"but found {len(parts2)}. Content: '{header_line2_content}'"
             )

        # Parse X column name and unit from the first part of line 2
        x_col_name_in_df, x_unit_display = parse_label_unit(parts2[0])
        metadata['X_col'] = x_col_name_in_df
        metadata['X_unit_display'] = x_unit_display
        metadata['X_label_with_unit'] = parts2[0] # Store the original string

        # Parse Y column name and unit from the second part of line 2
        y_col_name_in_df, y_unit_display = parse_label_unit(parts2[1])
        metadata['Y_col'] = y_col_name_in_df
        metadata['Y_unit_display'] = y_unit_display
        metadata['Y_label_with_unit'] = parts2[1] # Store the original string

        # Parse Size meaning and unit from the third part of line 2
        size_meaning, size_unit_display = parse_label_unit(parts2[2])
        metadata['Size_col'] = size_meaning # Use the descriptive part as the column name for size data
        metadata['Size Meaning'] = size_meaning # This is the descriptive meaning
        metadata['Size_unit_display'] = size_unit_display # This is the unit for size
        metadata['Size_label_with_unit'] = parts2[2] # Store the original string


        # Read the actual data, starting from the third line (skiprows=2)
        df = pd.read_csv(filepath, skiprows=2, header=None, encoding='utf-8', index_col=False)

        expected_data_cols = 3
        if df.shape[1] < expected_data_cols:
             raise ValueError(f"Data section in {filepath} has fewer than {expected_data_cols} columns.")
        elif df.shape[1] > expected_data_cols:
             print(f"Warning: Data section in {filepath} has more than {expected_data_cols} columns. Using first {expected_data_cols}.")
             df = df.iloc[:, :expected_data_cols]

        # Assign the correct column names derived from the metadata
        # The DataFrame columns should correspond to X, Y, and the conceptual Size metric
        df.columns = [metadata['X_col'], metadata['Y_col'], metadata['Size_col']]

        # Ensure data columns are numeric
        cols_to_convert = [metadata['X_col'], metadata['Y_col'], metadata['Size_col']]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    print(f"Warning: Column '{col}' in {filepath} contains non-numeric values after conversion. Invalid values treated as NaN.")
            else:
                 # This case should ideally not happen if df.columns assignment above was correct
                 # But including for robustness
                 raise ValueError(f"Logic Error: Column '{col}' not found after assignment for {filepath}.")


    except FileNotFoundError:
        raise
    except ValueError as ve:
        print(f"Error parsing header lines in {filepath}: {ve}") # More specific error message
        raise ve
    except Exception as e:
        raise Exception(f"Error reading or parsing CSV file {filepath}: {e}") from e

    return df, metadata
# --- END parse_bubble_csv ---


# Define a list of suitable single colors for drawing bubbles
single_draw_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- Format number with units ---
def format_number_with_unit(number: float) -> str:
    """
    Formats a number with appropriate unit suffixes (k, M, B, T)
    for better readability in labels.
    """
    if pd.isna(number):
        return ""
    abs_number = abs(number)
    if abs_number >= 1e12: return f"{number / 1e12:.1f}T"
    elif abs_number >= 1e9: return f"{number / 1e9:.1f}B"
    elif abs_number >= 1e6: return f"{number / 1e6:.1f}M"
    elif abs_number >= 1e3: return f"{number / 1e3:.1f}k"
    else:
        # Check if the number is essentially an integer
        if abs(number - round(number)) < 1e-9: return f"{int(round(number))}"
        # Check if the number has more than 2 decimal places and round if needed
        # This is a simple check, more complex logic might be needed for perfect handling
        elif round(number, 2) != number and abs(number - round(number, 2)) < 1e-9:
             # Use rstrip to remove trailing zeros and decimal point if unnecessary after rounding
             return f"{round(number, 2)}".rstrip('0').rstrip('.')
        else: return f"{number:.2f}".rstrip('0').rstrip('.')

# --- END Format number with units ---

# --- plot_bubble_chart (Reverted legend data selection) ---
def plot_bubble_chart(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[plt.Figure, str]:
    """
    Creates a bubble chart visualization from the DataFrame and metadata.
    Uses Little_Theme as the chart title.
    Applies line wrapping to the legend title.
    Adjusts whitespace to the right of the legend.
    """
    x_col = metadata['X_col']
    y_col = metadata['Y_col']
    size_col = metadata['Size_col']

    # Use Little_Theme directly as the chart title
    chart_title = metadata.get('Little_Theme', 'Bubble Chart Analysis')

    x_unit_str = str(metadata.get('X_unit_display', ''))
    y_unit_str = str(metadata.get('Y_unit_display', ''))

    apply_chart_formatting()
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # --- Scale bubble sizes for display ---
    min_display_radius = 10
    max_display_radius = 20
    min_display_size_area = np.pi * min_display_radius**2
    max_display_size_area = np.pi * max_display_radius**2

    valid_sizes = df[size_col].dropna().astype(float)
    q_min = valid_sizes.min() if len(valid_sizes) > 0 and pd.notna(valid_sizes.min()) else 0.0
    q_max = valid_sizes.max() if len(valid_sizes) > 0 and pd.notna(valid_sizes.max()) else 0.0

    display_sizes_area = np.full_like(df[size_col], (min_display_size_area + max_display_size_area) / 2.0, dtype=float)

    if len(valid_sizes) > 0 and q_min != q_max:
        # Use 5th and 95th percentile for scaling if enough data points (more robust to outliers)
        # Otherwise, use actual min/max
        if len(valid_sizes) >= 20:
             interp_q_min = valid_sizes.quantile(0.05)
             interp_q_max = valid_sizes.quantile(0.95)
             # Fallback to min/max if quantiles are NaN or identical
             if pd.isna(interp_q_min) or pd.isna(interp_q_max) or interp_q_min == interp_q_max:
                 interp_q_min, interp_q_max = q_min, q_max
        else:
             interp_q_min, interp_q_max = q_min, q_max


        if interp_q_min != interp_q_max:
            size_values_for_interp = df[size_col].fillna(valid_sizes.mean() if len(valid_sizes) > 0 else 0.0).astype(float)
            interp_range = (float(interp_q_min), float(interp_q_max))
            current_display_sizes_area = np.interp(size_values_for_interp,
                                       interp_range,
                                       (min_display_size_area, max_display_size_area))
            display_sizes_area = np.clip(current_display_sizes_area, min_display_size_area, max_display_size_area)
        else:
             print(f"Warning: Scaled size data range for column '{size_col}' is zero. Using uniform bubble size.")
    elif len(valid_sizes) > 0:
         print(f"Warning: All valid values in size column '{size_col}' are identical. Using uniform bubble size.")
    else:
        print(f"Warning: No valid numeric values in size column '{size_col}'. Using uniform bubble size.")

    plot_df = df.dropna(subset=[x_col, y_col, size_col]).copy() # Use .copy() to avoid SettingWithCopyWarning
    display_sizes_area_plot = pd.Series([], dtype=float) # Ensure dtype for empty series
    if not plot_df.empty:
         # Need to re-index display_sizes_area to match plot_df's index
         display_sizes_area_plot = pd.Series(display_sizes_area, index=df.index).loc[plot_df.index]
         # Ensure display_sizes_area_plot is aligned with plot_df after dropna
         display_sizes_area_plot = display_sizes_area_plot.reindex(plot_df.index)


    selected_color = random.choice(single_draw_colors) if single_draw_colors else '#1f77b4'
    fixed_alpha = 0.7

    if not plot_df.empty:
        plot_df_sorted = plot_df.sort_values(by=size_col, ascending=False).copy()
        display_sizes_area_plot_sorted = display_sizes_area_plot.loc[plot_df_sorted.index]

        ax.scatter(plot_df_sorted[x_col], plot_df_sorted[y_col],
                             s=display_sizes_area_plot_sorted,
                             alpha=fixed_alpha,
                             color=selected_color,
                             edgecolors='#000000',
                             linewidth=0.5)
        texts = []
        # Add annotations only if there are data points
        for index, row in plot_df_sorted.iterrows():
            x_val, y_val, size_val = row[x_col], row[y_col], row[size_col]
            if pd.notna(x_val) and pd.notna(y_val) and pd.notna(size_val):
                 annotation_text = format_number_with_unit(size_val)
                 # Use a slightly larger font for annotations if needed, adjust based on s value?
                 texts.append(ax.text(x_val, y_val, annotation_text, ha='center', va='center', color='#000000', fontsize=8, weight='bold'))
        if texts:
            # Only attempt adjust_text if there are text objects
            adjust_text.adjust_text(texts, force_points=(0.0, 0.0), force_text=(0.2, 0.5),
                                    expand_points=(1.0, 1.0), expand_text=(1.0, 1.0),
                                    arrowprops=None, only_move={'points': 'xy', 'text': 'xy'},
                                    lim=100, precision=0.01, avoid_self=False, ensure_inside_axes=True)

        x_min_data, x_max_data = plot_df_sorted[x_col].min(), plot_df_sorted[x_col].max()
        y_min_data, y_max_data = plot_df_sorted[y_col].min(), plot_df_sorted[y_col].max()
        padding_percentage = 0.15
        x_data_range = x_max_data - x_min_data if pd.notna(x_max_data) and pd.notna(x_min_data) else 0
        y_data_range = y_max_data - y_min_data if pd.notna(y_max_data) and pd.notna(y_min_data) else 0
        min_data_padding = 1.0 # A small fixed padding for very small ranges
        x_padding_data = max(x_data_range * padding_percentage, min_data_padding if abs(x_data_range) < 1e-6 else 0.1)
        y_padding_data = max(y_data_range * padding_percentage, min_data_padding if abs(y_data_range) < 1e-6 else 0.1)

        # Ensure padding doesn't make limits NaN if data is NaN
        if pd.notna(x_min_data) and pd.notna(x_max_data):
             ax.set_xlim(x_min_data - x_padding_data, x_max_data + x_padding_data)
        if pd.notna(y_min_data) and pd.notna(y_max_data):
             ax.set_ylim(y_min_data - y_padding_data, y_max_data + y_padding_data)

    else:
        # Set default limits if no data points
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        print("Warning: No valid data points to plot after dropping NaNs. Plot will be empty or show default limits.")

    # Handle offset text for axes labels
    x_offset_text, y_offset_text = "", ""
    if isinstance(ax.xaxis.get_major_formatter(), mticker.ScalarFormatter):
        offset_artist = ax.xaxis.get_offset_text()
        # Check if offset_artist is valid and contains non-whitespace characters other than the bullet point
        if offset_artist and offset_artist.get_text().strip() and offset_artist.get_text().strip() != '\x08':
             x_offset_text = offset_artist.get_text().strip()
             offset_artist.set_visible(False) # Hide the default offset text
    if isinstance(ax.yaxis.get_major_formatter(), mticker.ScalarFormatter):
         offset_artist = ax.yaxis.get_offset_text()
         if offset_artist and offset_artist.get_text().strip() and offset_artist.get_text().strip() != '\x08':
              y_offset_text = offset_artist.get_text().strip()
              offset_artist.set_visible(False) # Hide the default offset text

    # Construct full axis labels including units and offsets
    x_label_full = metadata.get('X_col', 'X Value')
    # Use the full label with unit from metadata if available, otherwise construct
    x_label_full = metadata.get('X_label_with_unit', x_label_full)
    if x_offset_text:
         # Append offset text if present, handle case where unit is already in label
         if '(' in x_label_full and x_label_full.endswith(')'):
              x_label_full = x_label_full[:-1] + f" {x_offset_text})"
         else:
              x_label_full += f" ({x_offset_text})"


    y_label_full = metadata.get('Y_col', 'Y Value')
    # Use the full label with unit from metadata if available, otherwise construct
    y_label_full = metadata.get('Y_label_with_unit', y_label_full)
    if y_offset_text:
         # Append offset text if present, handle case where unit is already in label
         if '(' in y_label_full and y_label_full.endswith(')'):
              y_label_full = y_label_full[:-1] + f" {y_offset_text})"
         else:
              y_label_full += f" ({y_offset_text})"


    ax.set_xlabel(x_label_full)
    ax.set_ylabel(y_label_full)
    ax.set_title(chart_title, pad=20, fontsize=plt.rcParams['axes.titlesize'])

    # --- Legend Creation ---
    legend_elements, legend_labels = [], []
    if len(valid_sizes) > 0:
        # --- REVERTED LOGIC: Use only min and max for legend data values ---
        legend_data_values = sorted(list(set([q_min, q_max]))) if q_min != q_max else [q_min]
        # --- END REVERTED LOGIC ---

        # Ensure unique values and handle potential NaNs from quantiles on edge cases
        legend_data_values = sorted(list(set(filter(pd.notna, legend_data_values))))

        # Scale the representative data values to display sizes
        # Use actual min/max for interpolation range, ensure range is not zero
        interp_range_for_legend = (float(valid_sizes.min()), float(valid_sizes.max())) if len(valid_sizes) > 1 and valid_sizes.min() != valid_sizes.max() else (0, 1)
        legend_display_sizes_area = np.full_like(legend_data_values, (min_display_size_area + max_display_size_area) / 2.0, dtype=float) # Default if interpolation fails

        if interp_range_for_legend[0] != interp_range_for_legend[1]:
             current_legend_display_sizes_area = np.interp(legend_data_values,
                                                 interp_range_for_legend,
                                                 (min_display_size_area, max_display_size_area))
             legend_display_sizes_area = np.clip(current_legend_display_sizes_area, min_display_size_area, max_display_size_area)
        elif len(legend_data_values) > 0: # Case where all data points are the same value
             legend_display_sizes_area = np.full_like(legend_data_values, (min_display_size_area + max_display_size_area) / 2.0, dtype=float)


        legend_size_scale_factor = 0.8 # Adjust bubble size in legend for aesthetics
        scaled_legend_display_sizes_area = legend_display_sizes_area * legend_size_scale_factor

        for i, data_value in enumerate(legend_data_values):
            if pd.isna(data_value): continue
            display_size = scaled_legend_display_sizes_area[i]
            formatted_label = format_number_with_unit(data_value)
            legend_elements.append(ax.scatter([], [], s=display_size, color=selected_color, alpha=fixed_alpha,
                                   edgecolors='#000000', linewidth=0.5, label=formatted_label))
            legend_labels.append(formatted_label)

        # --- Construct legend title with wrapping ---
        # Use the full label with unit from metadata for the legend title
        legend_title_str_raw = metadata.get('Size_label_with_unit', metadata.get('Size Meaning', size_col))
        MAX_LEGEND_TITLE_LINE_WIDTH = 25  # Max characters per line for the description

        # Wrap the description part, handle unit separately if present
        desc_part, unit_part = parse_label_unit(legend_title_str_raw)
        wrapped_desc = textwrap.fill(desc_part, width=MAX_LEGEND_TITLE_LINE_WIDTH)

        if unit_part:
            # Ensure unit part is on a new line
            legend_title_str = f"{wrapped_desc}\n({unit_part})"
        else:
            legend_title_str = wrapped_desc
        # --- END MODIFICATION for legend title ---

        if legend_elements:
            legend = ax.legend(
                handles=legend_elements, labels=legend_labels, title=legend_title_str,
                loc='upper right', bbox_to_anchor=(1.25, 1),  # Adjusted to upper right with more space
                borderaxespad=0., frameon=True,
                labelspacing=1.75, borderpad=2, handletextpad=2.0, handlelength=1.5)
            # Adjust legend title font size
            plt.setp(legend.get_title(), fontsize=plt.rcParams['legend.fontsize'] + 1)

    # --- Adjust right margin for legend by changing rect's right value ---
    # This leaves space on the right for the legend. Adjust the value (0.97) as needed.
    # A value closer to 1 means less space on the right.
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    # --- END Adjust right margin ---

    return fig, selected_color
# --- END plot_bubble_chart ---

# --- Saving ---
def save_chart(fig: plt.Figure, png_filepath: str, svg_filepath: str, dpi: int = 300):
    """Saves the figure to PNG and SVG formats to specified paths."""
    try:
        # tight_layout is called in plot_bubble_chart, bbox_inches='tight' might not be needed or could conflict
        fig.savefig(png_filepath, dpi=dpi) # Removed bbox_inches and pad_inches, rely on tight_layout
        print(f"Saved PNG to {png_filepath}")
    except Exception as e:
        print(f"Error saving PNG file {png_filepath}: {e}\nDetails: {e}")
    try:
        fig.savefig(svg_filepath, format='svg') # Removed bbox_inches and pad_inches
        print(f"Saved SVG to {svg_filepath}")
    except Exception as e:
        print(f"Error saving SVG file {svg_filepath}: {e}\nDetails: {e}")
    plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    OUTPUT_PNG_DIR = "./bubble/png"
    OUTPUT_SVG_DIR = "./bubble/svg"
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SVG_DIR, exist_ok=True)
    INPUT_CSV_DIR = "./bubble/csv"

    csv_files = glob.glob(os.path.join(INPUT_CSV_DIR, "bubble_*.csv"))

    if not csv_files:
        print(f"No 'bubble_*.csv' files found in the directory: {INPUT_CSV_DIR}")
    else:
        print(f"Found {len(csv_files)} 'bubble_*.csv' file(s) in '{INPUT_CSV_DIR}'.")
        try:
            def sort_key(f):
                base = os.path.basename(f)
                match = re.search(r'_(\d+)\.csv$', base)
                return int(match.group(1)) if match else base
            csv_files.sort(key=sort_key)
        except Exception:
             csv_files.sort()
             print("Warning: Could not sort files numerically. Sorting alphabetically.")

        processed_count = 0
        for input_csv_path in csv_files:
            print(f"\nProcessing file: {input_csv_path}")
            try:
                output_filename_base = os.path.splitext(os.path.basename(input_csv_path))[0]
                # --- Call the updated parse_bubble_csv ---
                df_data, metadata_dict = parse_bubble_csv(input_csv_path)
                print("CSV file parsed successfully.")

                # Check if the essential columns exist and have valid data after parsing
                required_cols = [metadata_dict.get('X_col'), metadata_dict.get('Y_col'), metadata_dict.get('Size_col')]
                required_cols = [col for col in required_cols if col is not None] # Filter out None if keys are missing
                if not required_cols or any(col not in df_data.columns for col in required_cols):
                     print(f"Skipping file {input_csv_path}: Required columns not found in DataFrame after parsing.")
                     continue

                # Check if there are any valid data points for plotting
                if df_data.empty or df_data.dropna(subset=required_cols).empty:
                     print(f"Skipping file {input_csv_path}: No valid data points after parsing/conversion/dropping NaNs.")
                     continue


                print("Generating bubble chart...")
                # --- Call the updated plot_bubble_chart ---
                fig_object, selected_color_hex = plot_bubble_chart(df_data, metadata_dict)
                print("Bubble chart generated.")

                png_output_path = os.path.join(OUTPUT_PNG_DIR, f"{output_filename_base}.png")
                svg_output_path = os.path.join(OUTPUT_SVG_DIR, f"{output_filename_base}.svg")
                save_chart(fig_object, png_output_path, svg_output_path, dpi=300)
                processed_count += 1

            except FileNotFoundError as e_fnf:
                print(f"Skipping file {input_csv_path}: {e_fnf}")
            except ValueError as e_val:
                print(f"Skipping file {input_csv_path} due to data format/parsing issue: {e_val}")
                # traceback.print_exc() # Uncomment for more detailed ValueError
            except Exception as e_generic:
                print(f"Skipping file {input_csv_path} due to unexpected error: {e_generic}")
                traceback.print_exc()
        print(f"\nVisualization process finished. Successfully processed {processed_count} out of {len(csv_files)} files found.")

