import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm # Keep import, though not directly used in final data generation
import random
from typing import Tuple, Dict, List, Any

# --- Configuration Data ---
# Coordinate axis labels and their typical ranges and units
# NOTE: These ranges and units are examples and might need adjustment for specific needs.
axis_info: Dict[str, Tuple[float, float, str]] = {

}

# Combinations of X, Y, and Size axes for Bubble Charts, linked to topics
# Format: (Topic, X_Axis_Name, Y_Axis_Name, Size_Axis_Name, Size_Description)
bubble_combinations: List[Tuple[str, str, str, str, str]] = [
   ]

# Dictionary to keep track of the sequence number for each topic
topic_counters: Dict[str, int] = {}

# Distribution types for bubble size
BUBBLE_SIZE_DISTRIBUTIONS = ['random', 'normal', 'long_tail', 'linear']

# --- Data Generation Functions ---

def random_distribution(count: int) -> np.ndarray:
    """Generates random bubble sizes."""
    sizes = np.random.rand(count) * 100 # Scale to 0-100 initially
    return sizes

def normal_distribution(count: int) -> np.ndarray:
    """Generates normally distributed bubble sizes."""
    # Generate values centered around 50 with std dev 20 (on a 0-100 scale base)
    sizes = np.random.normal(50, 20, count)
    return sizes

def long_tail_distribution(count: int) -> np.ndarray:
    """Generates long-tail (Pareto-like) distributed bubble sizes."""
    # Pareto distribution, shape parameter 3, scaled and shifted
    sizes = np.random.pareto(3, count) * 20 + 10 # Scale and shift to get values mostly in a low-mid range with some high outliers
    return sizes

def linear_distribution(count: int) -> np.ndarray:
    """Generates linearly increasing bubble sizes."""
    return np.linspace(10, 100, count) # From 10 to 100 over the count

def generate_bubble_data(num_records: int, x_range: Tuple[float, float],
                         y_range: Tuple[float, float], size_range: Tuple[float, float],
                         overlap_degree: float, distribution_type: str,
                         x_col_name: str, y_col_name: str, size_col_name: str) -> pd.DataFrame:
    """
    Generates bubble chart data.

    Args:
        num_records: Number of data points to generate.
        x_range: Tuple (min, max) for X axis data.
        y_range: Tuple (min, max) for Y axis data.
        size_range: Tuple (min, max) for bubble size data.
        overlap_degree: Degree of overlap/noise (0 to 1) applied to X and Y.
        distribution_type: Type of distribution for bubble size ('random', 'normal', 'long_tail', 'linear').
        x_col_name: Name for the X column.
        y_col_name: Name for the Y column.
        size_col_name: Name for the Size column.

    Returns:
        pd.DataFrame: The generated data.
    """
    if num_records < 1:
        raise ValueError("Number of records must be at least 1")
    if not 0 <= overlap_degree <= 1:
        raise ValueError("Overlap degree must be between 0 and 1")
    if distribution_type not in BUBBLE_SIZE_DISTRIBUTIONS:
         raise ValueError(f"Invalid distribution type. Must be one of {BUBBLE_SIZE_DISTRIBUTIONS}")

    # Generate base X and Y values within their target ranges
    x_values = np.random.uniform(x_range[0], x_range[1], num_records)
    y_values = np.random.uniform(y_range[0], y_range[1], num_records)

    # Add overlap noise to X and Y
    # The scale of noise is relative to the range size
    x_noise_scale = (x_range[1] - x_range[0]) * overlap_degree * 0.1 if (x_range[1] - x_range[0]) > 0 else 0.1 # 10% of range scaled by overlap, handle zero range
    y_noise_scale = (y_range[1] - y_range[0]) * overlap_degree * 0.1 if (y_range[1] - y_range[0]) > 0 else 0.1 # 10% of range scaled by overlap, handle zero range

    x_values += np.random.normal(0, x_noise_scale, num_records)
    y_values += np.random.normal(0, y_noise_scale, num_records)

    # Clip X and Y values to roughly stay within the target range +/- a small margin
    # Added check for zero range
    x_margin = (x_range[1] - x_range[0]) * 0.05 if (x_range[1] - x_range[0]) > 0 else 0.1 # 5% margin, handle zero range
    y_margin = (y_range[1] - y_range[0]) * 0.05 if (y_range[1] - y_range[0]) > 0 else 0.1 # 5% margin, handle zero range

    x_values = np.clip(x_values, x_range[0] - x_margin, x_range[1] + x_margin)
    y_values = np.clip(y_values, y_range[0] - y_margin, y_range[1] + y_margin)


    # Generate base bubble sizes (on a 0-100 internal scale)
    if distribution_type == 'random':
        size_values_base = random_distribution(num_records)
    elif distribution_type == 'normal':
        size_values_base = normal_distribution(num_records)
    elif distribution_type == 'long_tail':
        size_values_base = long_tail_distribution(num_records)
    elif distribution_type == 'linear':
        size_values_base = linear_distribution(num_records)
    else:
        # Should not happen due to validation, but as a fallback
        size_values_base = random_distribution(num_records)


    # Scale base size values (0-100) to the target size range
    size_min_base = np.min(size_values_base)
    size_max_base = np.max(size_values_base)
    size_min_target, size_max_target = size_range

    if size_min_base == size_max_base or size_min_target == size_max_target:
        # Handle cases where base size is constant or target range is zero
        # Set all sizes to the middle of the target range
        size_values_scaled = np.full_like(size_values_base, (size_min_target + size_max_target) / 2)
    else:
        # Avoid division by zero if size_max_base == size_min_base
        scale_factor = (size_max_target - size_min_target) / (size_max_base - size_min_base) if (size_max_base - size_min_base) != 0 else 0
        size_values_scaled = size_min_target + (size_values_base - size_min_base) * scale_factor


    # Ensure scaled sizes are within the target range
    size_values_scaled = np.clip(size_values_scaled, size_min_target, size_max_target)

    # Prepare the data as a DataFrame with dynamic column names
    data = pd.DataFrame({
        x_col_name: x_values,
        y_col_name: y_values,
        size_col_name: size_values_scaled
    })

    return data


def save_bubble_to_csv(data: pd.DataFrame, topic: str, x_col_name: str, y_col_name: str,
                       size_col_name: str, x_unit: str, y_unit: str, size_unit: str,
                       distribution_type: str, bubble_size_description: str) -> str:
    """
    Saves bubble chart data to CSV with customizable column names and metadata.

    Args:
        data: The DataFrame containing bubble data.
        topic: The data topic.
        x_col_name: Name of the X column.
        y_col_name: Name of the Y column.
        size_col_name: Name of the Size column.
        x_unit: Unit for the X axis.
        y_unit: Unit for the Y axis.
        size_unit: Unit for the Size axis.
        distribution_type: The distribution type used for bubble size.
        bubble_size_description: Description of what the bubble size represents.

    Returns:
        The path to the saved CSV file.
    """
    output_dir = './csv/bubble'
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize topic for filename
    sanitized_topic = topic.replace(' ', '_').replace('&', 'and').replace(',', '').replace(':', '').lower()

    # Get current topic sequence number and update counter
    if sanitized_topic not in topic_counters:
        topic_counters[sanitized_topic] = 0

    current_count = topic_counters[sanitized_topic] + 1
    topic_counters[sanitized_topic] = current_count

    # Construct filename (topic_xx.csv, xx from 1 without leading zero for single digits)
    filename_base = f"{sanitized_topic}_{current_count}"
    filename = os.path.join(output_dir, f"{filename_base}.csv")

    # Construct header description
    header_description = f"# Topic: {topic}, "
    header_description += f"X: {x_col_name} ({x_unit}), Y: {y_col_name} ({y_unit}), "
    header_description += f"Size: {size_col_name} ({size_unit}), "
    header_description += f"Size Distribution: {distribution_type.replace('_', ' ')}, " # Add distribution type
    header_description += f"Size Meaning: {bubble_size_description}"

    with open(filename, 'w', newline='') as f:
        f.write(header_description + "\n")
        data.to_csv(f, index=False, header=True)

    return filename

def plot_bubble_chart(data: pd.DataFrame, topic: str, x_col_name: str, y_col_name: str,
                      size_col_name: str, x_range: Tuple[float, float], y_range: Tuple[float, float]):
    """
    Plots the generated bubble chart data.

    Args:
        data: The DataFrame containing data for plotting.
        topic: The topic for the chart title.
        x_col_name: Name of the X column.
        y_col_name: Name of the Y column.
        size_col_name: Name of the Size column.
        x_range: The target range for the X axis data.
        y_range: The target range for the Y axis data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scale bubble sizes for display - map data range to a display range (e.g., 50 to 2000 points^2)
    # Ensure size column is not empty or constant
    if data[size_col_name].min() != data[size_col_name].max():
        display_sizes = np.interp(data[size_col_name],
                                  (data[size_col_name].min(), data[size_col_name].max()),
                                  (50, 2000)) # Map data size range to display size range
    else:
        # If all sizes are the same, use a default display size
        display_sizes = np.full_like(data[size_col_name], 500) # Default size for uniform data


    scatter = ax.scatter(data[x_col_name], data[y_col_name], s=display_sizes, alpha=0.6,
                         c=data[size_col_name], cmap='viridis')

    # Set axis limits based on the target ranges with padding
    x_padding = (x_range[1] - x_range[0]) * 0.1 if (x_range[1] - x_range[0]) > 0 else 1 # Add 10% padding, handle zero range
    y_padding = (y_range[1] - y_range[0]) * 0.1 if (y_range[1] - y_range[0]) > 0 else 1 # Add 10% padding, handle zero range

    # Add a small buffer even if padding is 0 due to zero range
    x_buffer = 1 if x_padding == 0 and (x_range[1] - x_range[0]) == 0 else 0
    y_buffer = 1 if y_padding == 0 and (y_range[1] - y_range[0]) == 0 else 0


    ax.set_xlim(x_range[0] - x_padding - x_buffer, x_range[1] + x_padding + x_buffer)
    ax.set_ylim(y_range[0] - y_padding - y_buffer, y_range[1] + y_padding + y_buffer)

    ax.set_xlabel(f"{x_col_name} ({axis_info.get(x_col_name, (0,0,''))[2]})") # Add unit to label
    ax.set_ylabel(f"{y_col_name} ({axis_info.get(y_col_name, (0,0,''))[2]})") # Add unit to label
    ax.set_title(f"{topic}: {x_col_name} vs {y_col_name} (Bubble size by {size_col_name})")

    # Add color bar representing the size value
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{size_col_name} ({axis_info.get(size_col_name, (0,0,''))[2]})") # Add unit to color bar label

    plt.grid(True, linestyle='--', alpha=0.6) # Add grid
    plt.tight_layout() # Adjust layout
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Customizable Parameters ---
    NUM_FILES_TO_GENERATE = 10 # Total number of CSV files to generate

    # Parameters for data generation (can be fixed or randomized per file in the loop)
    # To randomize per file, move these inside the loop in main()
    FIXED_NUM_RECORDS = None # Set to None to randomize per file, or set an integer (e.g., 50)
    RECORDS_RANGE = (5, 10) # Range for number of records if randomized

    FIXED_OVERLAP_DEGREE = None # Set to None to randomize per file, or set a float (e.g., 0.2)
    OVERLAP_RANGE = (0.0, 0.5) # Range for overlap degree if randomized (0 to 1)

    FIXED_DISTRIBUTION_TYPE = None # Set to None to randomize per file, or choose from BUBBLE_SIZE_DISTRIBUTIONS (e.g., 'normal')

    # --- End Customizable Parameters ---

    # Reset counters for a new run
    topic_counters.clear()

    generated_files = 0
    while generated_files < NUM_FILES_TO_GENERATE:
        try:
            # --- Randomly select combination and parameters for this file ---
            selected_combo = random.choice(bubble_combinations)
            topic, x_name, y_name, size_name, size_description = selected_combo

            # Get ranges and units from axis_info, use defaults if not found
            # CORRECTED UNPACKING HERE:
            x_info = axis_info.get(x_name, (0, 100, 'unit'))
            x_range = (x_info[0], x_info[1])
            x_unit = x_info[2]

            y_info = axis_info.get(y_name, (0, 100, 'unit'))
            y_range = (y_info[0], y_info[1])
            y_unit = y_info[2]

            size_info = axis_info.get(size_name, (0, 100, 'unit'))
            size_range = (size_info[0], size_info[1])
            size_unit = size_info[2]
            # END CORRECTED UNPACKING
            
            # Randomize parameters if not fixed
            num_records = FIXED_NUM_RECORDS if FIXED_NUM_RECORDS is not None else random.randint(*RECORDS_RANGE)
            overlap_degree = FIXED_OVERLAP_DEGREE if FIXED_OVERLAP_DEGREE is not None else random.uniform(*OVERLAP_RANGE)
            distribution_type = FIXED_DISTRIBUTION_TYPE if FIXED_DISTRIBUTION_TYPE is not None else random.choice(BUBBLE_SIZE_DISTRIBUTIONS)
            # ---------------------------------------------------------------

            # Generate data
            data = generate_bubble_data(
                num_records=num_records,
                x_range=x_range,
                y_range=y_range,
                size_range=size_range,
                overlap_degree=overlap_degree,
                distribution_type=distribution_type,
                x_col_name=x_name,
                y_col_name=y_name,
                size_col_name=size_name
            )

            # Save data to CSV
            output_file = save_bubble_to_csv(
                data=data,
                topic=topic,
                x_col_name=x_name,
                y_col_name=y_name,
                size_col_name=size_name,
                x_unit=x_unit,
                y_unit=y_unit,
                size_unit=size_unit,
                distribution_type=distribution_type,
                bubble_size_description=size_description
            )

            # Plot data
            # plot_bubble_chart(data, topic, x_name, y_name, size_name, x_range, y_range) # Uncomment to plot each file

            print(f"Generated file {generated_files + 1}/{NUM_FILES_TO_GENERATE}:")
            print(f"  Topic: {topic}")
            print(f"  X-axis: {x_name} ({x_unit}), Range: {x_range}")
            print(f"  Y-axis: {y_name} ({y_unit}), Range: {y_range}")
            print(f"  Size-axis: {size_name} ({size_unit}), Range: {size_range}")
            print(f"  Size Distribution: {distribution_type}")
            print(f"  Number of records: {num_records}")
            print(f"  Overlap Degree: {overlap_degree:.2f}")
            print(f"  Saved to {output_file}\n")

            generated_files += 1

        except ValueError as e:
            print(f"Error generating data (ValueError): {e}. Skipping this combination/parameters.")
            # This might happen if the chosen combination leads to invalid ranges or counts.
            # The loop will continue trying to generate the required number of files.
        except Exception as e:
             print(f"An unexpected error occurred: {e}. Skipping this file.")
             # Catch other potential errors during generation or saving


    print(f"Finished generating {NUM_FILES_TO_GENERATE} bubble chart data files.")
