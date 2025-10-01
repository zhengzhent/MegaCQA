import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import random
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import io
import re
# --- MODIFICATION START ---
# Import the necessary toolkit
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- MODIFICATION END ---

# (Keep the colormap lists and filtering logic as before)
# Define single-color and dual-color color map lists
# Explicitly removing colormaps known to produce very dark, black, deep blue, deep purple, or brown colors.
# The lists are maintained to exclude potentially problematic maps based on common usage and visual outcome.
single_color_cmaps = [
    # Removed maps that often have very dark ends or are perceptually uniform but go to black/deep colors
    # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', # Standard sequential, often have dark ends
    # 'viridis', 'plasma', 'inferno', 'magma', # Perceptually uniform, but have very dark ends
    # 'Greys', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'copper', 'hot', 'afmhot', 'gist_heat', # Black, grey, brown, or harsh dark ends
    # 'PuRd', 'RdPu', # Contain purple, might be too deep even with truncation
    # 'PuBu', 'PuBuGn', 'GnBu', 'YlGnBu', # Contain blue, might be too deep even with truncation
    # 'YlOrRd', 'OrRd', 'YlOrBr', # Contain red/orange/brown, OrRd/YlOrBr might be too dark/brown

    # Keeping maps that are generally lighter or transition to lighter shades, and avoid deep blues/purples/blacks/browns
    'YlGn',       # Yellow-Green (lighter greens)
    'Wistia',     # Pink-Yellow-Orange (generally bright)
    'summer',     # Green-Yellow (bright)
    'spring',     # Pink-Yellow (bright)
    'autumn',     # Red-Orange-Yellow (bright/warm)
    'Oranges',    # Re-added Oranges, truncation helps manage the dark end. Can be vibrant.
    'Reds',       # Re-added Reds, truncation helps manage the dark end. Can be vibrant.
    'Greens',     # Re-added Greens, truncation helps manage the dark end. Can be vibrant.
    'Blues'       # Re-added Blues, truncation helps manage the dark end. Can be vibrant.
    # Note: Re-adding Oranges, Reds, Greens, Blues relies more heavily on the truncation (0.2) to avoid deep ends.
    # YlGnBu, GnBu, PuRd, RdPu are still excluded as they are more prone to deep blues/purples.
]

dual_color_cmaps = [
    # Removed diverging maps known for dark or problematic ends (blue/purple/black/brown)
    # 'RdBu_r', 'PuOr', 'BrBG', 'RdGy', 'seismic', # Often have dark ends, or involve blue/purple
    # 'Spectral', # Can have deep blue/purple ends
    # 'twilight', 'twilight_shifted', 'hsv', # Cyclic, can have dark transitions or jarring colors
    # 'terrain', 'ocean', 'gist_earth', # Geophysical, often dark/deep blues/greens/browns
    # 'PRGn', 'PRGn_r', # Purple-Green (involve purple)
    # 'PiYG', # Pink-Yellow-Green (can lean purple)
    # 'RdYlBu', # Red-Yellow-Blue (involves blue)

    # Keeping diverging maps that are generally lighter or transition between less extreme colors
    'RdYlGn_r',   # Green-Yellow-Red (avoids deep blue/purple)
    'coolwarm',   # Blue-Red (can have blue/red ends, but truncation helps, common diverging map)
    'bwr',        # Blue-White-Red (similar to coolwarm, truncation helps)
    # Note: coolwarm and bwr still involve blue/red extremes, but are common diverging maps.
    # Truncation is crucial here.
]


# Ensure lists contain unique color map names
single_color_cmaps = list(set(single_color_cmaps))
dual_color_cmaps = list(set(dual_color_cmaps))

# Filter out any cmap names that might not exist in the current matplotlib version
# This is a safeguard
all_available_cmaps = list(plt.colormaps())
single_color_cmaps = [c for c in single_color_cmaps if c in all_available_cmaps]
dual_color_cmaps = [c for c in dual_color_cmaps if c in all_available_cmaps]

# Final check to ensure lists are not empty after filtering
if not single_color_cmaps and not dual_color_cmaps:
    print("Warning: No suitable color maps available after extensive filtering!")
    # As a fallback, add a few universally available and relatively light maps if possible
    fallback_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'bwr']
    for cmap_name in fallback_cmaps:
         if cmap_name in all_available_cmaps:
             if cmap_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
                 if cmap_name not in single_color_cmaps: single_color_cmaps.append(cmap_name)
             else: # coolwarm, bwr
                 if cmap_name not in dual_color_cmaps: dual_color_cmaps.append(cmap_name)
    print(f"Using fallback color maps: Single={single_color_cmaps}, Dual={dual_color_cmaps}")

# --- MODIFICATION START: visualize_heatmap signature changed ---
# Accepts specific output directories for png and svg
def visualize_heatmap(csv_file_path, png_output_dir, svg_output_dir):
    """
    Visualize heatmap data from CSV file and save as PNG and SVG to specified directories.

    Args:
        csv_file_path: Path to the input CSV file
        png_output_dir: Directory to save PNG files.
        svg_output_dir: Directory to save SVG files.
    """
    try:
        # Read the CSV file
        # Assuming the first row is metadata: Main theme, little theme, dimension, pattern
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            # Read the header line
            header_line = f.readline().strip()
            # Read the rest as DataFrame, skipping the header row
            df = pd.read_csv(f, header=None)

        # Check if DataFrame has enough columns (at least 3 for data)
        if df.shape[1] < 3:
            print(f"Error: CSV file {csv_file_path} does not have at least 3 data columns.")
            return

        # Manually set column names assuming the first three data columns are x_block, y_block, level
        df.rename(columns={0: 'x_block', 1: 'y_block', 2: 'level'}, inplace=True)

        # Parse metadata from the header line: Main theme, little theme, dimension, pattern
        metadata = {}
        parts = [part.strip() for part in header_line.split(',')]

        if len(parts) >= 4:
            metadata['topic'] = parts[0] # Main theme
            metadata['little_theme'] = parts[1] # Little theme
            metadata['dimension_str'] = parts[2] # Dimension string
            metadata['pattern'] = parts[3] # Pattern

            # Optional: Parse dimensions if needed later (using the third part)
            try:
                x_str, y_str = metadata['dimension_str'].split('x')
                metadata['x_blocks'] = int(x_str.strip())
                metadata['y_blocks'] = int(y_str.strip())
            except (IndexError, ValueError):
                metadata['x_blocks'] = None
                metadata['y_blocks'] = None

        else:
            print(f"Warning: Header line in {csv_file_path} does not match expected format (Main theme, little theme, dimension, pattern). Using default metadata.")
            metadata['topic'] = 'Unknown Topic'
            metadata['little_theme'] = 'Unknown Purpose' # Default little theme
            metadata['dimension_str'] = 'NxM'
            metadata['x_blocks'] = None
            metadata['y_blocks'] = None
            metadata['pattern'] = 'Unknown Pattern'


        if df.empty:
             print(f"Error: DataFrame from {csv_file_path} is empty after reading.")
             return

        # Ensure 'level' column is numeric and round to 2 decimal places
        df['level'] = pd.to_numeric(df['level'], errors='coerce')
        df['level'] = df['level'].round(2) # Explicitly round to 2 decimals


        # Ensure x_block and y_block columns exist and are suitable for pivoting index/columns
        # Convert them to string or integer if possible, handling potential errors
        if 'x_block' not in df.columns or 'y_block' not in df.columns or 'level' not in df.columns:
             print(f"Error: DataFrame from {csv_file_path} is missing required columns (x_block, y_block, level) after renaming.")
             return

        # Attempt to sort labels before pivoting to ensure consistent ordering
        # This is important if the labels are not inherently sortable as numbers
        # Get unique labels and sort them
        unique_x_labels = sorted(df['x_block'].unique().tolist())
        unique_y_labels = sorted(df['y_block'].unique().tolist())

        # Convert to categorical type with specified order to enforce sorting in pivot
        df['x_block'] = pd.Categorical(df['x_block'], categories=unique_x_labels, ordered=True)
        df['y_block'] = pd.Categorical(df['y_block'], categories=unique_y_labels, ordered=True)


        # Drop rows where x_block or y_block might have become NaN or are otherwise invalid
        df.dropna(subset=['x_block', 'y_block'], inplace=True)


        # Pivot the data for heatmap
        # Handle potential duplicates in x_block, y_block pairs by taking the mean value
        # pivot_table automatically ignores NaN values in the 'level' column during aggregation
        # Use the ordered categorical types for index and columns
        heatmap_data = df.pivot_table(index='y_block', columns='x_block', values='level', aggfunc='mean')

        if heatmap_data.empty:
             print(f"Error: Pivoted data from {csv_file_path} is empty or contains only NaNs.")
             return

        # Set up plot style
        plt.style.use('default')
        rcParams['font.family'] = 'Times New Roman'
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 1
        rcParams['axes.edgecolor'] = '#000000'
        rcParams['axes.labelcolor'] = '#000000'
        rcParams['xtick.color'] = '#000000'
        rcParams['ytick.color'] = '#000000'

        # Create figure with specified size (6.4x4.8 inches) - this might be adjusted by aspect='equal'
        # Consider adjusting figsize based on data shape if needed for aspect='equal'
        fig, ax = plt.subplots(figsize=(max(6, heatmap_data.shape[1]*0.8), max(4.8, heatmap_data.shape[0]*0.8)), dpi=300) # Adjust size based on dimensions

        # --- Random color map selection logic (remains the same) ---
        available_cmap_types = []
        if single_color_cmaps:
            available_cmap_types.append('single')
        if dual_color_cmaps:
            available_cmap_types.append('dual')

        if not available_cmap_types:
             print("Error: No suitable color maps available after filtering and fallback.")
             plt.close(fig)
             return

        weights = None
        if 'single' in available_cmap_types and 'dual' in available_cmap_types:
             weights = [7, 3]
        elif 'single' in available_cmap_types:
             weights = [1]
        elif 'dual' in available_cmap_types:
             weights = [1]

        chosen_type = random.choices(available_cmap_types, weights=weights, k=1)[0]
        selected_cmap_name = random.choice(single_color_cmaps) if chosen_type == 'single' else random.choice(dual_color_cmaps)
        cmap_type = "Single Color" if chosen_type == 'single' else "Dual Color"
        print(f"Processing with color map: {selected_cmap_name} ({cmap_type})")

        # --- Modify the selected colormap to reduce darkness ---
        original_cmap = plt.cm.get_cmap(selected_cmap_name)
        # Increased truncation value further to prevent very dark colors
        dark_color_truncation = 0.35 # Increased from 0.3

        cmap_sample_start = 0.0
        cmap_sample_stop = 1.0
        dark_at_start_maps_heuristic = selected_cmap_name.endswith('_r') or selected_cmap_name in ['hot', 'copper', 'gist_heat', 'afmhot', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone']

        if selected_cmap_name in dual_color_cmaps:
             truncation_per_end = dark_color_truncation / 2.0
             cmap_sample_start = truncation_per_end
             cmap_sample_stop = 1.0 - truncation_per_end
        elif dark_at_start_maps_heuristic:
             cmap_sample_start = dark_color_truncation
             cmap_sample_stop = 1.0
        else:
             cmap_sample_start = 0.0
             cmap_sample_stop = 1.0 - dark_color_truncation

        cmap_sample_start = max(0.0, cmap_sample_start)
        cmap_sample_stop = min(1.0, cmap_sample_stop)

        if cmap_sample_start >= cmap_sample_stop - 1e-9:
            print(f"Warning: Truncation resulted in invalid range [{cmap_sample_start:.2f}, {cmap_sample_stop:.2f}] for {selected_cmap_name}. Using original colormap.")
            modified_cmap = original_cmap
        else:
            new_colors = original_cmap(np.linspace(cmap_sample_start, cmap_sample_stop, 256))
            modified_cmap = LinearSegmentedColormap.from_list(f'{selected_cmap_name}_lightened', new_colors)

        # ----------------------------------------------------


        # Use vmin/vmax based on the range of values in the heatmap data
        # Consider using percentiles if outliers are an issue, but min/max is standard.
        # Ensure vmin/vmax are not equal if data has only one unique non-NaN value.
        data_min = heatmap_data.min().min()
        data_max = heatmap_data.max().max()

        if pd.isna(data_min) or pd.isna(data_max):
             # Fallback if min/max are NaN (e.g., heatmap_data was all NaNs despite pivot_table)
             vmin, vmax = 0, 1
             print("Warning: Heatmap data min/max are NaN. Using default colorbar range [0, 1].")
        elif data_min == data_max:
             # If all non-NaN values are the same, set a small range around that value
             vmin = data_min - 0.01 if data_min > 0.01 else 0
             vmax = data_max + 0.01 if data_max < 0.99 else 1
             if vmin == vmax: # Ensure vmin != vmax even with tiny range
                 vmin = max(0.0, vmin - 0.001)
                 vmax = min(1.0, vmax + 0.001)
             print(f"Warning: All non-NaN heatmap values are identical ({data_min}). Using colorbar range [{vmin:.2f}, {vmax:.2f}].")
        else:
             # Use the actual data range
             vmin, vmax = data_min, data_max


        # Create heatmap using the modified color map and data range
        im = ax.imshow(heatmap_data, cmap=modified_cmap, aspect='equal', vmin=vmin, vmax=vmax)

        # Set title - Use the little theme from metadata
        little_theme_name = metadata.get('little_theme', 'Unknown Purpose')
        ax.set_title(f"{little_theme_name}", fontsize=16, pad=20, color='#000000')

        # Set ticks and labels
        ax.set_xticks(range(len(unique_x_labels)))
        ax.set_yticks(range(len(unique_y_labels)))
        ax.set_xticklabels([str(label) for label in unique_x_labels], rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels([str(label) for label in unique_y_labels], fontsize=12)


        # --- MODIFICATION START: Use make_axes_locatable for colorbar placement ---
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append a new axes to the right of the main axes.
        # size="5%" means the new axes' width is 5% of the main axes' width.
        # pad=0.2 controls the space between the main axes and the new axes. Adjust as needed.
        cax = divider.append_axes("right", size="1%", pad=0.2) # Reduced size to 3% for thinner bar, increased pad

        # Create the colorbar in the new axes `cax`
        # Adjust ticks based on the actual data range if it's not [0, 1]
        if vmin != 0 or vmax != 1:
             # Use a reasonable number of ticks (e.g., 5) within the data range
             cbar_ticks = np.linspace(vmin, vmax, 5)
             cbar_ticks = [round(t, 2) for t in cbar_ticks] # Round ticks
             cbar = plt.colorbar(im, cax=cax, ticks=cbar_ticks)
        else:
             # Default ticks for [0, 1] range
             cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
        # --- MODIFICATION END ---

        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Level', fontsize=14)

        # Add value annotations
        text_color = "#000000"
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data.iloc[i, j]
                if pd.notna(value):
                     # Use a different text color for better contrast on darker cells (optional refinement)
                     # For simplicity and consistency with request, keeping black text for now.
                     # You could add logic here based on the cell's color or value.
                     text = ax.text(j, i, f"{value:.2f}",
                                  ha="center", va="center", color=text_color, fontsize=10)

        # Remove grid and frame
        ax.grid(False)
        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- MODIFICATION START ---
        # Remove or comment out plt.tight_layout() as make_axes_locatable handles placement
        # plt.tight_layout() # Optional: Can sometimes help with title/label spacing, but might interfere with cax placement. Test with and without.
        # --- MODIFICATION END ---

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_filename_base = f"{base_name}"

        # --- MODIFICATION START: Save to specified output directories ---
        # Save as PNG
        png_path = os.path.join(png_output_dir, f"{output_filename_base}.png")
        # Use bbox_inches='tight' to try and include all elements within the saved figure bounds
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved PNG to {png_path}")

        # Save as SVG
        svg_path = os.path.join(svg_output_dir, f"{output_filename_base}.svg")
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"Saved SVG to {svg_path}")
        # --- MODIFICATION END: Save to specified output directories ---

        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file {csv_file_path} is empty.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred while processing {csv_file_path}:")
        traceback.print_exc() # Print full traceback for debugging
        # In case of error, try to close all figures to prevent memory issues
        plt.close('all')


# --- MODIFICATION START: visualize_all_heatmaps signature changed ---
# Accepts specific output directories for png and svg
def visualize_all_heatmaps(input_dir, png_output_dir, svg_output_dir):
    """
    Visualize all heatmap CSV files in a directory and save to specified output directories.

    Args:
        input_dir: Directory containing CSV files.
        png_output_dir: Directory to save PNG files.
        svg_output_dir: Directory to save SVG files.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Get all CSV files in input directory
    # Sorting now happens in the generation script by filename
    # We can still sort here for processing order consistency if needed,
    # but the filenames are already sequential.
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    # Sort files numerically based on the sequence number in the filename (e.g., heatmap_XX.csv or heatmap_Topic_XX.csv)
    try:
        def sort_key(f):
            base = os.path.basename(f)
            match = re.search(r'_(\d+)\.csv$', base)
            if match:
                return int(match.group(1))
            return base # Fallback to alphabetical sort if number not found
        csv_files.sort(key=sort_key)
    except Exception:
         # Generic sort if the specific key fails
         csv_files.sort()
         print("Warning: Could not sort files numerically. Sorting alphabetically.")


    print(f"Found {len(csv_files)} CSV files in {input_dir}. Visualizing...")

    for csv_file in csv_files: # Process in sorted order
        csv_path = os.path.join(input_dir, csv_file)
        print(f"\n--- Processing {csv_file} ---")
        # Pass the specific output directories to visualize_heatmap
        visualize_heatmap(csv_path, png_output_dir, svg_output_dir)

    print("\nVisualization process completed.")
# --- MODIFICATION END: visualize_all_heatmaps signature changed ---


# --- Main Execution ---
if __name__ == "__main__":
    # Define the input and output directories
    INPUT_CSV_DIR = "./heatmap/csv" # Modified input directory
    OUTPUT_PNG_DIR = "./heatmap/png" # Modified PNG output directory
    OUTPUT_SVG_DIR = "./heatmap/svg" # Modified SVG output directory

    # Ensure the output directories exist
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SVG_DIR, exist_ok=True)

    # Add a check for scipy dependency for the diagonal pattern's blur effect (only relevant for generation, but keeping here for completeness)
    # try:
    #     import scipy
    # except ImportError:
    #     print("Optional dependency 'scipy' not found. Gaussian blur for diagonal pattern will be skipped if generated.")

    # Call the main visualization function with the specified directories
    visualize_all_heatmaps(input_dir=INPUT_CSV_DIR, png_output_dir=OUTPUT_PNG_DIR, svg_output_dir=OUTPUT_SVG_DIR)
