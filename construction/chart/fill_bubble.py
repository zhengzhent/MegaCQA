import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import os
import traceback
import re # Import re for parsing
from math import sqrt, pi, cos, sin, atan2
from matplotlib.colors import LinearSegmentedColormap
import random
from typing import Tuple, Dict, Any, List, Optional # Import necessary types
# import adjustText as adjust_text # Keeping this commented out as per "002gf" context

# --- Define layout parameters ---
BASE_LAYOUT_SCALE = 0.75 # This factor influenced the relative size within parent, keep it.
MIN_SPACING_FACTOR = 0.3
MAX_ITERATIONS = 25
REPULSION_DECAY = 0.7

# --- Fixed Figure and Font Parameters ---
FIXED_FIG_WIDTH_INCHES = 6.4 # Fixed width in inches
FIXED_FIG_HEIGHT_INCHES = 6.4 # Fixed height in inches
FIXED_DPI = 300 # Adjusted DPI to achieve 1920x1920 (6.4 * 300 = 1920)
PLOT_MARGIN_FACTOR = 0.03 # Factor for padding around the content within the fixed figure (e.g., 15%)

# Fixed font sizes for the first three layers
FIXED_FONT_SIZES = {
    1: 14, # Layer 1 (Depth 1)
    2: 12, # Layer 2 (Depth 2)
    3: 10  # Layer 3 (Depth 3)
}
DEFAULT_FONT_SIZE = 9 # Default for layers > 3

# --- Define Brighter Color Palette for Depths 1, 2, 3 ---
# Using distinct, brighter colors for the first three layers
BRIGHT_COLOR_PALETTE = {
    1: '#64B5F6', # Light Blue (for Depth 1 / Layer 1)
    2: '#FFB74D', # Amber (for Depth 2 / Layer 2)
    3: '#81C784', # Light Green (for Depth 3 / Layer 3)
    # Add default/fallback for depths > 3 if needed, maybe a lighter gray
    4: '#CCCCCC',
    5: '#BBBBBB',
    # ... and so on, or just use a single fallback color
}


# --- Label Collision Parameters ---
MAX_LABEL_ITERATIONS = 300 # Increased iterations for label adjustment with padding
LABEL_REPULSION_DECAY = 0.96 # Slightly adjusted decay
PARENT_ATTRACTION_STRENGTH = 0.01 # Strength of pull towards parent bubble center (applied per iteration)
LABEL_SHIFT_THRESHOLD = 0.005 # Minimum total displacement to continue iterations (in data units/inches)
# Heuristic factors for estimating text bounding box size (adjust as needed)
# These factors are rough estimates for average character width and line height relative to fontsize
# Used for fallback estimation only.
AVG_CHAR_WIDTH_FACTOR = 0.48
LINE_HEIGHT_FACTOR = 1.05

# Padding factor added to estimated bbox size *only for collision detection*
LABEL_COLLISION_PADDING_FACTOR = 0.2 # Add 15% padding to estimated bbox dimensions for collision

# --- Custom Shift Parameters (Added for this request) ---
# Shift the entire bubble layout area by this amount (in inches) relative to the Axes origin (approx. Figure center)
# Adjust these values to move the bubble cluster within the fixed figure.
BUBBLE_SHIFT_X_INCHES = -0.17 # Negative value shifts left. ADJUST THIS!
BUBBLE_SHIFT_Y_INCHES = 0.0 # 0 for no vertical shift. ADJUST THIS!

# Offset for the legend position (relative to the default 'upper right' anchor point)
# (x, y) offset in axes coordinates. (1,1) is upper right of axes.
LEGEND_ANCHOR_OFFSET_X = 0.09 # Positive value shifts right relative to (1,1). ADJUST THIS!
LEGEND_ANCHOR_OFFSET_Y = 0.0 # No vertical offset relative to (1,1). ADJUST THIS!


def parse_fill_bubble_csv(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads the fill_bubble CSV file, parses metadata from the first line,
    and loads the data from the rest, handling potential errors.
    Expected first line: Main theme, little theme, Total Nodes
    Expected data header (second line): size,father,depth,label
    """
    metadata: Dict[str, Any] = {}
    df = pd.DataFrame()
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")

        # Read the first line (metadata) separately
        with open(filepath, 'r', encoding='utf-8') as f:
            header_line_1 = f.readline().strip()

        # Parse metadata from the first line
        parts_1 = [part.strip() for part in header_line_1.split(',')]
        expected_parts_1 = 3
        if len(parts_1) != expected_parts_1:
            error_msg = f"Incorrect first header format in {filepath}. Expected {expected_parts_1} parts, found {len(parts_1)}. Header: '{header_line_1}'"
            # Check for BOM (Byte Order Mark) which can sometimes cause parsing issues
            if header_line_1 and header_line_1.startswith('\ufeff'):
                 error_msg += " (File might have a BOM)"
            raise ValueError(error_msg)

        metadata['Topic'] = parts_1[0]
        metadata['Little_Theme'] = parts_1[1]
        try:
            metadata['Total_Nodes'] = int(parts_1[2])
        except ValueError:
            print(f"Warning: Could not parse Total Nodes '{parts_1[2]}' in {filepath}. Storing as string.")
            metadata['Total_Nodes'] = parts_1[2]
        except IndexError:
            print(f"Warning: Missing Total Nodes in {filepath}.")
            metadata['Total_Nodes'] = "N/A"

        # Read the rest of the data starting from the second line, using the second line as header
        df = pd.read_csv(filepath, skiprows=1, header=0, encoding='utf-8', index_col=False)

        # Validate required columns
        required_cols = ['size', 'father', 'depth', 'label']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required data columns in {filepath}: {', '.join(missing)}. Found: {list(df.columns)}")

        if df.empty:
            print(f"Warning: DataFrame is empty after reading data (before cleaning) from {filepath}.")
            return df, metadata

        # Convert columns to appropriate types and handle missing/invalid data
        # Use errors='coerce' to turn invalid parsing into NaN
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        # Fill NaN in father/depth with defaults and convert to int
        df['father'] = pd.to_numeric(df['father'], errors='coerce').fillna(0).astype(int)
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce').fillna(1).astype(int)
        # FIX: Ensure .str is used for string methods on the Series
        df['label'] = df['label'].astype(str).str.strip() # Ensure label is string and strip whitespace

        # Drop rows with missing required values after conversion
        df.dropna(subset=['size', 'label'], inplace=True)

        # Filter out nodes with non-positive size or empty labels (if any slipped through)
        df = df[df['size'] > 0]
        df = df[df['label'] != '']

        if df.empty:
            print(f"Warning: DataFrame is empty after cleaning data from {filepath}.")

    except FileNotFoundError:
        # Re-raise the exception so the caller can handle it
        raise
    except ValueError as ve:
        # Re-raise the exception so the caller can handle it
        raise ve
    except Exception as e:
        # Wrap other exceptions with file context and re-raise
        raise Exception(f"Error reading or parsing CSV file {filepath}: {e}") from e

    return df, metadata


def normalize_sizes(nodes: Dict[int, Dict[str, Any]], root_nodes_for_scaling: List[Dict[str, Any]], base_size: float = 700.0): # Reduced base_size from 800 to 700
    if not nodes:
        return
    sizes = [n['size'] for n in nodes.values()]
    max_size = max(sizes) if sizes else 1.0
    scale_factor = base_size / max_size if max_size > 0 else 1.0
    for node_data in nodes.values():
        # Apply scaling factor to the 'size' used for layout calculation
        node_data['size'] *= scale_factor

# Helper function to apply absolute positions recursively
def apply_absolute_positions(node: Dict[str, Any], parent_absolute_x: float, parent_absolute_y: float):
    """
    Recursively calculates and sets the absolute x, y coordinates and radius for a node
    and its children based on their relative positions and the parent's absolute position.
    """
    node['x'] = parent_absolute_x + node['x_relative']
    node['y'] = parent_absolute_y + node['y_relative']
    node['radius'] = node['radius_relative'] # Store the calculated absolute radius (after scaling)
    for child_node_data in node['children']:
        apply_absolute_positions(child_node_data, node['x'], node['y'])


def calculate_layout(nodes: Dict[int, Dict[str, Any]], root_nodes_for_layout: List[Dict[str, Any]]):
    """
    Calculates the relative layout and then applies absolute positions for the bubble nodes.
    This function assumes nodes have 'size', 'children', and will add/modify
    'radius_relative', 'x_relative', 'y_relative', 'radius', 'x', 'y'.
    The resulting x, y, radius will be relative to the main root's initial position (0,0).
    """
    # Normalize sizes first to get base radius for layout calculation
    normalize_sizes(nodes, root_nodes_for_layout) # base_size is now 700

    def calculate_relative_layout(node: Dict[str, Any], level: int) -> float:
        """
        Recursively calculates the relative positions and radius for a node and its children.
        This is the force-directed layout part.
        """
        # Base radius based on size, scaled by BASE_LAYOUT_SCALE
        base_radius = sqrt(node['size']) * BASE_LAYOUT_SCALE
        base_radius = max(base_radius, 0.1) # Ensure minimum radius
        node['radius_relative'] = base_radius # Store the calculated relative radius

        # Initialize relative position to (0,0) relative to its parent
        node['x_relative'] = 0
        node['y_relative'] = 0

        if not node['children']:
            # If no children, the node's size determines its relative radius
            return node['radius_relative']

        # If node has children, calculate their relative layouts first
        children_radius_relative = []
        for child_node_data in node['children']:
            child_radius = calculate_relative_layout(child_node_data, level + 1)
            children_radius_relative.append(child_radius)

        # Determine the required space for children
        max_child_radius = max(children_radius_relative) if children_radius_relative else 0
        min_spacing_relative = max_child_radius * MIN_SPACING_FACTOR # Spacing between children

        # Calculate the radius the parent node needs to enclose its children
        # If only one child, parent needs to be big enough for child + spacing within its radius
        if len(node['children']) == 1:
            arrange_radius_relative = children_radius_relative[0] * 1.8 + min_spacing_relative # A bit more than child radius + spacing
        else:
            # For multiple children, approximate needed radius based on sum of squares of radii + spacing
            arrange_radius_relative = sqrt(sum(r**2 for r in children_radius_relative)) * 1.5 + min_spacing_relative


        # The parent's relative radius should be at least its base radius, but also large enough to contain children
        node['radius_relative'] = max(node['radius_relative'], arrange_radius_relative)

        # Calculate the radius of the circle on which children centers will be initially placed
        # This circle is smaller than the parent's radius
        layout_circle_radius_relative = max(
            node['radius_relative'] - max_child_radius - min_spacing_relative * 1.2, # Ensure circle is inside parent boundary
            max_child_radius * 1.2 # Ensure circle is large enough for children
        )
        layout_circle_radius_relative = max(layout_circle_radius_relative, 0.1) # Minimum circle radius

        # Initial placement of children on a circle
        if len(node['children']) > 1:
            golden_angle = np.pi * (3 - np.sqrt(5)) # Golden angle for even distribution
            phase_shift = random.uniform(0, 2 * np.pi) # Randomize starting angle
            initial_angles = [(i * golden_angle + phase_shift) % (2 * np.pi) for i in range(len(node['children']))]
        else:
            # If only one child, place it at a random angle
            initial_angles = [random.uniform(0, 2 * np.pi)]

        for i, (child_node_data, angle) in enumerate(zip(node['children'], initial_angles)):
            # Initial position on the layout circle
            rand_factor = 0.9 + 0.2 * random.random() # Add a bit of randomness to initial placement radius
            child_node_data['x_relative'] = layout_circle_radius_relative * cos(angle) * rand_factor
            child_node_data['y_relative'] = layout_circle_radius_relative * sin(angle) * rand_factor

        # Iterative force-directed layout for children
        for iteration in range(MAX_ITERATIONS):
            total_displacement = 0 # Track movement to check for convergence

            # Repulsion between children nodes
            for i in range(len(node['children'])):
                for j in range(i + 1, len(node['children'])):
                    ci = node['children'][i]
                    cj = node['children'][j]
                    dx = ci['x_relative'] - cj['x_relative']
                    dy = ci['y_relative'] - cj['y_relative']
                    dist = sqrt(dx**2 + dy**2) # Distance between centers
                    dist = dist or 1e-6 # Avoid division by zero if centers are identical

                    # Target distance includes radii and minimum spacing
                    target_dist = ci['radius_relative'] + cj['radius_relative'] + min_spacing_relative

                    overlap = target_dist - dist # How much they overlap or are too close

                    if overlap > 0: # If they are too close
                        # Calculate unit vector direction of repulsion
                        ux, uy = dx / dist, dy / dist
                        # Calculate shift proportional to overlap and decay over iterations
                        shift = overlap * 0.5 * (REPULSION_DECAY**iteration)
                        # Apply shifts in opposite directions
                        ci['x_relative'] += ux * shift; ci['y_relative'] += uy * shift
                        cj['x_relative'] -= ux * shift; cj['y_relative'] -= uy * shift
                        total_displacement += abs(shift) # Sum absolute shifts

            # Attraction/Constraint towards parent center/layout circle
            for child_node_data in node['children']:
                dist_to_center = sqrt(child_node_data['x_relative']**2 + child_node_data['y_relative']**2)

                # Calculate the maximum distance a child's center can be from the parent's center
                # while still being fully inside the parent (considering min spacing)
                safe_radius_for_child_center = node['radius_relative'] - child_node_data['radius_relative'] - (min_spacing_relative * 0.5)
                safe_radius_for_child_center = max(safe_radius_for_child_center, 0) # Cannot be negative

                if dist_to_center > safe_radius_for_child_center:
                    # If child center is too far from parent center, pull it back
                    overshoot = dist_to_center - safe_radius_for_child_center
                    # Calculate unit vector pointing away from parent center
                    ux = child_node_data['x_relative'] / dist_to_center if dist_to_center > 0 else 0
                    uy = child_node_data['y_relative'] / dist_to_center if dist_to_center > 0 else 0

                    # Pull child back towards the parent center
                    child_node_data['x_relative'] -= ux * overshoot * 0.8 # Apply a fraction of overshoot
                    child_node_data['y_relative'] -= uy * overshoot * 0.8
                    # total_displacement += overshoot # Add to total displacement - this might double count, let's just track repulsion for convergence

                # Optional: Gently pull children towards the ideal layout circle, especially later iterations
                if iteration > MAX_ITERATIONS // 2 and layout_circle_radius_relative > 0:
                    target_angle = atan2(child_node_data['y_relative'], child_node_data['x_relative'])
                    ideal_radius = layout_circle_radius_relative * 0.95 # Slightly smaller than the ideal circle radius
                    # current_radius = dist_to_center # Re-calculate distance to center - not needed for blend

                    # Interpolate between current position and ideal position on the circle
                    blend_factor = 0.05 # How strongly to pull towards the ideal circle
                    ideal_x = ideal_radius * cos(target_angle)
                    ideal_y = ideal_radius * sin(target_angle)

                    child_node_data['x_relative'] = child_node_data['x_relative'] * (1 - blend_factor) + ideal_x * blend_factor
                    child_node_data['y_relative'] = child_node_data['y_relative'] * (1 - blend_factor) + ideal_y * blend_factor


            # Check for convergence: if total movement is very small, stop iterating
            # Check repulsion displacement only for convergence of the packing
            if iteration > 5 and total_displacement < 0.01 * (min_spacing_relative or 1):
                break

        # Return the calculated relative radius of the current node
        return node['radius_relative']

    if not root_nodes_for_layout:
        return # Cannot calculate layout if no root nodes are provided

    # Start the recursive relative layout calculation from the main root
    # The initial call to apply_absolute_positions will set root's x,y to 0,0
    the_main_root = root_nodes_for_layout[0]
    calculate_relative_layout(the_main_root, 0) # Pass 0 for initial level

    # Apply the calculated relative positions as absolute positions, starting from (0,0) for the main root
    apply_absolute_positions(the_main_root, 0, 0)


def estimate_text_bbox_size_heuristic(text: str, fontsize: float) -> Tuple[float, float]:
    """
    Estimate the bounding box size of text based on heuristic factors (fallback).
    Returns size in fontsize units (points).
    """
    # These factors might need tuning based on the actual font and rendering
    width = len(text) * AVG_CHAR_WIDTH_FACTOR * fontsize
    height = LINE_HEIGHT_FACTOR * fontsize # Assumes single line
    return width, height

def resolve_label_collisions(label_list: List[Dict[str, Any]], visual_scale: float):
    """
    Iteratively adjusts label positions vertically and horizontally to prevent
    overlaps of their estimated bounding boxes (with padding), while also pulling them towards
    their original center positions.

    Args:
        label_list: A list of dictionaries, each representing a label with
                    'curr_x', 'curr_y', 'orig_x', 'orig_y', 'bbox_w', 'bbox_h'.
                    These will be modified in place. bbox_w/h are the *estimated* unpadded sizes in data units (inches).
        visual_scale: The final visual scaling factor applied to the plot content (data units per layout unit).
                      Used for scaling the convergence threshold.
    """
    if not label_list or len(label_list) < 2:
        return # No collisions possible with 0 or 1 label

    # print(f"Attempting to resolve collisions for {len(label_list)} labels...") # Comment out for cleaner output

    # Calculate padded bbox dimensions once (in data units/inches)
    for label_data in label_list:
        label_data['padded_bbox_w'] = label_data['bbox_w'] * (1 + LABEL_COLLISION_PADDING_FACTOR)
        label_data['padded_bbox_h'] = label_data['bbox_h'] * (1 + LABEL_COLLISION_PADDING_FACTOR)

    # LABEL_SHIFT_THRESHOLD is in data units (inches) already
    # PARENT_ATTRACTION_STRENGTH is applied as a fraction of the distance in data units (inches)

    for iteration in range(MAX_LABEL_ITERATIONS):
        total_displacement = 0 # Track total movement in this iteration (in data units)

        # --- Repulsion between labels ---
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                l1 = label_list[i]
                l2 = label_list[j]

                # Calculate bounding box corners using the *padded* estimated size (in data units)
                l1_left = l1['curr_x'] - l1['padded_bbox_w'] / 2
                l1_right = l1['curr_x'] + l1['padded_bbox_w'] / 2
                l1_bottom = l1['curr_y'] - l1['padded_bbox_h'] / 2
                l1_top = l1['curr_y'] + l1['padded_bbox_h'] / 2

                l2_left = l2['curr_x'] - l2['padded_bbox_w'] / 2
                l2_right = l2['curr_x'] + l2['padded_bbox_w'] / 2
                l2_bottom = l2['curr_y'] - l2['padded_bbox_h'] / 2
                l2_top = l2['curr_y'] + l2['padded_bbox_h'] / 2

                # Check for overlap
                x_overlap = max(0, min(l1_right, l2_right) - max(l1_left, l2_left))
                y_overlap = max(0, min(l1_top, l2_top) - max(l1_bottom, l2_bottom))

                if x_overlap > 0 and y_overlap > 0: # Overlap exists based on padded boxes
                    # Calculate shift amounts proportional to overlap and decay
                    # Apply a fraction (e.g., 0.5) of the overlap as the base shift
                    x_shift_amount = x_overlap * 0.5 * (LABEL_REPULSION_DECAY**iteration)
                    y_shift_amount = y_overlap * 0.5 * (LABEL_REPULSION_DECAY**iteration)

                    # Ensure minimum shift for very small overlaps to help escape local minima
                    # Threshold is in data units (inches)
                    min_shift = 0.001 # Absolute minimum shift in data units
                    x_shift_amount = max(x_shift_amount, min_shift if x_overlap > 1e-6 else 0) # Only apply min_shift if there's actual overlap
                    y_shift_amount = max(y_shift_amount, min_shift if y_overlap > 1e-6 else 0)


                    # Determine direction of shift based on relative position
                    # Shift l1 and l2 away from each other
                    if l1['curr_x'] > l2['curr_x']: # l1 is to the right of l2
                        l1['curr_x'] += x_shift_amount / 2
                        l2['curr_x'] -= x_shift_amount / 2
                    else: # l2 is to the right of l1 or they are aligned horizontally
                        l1['curr_x'] -= x_shift_amount / 2
                        l2['curr_x'] += x_shift_amount / 2

                    if l1['curr_y'] > l2['curr_y']: # l1 is above l2
                        l1['curr_y'] += y_shift_amount / 2
                        l2['curr_y'] -= y_shift_amount / 2
                    else: # l2 is above l1 or they are aligned vertically
                        l1['curr_y'] -= y_shift_amount / 2
                        l2['curr_y'] += y_shift_amount / 2

                    total_displacement += x_shift_amount + y_shift_amount # Sum absolute shifts

        # --- Attraction towards original position (bubble center) ---
        # This pull is applied after repulsion, scaled by the fixed attraction strength factor
        for label_data in label_list:
            pull_dx = label_data['orig_x'] - label_data['curr_x']
            pull_dy = label_data['orig_y'] - label_data['curr_y']

            # Apply a fraction of the pull distance
            label_data['curr_x'] += pull_dx * PARENT_ATTRACTION_STRENGTH
            label_data['curr_y'] += pull_dy * PARENT_ATTRACTION_STRENGTH

            # Add the magnitude of pull displacement to total displacement
            total_displacement += sqrt((pull_dx * PARENT_ATTRACTION_STRENGTH)**2 + (pull_dy * PARENT_ATTRACTION_STRENGTH)**2)


        # Check for convergence
        # Threshold is in data units (inches).
        if iteration > 20 and total_displacement < LABEL_SHIFT_THRESHOLD: # Add enough initial iterations before checking convergence
            # print(f"Label collision resolution converged after {iteration + 1} iterations.") # Comment out for cleaner output
            break

    # print(f"Label collision resolution finished after {iteration + 1} iterations. Final total displacement: {total_displacement:.4f}") # Comment out


def create_hierarchical_bubble_chart(csv_filepath: str, png_filepath: str, svg_filepath: str) -> bool:
    print(f"Processing {os.path.basename(csv_filepath)}...")
    try:
        df, metadata = parse_fill_bubble_csv(csv_filepath)

        if df.empty:
             print(f"Skipping file {csv_filepath}: No valid data points after parsing/cleaning.")
             return False

        # ... (节点初始化和层级构建不变) ...
        nodes: Dict[int, Dict[str, Any]] = {}
        for df_index, row in df.iterrows():
             node_id = df_index + 1 # Use 1-based index for node ID
             nodes[node_id] = {
                 'id': node_id, 'size': float(row['size']), 'original_size': float(row['size']),
                 'depth': int(row['depth']), 'label': str(row['label']).strip(), 'father': int(row['father']),
                 'children': [], 'radius': 0, 'x': 0, 'y': 0, 'radius_relative': 0, 'x_relative': 0, 'y_relative': 0
             }

        the_main_root_node: Optional[Dict[str, Any]] = None
        for node_data_val in nodes.values():
            if node_data_val['father'] == 0 and node_data_val['depth'] == 1:
                the_main_root_node = node_data_val
                if node_data_val['id'] != 1:
                     print(f"Warning: Main root found at ID {node_data_val['id']}, not ID 1, in {csv_filepath}.")
                break

        if not the_main_root_node:
            print(f"Critical Error: No main root node (father=0, depth=1) found in {csv_filepath}. Cannot create chart.")
            return False

        for node_id, node_data_val in nodes.items():
            if node_data_val['father'] != 0:
                parent_id = node_data_val['father']
                if parent_id in nodes:
                    nodes[parent_id]['children'].append(node_data_val)
                else:
                    print(f"Warning: Orphan node '{node_data_val['label']}' (ID: {node_id}) in {csv_filepath}. Parent ID {parent_id} not found.")

        calculate_layout(nodes, root_nodes_for_layout=[the_main_root_node])

        center_dx = -the_main_root_node['x']
        center_dy = -the_main_root_node['y']
        for node_data_val in nodes.values():
            node_data_val['x'] += center_dx
            node_data_val['y'] += center_dy

        all_coords_centered = [(n['x'], n['y'], n['radius']) for n in nodes.values() if n['radius'] > 0]
        min_x_layout, max_x_layout, min_y_layout, max_y_layout = 0,0,0,0
        if all_coords_centered:
             min_x_layout = min(x - r for (x, _, r) in all_coords_centered); max_x_layout = max(x + r for (x, _, r) in all_coords_centered)
             min_y_layout = min(y - r for (_, y, r) in all_coords_centered); max_y_layout = max(y + r for (_, y, r) in all_coords_centered)
             layout_width = max_x_layout - min_x_layout
             layout_height = max_y_layout - min_y_layout
        else:
             layout_width, layout_height = 100, 100

        # This is the target span for scaling the bubble *content*
        content_span_for_scaling_x = FIXED_FIG_WIDTH_INCHES * (1 - 2 * PLOT_MARGIN_FACTOR)
        content_span_for_scaling_y = FIXED_FIG_HEIGHT_INCHES * (1 - 2 * PLOT_MARGIN_FACTOR)

        scale_x = content_span_for_scaling_x / layout_width if layout_width > 1e-9 else 1.0
        scale_y = content_span_for_scaling_y / layout_height if layout_height > 1e-9 else 1.0
        visual_scale_calculated = min(scale_x, scale_y)
        if not np.isfinite(visual_scale_calculated) or visual_scale_calculated <= 0:
             visual_scale_calculated = 1.0
        print(f"Calculated visual_scale (to fit content into {content_span_for_scaling_x:.2f}x{content_span_for_scaling_y:.2f}): {visual_scale_calculated:.4f}")

        for node_data_val in nodes.values():
            node_data_val['x'] *= visual_scale_calculated
            node_data_val['y'] *= visual_scale_calculated
            node_data_val['radius'] *= visual_scale_calculated

        # Apply Global Bubble Shift (coordinates are now in inches, relative to the original (0,0) center of the content)
        for node_data_val in nodes.values():
             node_data_val['x'] += BUBBLE_SHIFT_X_INCHES
             node_data_val['y'] += BUBBLE_SHIFT_Y_INCHES
        print(f"Applied global bubble shift: ({BUBBLE_SHIFT_X_INCHES:.2f}, {BUBBLE_SHIFT_Y_INCHES:.2f}) inches.")

        # ... (字体大小和颜色分配不变) ...
        min_render_fontsize = 6
        for node_data_val in nodes.values():
            csv_depth = node_data_val['depth']
            node_data_val['fontsize'] = FIXED_FONT_SIZES.get(csv_depth, DEFAULT_FONT_SIZE)
            display_text = f"{node_data_val['label']}, {node_data_val['original_size']:.2f}"
            node_data_val['display_text'] = display_text
            node_data_val['fontsize'] = max(min_render_fontsize, node_data_val['fontsize'])

        depth_color_map: Dict[int, str] = {}
        for node_data_val in nodes.values():
            depth = node_data_val['depth']
            node_data_val['color'] = BRIGHT_COLOR_PALETTE.get(depth, '#CCCCCC')
            if depth not in depth_color_map:
                 depth_color_map[depth] = node_data_val['color']


        plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'figure.dpi': FIXED_DPI})
        fig, ax = plt.subplots(figsize=(FIXED_FIG_WIDTH_INCHES, FIXED_FIG_HEIGHT_INCHES))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')

        # OPTIMIZATION: Set Axes limits to cover the general figure area
        # This allows the scaled and shifted content to be placed within this broader view.
        fig_view_xlim_min = -FIXED_FIG_WIDTH_INCHES / 2
        fig_view_xlim_max =  FIXED_FIG_WIDTH_INCHES / 2
        fig_view_ylim_min = -FIXED_FIG_HEIGHT_INCHES / 2
        fig_view_ylim_max =  FIXED_FIG_HEIGHT_INCHES / 2

        ax.set_xlim(fig_view_xlim_min, fig_view_xlim_max)
        ax.set_ylim(fig_view_ylim_min, fig_view_ylim_max)
        print(f"Set broader Axes limits to X: [{fig_view_xlim_min:.2f}, {fig_view_xlim_max:.2f}], Y: [{fig_view_ylim_min:.2f}, {fig_view_ylim_max:.2f}] inches.")

        ax.set_aspect('equal', adjustable='box'); ax.axis('off')

        # ... (绘制气泡，准备和解析标签的代码不变) ...
        nodes_to_draw = sorted(nodes.values(), key=lambda n: n['depth'])
        for node_data_val in nodes_to_draw:
            if node_data_val['radius'] <= 0: continue
            ax.add_patch(patches.Circle(
                (node_data_val['x'], node_data_val['y']), radius=node_data_val['radius'],
                facecolor=node_data_val.get('color', '#cccccc'), edgecolor='#333333',
                linewidth=0.75, alpha=0.8, zorder=node_data_val['depth']
            ))

        label_list = []
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for node_data_val in nodes_to_draw:
             current_fontsize = node_data_val.get('fontsize', 0)
             display_text = node_data_val.get('display_text', '')
             if display_text and current_fontsize > 0:
                 dummy_text = ax.text(0, 0, display_text, fontsize=current_fontsize, ha='center', va='center', wrap=False)
                 try:
                     dummy_bbox_display = dummy_text.get_window_extent(renderer)
                     dummy_bbox_data = dummy_bbox_display.transformed(ax.transData.inverted())
                     estimated_text_width_data = dummy_bbox_data.width
                     estimated_text_height_data = dummy_bbox_data.height
                 except Exception:
                     estimated_text_width_data = estimate_text_bbox_size_heuristic(display_text, current_fontsize)[0] / 72.0
                     estimated_text_height_data = estimate_text_bbox_size_heuristic(display_text, current_fontsize)[1] / 72.0
                     print(f"Warning: Failed to get precise bbox for label '{display_text}', using heuristic fallback.")
                 dummy_text.remove()

                 label_list.append({
                     'node_id': node_data_val['id'],
                     'orig_x': node_data_val['x'], 'orig_y': node_data_val['y'],
                     'curr_x': node_data_val['x'], 'curr_y': node_data_val['y'],
                     'text': display_text, 'fontsize': current_fontsize,
                     'bbox_w': estimated_text_width_data, 'bbox_h': estimated_text_height_data,
                     'depth': node_data_val['depth']
                 })

        if label_list:
           resolve_label_collisions(label_list, visual_scale_calculated)

        label_list_to_draw = sorted(label_list, key=lambda l: l['depth'])
        for label_data in label_list_to_draw:
            # Draw the text artist with white stroke effect
            text_artist = ax.text(
                label_data['curr_x'], label_data['curr_y'], label_data['text'],
                ha='center', va='center', fontsize=label_data['fontsize'],
                color='#000000', zorder=label_data['depth'] + 100,
                wrap=False,
                path_effects=[path_effects.withStroke(linewidth=1.2, foreground='white')]
            )
            try:
                # Get the actual bounding box of the drawn text artist
                bbox_display = text_artist.get_window_extent(renderer)
                bbox_data = bbox_display.transformed(ax.transData.inverted())
                # Draw a rectangle matching the bounding box
                bbox_rect = patches.Rectangle(
                    (bbox_data.x0, bbox_data.y0), bbox_data.width, bbox_data.height,
                    linewidth=0.5,
                    # --- MODIFICATION START ---
                    # Change edgecolor to 'none' to hide the border around the label.
                    # The alpha=0.3 was for the facecolor (which is 'none'), but setting edgecolor='none'
                    # explicitly removes the border.
                    edgecolor='none', # Set edge color to none to hide the border
                    # --- MODIFICATION END ---
                    facecolor='none', alpha=0.3,
                    zorder=label_data['depth'] + 99
                )
                ax.add_patch(bbox_rect)
            except Exception as e:
                 print(f"Warning: Could not get or draw bbox for label '{label_data['text']}': {e}")
                 pass

        # ax.set_aspect('equal', adjustable='box'); ax.axis('off') # Moved earlier

        legend_elements: List[patches.Patch] = []
        present_depths = sorted(list(set(n['depth'] for n in nodes.values() if n['depth'] in BRIGHT_COLOR_PALETTE)))
        for depth_val in present_depths:
            color = BRIGHT_COLOR_PALETTE[depth_val]
            layer_label = f"Layer {depth_val}"
            legend_elements.append(patches.Patch(facecolor=color, edgecolor='#333333', label=layer_label))

        if legend_elements:
            base_legend_fontsize = FIXED_FONT_SIZES.get(3, DEFAULT_FONT_SIZE)
            legend_fontsize = base_legend_fontsize * 0.9
            legend_title_fontsize = base_legend_fontsize * 1.1
            ax.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(1.0 + LEGEND_ANCHOR_OFFSET_X, 1.0 + LEGEND_ANCHOR_OFFSET_Y),
                      bbox_transform=ax.transAxes, title="Hierarchy Layers",
                      fontsize=max(6, legend_fontsize), title_fontsize=max(7, legend_title_fontsize),
                      frameon=True, facecolor='#FFFFF0', edgecolor='gray', framealpha=0.8)
            print(f"Adjusted legend anchor by ({LEGEND_ANCHOR_OFFSET_X:.2f}, {LEGEND_ANCHOR_OFFSET_Y:.2f}) relative to axes upper right.")

        base_chart_title = metadata.get('Little_Theme', 'Hierarchical Bubble Chart')
        chart_title = base_chart_title
        title_fontsize = FIXED_FONT_SIZES.get(1, DEFAULT_FONT_SIZE) * 1.2
        ax.set_title(chart_title, fontsize=max(10, title_fontsize), pad=15)

        try:
            plt.savefig(png_filepath, facecolor='white')
        except Exception as e:
            print(f"Error saving PNG to {png_filepath}: {e}")
        try:
            plt.savefig(svg_filepath, format='svg', facecolor='white')
        except Exception as e:
             print(f"Error saving SVG to {svg_filepath}: {e}")

        plt.close(fig)
        print(f"Successfully created visualization for {os.path.basename(csv_filepath)}")
        return True
    except FileNotFoundError: print(f"Error: CSV file not found at {csv_filepath}")
    except ValueError as e: print(f"Skipping file {csv_filepath} due to data format issue: {e}\nDetails: {e}")
    except Exception as e:
        print(f"Skipping file {csv_filepath} due to unexpected error: {e}")
        traceback.print_exc()
        return False

def process_all_csv_files(input_dir: str, png_output_dir: str, svg_output_dir: str):
    """
    Visualize all fill_bubble CSV files in a directory and save to specified output directories.

    Args:
        input_dir: Directory containing CSV files.
        png_output_dir: Directory to save PNG files.
        svg_output_dir: Directory to save SVG files.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    files_to_process = [f for f in os.listdir(input_dir) if re.match(r'fill_bubble_.*_\d+\.csv$', f, re.IGNORECASE) and not f.startswith('.')]

    if not files_to_process:
        print(f"No 'fill_bubble_*_XX.csv' files found in {input_dir}")
        return

    print(f"Found {len(files_to_process)} 'fill_bubble_*_XX.csv' file(s) to process in {input_dir}")

    try:
        def sort_key(f):
            base = os.path.basename(f)
            match = re.search(r'_(\d+)\.csv$', base, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return base
        files_to_process.sort(key=sort_key)
    except Exception:
         files_to_process.sort()
         print("Warning: Could not sort files numerically based on suffix. Sorting alphabetically.")

    os.makedirs(png_output_dir, exist_ok=True)
    os.makedirs(svg_output_dir, exist_ok=True)

    for filename in files_to_process:
        csv_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        png_output_path = os.path.join(png_output_dir, f"{base_name}.png")
        svg_output_path = os.path.join(svg_output_dir, f"{base_name}.svg")

        create_hierarchical_bubble_chart(
            csv_filepath=csv_path,
            png_filepath=png_output_path,
            svg_filepath=svg_output_path
        )

    print("\nVisualization process completed.")


# --- Main Execution ---
if __name__ == "__main__":
    INPUT_CSV_DIR = './fill_bubble/csv'
    OUTPUT_PNG_DIR = './fill_bubble/png'
    OUTPUT_SVG_DIR = './fill_bubble/svg'

    os.makedirs(INPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SVG_DIR, exist_ok=True)

    process_all_csv_files(
        input_dir=INPUT_CSV_DIR,
        png_output_dir=OUTPUT_PNG_DIR,
        svg_output_dir=OUTPUT_SVG_DIR
    )
