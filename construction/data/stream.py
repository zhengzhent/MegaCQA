# line.py

import random
import os
import csv
import math
from typing import List, Dict, Tuple

# Attempt to import THEME_METRIC_PARAMS from data.py
# This requires data.py to be in the same directory as this script
try:
    from data import THEME_METRIC_PARAMS
except ImportError:
    print("Error: data.py not found or THEME_METRIC_PARAMS not defined in data.py.")
    print("Please ensure data.py is in the same directory and contains the THEME_METRIC_PARAMS dictionary.")
    # Exit the script if the essential data cannot be imported
    exit()
except NameError:
    # This might happen if data.py exists but THEME_METRIC_PARAMS is missing
    print("Error: THEME_METRIC_PARAMS dictionary not found within data.py.")
    print("Please ensure the dictionary is correctly defined in data.py.")
    exit()

TIME_RANGE = {"start_year": 2015, "end_year": 2024}

# Patterns available for synthetic time‑series
AVAILABLE_PATTERNS = [
    "stable_rising",
    "stable_falling",
    "exponential_rising",
    "exponential_falling",
    "periodic_stable",
    "volatile_rising",
    "volatile_falling",
]


def simulate_time_series_metric(years: List[int], params: Dict, override_pattern: str | None = None) -> List[float]:
    """
    Enhanced simulation function supporting multiple patterns.

    `override_pattern` allows callers to force a specific pattern.
    Parameters like growth, noise, period, amplitude will use defaults if not
    provided in `params` *and* are needed by the chosen pattern.

    Args:
        years: List of years for the time series.
        params: Dictionary containing base, noise, and potentially growth, period, amplitude.
        override_pattern: Optional pattern name to force.

    Returns:
        List of simulated data values for the given years.
    """
    values = []
    # Get base and noise from params, using defaults if not provided (though generate_series should provide these)
    current_value = params.get("base", 100.0)  # Use float default
    noise_level = params.get("noise", 0.1)  # Default noise if not specified

    # Determine the pattern to use
    # Use override_pattern if provided, otherwise get from params (less likely now), or default
    pattern = override_pattern if override_pattern else params.get("pattern", "stable_rising")

    # Get other parameters needed by specific patterns, using defaults if not in params
    # These defaults are important because generate_series might only pass base and noise
    growth_rate = params.get("growth", 0.01)  # Default growth for stable/volatile/exponential
    period = params.get("period", 5)  # Default period for periodic
    # Default amplitude for periodic, based on initial value or a fixed value if base is 0
    amplitude = params.get("amplitude", params.get("base", 100.0) * 0.1 if params.get("base", 100.0) != 0 else 10.0)

    for i, _ in enumerate(years):
        # Store or calculate the value before adding noise for pattern logic
        value_before_noise = current_value

        # Apply base pattern logic
        if pattern == "stable_rising":
            # Uses the growth_rate parameter
            growth = current_value * growth_rate * random.uniform(0.8, 1.2)
            value_before_noise = current_value + growth
        elif pattern == "stable_falling":
            # Uses the growth_rate parameter
            decline = current_value * abs(growth_rate) * random.uniform(0.8, 1.2)
            value_before_noise = current_value - decline
        elif pattern == "exponential_rising":
            # Uses the growth_rate parameter and base
            base_val_exp = params.get("base", 100.0)  # Ensure using the original base
            # Exponential growth is calculated based on the initial base and year index
            # Ensure growth_rate is positive for rising
            value_before_noise = base_val_exp * (1 + max(0.001, growth_rate)) ** i
        elif pattern == "exponential_falling":
            # Uses the growth_rate parameter and base
            base_val_exp = params.get("base", 100.0)  # Ensure using the original base
            # Exponential decay is calculated based on the initial base and year index
            # Ensure growth_rate is negative for falling, use abs()
            value_before_noise = base_val_exp * (1 - abs(growth_rate)) ** i
        elif pattern == "periodic_stable":
            # Uses period, amplitude, and base
            base_val_periodic = params.get("base", 100.0)  # Ensure using the original base
            # Ensure using original amplitude or default, handle base=0 case
            amp = params.get("amplitude", base_val_periodic * 0.1 if base_val_periodic != 0 else 10.0)
            cycle = math.sin(2 * math.pi * i / period) if period != 0 else 0
            value_before_noise = base_val_periodic + amp * cycle
        elif pattern == "volatile_rising":
            # Uses the growth_rate parameter with wider random range
            growth = current_value * growth_rate * random.uniform(0.5, 1.5)
            value_before_noise = current_value + growth
        elif pattern == "volatile_falling":
            # Uses the growth_rate parameter with wider random range
            decline = current_value * abs(growth_rate) * random.uniform(0.5, 1.5)
            value_before_noise = current_value - decline
        else:
            # Fallback to stable rising if pattern unknown (shouldn't happen with random.choice)
            print(
                f"Warning: Unsupported pattern '{pattern}' detected during simulation, defaulting to 'stable_rising'.")
            growth = current_value * growth_rate * random.uniform(0.8, 1.2)
            value_before_noise = current_value + growth
            # Do not change 'pattern' variable here, it's the one being processed

        # Update current_value for the next iteration's base calculation (for stable/volatile)
        # For exponential/periodic, the next value is calculated from the original base and index i+1
        # So, only update current_value if the pattern is stable or volatile
        if pattern in ["stable_rising", "stable_falling", "volatile_rising", "volatile_falling"]:
            current_value = value_before_noise  # This value will be used as the base for the next year's growth calculation

        # Add noise based on noise_level from params
        # Avoid division by zero or issues with zero current_value if noise is relative
        # Noise is applied *after* the pattern calculation for the current year
        # Use a base value for noise calculation if the calculated value before noise is 0
        noise_base = value_before_noise if value_before_noise != 0 else params.get("base", 100.0)
        noise = random.normalvariate(0, noise_level * abs(noise_base))

        final_value = value_before_noise + noise

        final_value = max(0.0, final_value)  # Ensure value doesn't go below 0
        values.append(final_value)

    return values


def generate_series_for_single_metric(years: List[int], metric: Dict) -> Tuple[List[Dict], List[List[float]], str]:
    """
    Generate between 2‑7 parallel time‑series for the SAME metric.
    Each series gets a RANDOM pattern from AVAILABLE_PATTERNS.
    Base and Noise are taken from the metric definition in data.py.

    Args:
        years: List of years for the time series.
        metric: The metric dictionary from THEME_METRIC_PARAMS (contains base, noise).

    Returns:
        Tuple containing:
          - metric_info: List of dictionaries, one for each generated series (name, unit, pattern).
          - time_series_matrix: List of lists, where each inner list is a time series.
          - subject: The subject name for this metric.
    """
    # --- MODIFIED: Generate between 2 and 7 series ---
    num_series = random.randint(2, 7)
    # --- END MODIFIED ---
    metric_info: List[Dict] = []
    time_series_matrix: List[List[float]] = []

    # Get required attributes with defaults from the metric dictionary from data.py
    subject = metric.get('subject', 'item')
    metric_name = metric.get('name', 'Metric')
    metric_unit = metric.get('unit', '')

    # Get base and noise from the metric's params in data.py
    metric_params = metric.get("params", {})
    base_from_data = metric_params.get("base", 100.0)  # Use float default
    noise_from_data = metric_params.get("noise", 0.1)  # Default noise if missing

    # Optional: Get growth, period, amplitude from data.py if they exist,
    # but these will be overridden by simulate_time_series_metric's defaults
    # unless the random pattern logic is changed to use them.
    # For this request, we rely on simulate_time_series_metric's defaults for
    # growth, period, and amplitude based on the *randomly chosen pattern*.

    for idx in range(num_series):
        # Choose a random growth pattern for THIS series, ignoring the pattern in data.py
        pattern_for_this_series = random.choice(AVAILABLE_PATTERNS)

        # Create a params dictionary for simulate_time_series_metric
        # This dict ONLY contains base and noise from data.py (with noise randomization)
        # simulate_time_series_metric will use its internal defaults for
        # growth, period, and amplitude based on pattern_for_this_series.
        params_for_simulation = {
            "base": base_from_data,
            # Slightly randomize noise level for this specific line
            "noise": noise_from_data * random.uniform(0.8, 1.2),
            # We could optionally pass growth, period, amplitude from data.py here,
            # but simulate_time_series_metric's internal defaults for the *chosen pattern*
            # are more appropriate given the requirement to ignore data.py's pattern.
            # E.g., a "stable_rising" in data.py might have growth=0.05, but if we
            # randomly pick "periodic_stable", growth=0.05 is irrelevant for the cycle.
            # Relying on simulate_time_series_metric's defaults for pattern characteristics
            # (like growth=0.01, period=5, amplitude=base*0.1) aligns better with
            # the idea of a *randomly chosen* pattern determining the shape.
        }

        # Generate series values using the randomly chosen pattern and params from data.py
        series_values = simulate_time_series_metric(years, params_for_simulation,
                                                    override_pattern=pattern_for_this_series)
        time_series_matrix.append(series_values)

        # Construct name as "[Metric Name] – [Subject Name] [Index]"
        series_name = f"{metric_name} – {subject} {idx + 1}"

        metric_info.append(
            {
                "name": series_name,
                "unit": metric_unit,
                "data_type": "currency" if (
                            "$" in metric_unit or "€" in metric_unit or "£" in metric_unit) else "count",
                # Basic currency check
                "pattern": pattern_for_this_series  # Record the RANDOMLY chosen pattern
            }
        )

    return metric_info, time_series_matrix, subject  # Return subject


# The rest of the functions (generate_theme_data, main) remain largely unchanged
# as their logic for selecting a metric and writing to CSV is still correct.
# generate_theme_data will call generate_series_for_single_metric, which now
# implements the desired random pattern logic per series and the new series count range.

def generate_theme_data(years: List[int], theme_name: str) -> Tuple[str, str, List[Dict], List[Dict]]:
    """
    Generates data for a theme, including theme name, unit, metric info, and yearly rows.
    Randomly selects ONE metric from the theme to generate series for.

    Args:
        years: List of years for the time series.
        theme_name: Name of the theme to generate data for.

    Returns:
        Tuple containing:
         - theme_name: The input theme name.
         - unit: The unit of the selected metric.
         - metric_info: List of dictionaries describing each generated series.
         - formatted_rows: List of dictionaries, each representing a year's data.
    """
    # Retrieve metrics for the theme from the central dictionary
    if theme_name not in THEME_METRIC_PARAMS:
        raise ValueError(f"Theme '{theme_name}' not found in THEME_METRIC_PARAMS (data.py).")

    theme_metrics = THEME_METRIC_PARAMS[theme_name]
    if not theme_metrics:
        # Handle case where a theme exists but has no metrics listed
        raise ValueError(f"No metrics listed under theme '{theme_name}' in data.py.")

    # Randomly select **one** metric definition from the theme
    # All generated series for this theme will be based on this single metric's
    # name, unit, subject, base, and noise, but will have random patterns.
    selected_metric = random.choice(theme_metrics)

    # Generate the series data using the selected metric's definition
    # The returned 'subject' from generate_series_for_single_metric is not needed here, hence '_'
    metric_info, time_series_matrix, _ = generate_series_for_single_metric(years, selected_metric)

    # Get unit from the selected metric definition
    unit = selected_metric.get('unit', 'N/A')

    # Format data into yearly records
    formatted_rows: List[Dict] = []
    for year_idx, year in enumerate(years):
        row = {"Year": year}
        for col_idx, info in enumerate(metric_info):
            series_name = info["name"]
            # Ensure year_idx is within bounds for the time_series_matrix
            # Check col_idx as well, although it should align with metric_info
            if col_idx < len(time_series_matrix) and year_idx < len(time_series_matrix[col_idx]):
                value = time_series_matrix[col_idx][year_idx]
                # Round appropriately based on data type and unit
                if info["data_type"] == "currency":
                    row[series_name] = round(value, 2)
                elif "%" in info["unit"]:
                    row[series_name] = round(value, 2)  # Keep precision for percentages
                elif value < 100 and value != 0 and info[
                    "data_type"] != "currency":  # Keep precision for small indices/ratios etc. (but not zero)
                    # Check if it looks like an integer despite being small
                    if abs(value - round(value)) < 0.001:
                        row[series_name] = int(round(value))
                    else:
                        row[series_name] = round(value, 2)
                else:  # Assume count or larger index/value, round to integer
                    row[series_name] = int(round(value))
            else:
                # Handle potential mismatch if generation failed for a series/year
                row[series_name] = ''  # Use empty string as placeholder

        formatted_rows.append(row)

    return theme_name, unit, metric_info, formatted_rows


def main():
    """
    Main function to generate CSV files for each theme defined in data.py.
    """
    # --- MODIFIED: Define output directory name ---
    output_dir = "csv/stream"
    # --- END MODIFIED ---
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    years_to_generate = list(range(TIME_RANGE["start_year"], TIME_RANGE["end_year"] + 1))
    print(f"Generating time‑series CSV files in '{output_dir}' with specified format...")

    # Check if the imported dictionary is usable
    if not THEME_METRIC_PARAMS or not isinstance(THEME_METRIC_PARAMS, dict):
        print("Error: THEME_METRIC_PARAMS imported from data.py is empty or not a dictionary.")
        return  # Stop execution if data is unusable

    # Iterate through each theme defined in the dictionary
    for theme_idx, current_theme_name in enumerate(THEME_METRIC_PARAMS.keys(), start=1):
        # Generate a file-system friendly name from the theme name (not used in final name format)
        # safe_theme_name = "".join(c if c.isalnum() else "_" for c in current_theme_name).lower() # This line is no longer strictly needed for the filename format

        # --- MODIFIED: Define output file name format ---
        file_name = f"stream_chart_{theme_idx}.csv"
        # --- END MODIFIED ---
        file_path = os.path.join(output_dir, file_name)

        try:
            # Generate data for the current theme using definitions from data.py
            # This will now generate series with random patterns based on the selected metric's base/noise
            theme_name, unit, metric_info, yearly_rows = generate_theme_data(years_to_generate, current_theme_name)

            # Check if data generation yielded results
            if not metric_info or not yearly_rows:
                print(f"⚠️ No data generated for theme '{current_theme_name}', skipping file.")
                continue  # Skip to the next theme if no data

            # --- CSV Writing Logic ---
            # 1. Prepare Headers according to the requested format
            header1_str = f"Theme: {theme_name},Unit: {unit}"  # Line 1: Theme and Unit
            header_trends = ["trend"] + [m["pattern"] for m in
                                         metric_info]  # Line 2: Trend patterns (now reflects random patterns)
            header_names = ["Year"] + [m["name"] for m in
                                       metric_info]  # Line 3: Year and Series Names ('Metric - Subject Index')

            # 2. Write to CSV file
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                # Write the custom first header line directly
                f.write(header1_str + "\n")

                # Use csv.writer for the remaining headers and data for proper quoting
                writer = csv.writer(f)

                # Write the trend header (Line 2)
                writer.writerow(header_trends)
                # Write the name header (Line 3)
                writer.writerow(header_names)

                # Write data rows
                for row_dict in yearly_rows:
                    # Create row values in the correct order based on header_names
                    # Use row_dict.get(col, '') to handle potential missing keys gracefully
                    row_values = [row_dict.get(col, '') for col in header_names]
                    writer.writerow(row_values)
            # --- End CSV Writing ---

            print(f"✔ Generated {file_path}")

        except ValueError as ve:  # Catch specific configuration errors from generate_theme_data
            print(f"✘ Skipping theme '{current_theme_name}' due to configuration error: {ve}")
        except Exception as exc:  # Catch other unexpected errors during generation/writing for this theme
            print(f"✘ Error processing theme '{current_theme_name}': {exc}")
            # Uncomment the lines below for a detailed traceback if needed for debugging
            # import traceback
            # print(traceback.format_exc())

    print("\nCSV generation process completed.")


if __name__ == "__main__":
    # This block ensures the main function runs only when the script is executed directly
    main()
# line.py

import random
import os
import csv
import math
from typing import List, Dict, Tuple

# Attempt to import THEME_METRIC_PARAMS from data.py
# This requires data.py to be in the same directory as this script
try:
    from data import THEME_METRIC_PARAMS
except ImportError:
    print("Error: data.py not found or THEME_METRIC_PARAMS not defined in data.py.")
    print("Please ensure data.py is in the same directory and contains the THEME_METRIC_PARAMS dictionary.")
    # Exit the script if the essential data cannot be imported
    exit()
except NameError:
    # This might happen if data.py exists but THEME_METRIC_PARAMS is missing
    print("Error: THEME_METRIC_PARAMS dictionary not found within data.py.")
    print("Please ensure the dictionary is correctly defined in data.py.")
    exit()

TIME_RANGE = {"start_year": 2015, "end_year": 2024}

# Patterns available for synthetic time‑series
AVAILABLE_PATTERNS = [
    "stable_rising",
    "stable_falling",
    "exponential_rising",
    "exponential_falling",
    "periodic_stable",
    "volatile_rising",
    "volatile_falling",
]


def simulate_time_series_metric(years: List[int], params: Dict, override_pattern: str | None = None) -> List[float]:
    """
    Enhanced simulation function supporting multiple patterns.

    `override_pattern` allows callers to force a specific pattern.
    Parameters like growth, noise, period, amplitude will use defaults if not
    provided in `params` *and* are needed by the chosen pattern.

    Args:
        years: List of years for the time series.
        params: Dictionary containing base, noise, and potentially growth, period, amplitude.
        override_pattern: Optional pattern name to force.

    Returns:
        List of simulated data values for the given years.
    """
    values = []
    # Get base and noise from params, using defaults if not provided (though generate_series should provide these)
    current_value = params.get("base", 100.0)  # Use float default
    noise_level = params.get("noise", 0.1)  # Default noise if not specified

    # Determine the pattern to use
    # Use override_pattern if provided, otherwise get from params (less likely now), or default
    pattern = override_pattern if override_pattern else params.get("pattern", "stable_rising")

    # Get other parameters needed by specific patterns, using defaults if not in params
    # These defaults are important because generate_series might only pass base and noise
    growth_rate = params.get("growth", 0.01)  # Default growth for stable/volatile/exponential
    period = params.get("period", 5)  # Default period for periodic
    # Default amplitude for periodic, based on initial value or a fixed value if base is 0
    amplitude = params.get("amplitude", params.get("base", 100.0) * 0.1 if params.get("base", 100.0) != 0 else 10.0)

    for i, _ in enumerate(years):
        # Store or calculate the value before adding noise for pattern logic
        value_before_noise = current_value

        # Apply base pattern logic
        if pattern == "stable_rising":
            # Uses the growth_rate parameter
            growth = current_value * growth_rate * random.uniform(0.8, 1.2)
            value_before_noise = current_value + growth
        elif pattern == "stable_falling":
            # Uses the growth_rate parameter
            decline = current_value * abs(growth_rate) * random.uniform(0.8, 1.2)
            value_before_noise = current_value - decline
        elif pattern == "exponential_rising":
            # Uses the growth_rate parameter and base
            base_val_exp = params.get("base", 100.0)  # Ensure using the original base
            # Exponential growth is calculated based on the initial base and year index
            # Ensure growth_rate is positive for rising
            value_before_noise = base_val_exp * (1 + max(0.001, growth_rate)) ** i
        elif pattern == "exponential_falling":
            # Uses the growth_rate parameter and base
            base_val_exp = params.get("base", 100.0)  # Ensure using the original base
            # Exponential decay is calculated based on the initial base and year index
            # Ensure growth_rate is negative for falling, use abs()
            value_before_noise = base_val_exp * (1 - abs(growth_rate)) ** i
        elif pattern == "periodic_stable":
            # Uses period, amplitude, and base
            base_val_periodic = params.get("base", 100.0)  # Ensure using the original base
            # Ensure using original amplitude or default, handle base=0 case
            amp = params.get("amplitude", base_val_periodic * 0.1 if base_val_periodic != 0 else 10.0)
            cycle = math.sin(2 * math.pi * i / period) if period != 0 else 0
            value_before_noise = base_val_periodic + amp * cycle
        elif pattern == "volatile_rising":
            # Uses the growth_rate parameter with wider random range
            growth = current_value * growth_rate * random.uniform(0.5, 1.5)
            value_before_noise = current_value + growth
        elif pattern == "volatile_falling":
            # Uses the growth_rate parameter with wider random range
            decline = current_value * abs(growth_rate) * random.uniform(0.5, 1.5)
            value_before_noise = current_value - decline
        else:
            # Fallback to stable rising if pattern unknown (shouldn't happen with random.choice)
            print(
                f"Warning: Unsupported pattern '{pattern}' detected during simulation, defaulting to 'stable_rising'.")
            growth = current_value * growth_rate * random.uniform(0.8, 1.2)
            value_before_noise = current_value + growth
            # Do not change 'pattern' variable here, it's the one being processed

        # Update current_value for the next iteration's base calculation (for stable/volatile)
        # For exponential/periodic, the next value is calculated from the original base and index i+1
        # So, only update current_value if the pattern is stable or volatile
        if pattern in ["stable_rising", "stable_falling", "volatile_rising", "volatile_falling"]:
            current_value = value_before_noise  # This value will be used as the base for the next year's growth calculation

        # Add noise based on noise_level from params
        # Avoid division by zero or issues with zero current_value if noise is relative
        # Noise is applied *after* the pattern calculation for the current year
        # Use a base value for noise calculation if the calculated value before noise is 0
        noise_base = value_before_noise if value_before_noise != 0 else params.get("base", 100.0)
        noise = random.normalvariate(0, noise_level * abs(noise_base))

        final_value = value_before_noise + noise

        final_value = max(0.0, final_value)  # Ensure value doesn't go below 0
        values.append(final_value)

    return values


def generate_series_for_single_metric(years: List[int], metric: Dict) -> Tuple[List[Dict], List[List[float]], str]:
    """
    Generate between 2‑7 parallel time‑series for the SAME metric.
    Each series gets a RANDOM pattern from AVAILABLE_PATTERNS.
    Base and Noise are taken from the metric definition in data.py.

    Args:
        years: List of years for the time series.
        metric: The metric dictionary from THEME_METRIC_PARAMS (contains base, noise).

    Returns:
        Tuple containing:
          - metric_info: List of dictionaries, one for each generated series (name, unit, pattern).
          - time_series_matrix: List of lists, where each inner list is a time series.
          - subject: The subject name for this metric.
    """
    # --- MODIFIED: Generate between 2 and 7 series ---
    num_series = random.randint(2, 7)
    # --- END MODIFIED ---
    metric_info: List[Dict] = []
    time_series_matrix: List[List[float]] = []

    # Get required attributes with defaults from the metric dictionary from data.py
    subject = metric.get('subject', 'item')
    metric_name = metric.get('name', 'Metric')
    metric_unit = metric.get('unit', '')

    # Get base and noise from the metric's params in data.py
    metric_params = metric.get("params", {})
    base_from_data = metric_params.get("base", 100.0)  # Use float default
    noise_from_data = metric_params.get("noise", 0.1)  # Default noise if missing

    # Optional: Get growth, period, amplitude from data.py if they exist,
    # but these will be overridden by simulate_time_series_metric's defaults
    # unless the random pattern logic is changed to use them.
    # For this request, we rely on simulate_time_series_metric's defaults for
    # growth, period, and amplitude based on the *randomly chosen pattern*.

    for idx in range(num_series):
        # Choose a random growth pattern for THIS series, ignoring the pattern in data.py
        pattern_for_this_series = random.choice(AVAILABLE_PATTERNS)

        # Create a params dictionary for simulate_time_series_metric
        # This dict ONLY contains base and noise from data.py (with noise randomization)
        # simulate_time_series_metric will use its internal defaults for
        # growth, period, and amplitude based on pattern_for_this_series.
        params_for_simulation = {
            "base": base_from_data,
            # Slightly randomize noise level for this specific line
            "noise": noise_from_data * random.uniform(0.8, 1.2),
            # We could optionally pass growth, period, amplitude from data.py here,
            # but simulate_time_series_metric's internal defaults for the *chosen pattern*
            # are more appropriate given the requirement to ignore data.py's pattern.
            # E.g., a "stable_rising" in data.py might have growth=0.05, but if we
            # randomly pick "periodic_stable", growth=0.05 is irrelevant for the cycle.
            # Relying on simulate_time_series_metric's defaults for pattern characteristics
            # (like growth=0.01, period=5, amplitude=base*0.1) aligns better with
            # the idea of a *randomly chosen* pattern determining the shape.
        }

        # Generate series values using the randomly chosen pattern and params from data.py
        series_values = simulate_time_series_metric(years, params_for_simulation,
                                                    override_pattern=pattern_for_this_series)
        time_series_matrix.append(series_values)

        # Construct name as "[Metric Name] – [Subject Name] [Index]"
        series_name = f"{metric_name} – {subject} {idx + 1}"

        metric_info.append(
            {
                "name": series_name,
                "unit": metric_unit,
                "data_type": "currency" if (
                            "$" in metric_unit or "€" in metric_unit or "£" in metric_unit) else "count",
                # Basic currency check
                "pattern": pattern_for_this_series  # Record the RANDOMLY chosen pattern
            }
        )

    return metric_info, time_series_matrix, subject  # Return subject


# The rest of the functions (generate_theme_data, main) remain largely unchanged
# as their logic for selecting a metric and writing to CSV is still correct.
# generate_theme_data will call generate_series_for_single_metric, which now
# implements the desired random pattern logic per series and the new series count range.

def generate_theme_data(years: List[int], theme_name: str) -> Tuple[str, str, List[Dict], List[Dict]]:
    """
    Generates data for a theme, including theme name, unit, metric info, and yearly rows.
    Randomly selects ONE metric from the theme to generate series for.

    Args:
        years: List of years for the time series.
        theme_name: Name of the theme to generate data for.

    Returns:
        Tuple containing:
         - theme_name: The input theme name.
         - unit: The unit of the selected metric.
         - metric_info: List of dictionaries describing each generated series.
         - formatted_rows: List of dictionaries, each representing a year's data.
    """
    # Retrieve metrics for the theme from the central dictionary
    if theme_name not in THEME_METRIC_PARAMS:
        raise ValueError(f"Theme '{theme_name}' not found in THEME_METRIC_PARAMS (data.py).")

    theme_metrics = THEME_METRIC_PARAMS[theme_name]
    if not theme_metrics:
        # Handle case where a theme exists but has no metrics listed
        raise ValueError(f"No metrics listed under theme '{theme_name}' in data.py.")

    # Randomly select **one** metric definition from the theme
    # All generated series for this theme will be based on this single metric's
    # name, unit, subject, base, and noise, but will have random patterns.
    selected_metric = random.choice(theme_metrics)

    # Generate the series data using the selected metric's definition
    # The returned 'subject' from generate_series_for_single_metric is not needed here, hence '_'
    metric_info, time_series_matrix, _ = generate_series_for_single_metric(years, selected_metric)

    # Get unit from the selected metric definition
    unit = selected_metric.get('unit', 'N/A')

    # Format data into yearly records
    formatted_rows: List[Dict] = []
    for year_idx, year in enumerate(years):
        row = {"Year": year}
        for col_idx, info in enumerate(metric_info):
            series_name = info["name"]
            # Ensure year_idx is within bounds for the time_series_matrix
            # Check col_idx as well, although it should align with metric_info
            if col_idx < len(time_series_matrix) and year_idx < len(time_series_matrix[col_idx]):
                value = time_series_matrix[col_idx][year_idx]
                # Round appropriately based on data type and unit
                if info["data_type"] == "currency":
                    row[series_name] = round(value, 2)
                elif "%" in info["unit"]:
                    row[series_name] = round(value, 2)  # Keep precision for percentages
                elif value < 100 and value != 0 and info[
                    "data_type"] != "currency":  # Keep precision for small indices/ratios etc. (but not zero)
                    # Check if it looks like an integer despite being small
                    if abs(value - round(value)) < 0.001:
                        row[series_name] = int(round(value))
                    else:
                        row[series_name] = round(value, 2)
                else:  # Assume count or larger index/value, round to integer
                    row[series_name] = int(round(value))
            else:
                # Handle potential mismatch if generation failed for a series/year
                row[series_name] = ''  # Use empty string as placeholder

        formatted_rows.append(row)

    return theme_name, unit, metric_info, formatted_rows


def main():
    """
    Main function to generate CSV files for each theme defined in data.py.
    """
    # --- MODIFIED: Define output directory name ---
    output_dir = "csv/stream"
    # --- END MODIFIED ---
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    years_to_generate = list(range(TIME_RANGE["start_year"], TIME_RANGE["end_year"] + 1))
    print(f"Generating time‑series CSV files in '{output_dir}' with specified format...")

    # Check if the imported dictionary is usable
    if not THEME_METRIC_PARAMS or not isinstance(THEME_METRIC_PARAMS, dict):
        print("Error: THEME_METRIC_PARAMS imported from data.py is empty or not a dictionary.")
        return  # Stop execution if data is unusable

    # Iterate through each theme defined in the dictionary
    for theme_idx, current_theme_name in enumerate(THEME_METRIC_PARAMS.keys(), start=1):
        # Generate a file-system friendly name from the theme name (not used in final name format)
        # safe_theme_name = "".join(c if c.isalnum() else "_" for c in current_theme_name).lower() # This line is no longer strictly needed for the filename format

        # --- MODIFIED: Define output file name format ---
        file_name = f"stream_chart_{theme_idx}.csv"
        # --- END MODIFIED ---
        file_path = os.path.join(output_dir, file_name)

        try:
            # Generate data for the current theme using definitions from data.py
            # This will now generate series with random patterns based on the selected metric's base/noise
            theme_name, unit, metric_info, yearly_rows = generate_theme_data(years_to_generate, current_theme_name)

            # Check if data generation yielded results
            if not metric_info or not yearly_rows:
                print(f"⚠️ No data generated for theme '{current_theme_name}', skipping file.")
                continue  # Skip to the next theme if no data

            # --- CSV Writing Logic ---
            # 1. Prepare Headers according to the requested format
            header1_str = f"Theme: {theme_name},Unit: {unit}"  # Line 1: Theme and Unit
            header_trends = ["trend"] + [m["pattern"] for m in
                                         metric_info]  # Line 2: Trend patterns (now reflects random patterns)
            header_names = ["Year"] + [m["name"] for m in
                                       metric_info]  # Line 3: Year and Series Names ('Metric - Subject Index')

            # 2. Write to CSV file
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                # Write the custom first header line directly
                f.write(header1_str + "\n")

                # Use csv.writer for the remaining headers and data for proper quoting
                writer = csv.writer(f)

                # Write the trend header (Line 2)
                writer.writerow(header_trends)
                # Write the name header (Line 3)
                writer.writerow(header_names)

                # Write data rows
                for row_dict in yearly_rows:
                    # Create row values in the correct order based on header_names
                    # Use row_dict.get(col, '') to handle potential missing keys gracefully
                    row_values = [row_dict.get(col, '') for col in header_names]
                    writer.writerow(row_values)
            # --- End CSV Writing ---

            print(f"✔ Generated {file_path}")

        except ValueError as ve:  # Catch specific configuration errors from generate_theme_data
            print(f"✘ Skipping theme '{current_theme_name}' due to configuration error: {ve}")
        except Exception as exc:  # Catch other unexpected errors during generation/writing for this theme
            print(f"✘ Error processing theme '{current_theme_name}': {exc}")
            # Uncomment the lines below for a detailed traceback if needed for debugging
            # import traceback
            # print(traceback.format_exc())

    print("\nCSV generation process completed.")


if __name__ == "__main__":
    # This block ensures the main function runs only when the script is executed directly
    main()

