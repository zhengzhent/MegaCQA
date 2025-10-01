import os
# import sys # Removed as no longer needed for command-line arguments
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any, List, Union
import re
import traceback

from win32con import FLASHW_TIMER

# English generation types
GENERATION_TYPES = ['Positive Linear', 'Negative Linear', 'U-shape', 'Inverted U-shape', 'Random']

# (topics_axes and axis_info dictionaries remain the same)
topics_axes = {
    'Education and Academics': {
        'x_axis': ['Household income', 'Parental education', 'School funding', 'Study hours', 'Class size'],
        'y_axis': ['Academic performance', 'Graduation rate', 'Test scores', 'College admission', 'Literacy rate']
    },
    'Transportation and Logistics': {
        'x_axis': ['Distance traveled', 'Fuel cost', 'Delivery time', 'Vehicle age', 'Traffic volume'],
        'y_axis': ['Transport efficiency', 'Cost per mile', 'On-time rate', 'Fuel efficiency', 'Customer satisfaction']
    },
    'Healthcare and Medicine': {
        'x_axis': ['Patient age', 'BMI', 'Blood pressure', 'Treatment dosage', 'Hospital stay duration'],
        'y_axis': ['Recovery time_Healthcare', 'Treatment cost', 'Patient satisfaction', 'Readmission rate', 'Survival rate'] 
    },
    'Finance and Economics': {
        'x_axis': ['Interest rate', 'Inflation rate', 'GDP growth', 'Unemployment rate', 'Stock price'],
        'y_axis': ['Consumer spending', 'Investment rate', 'Business confidence', 'Housing starts', 'Exchange rate']
    },
     'Environmental Science and Ecology': {
        'x_axis': ['CO2 emissions', 'Average temperature', 'Rainfall amount', 'Forest cover', 'Pollution level'],
        'y_axis': ['Species diversity', 'Crop yield', 'Water quality', 'Air quality', 'Glacier size']
    },
    'Technology and Innovation': {
        'x_axis': ['R&D investment', 'Patent count', 'Internet speed', 'Smartphone penetration', 'Data usage'],
        'y_axis': ['Productivity growth', 'Startup success rate', 'Digital literacy', 'E-commerce sales', 'Innovation index']
    },
    'Marketing and Sales': {
        'x_axis': ['Advertising spend', 'Customer acquisition cost', 'Website traffic', 'Email open rate', 'Sales team size'],
        'y_axis': ['Conversion rate', 'Customer lifetime value', 'Bounce rate', 'Click-through rate', 'Revenue']
    },
    'Human Resources and Workplace': {
        'x_axis': ['Employee training hours', 'Years of experience', 'Salary', 'Commute time', 'Team size'],
        'y_axis': ['Job satisfaction', 'Employee retention', 'Productivity', 'Absenteeism', 'Promotion rate']
    },
    'Agriculture and Food Production': {
        'x_axis': ['Fertilizer usage', 'Irrigation amount', 'Sunlight hours', 'Pest control spending', 'Farm size'],
        'y_axis': ['Crop yield', 'Food production cost', 'Soil quality', 'Product freshness', 'Livestock health']
    },
    'Energy and Utilities': {
        'x_axis': ['Energy consumption', 'Renewable energy share', 'Electricity price', 'Grid stability', 'Infrastructure age'],
        'y_axis': ['Carbon footprint', 'Energy efficiency', 'Customer outage duration', 'Service cost', 'Reliability index']
    },
    'Retail and Consumer Behavior': {
        'x_axis': ['Store size', 'Product price', 'Promotional spending', 'Customer reviews count', 'Checkout time'],
        'y_axis': ['Sales volume', 'Customer satisfaction', 'Average transaction value', 'Return rate', 'Foot traffic']
    },
    'Sports and Fitness': {
        'x_axis': ['Training hours', 'Athlete age', 'Nutrition intake', 'Sleep duration', 'Injury frequency'],
        'y_axis': ['Performance score', 'Recovery time_Sports', 'Endurance level', 'Strength gain', 'Win rate'] 
    },
     'Tourism and Hospitality': {
        'x_axis': ['Hotel price', 'Attraction popularity', 'Travel distance', 'Marketing budget', 'Seasonality index'],
        'y_axis': ['Occupancy rate', 'Customer reviews score', 'Visitor spending', 'Booking conversion', 'Repeat visitor rate']
    },
    'Real Estate and Construction': {
        'x_axis': ['Property size', 'Construction cost', 'Location score', 'Interest rate', 'Development density'],
        'y_axis': ['Property value', 'Rental yield', 'Construction time', 'Vacancy rate', 'Sales price per mÂ³']
    },
     'Social Media and Digital Media and Streaming': {
        'x_axis': ['Content length', 'Posting frequency', 'Hashtag use count', 'Follower count', 'Ad spend'],
        'y_axis': ['Engagement rate', 'View count', 'Shares', 'New followers', 'Conversion rate']
    }
}

axis_info: Dict[str, Tuple[float, float, str]] = {
    'Household income': (20000, 200000, '$'), 'Parental education': (0, 20, 'years'),
    'School funding': (5000, 25000, '$/student'), 'Study hours': (1, 40, 'hours/week'),
    'Class size': (10, 40, 'students'), 'Academic performance': (0, 100, 'score'),
    'Graduation rate': (50, 100, '%'), 'Test scores': (400, 1600, 'score'),
    'College admission': (0, 100, '%'), 'Literacy rate': (70, 100, '%'),
    'Distance traveled': (1, 1000, 'miles'), 'Fuel cost': (2, 5, '$/gallon'),
    'Delivery time': (0.5, 10, 'hours'), 'Vehicle age': (0, 15, 'years'),
    'Traffic volume': (1000, 100000, 'vehicles/day'), 'Transport efficiency': (50, 95, '%'),
    'Cost per mile': (0.5, 5, '$'), 'On-time rate': (70, 100, '%'),
    'Fuel efficiency': (15, 50, 'mpg'), 'Customer satisfaction': (1, 5, 'rating'),
    'Patient age': (0, 100, 'years'), 'BMI': (15, 40, 'kg/mÂ²'), # mÂ² might cause print issues on GBK console
    'Blood pressure': (80, 180, 'mmHg'), 'Treatment dosage': (10, 1000, 'mg'),
    'Hospital stay duration': (1, 30, 'days'),
    'Recovery time_Healthcare': (1, 90, 'days'), 
    'Treatment cost': (100, 50000, '$'), 'Patient satisfaction': (1, 5, 'rating'), # Note: 'Customer satisfaction' is also (1,5,rating)
    'Readmission rate': (0, 30, '%'), 'Survival rate': (50, 100, '%'),
    'Interest rate': (0.1, 10, '%'), 'Inflation rate': (-2, 10, '%'),
    'GDP growth': (-5, 10, '%'), 'Unemployment rate': (2, 20, '%'),
    'Stock price': (1, 5000, '$'), 'Consumer spending': (1000, 100000, '$'),
    'Investment rate': (5, 50, '%'), 'Business confidence': (0, 100, 'index'),
    'Housing starts': (50000, 200000, 'units/month'), 'Exchange rate': (0.5, 2.0, 'currency/USD'),
    'CO2 emissions': (1, 100, 'tons/year'), 'Average temperature': (0, 30, 'Â°C'),
    'Rainfall amount': (100, 2000, 'mm/year'), 'Forest cover': (10, 90, '%'),
    'Pollution level': (10, 100, 'index'), 'Species diversity': (1, 1000, 'count'),
    'Crop yield': (1, 10, 'tons/hectare'), 'Water quality': (0, 100, 'index'),
    'Air quality': (0, 100, 'index'), 'Glacier size': (1, 1000, 'kmÂ²'), # kmÂ²
    'R&D investment': (100000, 100000000, '$'), 'Patent count': (1, 1000, 'count'),
    'Internet speed': (10, 1000, 'Mbps'), 'Smartphone penetration': (20, 95, '%'),
    'Data usage': (1, 1000, 'GB/month'), 'Productivity growth': (0, 10, '%'),
    'Startup success rate': (5, 50, '%'), 'Digital literacy': (0, 100, 'index'),
    'E-commerce sales': (10000, 1000000000, '$'), 'Innovation index': (0, 100, 'score'),
    'Advertising spend': (1000, 1000000, '$'), 'Customer acquisition cost': (10, 500, '$'),
    'Website traffic': (1000, 10000000, 'visits/month'), 'Email open rate': (5, 50, '%'),
    'Sales team size': (1, 100, 'people'), 'Conversion rate': (0, 10, '%'),
    'Customer lifetime value': (50, 5000, '$'), 'Bounce rate': (10, 80, '%'),
    'Click-through rate': (0.1, 10, '%'), 'Revenue': (10000, 1000000000, '$'),
    'Employee training hours': (1, 50, 'hours/year'), 'Years of experience': (0, 30, 'years'),
    'Salary': (30000, 200000, '$/year'), 'Commute time': (5, 90, 'minutes'),
    'Team size': (2, 20, 'people'), 'Job satisfaction': (1, 5, 'rating'),
    'Employee retention': (50, 95, '%'), 'Productivity': (50, 150, 'index'),
    'Absenteeism': (0, 10, 'days/year'), 'Promotion rate': (0, 20, '%'),
    'Fertilizer usage': (10, 500, 'kg/hectare'), 'Irrigation amount': (100, 2000, 'mm/season'),
    'Sunlight hours': (4, 12, 'hours/day'), 'Pest control spending': (10, 500, '$/hectare'),
    'Farm size': (1, 1000, 'hectares'), 'Food production cost': (100, 2000, '$/ton'),
    'Soil quality': (0, 100, 'index'), 'Product freshness': (1, 5, 'rating'),
    'Livestock health': (1, 5, 'rating'),
    'Energy consumption': (1000, 1000000, 'kWh/month'), 'Renewable energy share': (0, 100, '%'),
    'Electricity price': (0.1, 0.5, '$/kWh'), 'Grid stability': (0, 100, 'index'),
    'Infrastructure age': (0, 50, 'years'), 'Carbon footprint': (1, 50, 'tons CO2e/year'),
    'Energy efficiency': (0, 100, '%'), 'Customer outage duration': (0, 24, 'hours/year'),
    'Service cost': (50, 500, '$/month'), 'Reliability index': (0, 100, 'score'),
    'Store size': (100, 10000, 'mÂ³'), # mÂ³
    'Product price': (1, 1000, '$'),
    'Promotional spending': (100, 100000, '$'), 'Customer reviews count': (1, 10000, 'count'),
    'Checkout time': (0, 10, 'minutes'), 'Sales volume': (100, 1000000, 'units/month'),
    'Average transaction value': (10, 500, '$'), 'Return rate': (0, 20, '%'),
    'Foot traffic': (100, 100000, 'people/day'),
    'Training hours': (1, 40, 'hours/week'), 'Athlete age': (15, 40, 'years'),
    'Nutrition intake': (1000, 5000, 'calories/day'), 'Sleep duration': (4, 10, 'hours/night'),
    'Injury frequency': (0, 5, 'injuries/year'), 'Performance score': (0, 100, 'score'),
    'Recovery time_Sports': (0.5, 7, 'days'), 
    'Endurance level': (0, 100, 'index'), 'Strength gain': (0, 50, '%'), 'Win rate': (0, 100, '%'),
    'Hotel price': (50, 1000, '$/night'), 'Attraction popularity': (100, 1000000, 'visitors/year'),
    'Travel distance': (10, 5000, 'miles'), 'Marketing budget': (1000, 1000000, '$'),
    'Seasonality index': (0, 10, 'index'), 'Occupancy rate': (30, 100, '%'),
    'Customer reviews score': (1, 5, 'rating'), 'Visitor spending': (50, 5000, '$/day'),
    'Booking conversion': (0, 20, '%'), 'Repeat visitor rate': (0, 80, '%'),
    'Property size': (500, 10000, 'mÂ²'), # mÂ²
    'Construction cost': (100, 500, '$/mÂ²'), # $/mÂ²
    'Location score': (1, 10, 'score'), 'Development density': (1, 100, 'units/acre'),
    'Property value': (50000, 5000000, '$'), 'Rental yield': (1, 10, '%'),
    'Construction time': (1, 24, 'months'), 'Vacancy rate': (0, 20, '%'),
    'Sales price per mÂ³': (50, 1000, '$/mÂ³'), # $/mÂ³
    'Content length': (1, 60, 'minutes'), 'Posting frequency': (1, 20, 'posts/day'),
    'Hashtag use count': (1, 20, 'count/post'), 'Follower count': (100, 10000000, 'count'),
    'Ad spend': (100, 100000, '$'), 'Engagement rate': (0, 20, '%'),
    'View count': (100, 10000000, 'count'), 'Shares': (10, 100000, 'count'),
    'New followers': (10, 100000, 'count/day'),
}

ENGLISH_DESCRIPTORS = [
    "Relationship", "Correlation", "Distribution", "Analysis", "Overview",
    "Comparison", "Interaction", "Influence", "Pattern", "Dynamics", "Trend"
]

def sanitize_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[ /]', '_', text)
    text = re.sub(r'[^\w-]', '', text)
    return text

def sanitize_for_print(obj: Any) -> Any:
    """Recursively sanitizes an object for printing by replacing non-ASCII characters in strings."""
    if isinstance(obj, dict):
        return {sanitize_for_print(k): sanitize_for_print(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_print(i) for i in obj]
    elif isinstance(obj, str):
        return obj.encode('ascii', 'replace').decode('ascii')
    return obj

def generate_scatter_data(generation_type: str, overlap_degree: float,
                          strength_range: Tuple[float, float], num_records: int,
                          x_min_target: float, x_max_target: float,
                          y_min_target: float, y_max_target: float,
                          x_col_name: str = 'x_value', y_col_name: str = 'y_value') -> Tuple[pd.DataFrame, float]:
    strength = random.uniform(*strength_range)
    
    if num_records < 2: raise ValueError(f"Number of records must be at least 2. Got {num_records}")
    if x_min_target >= x_max_target: x_max_target = x_min_target + 1.0 
    if y_min_target >= y_max_target: y_max_target = y_min_target + 1.0 

    if generation_type not in GENERATION_TYPES: raise ValueError(f"Invalid generation_type: {generation_type}")
    if not 0 <= overlap_degree <= 1: raise ValueError("Overlap degree out of range")
    if any(not 0 <= s <= 1 for s in strength_range): raise ValueError("Strength range out of range")
    
    if num_records <= 0: return pd.DataFrame(columns=[x_col_name, y_col_name]), strength

    x_values_base = np.linspace(0, 100, num_records) 
    y_values_base = np.zeros(num_records) 
    
    noise_strength = 1 - strength
    
    if generation_type == 'Positive Linear':
        noise = np.random.normal(0, noise_strength * 25, num_records)
        y_values_base = strength * x_values_base + noise
    elif generation_type == 'Negative Linear':
        noise = np.random.normal(0, noise_strength * 25, num_records)
        y_values_base = strength * (100 - x_values_base) + noise
    elif generation_type == 'U-shape':
        noise = np.random.normal(0, noise_strength * 25, num_records)
        y_values_base = strength * 0.04 * (x_values_base - 50)**2 + noise
    elif generation_type == 'Inverted U-shape':
        noise = np.random.normal(0, noise_strength * 25, num_records)
        y_values_base = -strength * 0.04 * (x_values_base - 50)**2 + 100 + noise
    elif generation_type == 'Random':
        y_values_base = np.random.uniform(0, 100, num_records)
    
    if not isinstance(y_values_base, np.ndarray) or len(y_values_base) != num_records:
        y_values_base = np.full(num_records, 50.0)

    if overlap_degree > 0 and num_records > 0 : 
        num_overlap_points = int(num_records * overlap_degree)
        if 0 < num_overlap_points <= num_records: 
            overlap_indices = np.random.choice(num_records, num_overlap_points, replace=False)
            y_values_base[overlap_indices] += np.random.normal(0, 15 * overlap_degree, len(overlap_indices))

    if x_values_base.size == 0: return pd.DataFrame(columns=[x_col_name, y_col_name]), strength

    x_values_scaled = x_min_target + (x_values_base / 100.0) * (x_max_target - x_min_target) 

    if np.all(np.isnan(y_values_base)): 
        y_values_scaled = np.full(num_records, np.nan) 
    else:
        y_min_base_actual = np.nanmin(y_values_base) 
        y_max_base_actual = np.nanmax(y_values_base)
        if y_min_base_actual == y_max_base_actual or np.isnan(y_min_base_actual) or np.isnan(y_max_base_actual): 
             y_values_scaled = np.full(num_records, (y_min_target + y_max_target) / 2.0) 
        else:
             y_values_scaled = y_min_target + ((y_values_base - y_min_base_actual) * (y_max_target - y_min_target) / (y_max_base_actual - y_min_base_actual))
    
    y_values_scaled = np.nan_to_num(y_values_scaled, nan=(y_min_target + y_max_target) / 2.0, 
                                    posinf=y_max_target, neginf=y_min_target)

    if len(x_values_scaled) != num_records: x_values_scaled = np.full(num_records, x_min_target) 
    if len(y_values_scaled) != num_records: y_values_scaled = np.full(num_records, y_min_target) 
        
    df_generated = pd.DataFrame({x_col_name: x_values_scaled, y_col_name: y_values_scaled})
    
    assert len(df_generated) == num_records, f"DF length {len(df_generated)} != num_records {num_records}"
    assert df_generated.shape[1] == 2, f"DF columns {df_generated.shape[1]} != 2. Cols: {df_generated.columns.tolist()}"
            
    return df_generated, strength

def save_scatter_to_csv(data: pd.DataFrame, topic: str, topic_seq_num: int,
                        generation_type: str,
                        x_col_name: str, y_col_name: str, x_unit: str, y_unit: str) -> Tuple[Union[str, None], bool]:
    output_dir = './csv'
    sanitized_topic = sanitize_filename(topic)
    filename_candidate = os.path.join(output_dir, f"scatter_{sanitized_topic}_{topic_seq_num}.csv")
    
    data_to_save = data.copy()
    try:
        data_to_save = data_to_save[[x_col_name, y_col_name]]
    except KeyError as e:
        print(f"CRITICAL ERROR (save_scatter_to_csv for {filename_candidate}): Key error selecting columns. "
              f"Available: {data.columns.tolist()}. Expected: ['{x_col_name}', '{y_col_name}']. Error: {e}")
        raise e 

    if data_to_save.empty:
        print(f"DEBUG (save_scatter_to_csv for {filename_candidate}): 'data_to_save' is EMPTY. No file will be created.")
        return None, False

    os.makedirs(output_dir, exist_ok=True) 

    selected_descriptor = random.choice(ENGLISH_DESCRIPTORS)
    little_theme = f"the {selected_descriptor} of {topic}"
    header_line_1 = f"{topic},{little_theme},{generation_type}"
    header_line_2 = f"{x_col_name} ({x_unit}),{y_col_name} ({y_unit})"
    
    data_to_save[x_col_name] = data_to_save[x_col_name].round(2)
    data_to_save[y_col_name] = data_to_save[y_col_name].round(2)

    try:
        with open(filename_candidate, 'w', newline='', encoding='utf-8') as f:
            f.write(header_line_1 + "\n")
            f.write(header_line_2 + "\n")
            data_to_save.to_csv(f, index=False, header=False)
        return filename_candidate, True
    except Exception as e_write:
        print(f"ERROR (save_scatter_to_csv for {filename_candidate}): Failed during file write. Error: {e_write}")
        if os.path.exists(filename_candidate):
            try:
                os.remove(filename_candidate)
                print(f"  Removed potentially corrupted file: {filename_candidate}")
            except OSError as e_remove:
                print(f"  Error removing corrupted file {filename_candidate}: {e_remove}")
        return None, False

def main(num_files: int, test_mode: bool = False):
    if test_mode:
        types_to_generate = GENERATION_TYPES[:]
        num_files = len(types_to_generate)
    else:
        types_to_generate = None

    strength_range = (0.6, 0.95) 
    topic_file_counters: Dict[str, int] = {} 
    generated_files = 0
    print(f"Starting data generation for {num_files} scatter plot files.")
    attempts = 0
    max_attempts = num_files * 20 

    while generated_files < num_files and attempts < max_attempts:
        attempts += 1
        current_iter_params = {} 
        try:
            topic = random.choice(list(topics_axes.keys()))
            current_iter_params['topic'] = topic
            
            possible_x = [ax for ax in topics_axes[topic]['x_axis'] if ax in axis_info]
            possible_y = [ax for ax in topics_axes[topic]['y_axis'] if ax in axis_info]
            if not possible_x or not possible_y:
                print(f"Warning (Attempt {attempts}): Skipping topic '{topic}' (missing axis info).")
                continue 

            x_name = random.choice(possible_x); current_iter_params['x_name'] = x_name
            y_name = random.choice(possible_y); current_iter_params['y_name'] = y_name

            if test_mode and types_to_generate:
                generation_type = types_to_generate.pop(0)
            else:
                generation_type = random.choice(GENERATION_TYPES)
            current_iter_params['generation_type'] = generation_type

            overlap_degree = random.randint(0, 7) / 10.0
            num_records = random.randint(30, 100); current_iter_params['num_records'] = num_records

            x_min_target, x_max_target, x_unit = axis_info[x_name]; current_iter_params['x_unit'] = x_unit
            y_min_target, y_max_target, y_unit = axis_info[y_name]; current_iter_params['y_unit'] = y_unit

            data, actual_strength = generate_scatter_data(
                generation_type=generation_type, overlap_degree=overlap_degree,
                strength_range=strength_range, num_records=num_records, 
                x_min_target=x_min_target, x_max_target=x_max_target,
                y_min_target=y_min_target, y_max_target=y_max_target,
                x_col_name=x_name, y_col_name=y_name
            )
            
            assert isinstance(data, pd.DataFrame), f"generate_scatter_data returned non-DataFrame. Params: {sanitize_for_print(current_iter_params)}"

            is_compliant = True; error_detail = ""
            if len(data) != num_records:
                error_detail = f"has {len(data)} rows, expected {num_records}"; is_compliant = False
            elif data.shape[1] != 2: 
                error_detail = f"has {data.shape[1]} columns, expected 2. Cols: {data.columns.tolist()}"; is_compliant = False
            elif x_name not in data.columns or y_name not in data.columns:
                error_detail = f"expected columns ['{x_name}', '{y_name}'] not in DF columns: {data.columns.tolist()}"; is_compliant = False
            
            if not is_compliant:
                safe_params = sanitize_for_print(current_iter_params)
                print(f"CRITICAL WARNING (Attempt {attempts}): Data from generate_scatter_data for {safe_params} "
                      f"is non-compliant. Detail: {error_detail}. Skipping.")
                continue 
            
            if topic not in topic_file_counters: topic_file_counters[topic] = 0
            topic_file_counters[topic] += 1
            current_topic_seq_num = topic_file_counters[topic]

            output_file_path, data_was_written = save_scatter_to_csv( 
                data=data, topic=topic, topic_seq_num=current_topic_seq_num, 
                generation_type=generation_type,
                x_col_name=x_name, y_col_name=y_name, x_unit=x_unit, y_unit=y_unit,
            )
            
            if data_was_written and output_file_path is not None:
                generated_files += 1 
                print(f"Generated file {generated_files}/{num_files} (Attempt {attempts}): {output_file_path}")
            else:
                safe_params = sanitize_for_print(current_iter_params)
                print(f"Warning (Attempt {attempts}): save_scatter_to_csv for {safe_params} "
                      f"did not save a file with data. Path: {output_file_path}. Not counted.")

        except (AssertionError, ValueError) as e: 
            safe_params = sanitize_for_print(current_iter_params)
            print(f"HANDLED ERROR (Attempt {attempts}) with params {safe_params}: {str(e).encode('ascii', 'replace').decode('ascii')}. Skipping.")
        except Exception as e: 
             safe_params = sanitize_for_print(current_iter_params)
             print(f"UNEXPECTED ERROR (Attempt {attempts}) with params {safe_params}: {str(e).encode('ascii', 'replace').decode('ascii')}. Skipping.")
             traceback.print_exc()
    
    if attempts >= max_attempts and generated_files < num_files:
        print(f"Warning: Max attempts ({max_attempts}) reached. Generated {generated_files}/{num_files} files.")
    print(f"Finished. Generated {generated_files} valid files after {attempts} attempts.")

if __name__ == "__main__":
    # --- Configuration ---
    # Set to True to generate 5 sample files (one for each type) for quick review.
    # Set to False to generate the full number of files specified below.
    RUN_IN_TEST_MODE = False

    if RUN_IN_TEST_MODE:
        # Test mode: generate 5 files, one for each type.
        main(num_files=5, test_mode=True)
    else:
        # Normal mode: generate a large number of files.
        num_files = 10000
        main(num_files=num_files, test_mode=False)
