import os
import pandas as pd
import json
from collections import defaultdict
import random

def generate_funnel_qa(input_folder, output_folder):
    """
    Reads all CSV files in the input_folder, generates QA pairs based on the data,
    and saves the QA pairs as JSON files in the output_folder.

    Args:
        input_folder (str): The path to the folder containing the input CSV files.
        output_folder (str): The path to the folder where the output JSON files
                             will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    all_files = os.listdir(input_folder)

    # Filter for CSV files
    csv_files = [f for f in all_files if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in folder: {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_folder}. Processing...")

    for csv_filename in csv_files:
        csv_filepath = os.path.join(input_folder, csv_filename)
        json_filename = csv_filename.replace(".csv", ".json")
        json_filepath = os.path.join(output_folder, json_filename)

        # print(f"Processing {csv_filename}...")

        try:
            # === Step 1: Read csv ===
            with open(csv_filepath, encoding="utf-8") as f:
                lines = f.readlines()

            # Extract metadata from the first line
            if len(lines) < 3:
                print(f"Skipping {csv_filename}: File too short or missing data.")
                continue

            meta_parts = lines[0].strip().split(",")
            if len(meta_parts) < 4:
                 print(f"Skipping {csv_filename}: Metadata line incomplete.")
                 continue
            topic, sub_topic, unit, decay_model = meta_parts
            is_percentage = (unit == '%')

            # Data extraction from lines starting from the third line
            data = [line.strip().split(",") for line in lines[2:] if line.strip()]
            if not data:
                print(f"Skipping {csv_filename}: No data rows found.")
                continue

            df = pd.DataFrame(data, columns=["Phase", "Value"])
            # Ensure 'Value' column can be converted to int, handle potential errors
            try:
                df["Value"] = df["Value"].astype(int)
            except ValueError:
                print(f"Skipping {csv_filename}: 'Value' column contains non-integer data.")
                continue


            # === Step 2: QA distribution ===
            # Define QA generators within the function scope or ensure they are accessible
            qa_generators = []

            def qa_generator(category):
                def wrapper(func):
                    qa_generators.append((category, func))
                    return func
                return wrapper

            # Begin QA Category CTR(Chart Type Recognition) Done.
            @qa_generator("CTR")
            def chart_type_qa(df):
                return {
                    "Q": "What type of chart is this?",
                    "A": "This chart is a {funnel} chart."
                }
            # End QA Category CTR


            # Begin QA Category VEC(Visual Element Count) Done.
            @qa_generator("VEC")
            def stage_count_qa(df):
                count = len(df)
                return {
                    "Q": "How many stages are there in this funnel chart?",
                    "A": f"There are {{{count}}} stages."
                }
            # End QA Category VEC


            # Begin QA Category SRP(Spatial Relationship Perception)
            #       Q: Is the Purchase stage positioned above or below the Payment stage?
            #       A: The Purchase stage is positioned below the Payment stage.
            @qa_generator("SRP")
            def spatial_relation_qa(df):
                stages = df["Phase"].tolist()
                stage1, stage2 = random.sample(stages, 2)
                idx1 = stages.index(stage1)
                idx2 = stages.index(stage2)
                if idx1 < idx2:
                    relation = "above"
                else:
                    relation = "below"

                return {
                    "Q": f"Is the {stage1} stage positioned above or below the {stage2} stage?",
                    "A": f"The {stage1} stage is positioned {{{relation}}} the {stage2} stage."
                }
            # End QA Category SRP


            # Begin QA Category VPR(Visual Pattern Recognition) Done.
            @qa_generator("VPR")
            def visual_pattern_qa(df):
                # decay_model is read from the metadata
                return {
                    "Q": "What pattern does the data reduction follow in this funnel chart?",
                    "A": f"The data reduction follows a {{{decay_model}}} pattern."
                }
            # End QA Category VPR


            # Begin QA Category VE(Value Extraction) Almost Done.
            # Generating one QA for each stage value extraction
            @qa_generator("VE")
            def each_stage_qa(df):
                if df.empty: return []  # Handle empty dataframe
                num_samples = random.randint(2, 3)  # Randomly choose 2 or 3 stages
                sampled_rows = df.sample(n=min(num_samples, len(df)))  # Sample rows without exceeding the dataframe size
                return [
                    {
                        "Q": f"What is the value of stage {row.Phase}?",
                        "A": f"The value of stage {row.Phase} is {{{row.Value}{unit if is_percentage else ''}}}."
                    }
                    for _, row in sampled_rows.iterrows()
                ]
            # End QA Category VE


            # Begin QA Category EVJ(Exaggerated Value Judgment) Done.
            @qa_generator("EVJ")
            def max_value_qa(df):
                if df.empty: return {} # Handle empty dataframe
                row = df.loc[df["Value"].idxmax()]
                return {
                    "Q": "What is the maximum value in the funnel chart?",
                    "A": f"The stage with the highest value is {row.Phase}, with a value of {{{row.Value}{unit if is_percentage else ''}}}."
                }

            @qa_generator("EVJ")
            def min_value_qa(df):
                if df.empty: return {} # Handle empty dataframe
                row = df.loc[df["Value"].idxmin()]
                return {
                    "Q": "What is the minimum value in the funnel chart?",
                    "A": f"The stage with the lowest value is {row.Phase}, with a value of {{{row.Value}{unit if is_percentage else ''}}}."
                }
            # End QA Category EVJ


            # Begin QA Category SC(Statistic Calculate) Done.
            @qa_generator("SC")
            def conversion_or_decrease_qa(df):
                if len(df) < 2: return []  # Need at least 2 stages
                questions = []
                stage_indices = random.sample(range(len(df) - 1), k=random.randint(1, 3))  # Randomly select 1 to 3 unique stages

                for stage_index in stage_indices:
                    first_value = df.iloc[stage_index]["Value"]
                    second_value = df.iloc[stage_index + 1]["Value"]

                    if is_percentage:
                        # Calculate conversion rate
                        conversion_rate = first_value - second_value
                        questions.append({
                            "Q": f"What is the percentage reduction from stage {df.iloc[stage_index]['Phase']} to stage {df.iloc[stage_index + 1]['Phase']}?",
                            "A": f"The percentage reduction from stage {df.iloc[stage_index]['Phase']} to stage {df.iloc[stage_index + 1]['Phase']} is {{{conversion_rate}%}}."
                        })
                    else:
                        # Calculate decrease amount
                        decrease_amount = first_value - second_value
                        questions.append({
                            "Q": f"How much did it decrease from stage {df.iloc[stage_index]['Phase']} to stage {df.iloc[stage_index + 1]['Phase']}?",
                            "A": f"Decreased by {{{decrease_amount}}} {unit} from stage {df.iloc[stage_index]['Phase']} to stage {df.iloc[stage_index + 1]['Phase']}."
                        })

                return questions
            # End QA Category SC


            # Begin QA Category NF(Numerical Filtering)
            @qa_generator("NF")
            def filter_above_dynamic_threshold_qa(df):
                if df.empty: return {}  # Handle empty dataframe
                values = df["Value"].sort_values(ascending=False).tolist()
                num_to_filter = random.randint(1, 3)
                upper_idx = num_to_filter - 1
                lower_idx = num_to_filter
                upper_idx = min(upper_idx, len(values) - 1)
                lower_idx = min(lower_idx, len(values) - 1)
                threshold = (values[upper_idx] + values[lower_idx]) // 2

                filtered = df[df["Value"] > threshold]
                stages_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        stages_list.append(f"The {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                    else:
                        stages_list.append(f", the {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                stages_list.append('.')
                return {
                    "Q": f"In which stages of the funnel chart does the value exceed {threshold}{unit if is_percentage else ''}? Please list the stage labels and corresponding values.",
                    "A": "".join(stages_list) if stages_list else "No stages exceed the threshold."
                }

            @qa_generator("NF")
            def filter_below_dynamic_threshold_qa(df):

                num_to_filter = random.randint(1, 3)
                values = df["Value"].sort_values(ascending=True).tolist()
                
                upper_idx = num_to_filter - 1
                lower_idx = num_to_filter
                upper_idx = min(upper_idx, len(values) - 1)
                lower_idx = min(lower_idx, len(values) - 1)
                threshold = (values[upper_idx] + values[lower_idx]) // 2    

                filtered = df[df["Value"] < threshold]
                stages_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        stages_list.append(f"The {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                    else:
                        stages_list.append(f", the {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                stages_list.append('.')
                return {
                    "Q": f"In which stages of the funnel chart does the value fall below {threshold}{unit if is_percentage else ''}? Please list the stage labels and corresponding values.",
                    "A": "".join(stages_list)
                }

            @qa_generator("NF")
            def filter_between_dynamic_thresholds_qa(df):
                if df.empty: return {}  # Handle empty dataframe
                values = df["Value"].sort_values().tolist()
                num_to_filter = random.randint(1, 3)
                
                # Find the indices for the middle num_to_filter values
                mid_start = (len(values) - num_to_filter) // 2
                mid_end = mid_start + num_to_filter - 1
                # Calculate lower_threshold as the average of values[mid_start] and its previous value (or 0 if out of bounds)
                prev_idx = mid_start - 1
                prev_value = values[prev_idx] if prev_idx >= 0 else 0
                lower_threshold = (values[mid_start] + prev_value) // 2

                # Calculate upper_threshold as the average of values[mid_end] and its next value (or itself if out of bounds)
                next_idx = mid_end + 1
                next_value = values[next_idx] if next_idx < len(values) else values[mid_end]
                upper_threshold = (values[mid_end] + next_value) // 2
                
                filtered = df[(df["Value"] >= lower_threshold) & (df["Value"] <= upper_threshold)]

                stages_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        stages_list.append(f"The {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                    else:
                        stages_list.append(f", the {{{row.Phase}}} stage has {{{row.Value}{unit if is_percentage else ''}}}")
                stages_list.append('.')
                return {
                    "Q": f"In which stages of the funnel chart does the value fall between {lower_threshold}{unit if is_percentage else ''} and {upper_threshold}{unit if is_percentage else ''}? Please list the stage labels and corresponding values.",
                    "A": "".join(stages_list) if stages_list else "No stages fall within the range."
                }
            # End QA Category NF


            # Begin QA Category NC(Numerical Compare)
            @qa_generator("NC")
            def compare_reductions_qa(df):
                qa_pairs = []

                if len(df) < 3:
                    return []

                stage_pairs = [(i, i + 1) for i in range(len(df) - 1)]

                num_questions = min(random.choice([1, 2]), len(stage_pairs) - 1)
                random.shuffle(stage_pairs)

                compare_sets = []
                for i in range(len(stage_pairs)):
                    for j in range(i + 1, len(stage_pairs)):
                        compare_sets.append((stage_pairs[i], stage_pairs[j]))

                if not compare_sets:
                    return []

                random.shuffle(compare_sets)
                for i in range(num_questions):
                    (a_idx, b_idx), (c_idx, d_idx) = compare_sets[i]

                    value_a = df.iloc[a_idx]['Value']
                    value_b = df.iloc[b_idx]['Value']
                    value_c = df.iloc[c_idx]['Value']
                    value_d = df.iloc[d_idx]['Value']

                    phase_a = df.iloc[a_idx]['Phase']
                    phase_b = df.iloc[b_idx]['Phase']
                    phase_c = df.iloc[c_idx]['Phase']
                    phase_d = df.iloc[d_idx]['Phase']

                    reduction1 = value_a - value_b
                    reduction2 = value_c - value_d

                    question = f"Which reduction is greater? from stage {phase_a} to {phase_b}, or from stage {phase_c} to {phase_d}."

                    if reduction1 > reduction2:
                        answer = f"The reduction from stage {{{phase_a}}} to {{{phase_b}}} is greater."
                    elif reduction1 < reduction2:
                        answer = f"The reduction from stage {{{phase_c}}} to {{{phase_d}}} is greater."
                    else:
                        answer = f"The reduction from stage {{{phase_a}}} to {{{phase_b}}} is equal to the reduction from stage {phase_c} to {phase_d}."

                    qa_pairs.append({"Q": question, "A": answer})

                return qa_pairs

            @qa_generator("NC")
            def compare_three_reductions_qa(df):
                if len(df) < 4:
                    return []

                stage_pairs = [(i, i + 1) for i in range(len(df) - 1)]
                random.shuffle(stage_pairs)

                num_questions = min(random.choice([1, 2]), len(stage_pairs) // 3)
                qa_pairs = []

                for i in range(num_questions):
                    triplet = stage_pairs[i*3:(i+1)*3]
                    if len(triplet) < 3:
                        break

                    (a, b), (c, d), (e, f) = triplet

                    v1, v2, v3 = df.iloc[a]['Value'] - df.iloc[b]['Value'], df.iloc[c]['Value'] - df.iloc[d]['Value'], df.iloc[e]['Value'] - df.iloc[f]['Value']
                    p1a, p1b = df.iloc[a]['Phase'], df.iloc[b]['Phase']
                    p2a, p2b = df.iloc[c]['Phase'], df.iloc[d]['Phase']
                    p3a, p3b = df.iloc[e]['Phase'], df.iloc[f]['Phase']

                    question = f"Which reduction is the greatest? from stage {p1a} to {p1b}, from stage {p2a} to {p2b}, or from stage {p3a} to {p3b}."

                    if v1 > v2 and v1 > v3:
                        answer = f"The reduction from stage {{{p1a}}} to stage {{{p1b}}} is the greatest."
                    elif v2 > v1 and v2 > v3:
                        answer = f"The reduction from stage {{{p2a}}} to stage {{{p2b}}} is the greatest."
                    elif v3 > v1 and v3 > v2:
                        answer = f"The reduction from stage {{{p3a}}} to stage {{{p3b}}} is the greatest."
                    else:
                        answer = f"The reductions from stage {{{p1a}}} to {{{p1b}}}, {{{p2a}}} to {{{p2b}}}, and {{{p3a}}} to {{{p3b}}} are equal."

                    qa_pairs.append({"Q": question, "A": answer})

                return qa_pairs
            # End QA Category NC


            # Begin QA Category MSR(Multi-Step Reasoning)
            #       Q: Between which two consecutive stages in the funnel chart is the conversion rate drop the largest?
            #       A: The largest drop in conversion rate occurs between the X and Y stages.
            #       Q: Between which two consecutive stages in the funnel chart is the conversion rate drop the smallest?
            #       A: The smallest drop in conversion rate occurs between the X and Y stages.
            @qa_generator("MSR")
            def msr_placeholder_qa(df):
                drops = []
                for i in range(len(df) - 1):
                    phase1 = df.iloc[i]['Phase']
                    phase2 = df.iloc[i + 1]['Phase']
                    value1 = df.iloc[i]['Value']
                    value2 = df.iloc[i + 1]['Value']
                    drop = value1 - value2
                    drops.append({'pair': (phase1, phase2), 'drop': drop})

                max_drop_pair = max(drops, key=lambda x: x['drop'])['pair']
                min_drop_pair = min(drops, key=lambda x: x['drop'])['pair']

                return [
                    {
                        "Q": "Between which two consecutive stages in the funnel chart is the conversion rate drop the largest?",
                        "A": f"The largest drop in conversion rate occurs between the {{{max_drop_pair[0]}}} and {{{max_drop_pair[1]}}} stages."
                    },
                    {
                        "Q": "Between which two consecutive stages in the funnel chart is the conversion rate drop the smallest?",
                        "A": f"The smallest drop in conversion rate occurs between the {{{min_drop_pair[0]}}} and {{{min_drop_pair[1]}}} stages."
                    }
                ]
            # End QA Category MSR

            # Begin QA Category VA
            @qa_generator("VA")
            def va_placeholder_qa(df):
                 return 
            # End QA Category VA

            # === Step 3: 构建 QA JSON ===
            qa_json = defaultdict(list)

            # Iterate through the registered QA generators and execute them
            for category, func in qa_generators:
                result = func(df)
                # Check if the result is a dictionary (single QA) or a list of dictionaries (multiple QAs)
                if category not in qa_json:
                    qa_json[category] = []
                if isinstance(result, list):
                    # Extend the list for the category if the result is a list
                    qa_json[category].extend(result)
                elif isinstance(result, dict) and result: # Add if it's a non-empty dictionary
                    # Append the dictionary to the list for the category
                    qa_json[category].append(result)
                # Ignore empty results (e.g., from functions returning {} for invalid conditions)


            # === Step 4: 保存 JSON ===
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(qa_json, f, ensure_ascii=False, indent=4)

            print(f"Successfully generated {json_filename}")

        except Exception as e:
            print(f"Error processing {csv_filename}: {e}")


INPUT_DIR = "csv/funnel"
OUTPUT_DIR = "json/funnel"

if __name__ == '__main__':
    generate_funnel_qa(INPUT_DIR, OUTPUT_DIR)


