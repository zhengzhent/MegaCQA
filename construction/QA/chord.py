import os
import pandas as pd
import json
from collections import defaultdict
import random

def generate_chord_qa(input_folder, output_folder):
    """
    Reads all CSV files in the input_folder (assuming Chord diagram format),
    generates QA pairs based on the data, and saves the QA pairs as JSON files
    in the output_folder.

    Assumes CSV format:
    Metadata line (ignored for specific values, but structure is assumed)
    Source,Target,Value
    ... data rows ...

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

            # Extract metadata from the first line (optional for chord, but keeping structure)
            if len(lines) < 2: # Need at least a header and one data row
                print(f"Skipping {csv_filename}: File too short or missing data.")
                continue

            # Assuming the first line is metadata, read data from the second line onwards
            # Assuming header is always "Source,Target,Value" on the second line
            data = [line.strip().split(",") for line in lines[2:] if line.strip()]
            if not data:
                print(f"Skipping {csv_filename}: No data rows found.")
                continue

            # Read data into DataFrame with specific columns
            df = pd.DataFrame(data, columns=["Source", "Target", "Value"])

            # Ensure 'Value' column can be converted to int
            try:
                df["Value"] = df["Value"].astype(int)
            except ValueError:
                print(f"Skipping {csv_filename}: 'Value' column contains non-integer data.")
                continue

            # Determine the unit from metadata (assuming it's the 3rd part of the first line)
            # Default to empty string if metadata is not as expected
            unit = ""
            # Removed is_percentage logic as we assume concrete numerical unit
            if len(lines[0].strip().split(",")) > 2:
                 unit = lines[0].strip().split(",")[2].strip()


            # === Step 2: QA distribution ===
            # Define QA generators within the function scope
            qa_generators = []

            def qa_generator(category):
                def wrapper(func):
                    qa_generators.append((category, func))
                    return func
                return wrapper

            # Begin QA Category CTR(Chart Type Recognition)
            @qa_generator("CTR")
            def chart_type_qa(df):
                return {
                    "Q": "What type of chart is this?",
                    "A": "This chart is a {chord} chart."
                }
            # End QA Category CTR


            # Begin QA Category VEC(Visual Element Count)
            @qa_generator("VEC")
            def chord_count_qa(df):
                count = len(df) // 2
                return {
                    "Q": "How many chords are in this chord chart?",
                    "A": f"There are {{{count}}} chords."
                }

            @qa_generator("VEC")
            def outer_arc_count_qa(df):
                # Outer arcs correspond to unique source and target nodes
                unique_nodes = pd.concat([df['Source'], df['Target']]).unique()
                count = len(unique_nodes)
                return {
                    "Q": "How many outer arcs are in this chord chart?",
                    "A": f"There are {{{count}}} outer arcs."
                }
            # End QA Category VEC


            # Begin QA Category SRP(Spatial Relationship Perception)
            #       Q:  In the chord chart, what is the second arc in the clockwise direction from arc A?
            #       A: The second arc clockwise from arc A is arc D.
            #       Q:  In the chord chart, what is the second arc in the counterclockwise direction from arc A?
            #       A: The second arc counterclockwise from arc A is arc D.
            @qa_generator("SRP")
            def spatial_relation(df):
                nodes = []
                for node in pd.concat([df['Source'], df['Target']]):
                    if node not in nodes:
                        nodes.append(node)

                start_node_1 = random.choice(nodes)
                idx_1 = nodes.index(start_node_1)

                clockwise_idx = (idx_1 + 1) % len(nodes)
                clockwise_node = nodes[clockwise_idx]

                start_node_2 = random.choice(nodes)
                idx_2 = nodes.index(start_node_2)

                counter_idx = (idx_2 - 1) % len(nodes)
                counter_node = nodes[counter_idx]

                return [
                    {
                        "Q": f"In the chord chart, what is the next arc in the clockwise direction from arc {start_node_1}?",
                        "A": f"The next arc clockwise from arc {start_node_1} is arc {{{clockwise_node}}}."
                    },
                    {
                        "Q": f"In the chord chart, what is the next arc in the counterclockwise direction from arc {start_node_2}?",
                        "A": f"The next arc counterclockwise from arc {start_node_2} is arc {{{counter_node}}}."
                    }
                ]
            # End QA Category SRP


            # Begin QA Category VPR(Visual Pattern Recognition)
            @qa_generator("VPR")
            def largest_output_flow_qa(df):
                if df.empty: return {}
                # Find the row with the maximum Value (output flow)
                max_row = df.loc[df['Value'].idxmax()]
                source = max_row['Source']
                return {
                    "Q": "Which arc has the largest output flow in this chord chart?",
                    "A": f"{{{source}}} has the largest output flow."
                }

            @qa_generator("VPR")
            def largest_input_flow_qa(df):
                if df.empty: return {}
                # Find the row with the maximum Value (input flow)
                max_row = df.loc[df['Value'].idxmax()]
                target = max_row['Target']
                return {
                    "Q": "Which arc has the largest input flow in this chord chart?",
                    "A": f"{{{target}}} has the largest input flow."
                }
            # End QA Category VPR


            # Begin QA Category VE(Value Extraction)
            @qa_generator("VE")
            def specific_connection_value_qa(df):
                if df.empty: return {}
                # Select 2-3 random rows (chords)
                num_questions = random.randint(2, 3)
                random_chords = df.sample(num_questions)
                questions = []

                for _, chord in random_chords.iterrows():
                    source = chord['Source']
                    target = chord['Target']
                    value = chord['Value']
                    questions.append({
                        "Q": f"What is the flow value from {source} to {target} in the chord chart?",
                        "A": f"The flow value is {{{value}}}." # Modified unit output
                    })

                return questions
            # End QA Category VE


            # Begin QA Category EVJ(Exaggerated Value Judgment)
            @qa_generator("EVJ")
            def max_min_flow_between_nodes_qa(df):
  
                chords = df.sample(2, replace=False)
                chord1 = chords.iloc[0]
                chord2 = chords.iloc[1]

                source1, target1 = chord1['Source'], chord1['Target']
                flows1 = df[((df['Source'] == source1) & (df['Target'] == target1)) | ((df['Source'] == target1) & (df['Target'] == source1))]['Value']
                max_value1 = flows1.max()

                source2, target2 = chord2['Source'], chord2['Target']
                flows2 = df[((df['Source'] == source2) & (df['Target'] == target2)) | ((df['Source'] == target2) & (df['Target'] == source2))]['Value']
                min_value2 = flows2.min()

                return [
                    {
                        "Q": f"What is the maximum flow value between {source1} and {target1} in the chord chart?",
                        "A": f"The maximum flow value between {source1} and {target1} is {{{max_value1}}}."
                    },
                    {
                        "Q": f"What is the minimum flow value between {source2} and {target2} in the chord chart?",
                        "A": f"The minimum flow value between {source2} and {target2} is {{{min_value2}}}."
                    }
                ]
            
            @qa_generator("EVJ")
            def max_flow_value_qa(df):
                if df.empty: return {}
                max_value = df['Value'].max()
                # Find all connections with this max value
                max_connections = df[df['Value'] == max_value]
                if max_connections.empty: return {} # Should not happen if df is not empty

                return {
                    "Q": "What is the maximum flow value in the chord chart?",
                    "A": f"The maximum flow value is {{{max_value}}}." # Modified unit output
                }

            @qa_generator("EVJ")
            def min_flow_value_qa(df):
                if df.empty: return {}
                min_value = df['Value'].min()
                 # Find all connections with this min value
                min_connections = df[df['Value'] == min_value]
                if min_connections.empty: return {} # Should not happen if df is not empty

                return {
                    "Q": "What is the minimum flow value in the chord chart?",
                    "A": f"The minimum flow value is {{{min_value}}}." # Modified unit output
                }
            # End QA Category EVJ


            # Begin QA Category SC(Statistic Calculate)
            @qa_generator("SC")
            def total_output_flow_qa(df):
                if df.empty: return {}
                # Get unique source nodes that actually have output flow
                sources_with_output = df['Source'].unique()
                if len(sources_with_output) == 0: return {} # No sources with output

                # Select a random source node
                random_source = random.choice(sources_with_output)
                total_output = df[df['Source'] == random_source]['Value'].sum()
                return {
                    "Q": f"What is the total flow output from {random_source}?",
                    "A": f"The total flow output from {random_source} is {{{total_output}}}."
                }

            @qa_generator("SC")
            def total_input_flow_qa(df):
                if df.empty: return {}
                # Get unique target nodes that actually have input flow
                targets_with_input = df['Target'].unique()
                if len(targets_with_input) == 0: return {} # No targets with input

                # Select a random target node
                random_target = random.choice(targets_with_input)
                total_input = df[df['Target'] == random_target]['Value'].sum()
                return {
                    "Q": f"What is the total flow input by {random_target}?",
                    "A": f"The total flow input by {random_target} is {{{total_input}}}."
                }
            # End QA Category SC


            # Begin QA Category NF(Numerical Filtering)
            @qa_generator("NF")
            def filter_flow_above_dynamic_threshold_qa(df):

                num_to_filter = random.randint(1, 3)
                values = df["Value"].sort_values(ascending=False).tolist()
                if len(values) < 2 or num_to_filter >= len(values):
                    max_value = df["Value"].max()
                    if max_value == 0:
                        threshold = 1
                    else:
                        threshold = int(random.uniform(0.75 * max_value, 0.9 * max_value))
                else:
                    lower_idx = num_to_filter
                    upper_idx = num_to_filter - 1
                    lower_idx = min(lower_idx, len(values) - 1)
                    upper_idx = min(upper_idx, len(values) - 1)
                    threshold = (values[upper_idx] + values[lower_idx]) // 2

                filtered = df[df["Value"] > threshold]
                if filtered.empty:
                     return {
                        "Q": f"Which pairs of nodes in the chord chart have a flow greater than {threshold}? Please list the source node, target node, and the corresponding flow value.", # Modified unit output
                        "A": "No pairs have a flow greater than the threshold."
                    }

                flows_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        flows_list.append(f"There is a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                    else:
                        flows_list.append(f", a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                flows_list.append('.')
                return {
                    "Q": f"Which pairs of nodes in the chord chart have a flow greater than {threshold}? Please list the source node, target node, and the corresponding flow value.", # Modified unit output
                    "A": "".join(flows_list)
                }

            @qa_generator("NF")
            def filter_flow_below_dynamic_threshold_qa(df):
                if df.empty:
                    return {}
                num_to_filter = random.randint(1, 3)
                values = df["Value"].sort_values(ascending=True).tolist() # Sort ascending for "less than" threshold
                if len(values) < 2 or num_to_filter >= len(values):
                    max_value = df["Value"].max()
                    if max_value == 0:
                        threshold = 1
                    else:
                        threshold = int(random.uniform(0.25 * max_value, 0.3 * max_value)) # Adjust range for lower threshold
                else:
                    lower_idx = num_to_filter - 1 # Adjust index for "less than" threshold
                    upper_idx = num_to_filter
                    lower_idx = min(lower_idx, len(values) - 1)
                    upper_idx = min(upper_idx, len(values) - 1)
                    threshold = (values[upper_idx] + values[lower_idx]) // 2
                filtered = df[df["Value"] < threshold] # Filter for values less than the threshold
                if filtered.empty:
                    return {
                        "Q": f"Which pairs of nodes in the chord chart have a flow less than {threshold}? Please list the source node, target node, and the corresponding flow value.",
                        "A": "No pairs have a flow less than the threshold."
                    }
                flows_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        flows_list.append(f"There is a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                    else:
                        flows_list.append(f", a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                flows_list.append('.')
                return {
                    "Q": f"Which pairs of nodes in the chord chart have a flow less than {threshold}? Please list the source node, target node, and the corresponding flow value.",
                    "A": "".join(flows_list)
                }

            @qa_generator("NF")
            def filter_flow_between_dynamic_thresholds_qa(df):
                if df.empty:
                    return {}
                num_to_filter = random.randint(1, 3)
                values = df["Value"].sort_values().tolist()

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
                if filtered.empty:
                    return {
                        "Q": f"Which pairs of nodes in the chord chart have a flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and the corresponding flow value.",
                        "A": "No pairs have a flow within the range."
                    }
                flows_list = []
                for idx, row in enumerate(filtered.itertuples()):
                    if idx == 0:
                        flows_list.append(f"There is a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                    else:
                        flows_list.append(f", a flow of {{{row.Value}}} from {{{row.Source}}} to {{{row.Target}}}")
                flows_list.append('.')
                return {
                    "Q": f"Which pairs of nodes in the chord chart have a flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and the corresponding flow value.",
                    "A": "".join(flows_list)
                }
            # End QA Category NF


            # Begin QA Category NC(Numerical Compare)
            @qa_generator("NC")
            def compare_flows_qa(df):
                # Need at least 2 flows to compare
                if len(df) < 2:
                    return {
                        "Q": "Cannot compare flows.",
                        "A": "Requires at least 2 flows."
                    }

                # Select two random distinct flows (rows)
                if len(df) == 2:
                     flow1, flow2 = df.iloc[0], df.iloc[1]
                else:
                    flow1, flow2 = df.sample(2).iloc[0], df.sample(2).iloc[1]


                source1, target1, value1 = flow1['Source'], flow1['Target'], flow1['Value']
                source2, target2, value2 = flow2['Source'], flow2['Target'], flow2['Value']

                # Ensure the two selected flows are different
                while source1 == source2 and target1 == target2:
                     if len(df) < 3: # If only 2 flows, and they are the same, cannot pick two different ones
                         return {
                            "Q": "Cannot compare flows.",
                            "A": "Requires at least 2 distinct flows."
                         }
                     flow2 = df.sample(1).iloc[0]
                     source2, target2, value2 = flow2['Source'], flow2['Target'], flow2['Value']


                if value1 > value2:
                    comparison = f"The flow from {{{source1}}} to {{{target1}}} is larger."
                elif value1 < value2:
                    comparison = f"The flow from {{{source2}}} to {{{target2}}} is larger."
                else:
                    comparison = f"The flow from {{{source1}}} to {{{target1}}} is equal to the flow from {{{source2}}} to {{{target2}}}."

                question = f"Which flow is larger: from {source1} to {target1} or from {source2} to {target2}?"
                answer = comparison

                return {
                    "Q": question,
                    "A": answer
                }
            
            @qa_generator("NC")
            def compare_three_flows_qa(df):
                # Need at least 3 flows to compare
                if len(df) < 3:
                    return {
                        "Q": "Cannot compare three flows.",
                        "A": "Requires at least 3 flows."
                    }

                # Select three random distinct flows (rows)
                flows = df.sample(3)
                flow1, flow2, flow3 = flows.iloc[0], flows.iloc[1], flows.iloc[2]

                source1, target1, value1 = flow1['Source'], flow1['Target'], flow1['Value']
                source2, target2, value2 = flow2['Source'], flow2['Target'], flow2['Value']
                source3, target3, value3 = flow3['Source'], flow3['Target'], flow3['Value']

                # Compare the three flows
                if value1 > value2 and value1 > value3:
                    largest = f"The flow from {{{source1}}} to {{{target1}}} is the largest."
                elif value2 > value1 and value2 > value3:
                    largest = f"The flow from {{{source2}}} to {{{target2}}} is the largest."
                elif value3 > value1 and value3 > value2:
                    largest = f"The flow from {{{source3}}} to {{{target3}}} is the largest."
                else:
                    largest = "There is a tie for the largest flow."

                question = (
                    f"Which flow is the largest? from {source1} to {target1}, from {source2} to {target2}, or from {source3} to {target3}."
                )
                answer = largest

                return {
                    "Q": question,
                    "A": answer
                }
            # End QA Category NC


            # Begin QA Category MSR(Multi-Step Reasoning)
            # Q: Which category has the highest proportion of outgoing flow relative to its total flow?
            # A: The category with the highest proportion of outgoing flow relative to its total flow is {X}.
            # Q: Which pair has the strongest chord connection in this chord chart?
            # A: The strongest chord connection is between {A} and {B}.
            @qa_generator("MSR")
            def multi_step_reasoning(df):
                if df.empty:
                    return []
                # 1. Highest proportion of outgoing flow relative to total flow
                nodes = pd.concat([df['Source'], df['Target']]).unique()
                max_ratio = -1
                max_node = None
                for node in nodes:
                    outgoing = df[df['Source'] == node]['Value'].sum()
                    incoming = df[df['Target'] == node]['Value'].sum()
                    total = outgoing + incoming
                    ratio = outgoing / total
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_node = node
                qa1 = {
                    "Q": "Which category has the highest proportion of outgoing flow relative to its total flow?",
                    "A": f"The category with the highest proportion of outgoing flow relative to its total flow is {{{max_node}}}."
                }
                # 2. Strongest chord connection (max value between any pair, regardless of direction)
                df_pairs = df.copy()
                df_pairs['Pair'] = df_pairs.apply(lambda row: tuple(sorted([row['Source'], row['Target']])), axis=1)
                grouped = df_pairs.groupby('Pair')['Value'].sum()
                if not grouped.empty:
                    strongest_pair = grouped.idxmax()
                    qa2 = {
                        "Q": "Which pair has the strongest chord connection in this chord chart?",
                        "A": f"The strongest chord connection is between {{{strongest_pair[0]}}} and {{{strongest_pair[1]}}}."
                    }
                else:
                    qa2 = {}
                return [qa1, qa2] if qa2 else [qa1]
            # End QA Category MSR


            # Begin QA Category VA
            """
            使用csv数据可做。基于邻接矩阵的贪心排序（解释见右），优化节点的环形布局，
            最小化交叉：从一个起点开始，每次选择一个尚未被排序的节点，使得它与当前已排序序列的连接总权重最大（或交叉最小），然后将它添加到序列中。
            Q: To minimize the crossings between chords, please reorder the outer arcs and list the reorder arc order in clockwise direction.
            A: The reordered arc order in clockwise direction is: Arc C, Arc B, Arc A.
            """
            @qa_generator("VA")
            def va_avoid_crossing(df):
                # Build adjacency matrix
                nodes = list(pd.concat([df['Source'], df['Target']]).unique())
                n = len(nodes)
                adj = pd.DataFrame(0, index=nodes, columns=nodes)
                for _, row in df.iterrows():
                    adj.at[row['Source'], row['Target']] += row['Value']
                    adj.at[row['Target'], row['Source']] += row['Value']

                # Greedy ordering: start from node with max total connection
                used = set()
                order = []

                start = random.choice(nodes)
                order.append(start)
                used.add(start)

                while len(order) < n:
                    last = order[-1]
                    # Find unused node with max connection to current order
                    candidates = [(node, adj.loc[node, order].sum()) for node in nodes if node not in used]
                    if not candidates:
                        break
                    next_node = max(candidates, key=lambda x: x[1])[0]
                    order.append(next_node)
                    used.add(next_node)

                # Format answer
                arc_order = ', '.join(f'Arc {node}' for node in order)
                return {
                    "Q": "To minimize the crossings between chords, please reorder the outer arcs and list the reorder arc order in clockwise direction.",
                    "A": f"The reordered arc order in clockwise direction is: {{{arc_order}}}."
                }
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


INPUT_DIR = "./csv"
OUTPUT_DIR = "./QA"
if __name__ == '__main__':
    generate_chord_qa(INPUT_DIR, OUTPUT_DIR)

