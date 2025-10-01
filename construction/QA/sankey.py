import os
import pandas as pd
import json
from collections import defaultdict
import random
import itertools

def determine_and_add_layers(df, suggested_layer_names):
    """
    Automatically determines node layers based on flow in the DataFrame
    and adds 'Source_Layer' and 'Target_Layer' columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Source', 'Target', and 'Value' columns.
        suggested_layer_names (list): List of suggested layer names from metadata.

    Returns:
        tuple: (modified_df, determined_layer_names, num_determined_layers)
               - modified_df (pd.DataFrame): DataFrame with layer information added.
               - determined_layer_names (list): List of determined layer names (auto-generated or suggested).
               - num_determined_layers (int): The total number of determined layers.
    """
    node_layers = {}
    nodes_in_current_layer = set()
    nodes_assigned = set()

    # Get unique nodes
    all_nodes = pd.concat([df['Source'], df['Target']]).unique()
    if len(all_nodes) == 0:
        return df, [], 0 # Return empty if no nodes

    # Identify nodes with no incoming edges (these are the starting nodes, Layer 0)
    all_targets = set(df['Target'].unique())
    potential_layer_0_nodes = set(df['Source'].unique()) - all_targets

    # Start with nodes that are only sources (no incoming edges) as Layer 0
    nodes_in_current_layer = set(df['Source'].unique()) - set(df['Target'].unique())
    layer_index = 0

    while nodes_in_current_layer:
        # Assign current layer to these nodes
        for node in nodes_in_current_layer:
            node_layers[node] = layer_index
            nodes_assigned.add(node)

        # Find nodes for the next layer: targets of current layer's sources
        next_layer_nodes = set()
        current_layer_sources = [node for node in nodes_in_current_layer if node in df['Source'].unique()]

        if current_layer_sources:
            links_from_current_layer = df[df['Source'].isin(current_layer_sources)]
            next_layer_nodes = set(links_from_current_layer['Target'].unique()) - nodes_assigned

        nodes_in_current_layer = next_layer_nodes
        layer_index += 1

    # Handle nodes that were only targets and not sources, or disconnected nodes
    unassigned_nodes = set(all_nodes) - nodes_assigned
    if unassigned_nodes and layer_index > 0:
         # Assign remaining unassigned nodes to the last determined layer
         for node in unassigned_nodes:
              node_layers[node] = layer_index - 1 # Assign to the last layer
    elif unassigned_nodes and layer_index == 0:
         # If no layers were determined but there are nodes, assign them to layer 0
         for node in unassigned_nodes:
              node_layers[node] = 0


    # Determine the actual number of layers found
    num_determined_layers = max(node_layers.values()) + 1 if node_layers else 0
    determined_layer_names = [f"Layer {i+1}" for i in range(num_determined_layers)]

    # Use suggested layer names if they match the number of determined layers
    if len(suggested_layer_names) == num_determined_layers:
         display_layer_names = suggested_layer_names
    else:
         display_layer_names = determined_layer_names
         if suggested_layer_names:
              print(f"Warning: Number of determined layers ({num_determined_layers}) does not match suggested layer names ({len(suggested_layer_names)}). Using auto-generated layer names.")

    # === Add layer information to the DataFrame ===
    # Map Source and Target nodes to their determined layers
    df['Source_Layer'] = df['Source'].map(node_layers)
    df['Target_Layer'] = df['Target'].map(node_layers)

    # Handle potential NaN values if some nodes were not in node_layers (shouldn't happen with current logic but as safeguard)
    df['Source_Layer'] = df['Source_Layer'].fillna(-1).astype(int) # Use -1 or another indicator for unassigned
    df['Target_Layer'] = df['Target_Layer'].fillna(-1).astype(int) # Use -1 or another indicator for unassigned

    return df, display_layer_names, num_determined_layers


def generate_sankey_qa(input_folder, output_folder):
    """
    Reads all CSV files in the input_folder (assuming Sankey diagram format with layers),
    generates QA pairs based on the data, and saves the QA pairs as JSON files
    in the output_folder.

    Assumes CSV format:
    Metadata line (e.g., Theme Big,Theme Specific,Data Unit,Pattern,Layer1 Name,Layer2 Name,...)
    Header line: Source,Target,Value
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

            # Extract metadata from the first line
            # Assuming at least Theme Big, Theme Specific, Data Unit, Pattern are present
            if len(lines) < 3: # Need metadata, header, and at least one data row
                print(f"Skipping {csv_filename}: File too short or missing data.")
                continue

            meta_parts = lines[0].strip().split(",")
            # Extract unit from metadata (assuming it's the 3rd part)
            unit = meta_parts[2].strip() if len(meta_parts) > 2 else ""
            # Extract layer names (from the 5th element onwards)
            layer_names = [part.strip() for part in meta_parts[4:] if part.strip()]

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

            # Get unique nodes
            all_nodes = pd.concat([df['Source'], df['Target']]).unique()
            if len(all_nodes) == 0:
                 print(f"Skipping {csv_filename}: No nodes found in data.")
                 continue

            # === Determine Node Layers and add to DataFrame ===
            df, display_layer_names, num_determined_layers = determine_and_add_layers(df, layer_names)

            # Check if layer determination was successful and we have data
            if df.empty or num_determined_layers == 0:
                 print(f"Skipping {csv_filename}: Could not determine layers or data is empty after processing.")
                 continue

            # === Step 2: QA distribution ===
            # Define QA generators within the function scope
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
                    "A": "This chart is a {sankey} chart."
                }
            # End QA Category CTR


            # Begin QA Category VEC(Visual Element Count) Done.
            # Collect all VEC QA functions in a list for random sampling
            vec_qa_funcs = []

            @qa_generator("VEC")
            def link_count_qa(df):
                count = len(df) # Each row is a link
                return {
                    "Q": "How many links are in this sankey chart?",
                    "A": f"There are {{{count}}} links."
                }
            vec_qa_funcs.append(link_count_qa)

            @qa_generator("VEC")
            def node_count_qa(df):
                # Nodes are unique source and target values
                unique_nodes = pd.concat([df['Source'], df['Target']]).unique()
                count = len(unique_nodes)
                return {
                    "Q": "How many nodes are in this sankey chart?",
                    "A": f"There are {{{count}}} nodes."
                }
            vec_qa_funcs.append(node_count_qa)

            @qa_generator("VEC")
            def nodes_in_layer_qa(df):
                if not layer_names: return {}  # Skip if no layer names
                random_layer_index = random.randint(0, len(layer_names) - 1)
                random_layer_name = layer_names[random_layer_index]

                # Find nodes associated with the selected layer
                nodes_in_layer = set(df.loc[df['Source_Layer'] == random_layer_index, 'Source']).union(
                    df.loc[df['Target_Layer'] == random_layer_index, 'Target']
                )

                if not nodes_in_layer:
                    return {
                        "Q": f"How many nodes are in the column {random_layer_name} of this sankey chart?",
                        "A": f"There are no nodes in the column {random_layer_name}."
                    }

                node_count = len(nodes_in_layer)
                return {
                    "Q": f"How many nodes are in the column {random_layer_name} of this sankey chart?",
                    "A": f"There are {{{node_count}}} nodes in the column {random_layer_name}."
                }
            vec_qa_funcs.append(nodes_in_layer_qa)

            @qa_generator("VEC")
            def links_out_of_column_qa(df):
                unique_layers = df['Source_Layer'].unique()
                if len(unique_layers) == 0: return {}
                random_layer = random.choice(unique_layers)
                layer_name = layer_names[random_layer] if random_layer < len(layer_names) else f"Layer {random_layer + 1}"
                count = df[df['Source_Layer'] == random_layer].shape[0]
                return {
                    "Q": f"How many links flow out of column {layer_name} of this sankey chart?",
                    "A": f"There are {{{count}}} links flow out of column {layer_name}."
                }
            vec_qa_funcs.append(links_out_of_column_qa)

            @qa_generator("VEC")
            def links_into_column_qa(df):
                unique_layers = df['Target_Layer'].unique()
                if len(unique_layers) == 0: return {}
                random_layer = random.choice(unique_layers)
                layer_name = layer_names[random_layer] if random_layer < len(layer_names) else f"Layer {random_layer + 1}"
                count = df[df['Target_Layer'] == random_layer].shape[0]
                return {
                    "Q": f"How many links flow into column {layer_name} of this sankey chart?",
                    "A": f"There are {{{count}}} links flow into column {layer_name}."
                }
            vec_qa_funcs.append(links_into_column_qa)

            @qa_generator("VEC")
            def links_into_node_qa(df):
                if df.empty:
                    return {}
                random_node = random.choice(df['Target'].unique())
                count = df[df['Target'] == random_node].shape[0]
                return {
                    "Q": f"How many links flow into the node {random_node} in this sankey chart?",
                    "A": f"There are {{{count}}} links flowing into the node {random_node}."
                }
            vec_qa_funcs.append(links_into_node_qa)

            @qa_generator("VEC")
            def links_out_of_node_qa(df):
                if df.empty:
                    return {}
                random_node = random.choice(df['Source'].unique())
                count = df[df['Source'] == random_node].shape[0]
                return {
                    "Q": f"How many links flow out of the node {random_node} in this sankey chart?",
                    "A": f"There are {{{count}}} links flowing out of the node {random_node}."
                }
            vec_qa_funcs.append(links_out_of_node_qa)

            # Replace the default VEC QA generator with a random sampler
            def vec_qa_random_sampler(df):
                # Randomly select 2-4 unique VEC QA functions
                num_questions = random.randint(2, 4)
                sampled_funcs = random.sample(vec_qa_funcs, min(num_questions, len(vec_qa_funcs)))
                qa_list = []
                for func in sampled_funcs:
                    qa = func(df)
                    if qa:
                        qa_list.append(qa)
                return qa_list

            # Remove all previous VEC entries from qa_generators and add only the sampler
            qa_generators[:] = [(cat, f) for cat, f in qa_generators if cat != "VEC"]
            qa_generators.append(("VEC", vec_qa_random_sampler))
            # End QA Category VEC


            # Begin QA Category SRP(Spatial Relationship Perception)
            @qa_generator("SRP")
            def leftmost_axis_category_qa(df):
                # The leftmost axis corresponds to the first layer
                leftmost_layer_name = layer_names[0]
                return {
                    "Q": "Which category does the leftmost axis represent in this sankey chart?",
                    "A": f"The leftmost axis represents {{{leftmost_layer_name}}}."
                }

            @qa_generator("SRP")
            def rightmost_axis_category_qa(df):
                # The rightmost axis corresponds to the last layer
                rightmost_layer_name = layer_names[-1]
                return {
                    "Q": "Which category does the rightmost axis represent in this sankey chart?",
                    "A": f"The rightmost axis represents {{{rightmost_layer_name}}}."
                }
            # End QA Category SRP


            # Begin QA Category VPR(Visual Pattern Recognition)
            @qa_generator("VPR")
            def largest_flow_pair_qa(df):
                if df.empty: return {}
                # Find the row with the maximum 'Value'
                largest_link = df.loc[df['Value'].idxmax()]
                source = largest_link['Source']
                target = largest_link['Target']
                # Value is asked in VE/EVJ, VPR is about identifying the pair
                return {
                    "Q": "Which pair of nodes shows the largest flow in this sankey chart?",
                    "A": f"The largest flow is between {{{source}}} and {{{target}}}."
                }

            @qa_generator("VPR")
            def smallest_flow_pair_qa(df):
                if df.empty: return {}
                 # Find the row with the minimum 'Value'
                smallest_link = df.loc[df['Value'].idxmin()]
                source = smallest_link['Source']
                target = smallest_link['Target']
                 # Value is asked in VE/EVJ, VPR is about identifying the pair
                return {
                    "Q": "Which pair of nodes shows the smallest flow in this sankey chart?",
                    "A": f"The smallest flow is between {{{source}}} and {{{target}}}."
                }
            # End QA Category VPR


            # Begin QA Category VE(Value Extraction)
            @qa_generator("VE")
            def specific_flow_value_qa(df):
                if df.empty: return {}
                # Select 2-3 random rows (links)
                num_samples = random.randint(2, 3)  # Randomly choose 2 or 3 rows
                random_links = df.sample(num_samples)
                questions_answers = []

                for _, row in random_links.iterrows():
                    source = row['Source']
                    target = row['Target']
                    value = row['Value']
                    questions_answers.append({
                        "Q": f"What is the flow value from the node {source} to the node {target}?",
                        "A": f"The flow value from the node {source} to the node {target} is {{{value}}}."
                    })

                return questions_answers
            # End QA Category VE


            # Begin QA Category EVJ(Exaggerated Value Judgment)
            @qa_generator("EVJ")
            def max_flow_from_node_qa(df):
                if df.empty: return {}
                random_source = random.choice(df['Source'].unique())
                max_value = df[df['Source'] == random_source]['Value'].max()
                return {
                    "Q": f"What is the maximum flow value from the node {random_source} to the next stage in the sankey chart?",
                    "A": f"The maximum flow value from the node {random_source} to the next stage is {{{max_value}}}."
                }

            @qa_generator("EVJ")
            def min_flow_from_node_qa(df):
                if df.empty: return {}
                random_source = random.choice(df['Source'].unique())
                min_value = df[df['Source'] == random_source]['Value'].min()
                return {
                    "Q": f"What is the minimum flow value from the node {random_source} to the next stage in the sankey chart?",
                    "A": f"The minimum flow value from the node {random_source} to the next stage is {{{min_value}}}."
                }


            # Begin QA Category SC(Statistic Calculate)
            @qa_generator("SC")
            def total_output_flow_qa(df):
                if df.empty: return {}
                # Get unique source nodes that actually have output flow
                source_nodes = df['Source'].unique()
                if len(source_nodes) == 0: return {} # No sources with output

                # Select a random source node
                random_source = random.choice(source_nodes)
                total_output = df[df['Source'] == random_source]['Value'].sum()
                return {
                    "Q": f"What is the total flow output from node {random_source}?",
                    "A": f"The total flow output from node {random_source} is {{{total_output}}}."
                }

            @qa_generator("SC")
            def total_input_flow_qa(df):
                if df.empty: return {}
                 # Get unique target nodes that actually have input flow
                target_nodes = df['Target'].unique()
                if len(target_nodes) == 0: return {} # No targets with input

                # Select a random target node
                random_target = random.choice(target_nodes)
                total_input = df[df['Target'] == random_target]['Value'].sum()
                return {
                    "Q": f"What is the total flow input by node {random_target}?",
                    "A": f"The total flow input by node {random_target} is {{{total_input}}}."
                }
            # End QA Category SC


            # Begin QA Category NF(Numerical Filtering)
            @qa_generator("NF")
            def nf_single_outflow_exceeds_threshold_qa(df):
                candidate_layers = list(range(len(layer_names) - 1))
                chosen_layer = random.choice(candidate_layers)
                layer_name = layer_names[chosen_layer]

                layer_links = df[df['Source_Layer'] == chosen_layer]
                if layer_links.empty:
                    return {}

                values = layer_links['Value'].sort_values(ascending=False).tolist()
                num_to_filter = random.randint(1, 3)

                lower_idx = num_to_filter
                upper_idx = num_to_filter - 1
                lower_idx = min(lower_idx, len(values) - 1)
                upper_idx = min(upper_idx, len(values) - 1)

                threshold = (values[upper_idx] + values[lower_idx]) // 2
                filtered = layer_links[layer_links['Value'] > threshold]
                
                if filtered.empty:
                    return {}
                node_values = [(row['Source'], row['Value']) for _, row in filtered.iterrows()]
                q = f"In the column {layer_name} of the sankey chart, which nodes have an outflow exceeded {threshold}? Please list the node labels and their corresponding outflow values."
                a_list = []
                for idx, (node, val) in enumerate(node_values):
                    if idx == 0:
                        a_list.append(f"{{{node}}} has an outflow of {{{val}}}")
                    else:
                        a_list.append(f", {{{node}}} has an outflow of {{{val}}}")
                a_list.append('.')
                a = "".join(a_list)
                return {"Q": q, "A": a}

            @qa_generator("NF")
            def nf_layer_flow_in_range_qa(df):
                # choose a random layer
                candidate_layers = list(range(len(layer_names) - 1))
                chosen_layer = random.choice(candidate_layers)
                source_layer_name = layer_names[chosen_layer]
                target_layer_name = layer_names[chosen_layer+1]

                # get selected layer links 
                layer_links = df[df['Source_Layer'] == chosen_layer]
                if layer_links.empty:
                    return {}

                values = layer_links['Value'].sort_values(ascending=True).tolist()
                num_to_filter = random.randint(1, min(3, len(values)))

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


                filtered = layer_links[(layer_links['Value'] >= lower_threshold) &(layer_links['Value']<upper_threshold)]
                if filtered.empty:
                    return {
                        "Q": f"Between which pairs of nodes in column {source_layer_name} and column {target_layer_name} is the flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and the corresponding flow value.",
                        "A": f"No nodes have an flow between {lower_threshold} and {upper_threshold}."
                    }

                node_values = [(row['Source'],row['Target'], row['Value']) for _, row in filtered.iterrows()]

                q = f"Between which pairs of nodes in column {source_layer_name} and column {target_layer_name} is the flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and the corresponding flow value."

                a_list = []
                for idx, (src,tgt, val) in enumerate(node_values):
                    if idx == 0:
                        a_list.append(f"There is a flow of {{{val}}} from node {{{src}}} to {{{tgt}}}")
                    else:
                        a_list.append(f", a flow of {{{val}}} from node {{{src}}} to {{{tgt}}}")
                a_list.append('.')
                a = "".join(a_list)
                return {"Q": q, "A": a}
            
            @qa_generator("NF")
            def nf_outflow_in_range_qa(df):
                if df.empty:
                    return {}

                values = df['Value'].sort_values(ascending=True).tolist()
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

                # Filter links with flow values within the specified range
                filtered_links = df[(df['Value'] >= lower_threshold) & (df['Value'] < upper_threshold)]

                if filtered_links.empty:
                    return {
                        "Q": f"In the sankey chart, which links have a flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and their corresponding flow values.",
                        "A": f"No links have a flow between {lower_threshold} and {upper_threshold}."
                    }

                q = f"In the sankey chart, which links have a flow between {lower_threshold} and {upper_threshold}? Please list the source node, target node, and their corresponding flow values."

                node_values = [(row['Source'],row['Target'], row['Value']) for _, row in filtered_links.iterrows()]
                a_list = []
                for idx, (src,tgt, val) in enumerate(node_values):
                    if idx == 0:
                        a_list.append(f"There is a flow of {{{val}}} from node {{{src}}} to {{{tgt}}}")
                    else:
                        a_list.append(f", a flow of {{{val}}} from node {{{src}}} to {{{tgt}}}")
                a_list.append('.')
                a = "".join(a_list)
                return {"Q": q, "A": a}
            # End QA Category NF



            # Begin QA Category NC(Numerical Compare)
            @qa_generator("NC")
            def compare_multiple_flows_qa(df):
                if len(df) < 3:
                    return {
                        "Q": "Cannot compare flows.",
                        "A": "Requires at least 3 distinct links."
                    }

                sampled_links = df.sample(3, replace=False).reset_index(drop=True)
                
                pairs = list(itertools.combinations(range(3), 2))
                
                num_comparisons = random.randint(1, 2)
                selected_pairs = random.sample(pairs, num_comparisons)

                qa_pairs = []

                for i1, i2 in selected_pairs:
                    link1, link2 = sampled_links.iloc[i1], sampled_links.iloc[i2]
                    source1, target1, value1 = link1['Source'], link1['Target'], link1['Value']
                    source2, target2, value2 = link2['Source'], link2['Target'], link2['Value']

                    if value1 > value2:
                        comparison = f"The flow from node {{{source1}}} to node {{{target1}}} is larger."
                    elif value1 < value2:
                        comparison = f"The flow from node {{{source2}}} to node {{{target2}}} is larger."
                    else:
                        comparison = f"The flow from node {{{source1}}} to node {{{target1}}} is equal to the flow from node {{{source2}}} to node {{{target2}}}."

                    question = f"Between the flows from node {source1} to node {target1} and from node {source2} to node {target2}, which path has a larger flow?"
                    answer = comparison

                    qa_pairs.append({
                        "Q": question,
                        "A": answer
                    })

                return qa_pairs

            
            @qa_generator("NC")
            def compare_three_flows_qa(df):
                if len(df) < 6:
                    return {
                        "Q": "Cannot compare three flows.",
                        "A": "Requires at least 6 distinct links to form two triplets."
                    }

                sampled_links = df.sample(6, replace=False).reset_index(drop=True)

                triplet1 = sampled_links.iloc[0:3]
                triplet2 = sampled_links.iloc[3:6]

                num_triplets_to_use = random.randint(1, 2)
                selected_triplets = [triplet1] if num_triplets_to_use == 1 else [triplet1, triplet2]

                qa_pairs = []

                for triplet in selected_triplets:
                    source1, target1, value1 = triplet.iloc[0]['Source'], triplet.iloc[0]['Target'], triplet.iloc[0]['Value']
                    source2, target2, value2 = triplet.iloc[1]['Source'], triplet.iloc[1]['Target'], triplet.iloc[1]['Value']
                    source3, target3, value3 = triplet.iloc[2]['Source'], triplet.iloc[2]['Target'], triplet.iloc[2]['Value']

                    if value1 > value2 and value1 > value3:
                        answer = f"The flow from node {{{source1}}} to node {{{target1}}} is the largest."
                    elif value2 > value1 and value2 > value3:
                        answer = f"The flow from node {{{source2}}} to node {{{target2}}} is the largest."
                    elif value3 > value1 and value3 > value2:
                        answer = f"The flow from node {{{source3}}} to node {{{target3}}} is the largest."
                    else:
                        answer = "There is a tie between the flows."

                    question = (
                        f"Among the flows from node {source1} to node {target1}, from node {source2} to node {target2}, "
                        f"and from node {source3} to node {target3}, which path has the largest flow?"
                    )

                    qa_pairs.append({
                        "Q": question,
                        "A": answer
                    })

                return qa_pairs
            # End QA Category NC


            # Begin QA Category MSR(Multi-Step Reasoning)
            #       Q: Which node in the second column of the sankey chart is the most critical?
            #       A: The most critical node in the second column is China.

            @qa_generator("MSR")
            def msr_most_critical_node_random_column_qa(df):
                # Randomly select a layer (except the first and last, if possible)
                candidate_layers = sorted(set(df['Source_Layer']).union(set(df['Target_Layer'])))
   
                chosen_layer = random.choice(candidate_layers)
                # Find all nodes in the chosen layer (either as Source or Target)
                nodes_in_layer = set(df.loc[df['Source_Layer'] == chosen_layer, 'Source']).union(
                    df.loc[df['Target_Layer'] == chosen_layer, 'Target']
                )
    
                # For each node, sum all incoming and outgoing flows
                node_scores = {}
                for node in nodes_in_layer:
                    in_flow = df[df['Target'] == node]['Value'].sum()
                    out_flow = df[df['Source'] == node]['Value'].sum()
                    node_scores[node] = in_flow + out_flow
                # Find the node with the highest score
                most_critical_node = max(node_scores, key=node_scores.get)
                layer_name = layer_names[chosen_layer] if chosen_layer < len(layer_names) else f"Layer {chosen_layer+1}"
                return {
                    "Q": f"Which node in the column {layer_name} of the sankey chart is the most critical?",
                    "A": f"The most critical node in the column {layer_name} is {{{most_critical_node}}}."
                }
            
            #       Q: What is the path with the highest flow from source node China to target node USA? Please list all node labels along this path.
            #       A: The path with the highest flow from node China to node USA is: China, B, C, USA.
            @qa_generator("MSR")
            def msr_highest_flow_path_qa_sum(df):
                # Randomly select a source node (from first layer) and a target node (from last layer)
                first_layer = df['Source_Layer'].min()
                last_layer = df['Target_Layer'].max()
                source_candidates = df[df['Source_Layer'] == first_layer]['Source'].unique()
                target_candidates = df[df['Target_Layer'] == last_layer]['Target'].unique()

                source_node = random.choice(list(source_candidates))
                target_node = random.choice(list(target_candidates))
                # Build adjacency list with values
                graph = defaultdict(list)
                for _, row in df.iterrows():
                    graph[row['Source']].append((row['Target'], row['Value']))
                # DFS to find all paths from source_node to target_node, tracking the sum of flow along each path
                max_total_flow = -float('inf')
                best_path = []
                # DFS 函数定义，用于寻找总流量最大的路径
                def dfs(current, path, current_total_flow):
                    nonlocal max_total_flow, best_path
                    if current == target_node:
                        if current_total_flow > max_total_flow: 
                            max_total_flow = current_total_flow 
                            best_path = path[:] 
                        return 
                    for neighbor, value in graph.get(current, []): 
                        if neighbor not in path:  
                            dfs(neighbor, path + [neighbor], current_total_flow + value)

                dfs(source_node, [source_node], 0) # 从源节点开始 DFS，初始路径为 [源节点]，初始总流量为 0

  
                path_str = ", ".join(best_path)
                return {
                    "Q": f"What is the path with the highest total flow from source node {source_node} to target node {target_node}? Please list all node labels along this path.",
                    "A": f"The path with the highest total flow from node {source_node} to node {target_node} is: {{{path_str}}}."
                }
            # End QA Category MSR

            
            # Begin QA Category VA
            """
             使用csv数据可做。使用Barycentric Heuristic（重心启发式算法）减少相邻两列的交叉, 参考右侧附件
            Q: To minimize the crossings between column 2 and column 3, please reorder the nodes in column 2 based on Barycenter Heuristic, and list the sorted node order (from top to bottom).
            A: The reordered node order from top to bottom is: Node C, Node B, Node A.
            """
            @qa_generator("VA")
            def va_avoid_crossing_qa(df):
                return
                # # Step 1: 随机选择两个相邻层
                # layer_idx = random.randint(0, num_determined_layers - 2)
                # left_layer = layer_idx
                # right_layer = layer_idx + 1
                # left_layer_name = layer_names[left_layer]
                # right_layer_name = layer_names[right_layer]

                # # Step 2: 获取两个层之间的全部流
                # flow_df = df[(df['Source_Layer'] == left_layer) & (df['Target_Layer'] == right_layer)]

                # # Step 3: 构造节点列表和右层节点位置映射
                # left_nodes = sorted(set(flow_df['Source']))
                # right_nodes = sorted(set(flow_df['Target']))
                # right_node_pos = {node: idx for idx, node in enumerate(right_nodes)}
                # right_node_count = len(right_nodes)

                # # Step 4: 每个左层节点使用 1/N 阈值保留主连接
                # barycenters = {}
                # for node in left_nodes:
                #     node_flows = flow_df[flow_df['Source'] == node]
                #     total_out = node_flows['Value'].sum()

                #     if total_out == 0 or right_node_count == 0:
                #         barycenters[node] = float('inf')
                #         continue

                #     threshold = total_out / right_node_count  # 比如 5个右节点 → 阈值 = sum/5
                #     strong_links = node_flows[node_flows['Value'] >= threshold]
                #     targets = strong_links['Target'].tolist()
                #     positions = [right_node_pos[t] for t in targets if t in right_node_pos]

                #     barycenters[node] = sum(positions) / len(positions) if positions else float('inf')

                # # Step 5: 排序并生成 QA 对
                # sorted_nodes = sorted(left_nodes, key=lambda n: (barycenters[n], str(n)))
                # node_order_str = ", ".join(sorted_nodes)

                # q = (
                #     f"To reduce crossings between column {left_layer_name} and column {right_layer_name}, "
                #     f"reorder the nodes in column {left_layer_name} based on Barycenter Heuristic. "
                #     f"Only consider flows whose values exceed the average outflow from the node, "
                #     f"list the sorted node order (from top to bottom)."
                # )
                # a = f"The reordered node order from top to bottom is: {{{node_order_str}}}."

                # return {"Q": q, "A": a}
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


INPUT_DIR = "csv/sankey"
OUTPUT_DIR = "json/sankey"
if __name__ == '__main__':
    generate_sankey_qa(INPUT_DIR, OUTPUT_DIR)
