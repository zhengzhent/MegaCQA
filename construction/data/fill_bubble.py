import random
import csv
from typing import List, Dict, Any, Optional, Tuple
import os
import io # Import io for handling BOM

# --- Configuration Data ---

# Define 15 topics and their hierarchical labels (up to 3 layers)
# Format: { "Topic Name": { "Depth 2 Label": ["Depth 3 Label 1", "Depth 3 Label 2", ...], ... }, ... }
hierarchical_topics: Dict[str, Dict[str, List[str]]] = {
    "Technology and Software": {
        "Software Development": ["Frontend", "Backend", "Mobile", "Database", "DevOps", "Testing"],
        "Artificial Intelligence": ["Machine Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Neural Networks"],
        "Cloud Computing": ["AWS", "Azure", "Google Cloud", "Cloud Native", "Serverless"],
        "Cybersecurity": ["Network Security", "Endpoint Security", "Cryptography", "Threat Intelligence", "Vulnerability Management"],
        "Data Science": ["Data Analysis", "Big Data", "Data Visualization", "Statistical Modeling", "Data Engineering"]
    },
    "Healthcare and Medicine": {
        "Medical Specialties": ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Dermatology", "Psychiatry"],
        "Medical Procedures": ["Surgery", "Diagnosis", "Therapy", "Rehabilitation", "Preventive Care"],
        "Pharmaceuticals": ["Drug Discovery", "Clinical Trials", "Pharmacology", "Pharmacy Practice"],
        "Hospital Management": ["Patient Care", "Administration", "Healthcare Informatics", "Quality Control"]
    },
    "Business and Finance": {
        "Investments": ["Stocks", "Bonds", "Real Estate", "Cryptocurrency", "Mutual Funds", "ETFs"],
        "Banking": ["Retail Banking", "Commercial Banking", "Investment Banking", "Digital Banking"],
        "Financial Markets": ["Stock Market", "Forex Market", "Commodity Market", "Derivatives"],
        "Accounting and Audit": ["Taxation", "Compliance", "Financial Reporting", "Auditing"]
    },
    "Education and Academics": {
        "Sciences": ["Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Environmental Science"],
        "Humanities": ["History", "Literature", "Philosophy", "Linguistics", "Archaeology", "Cultural Studies"],
        "Engineering": ["Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Chemical Engineering", "Aerospace Engineering"],
        "Social Sciences": ["Psychology", "Sociology", "Economics", "Political Science", "Anthropology"]
    },
    "Food and Beverage": {
        "Cuisines": ["Italian Cuisine", "Mexican Cuisine", "Asian Cuisine", "Mediterranean Cuisine", "French Cuisine", "Indian Cuisine"],
        "Ingredients": ["Fruits and Vegetables", "Meats and Poultry", "Dairy and Eggs", "Grains and Legumes", "Spices and Herbs", "Seafood"],
        "Beverages": ["Coffee and Tea", "Alcoholic Drinks", "Juices and Smoothies", "Soft Drinks", "Water"]
    },
    "Transportation and Logistics": {
        "Modes of Transport": ["Road Transport", "Rail Transport", "Air Transport", "Maritime Transport", "Pipeline Transport"],
        "Logistics Operations": ["Warehousing", "Inventory Management", "Supply Chain", "Freight Forwarding", "Last-Mile Delivery"],
        "Vehicle Technology": ["Electric Vehicles", "Autonomous Vehicles", "Aerospace Technology", "Marine Engineering"]
    },
    "Environmental Science": {
        "Conservation": ["Wildlife Conservation", "Forest Conservation", "Marine Conservation", "Habitat Restoration"],
        "Renewable Energy": ["Solar Energy", "Wind Energy", "Hydro Energy", "Geothermal Energy", "Bioenergy"],
        "Pollution Control": ["Air Pollution", "Water Pollution", "Waste Management", "Soil Remediation"]
    },
    "Arts and Culture": {
        "Visual Arts": ["Painting", "Sculpture", "Photography", "Drawing", "Digital Art"],
        "Performing Arts": ["Music", "Theater", "Dance", "Opera", "Circus Arts"],
        "Literature and Writing": ["Fiction", "Non-Fiction", "Poetry", "Playwriting", "Screenwriting"],
        "Museums and Galleries": ["Collections", "Exhibitions", "Conservation", "Curatorial Studies"]
    },
    "Sports and Recreation": {
        "Team Sports": ["Soccer", "Basketball", "Baseball", "Volleyball", "American Football", "Hockey"],
        "Individual Sports": ["Tennis", "Swimming", "Athletics", "Gymnastics", "Cycling", "Badminton"],
        "Outdoor Activities": ["Hiking", "Camping", "Skiing", "Snowboarding", "Rock Climbing", "Kayaking"]
    },
    "Real Estate and Construction": {
        "Property Types": ["Residential Real Estate", "Commercial Real Estate", "Industrial Real Estate", "Land Development"],
        "Construction Phases": ["Planning and Design", "Building", "Finishing", "Inspection"],
        "Market Analysis": ["Valuation", "Trends", "Investment Analysis"],
        "Building Materials": ["Concrete", "Steel", "Wood", "Masonry"]
    },
     "Media and Communication": {
        "Social Media Platforms": ["Facebook", "Instagram", "Twitter", "TikTok", "LinkedIn", "Pinterest"],
        "Traditional Media": ["Television", "Radio", "Print Journalism", "Film"],
        "Digital Content": ["Streaming Services", "Podcasting", "Online News", "Blogging", "Vlogging"]
    },
    "Government and Politics": {
        "Levels of Government": ["Federal Government", "State Government", "Local Government", "International Relations"],
        "Policy Areas": ["Economic Policy", "Social Policy", "Foreign Policy", "Environmental Policy", "Healthcare Policy"],
        "Political Processes": ["Elections", "Legislation", "Lobbying", "Public Administration"]
    },
    "Manufacturing and Industry": {
        "Industry Sectors": ["Automotive Manufacturing", "Electronics Manufacturing", "Textile Manufacturing", "Food Processing", "Chemical Manufacturing"],
        "Production Processes": ["Assembly Line", "Quality Control", "Supply Chain Management", "Automation", "Lean Manufacturing"],
        "Materials Science": ["Metals", "Plastics", "Composites", "Ceramics"]
    },
    "Tourism and Hospitality": {
        "Accommodation": ["Hotels", "Resorts", "Vacation Rentals", "Hostels", "Bed and Breakfasts"],
        "Tourism Types": ["Adventure Tourism", "Cultural Tourism", "Eco-Tourism", "Business Tourism", "Medical Tourism"],
        "Destinations": ["Urban Destinations", "Coastal Destinations", "Mountain Destinations", "Rural Destinations"]
    },
    "Retail and E-commerce": {
        "Retail Channels": ["Physical Stores", "Online Stores", "Mobile Commerce", "Pop-up Shops"],
        "Product Categories": ["Apparel", "Electronics", "Home Goods", "Groceries", "Health and Beauty", "Books and Media"],
        "Customer Experience": ["Customer Service", "Personalization", "Loyalty Programs", "User Interface (UI)"]
    }
}

# Dictionary to keep track of the sequence number for each topic
topic_counters: Dict[str, int] = {}

# --- Data Generation Class ---

class DataGenerator:
    def __init__(self,
                 topic: str,
                 num_files: int = 1,
                 root_size_range: Tuple[float, float] = (50, 100),
                 d2_size_range: Tuple[float, float] = (15, 40),
                 d3_size_range: Tuple[float, float] = (5, 15),
                 total_children_range: Tuple[int, int] = (3, 10),
                 d2_count_range_base: Tuple[int, int] = (1, 6) # Base range for number of D2 nodes
                ):
        """
        Initializes the data generator for hierarchical bubble charts.

        Args:
            topic: The root topic for data generation (must be a key in hierarchical_topics).
            num_files: The number of CSV files to generate for this topic.
            root_size_range: Tuple (min, max) for root node size.
            d2_size_range: Tuple (min, max) for depth 2 node size.
            d3_size_range: Tuple (min, max) for depth 3 node size.
            total_children_range: Tuple (min, max) for the total number of child nodes (D2 + D3).
            d2_count_range_base: Tuple (min, max) for the base random range of depth 2 node count.
        """
        if topic not in hierarchical_topics:
            raise ValueError(f"Unknown topic: {topic}. Available topics are: {list(hierarchical_topics.keys())}")

        self.topic = topic
        self.num_files = num_files
        self.hierarchy_structure = hierarchical_topics[topic]
        self.root_size_range = root_size_range
        self.d2_size_range = d2_size_range
        self.d3_size_range = d3_size_range
        self.total_children_range = total_children_range
        self.d2_count_range_base = d2_count_range_base


    def generate_single_tree_data(self) -> List[Dict[str, Any]]:
        """
        Generates data for a single hierarchical tree based on the chosen topic.
        The structure will have 1 root (topic) and 3 to 10 child nodes total (Depth 2 + Depth 3).
        Node labels (except root) will be unique within the generated tree.
        """
        nodes_with_temp_id: List[Dict[str, Any]] = []
        node_counter = 1 # 1-based temporary ID counter
        used_labels: set[str] = set() # To ensure uniqueness of labels (except root)

        # --- Depth 1 (Root Node) ---
        root_label = self.topic
        # Root node size
        root_size = random.uniform(*self.root_size_range)
        root_node = {"size": round(root_size, 1), "father": 0, "depth": 1, "label": root_label, "temp_id": node_counter}
        nodes_with_temp_id.append(root_node)
        node_counter += 1
        # --------------------------

        # --- Decide Total Children and Distribution ---
        total_children = random.randint(*self.total_children_range) # Total number of nodes at Depth 2 and Depth 3 combined

        # Determine available Depth 2 labels for this topic
        available_d2_labels = list(self.hierarchy_structure.keys())
        random.shuffle(available_d2_labels)

        # Decide how many Depth 2 nodes will be generated directly under the root
        # Use the base range, but clamp by 1, total_children, and available D2 labels
        min_d2_count = max(1, self.d2_count_range_base[0]) # Must have at least 1 D2 node if total_children > 0
        max_d2_count = min(total_children, len(available_d2_labels), self.d2_count_range_base[1])

        # Ensure min_d2_count is not greater than max_d2_count if constraints are tight
        if min_d2_count > max_d2_count:
            # This means based on total_children or available labels, we can't even meet the minimum D2 count from the base range.
            # In this case, set max_d2_count as the target number of D2 nodes.
            num_d2_to_generate = max_d2_count
        else:
            num_d2_to_generate = random.randint(min_d2_count, max_d2_count)

        # Decide how many Depth 3 nodes will be generated
        num_d3_to_generate = total_children - num_d2_to_generate


        # --- Generate Depth 2 Nodes ---
        generated_d2_nodes_with_temp_id: List[Dict[str, Any]] = []
        d2_labels_to_use = available_d2_labels[:num_d2_to_generate] # Select the labels

        for d2_label in d2_labels_to_use:
            # Ensure label hasn't been used (defensive check)
            if d2_label not in used_labels:
                # Depth 2 node size
                d2_size = random.uniform(*self.d2_size_range)
                d2_node = {"size": round(d2_size, 1), "father": root_node["temp_id"], "depth": 2, "label": d2_label, "temp_id": node_counter}
                nodes_with_temp_id.append(d2_node)
                generated_d2_nodes_with_temp_id.append(d2_node)
                used_labels.add(d2_label)
                node_counter += 1
            # If label was already used or not enough available, just skip.


        # --- Generate Depth 3 Nodes ---
        # We need to generate num_d3_to_generate nodes, assigning them to generated D2 parents
        # and using unique labels from the available D3 labels under those parents.

        # Create a pool of available (d2_node_temp_id, d3_label) tuples based on generated D2 nodes and unused labels
        available_d3_options: List[Tuple[int, str]] = [] # List of (parent_temp_id, d3_label)
        for d2_node in generated_d2_nodes_with_temp_id:
            parent_temp_id = d2_node['temp_id']
            d2_label = d2_node['label']
            potential_d3_labels = self.hierarchy_structure.get(d2_label, [])
            for d3_label in potential_d3_labels:
                if d3_label not in used_labels:
                    available_d3_options.append((parent_temp_id, d3_label))

        # Shuffle available D3 options and select up to num_d3_to_generate unique ones
        random.shuffle(available_d3_options)
        actual_d3_nodes_to_generate = min(num_d3_to_generate, len(available_d3_options))

        d3_options_to_use = available_d3_options[:actual_d3_nodes_to_generate]

        for parent_temp_id, d3_label in d3_options_to_use:
             if d3_label not in used_labels: # Double check uniqueness (should be handled by available_d3_options logic)
                # Depth 3 node size
                d3_size = random.uniform(*self.d3_size_range)
                d3_node = {"size": round(d3_size, 1), "father": parent_temp_id, "depth": 3, "label": d3_label, "temp_id": node_counter}
                nodes_with_temp_id.append(d3_node)
                used_labels.add(d3_label)
                node_counter += 1


        # --- Final Data Preparation: Re-index Father IDs ---
        # Now that all nodes are generated with temporary IDs and father links using temp IDs,
        # create the final list with father IDs pointing to the 1-based index in the final list.

        final_data_list: List[Dict[str, Any]] = []
        # Map temp_id to final 1-based index (0 maps to 0 for root's father)
        temp_id_to_final_index: Dict[int, int] = {0: 0}

        for i, node in enumerate(nodes_with_temp_id):
             # Ensure 'temp_id' key exists before accessing
             if 'temp_id' not in node:
                 # This indicates a logic error in node creation if it happens
                 # print(f"Error: Node is missing 'temp_id' key: {node}") # Debugging print
                 continue # Skip this node or handle error appropriately

             temp_id = node['temp_id']
             temp_father_id = node['father']

             # Find the final 1-based index of the father
             # Use .get with a default of 0 in case father is the original root's temp_id (which is 1)
             # Or if there's an unexpected father ID (shouldn't happen with current logic)
             final_father_index = temp_id_to_final_index.get(temp_father_id, 0)

             final_node = {
                 "size": node['size'],
                 "father": final_father_index,
                 "depth": node['depth'],
                 "label": node['label']
             }
             final_data_list.append(final_node)
             # Map the current node's temporary ID to its final 1-based index
             temp_id_to_final_index[temp_id] = len(final_data_list) # len(list) is the next index, so it's the current node's 1-based index


        # Check if the generated data list size is within the desired range (4-11)
        # This check is also done in generate_all's while loop, but an extra check here is fine.
        # if not (4 <= len(final_data_list) <= 11):
        #     print(f"Debug: Generated tree has {len(final_data_list)} nodes. Expected 4-11.")


        return final_data_list


    def save_to_csv(self, data: List[Dict[str, Any]], file_index: int):
        """
        Saves the data to a CSV file with metadata comments on the first line.

        Args:
            data: The list of dictionaries representing the hierarchical data.
            file_index: The sequence number for the file (1-based).
        """
        # Create directory if it doesn't exist
        output_dir = './csv/fill_bubble'
        os.makedirs(output_dir, exist_ok=True)

        # Sanitize topic for filename
        sanitized_topic = self.topic.replace(' ', '_').replace('&', 'and').replace(',', '').replace(':', '').lower()

        # Construct filename (topic_xx.csv, xx from 1, no leading zero for single digits)
        filename = os.path.join(output_dir, f"{sanitized_topic}_{file_index}.csv")

        # Construct the single comment line
        # Fields: size (node size), father (parent node ID, 0 for root), depth (hierarchy level), label (node name)
        # REMOVED "Hierarchy Depth: 1-3," as requested
        comment_line = f"# Topic: {self.topic}, Total Nodes: {len(data)}, Fields: size (node size), father (parent node ID, 0 for root), depth (hierarchy level), label (node name)"

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Use csvfile.write() to write the comment line directly, avoiding csv.writer's quoting
            csvfile.write(comment_line + '\n')

            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["size", "father", "depth", "label"])
            # Write data
            for row in data:
                # Ensure order matches header and handle potential missing keys defensively
                writer.writerow([row.get("size", ""), row.get("father", ""), row.get("depth", ""), row.get("label", "")])

        print(f"Generated {filename}")

    def generate_all(self):
        """Generates the specified number of data files for the topic."""
        for i in range(1, self.num_files + 1):
            # Regenerate data until the total node count is within the desired range (4-11)
            while True:
                data = self.generate_single_tree_data()
                if 4 <= len(data) <= 11: # Check against the fixed requirement of 4-11 total nodes
                    break
                # Optional: print regeneration info for debugging
                # print(f"Generated {len(data)} nodes for {self.topic}, file {i}. Regenerating to meet 4-11 node count.")

            self.save_to_csv(data, i)


# --- Main Execution ---
if __name__ == "__main__":
    # === Customizable Parameters ===
    # >>> USER: Configure data generation here <<<

    # List of topics to generate data for. Choose from keys in hierarchical_topics.
    # Example: Generate for all 15 topics
    topics_to_generate: List[str] = list(hierarchical_topics.keys())

    # Number of files to generate for *each* topic in topics_to_generate list.
    NUM_FILES_PER_TOPIC = 1 # Example: Generate 5 files per topic

    # Random range for node sizes
    ROOT_SIZE_RANGE: Tuple[float, float] = (50.0, 100.0)
    D2_SIZE_RANGE: Tuple[float, float] = (15.0, 40.0)
    D3_SIZE_RANGE: Tuple[float, float] = (5.0, 15.0)

    # Random range for the total number of child nodes (Depth 2 + Depth 3)
    # Total nodes will be 1 (root) + random value from this range.
    # The while loop in generate_all ensures total nodes are 4-11.
    # Setting this range to (3, 10) directly targets 4-11 total nodes.
    TOTAL_CHILDREN_RANGE: Tuple[int, int] = (3, 10)


    # Base random range for the number of Depth 2 nodes generated directly under the root.
    # The actual number generated will also be limited by the total_children and available labels.
    # E.g., (1, 6) means attempt to generate 1 to 6 D2 nodes, but not more than total_children or available D2 labels.
    D2_COUNT_RANGE_BASE: Tuple[int, int] = (1, 6)


    # === End Customizable Parameters ===


    # --- Generate data for each specified topic ---
    total_topics = len(topics_to_generate)
    print(f"Starting data generation for {total_topics} topics, {NUM_FILES_PER_TOPIC} file(s) per topic.")

    for i, topic_name in enumerate(topics_to_generate):
        print(f"\n--- Generating data for topic: '{topic_name}' ({i + 1}/{total_topics}) ---")
        try:
            generator = DataGenerator(
                topic=topic_name,
                num_files=NUM_FILES_PER_TOPIC,
                root_size_range=ROOT_SIZE_RANGE,
                d2_size_range=D2_SIZE_RANGE,
                d3_size_range=D3_SIZE_RANGE,
                total_children_range=TOTAL_CHILDREN_RANGE,
                d2_count_range_base=D2_COUNT_RANGE_BASE
            )
            generator.generate_all()
        except ValueError as e:
            print(f"Error initializing generator for topic '{topic_name}': {e}. Skipping.")
        except Exception as e:
            # Catching general exceptions to print the error and skip the topic
            print(f"An unexpected error occurred while generating data for topic '{topic_name}': {e}. Skipping.")

    print("\nData generation process finished.")
