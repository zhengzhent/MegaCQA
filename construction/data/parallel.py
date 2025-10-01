import os
import random
import csv
import numpy as np
from faker import Faker

# Constants
TREND_PATTERNS = [
    "Strong Correlation",
    "Weak Correlation",
    "No Correlation",
    "Clustering"
]
###################################################################################
theme = "Social Media and Digital Media and Streaming"
sub_theme = "Content Performance"
dimensions = [
    ["View count (millions)", [0.001, 10000]],
    ["Completion rate (%)", [10, 100]],
    ["Content freshness (days)", [0, 365]],
    ["Personalization accuracy (%)", [50, 95]],
    ["Buffer time (seconds)", [0, 10]],
    ["Recommendation effectiveness (%)", [10, 50]],
    ["Creator diversity index (1-10)", [1, 10]]
]


###################################################################################
def generate_dataset(num_categories=None, num_dimensions=None, min_overlap=0.3, max_overlap=0.7):
    """Generate radar chart dataset with specified overlap constraints"""
    fake = Faker()

    if num_categories is None:
        num_categories = random.randint(5, 7)
    if num_dimensions is None:
        num_dimensions = random.randint(3, 7)

    if len(dimensions) < num_dimensions:
        raise ValueError(f"Not enough dimensions defined. Need {num_dimensions}, but only {len(dimensions)} available.")

    selected_dims = random.sample(dimensions, num_dimensions)
    dim_names = [dim[0] for dim in selected_dims]
    dim_ranges = [dim[1] for dim in selected_dims]

    category_names = [fake.company() for _ in range(num_categories)]

    dimension_patterns = []
    for i in range(num_dimensions - 1):
        pattern = random.choice(TREND_PATTERNS)
        dimension_patterns.append((dim_names[i], dim_names[i + 1], pattern))

    data = np.zeros((num_categories, num_dimensions))

    base_data = np.random.rand(num_categories, num_dimensions)

    for i in range(num_dimensions - 1):
        pattern = dimension_patterns[i][2]
        dim1_data = base_data[:, i]
        dim2_data = base_data[:, i + 1]

        if pattern == "Strong Correlation":
            noise = np.random.normal(0, 0.05, num_categories)
            dim2_data = dim1_data * (0.8 + 0.4 * np.random.rand()) + noise

        elif pattern == "Weak Correlation":
            noise = np.random.normal(0, 0.2, num_categories)
            dim2_data = dim1_data * (0.3 + 0.4 * np.random.rand()) + noise

        elif pattern == "No Correlation":
            dim2_data = np.random.rand(num_categories)

        elif pattern == "Clustering":
            centers = np.random.rand(3)
            for j in range(num_categories):
                cluster = j % 3
                dim2_data[j] = centers[cluster] + np.random.normal(0, 0.1)

        base_data[:, i + 1] = dim2_data

    data = base_data

    enforce_overlap_constraints(data, min_overlap, max_overlap)

    scaled_data = scale_data_to_ranges(data, dim_ranges)
    scaled_data = np.round(scaled_data, 2)
    return category_names, dim_names, scaled_data, dimension_patterns


def enforce_overlap_constraints(data, min_overlap, max_overlap):
    """Adjust data to meet overlap constraints"""
    num_categories, num_dimensions = data.shape

    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            overlap = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))

            if overlap < min_overlap or overlap > max_overlap:
                target_overlap = random.uniform(min_overlap, max_overlap)

                current_cos_theta = overlap
                target_cos_theta = target_overlap

                current_angle = np.arccos(np.clip(current_cos_theta, -1, 1))
                target_angle = np.arccos(np.clip(target_cos_theta, -1, 1))
                rotation_angle = target_angle - current_angle

                if np.linalg.norm(data[i]) > 1e-6 and np.linalg.norm(data[j]) > 1e-6:
                    j_orth = data[j] - (np.dot(data[j], data[i]) / np.dot(data[i], data[i])) * data[i]

                    if np.linalg.norm(j_orth) > 1e-6:
                        u = data[i] / np.linalg.norm(data[i])
                        v = j_orth / np.linalg.norm(j_orth)

                        data[j] = np.cos(rotation_angle) * u * np.linalg.norm(data[j]) + \
                                  np.sin(rotation_angle) * v * np.linalg.norm(data[j])


def scale_data_to_ranges(data, dim_ranges):
    """Scale data to specified dimension ranges"""
    num_categories, num_dimensions = data.shape
    scaled_data = np.zeros_like(data)

    for i in range(num_dimensions):
        dim_min, dim_max = dim_ranges[i]
        col_data = data[:, i]

        if np.ptp(col_data) == 0:
            scaled_col = np.full_like(col_data, (dim_min + dim_max) / 2)
        else:
            scaled_col = (col_data - np.min(col_data)) / np.ptp(col_data) * (dim_max - dim_min) + dim_min

        scaled_data[:, i] = scaled_col

    return scaled_data


def save_dataset(category_names, dimensions, data, dimension_patterns, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        csvfile.write(f"{theme},{sub_theme}\n")
        writer.writerow(["labels"] + dimensions)

        pattern_row = ["Pattern"]
        for i in range(len(dimensions) - 1):
            if i < len(dimension_patterns):
                pattern_row.append(f"{dimension_patterns[i][2]}")
        writer.writerow(pattern_row)

        for name, values in zip(category_names, data):
            writer.writerow([name] + list(values))


def main():
    num_datasets = 10
    save_path = "csv/parallel_coordinates"

    for i in range(1, num_datasets + 1):
        categories, dims, data, dimension_patterns = generate_dataset()
        filename = f"Parallel_graph_{i}.csv"
        save_dataset(categories, dims, data, dimension_patterns, save_path, filename)

    print(f"\nGenerated {num_datasets} datasets to: {save_path}")


if __name__ == "__main__":
    main()

