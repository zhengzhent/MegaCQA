# 雷达图数据生成模型
# 2变量
import os
import numpy as np
import pandas as pd
from faker import Faker
import matplotlib.pyplot as plt


def generate_data(num_datasets, save_path):
    os.makedirs(save_path, exist_ok=True)
    fake = Faker()

    #########################################################theme&dimensions
    theme = ""
    dimensions = [
    ]
    unit = ""
    ###########################################################################

    for dataset_idx in range(1, num_datasets + 1):
        print(f"\nGenerating dataset {dataset_idx}/{num_datasets}...")

        num_categories = np.random.randint(1, 4)

        # fake data
        category_names = [fake.city() for _ in range(num_categories)]

        num_dimensions = np.random.randint(5, 9)
        selected_dims = np.random.choice(dimensions, num_dimensions, replace=False)

        min_val, max_val = 20, 100
        overlap = np.random.uniform(0.1, 0.4)

        data = {}
        for i in range(num_categories):
            base_values = np.random.randint(min_val, max_val + 1, num_dimensions)

            if i > 0:
                overlap_values = np.round(data[category_names[i - 1]] * overlap).astype(int)
                new_values = np.random.randint(min_val, max_val + 1, num_dimensions)
                non_overlap_values = np.round(new_values * (1 - overlap)).astype(int)
                base_values = overlap_values + non_overlap_values
                base_values = np.clip(base_values, min_val, max_val)

            data[category_names[i]] = base_values

        df = pd.DataFrame(data, index=selected_dims).T
        csv_filename = f'{theme}_{dataset_idx}.csv'
        csv_path = os.path.join(save_path, csv_filename)

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(f"{theme} , {unit}\n")
        f.write("labels," + ",".join(df.columns) + "\n")
        df.to_csv(f, header=False)

        print(f"Data saved to: {csv_path}")

    print(f"\nCompleted generation of {num_datasets} datasets")


if __name__ == "__main__":
    generate_data(num_datasets=1, save_path="csv/radar_chart")