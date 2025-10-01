import numpy as np
import pandas as pd
from faker import Faker
import random
import os


def generate_ridgeline_data(num_points, num_categories, trend_pattern, value_range):
    min_value, max_value = value_range
    data = {}

    for i in range(num_categories):
        if trend_pattern == "Single Peak Distribution":
            x = np.linspace(-3, 3, num_points)
            y = np.exp(-x ** 2)
        elif trend_pattern == "Concentrated distribution pattern":
            y = np.random.normal(0, 1, num_points) * 0.5
        elif trend_pattern == "Decentralized distribution pattern":
            y = np.random.normal(0, 1, num_points) * 1.5
        elif trend_pattern == "Periodic fluctuation":
            freq = random.uniform(0.5, 2)
            y = np.sin(np.linspace(0, 2 * np.pi, num_points) * freq) * (1 + 0.2 * np.random.randn(num_points))
        elif trend_pattern == "Random fluctuation":
            y = np.cumsum(np.random.randn(num_points))

        y_min, y_max = y.min(), y.max()
        y_scaled = (y - y_min) / (y_max - y_min) * (max_value - min_value) + min_value
        data[f"Category_{i + 1}"] = y_scaled

    return pd.DataFrame(data)


def main(num_datasets, num_points):
    save_path = "csv/ridgeline_chart"
    os.makedirs(save_path, exist_ok=True)

    for t in range(num_datasets):

        #################################################################################### 生成数据变量部分
        # 主题  内容名(小主题)  数据范围  单位  labels
        theme = "Healthcare and Health"
        name = "Disease Incidence Rate"
        value_range = [0, 2000]
        unit = "cases/100k"
        category_names = ["Infectious Diseases", "Non-communicable Diseases", "Seasonal Illnesses",
                          "Occupational Hazards", "Zoonotic Diseases", "Lifestyle Diseases", "Nutritional Deficiencies"]
        #######################################################################################################

        num_categories = random.randint(3, 7)
        pattern_options = [
            "Single Peak Distribution",
            "Concentrated distribution pattern",
            "Decentralized distribution pattern",
            "Periodic fluctuation",
            "Random fluctuation"
        ]
        selected_pattern = random.choice(pattern_options)
        category_name = random.sample(category_names, num_categories)

        df = generate_ridgeline_data(num_points, num_categories, selected_pattern, value_range)

        for i in range(num_categories):
            df = df.rename(columns={f"Category_{i + 1}": category_name[i]})

        csv_path = os.path.join(save_path, f"{theme}_{name}_{t + 1}.csv")
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(f"{theme},{name},{unit},{selected_pattern}\n")
            df.to_csv(f)

        print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
    main(num_points=50, num_datasets=1)
