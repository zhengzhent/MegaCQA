# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import os
from typing import List

class BoxPlotCsvGenerator:
    def __init__(
        self,
        output_dir: str = "./csv/box_plot",
        topic_name: str = "Tourism and Hospitality",
        file_prefix: str = "box_plot_topic2",
        distribution_types: List[str] = None
    ):
        self.output_dir = output_dir
        self.topic_name = topic_name
        self.file_prefix = file_prefix
        self.distribution_types = distribution_types or ['normal', 't', 'skewed']
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_distribution_data(self, distribution_type: str, n_points: int) -> np.ndarray:
        if distribution_type == 'normal':
            # Base price between 300 and 800 USD
            data = np.random.normal(loc=550, scale=150, size=n_points)
        elif distribution_type == 't':
            # More volatile price distribution
            df = random.choice([3, 5, 8])
            data = np.random.standard_t(df=df, size=n_points) * 150 + 550
        elif distribution_type == 'skewed':
            # Right-skewed distribution for housing prices
            data = np.random.gamma(shape=2, scale=200, size=n_points) + 300
        else:
            raise ValueError(f"Unsupported distribution: {distribution_type}")

        # 删除所有小于 0 的值
        data = data[(data > 0)  & (data < np.percentile(data, 99))]

        return data

    def generate_dataframe(
        self,
        distribution_type: str,
        n_points: int,
        labels: List[str]
    ) -> pd.DataFrame:
        """
        为每个标签生成经过过滤的数据列，并统一截断到相同长度（最小列长度）。
        返回整型 DataFrame。
        """
        series_list = []
        for label in labels:
            arr = self.generate_distribution_data(distribution_type, n_points)
            series_list.append(pd.Series(arr, name=label))
        # 找到最小长度
        min_len = min(s.size for s in series_list)
        # 截断所有列到最小长度
        df = pd.concat([s.iloc[:min_len] for s in series_list], axis=1)
        return df.round().astype(int)

    def save_data_to_csv(
        self,
        df: pd.DataFrame,
        file_index: int,
        theme: str,
        unit: str,
        mode: str
    ) -> None:
        file_name = os.path.join(
            self.output_dir,
            f"{self.file_prefix}_{file_index}.csv"
        )
        # 写入 header
        header = f"{self.topic_name},{theme},{unit},{mode}"
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            f.write(header + '\n')
            f.write(','.join(df.columns) + '\n')
            df.to_csv(f, index=False, header=False)

    def generate_and_save(
        self,
        labels: List[str],
        num_sets: int = 15,
        num_points: int = 100,
        theme: str = "Daily Visitor Count at Tourist Attractions",
        unit: str = "Visitors"
    ) -> None:

        for i in range(1, num_sets + 1):
            selected = random.sample(labels, random.randint( 2, min( 7, len(labels) ) ) )
            mode = random.choice(self.distribution_types)
            df = self.generate_dataframe(mode, num_points, selected)
            self.save_data_to_csv(df, i, theme, unit, mode)
            if i % 100 == 0:
                print(f"Saved {i} CSV files (mode={mode}, labels={len(selected)})")


if __name__ == '__main__':
    # 示例用法
    labels = [
        'Single Family', 'Townhouse', 'Condo', 'Apartment', 'Luxury Villa',
        'Studio', 'Duplex', 'Triplex', 'Penthouse', 'Loft',
        'Co-op', 'Mobile Home', 'Tiny House', 'Mansion', 'Bungalow',
        'Cottage', 'Farmhouse', 'Beach House', 'Mountain Cabin', 'Urban Loft'
    ]
    generator = BoxPlotCsvGenerator(
        output_dir='./box_plot/csv',
        topic_name='Real Estate and Housing Market',
        file_prefix='box_plot_Real_Estate_and_Housing_Market'
    )
    generator.generate_and_save(
        labels=labels,
        num_sets=8,
        num_points=120,
        theme='Virtual Housing Prices by Property Type',
        unit='thousand USD'
    )