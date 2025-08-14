# extract_embeddings.py

import os
import json
import random
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from transformers import TapasTokenizer, TapasModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TableEmbedder:

    def __init__(self, model_name="google/tapas-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasModel.from_pretrained(model_name).to(self.device)

    def embed_table(self, table: pd.DataFrame):
        table = table.astype(str).fillna("")
        inputs = self.tokenizer(
            table=table,
            queries=["What is in the table?"],
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return embedding


class ChartEmbeddingExtractor:
    # ================================
    # 📊 图表类型预处理规则定义
    # 格式: {图表类型: {'skip_rows': [行索引], 'drop_cols': [列索引]}}
    # 注意：索引从 0 开始
    # ================================

    """
    仅提取嵌入，不执行 t-SNE
    """

    def __init__(self, root_folder, save_path, sample_per_chart=50, base_path=None):
        self.root_folder = root_folder
        self.save_path = save_path
        self.sample_per_chart = sample_per_chart
        self.base_path = base_path or root_folder
        self.embedder = TableEmbedder()
        self.results = []
        self.PREPROCESS_RULES = {
            'bar': {'skip_rows': [0, 1], 'drop_cols': [0]},
            'box': {'skip_rows': [0, 1], 'drop_cols': []},
            'bubble': {'skip_rows': [0, 1], 'drop_cols': []},
            'chord': {'skip_rows': [0, 1], 'drop_cols': [0]},
            'fill_bubble': {'skip_rows': [0, 1], 'drop_cols': [3]},
            'funnel': {'skip_rows': [0, 1], 'drop_cols': []},
            'heatmap': {'skip_rows': [0], 'drop_cols': [0, 1]},
            'line': {'skip_rows': [0, 1, 2], 'drop_cols': []},
            'node_link': {'skip_rows': [0, 1], 'drop_cols': []},
            'parallel': {'skip_rows': [0, 1, 2], 'drop_cols': [0]},
            'pie': {'skip_rows': [0, 1], 'drop_cols': [0]},
            'radar': {'skip_rows': [0, 1], 'drop_cols': []},
            'ridgeline': {'skip_rows': [0, 1, 2], 'drop_cols': []},
            'sankey': {'skip_rows': [0, 1], 'drop_cols': [0, 1]},
            'scatter': {'skip_rows': [0, 1], 'drop_cols': []},
            'stacked_area': {'skip_rows': [0, 1, 2], 'drop_cols': []},
            'stacked_bar': {'skip_rows': [0, 1], 'drop_cols': [0]},
            'stream': {'skip_rows': [0, 1, 2], 'drop_cols': []},
            'sunburst': {'skip_rows': [0, 1], 'drop_cols': [0, 1]},
            'treemap': {'skip_rows': [0, 1, 2], 'drop_cols': [0, 1]},
            'treemap_D3': {'skip_rows': [0, 1, 2], 'drop_cols': [0, 1]},
            'violin': {'skip_rows': [0, 1], 'drop_cols': []},
        }
    
    def preprocess_table(self, table: pd.DataFrame, chart_type: str) -> pd.DataFrame:
        """
        根据图表类型对表格进行预处理：跳过指定行，删除指定列。

        Args:
            table (pd.DataFrame): 原始表格。
            chart_type (str): 图表类型。

        Returns:
            pd.DataFrame: 预处理后的表格。
        """
        if chart_type not in self.PREPROCESS_RULES:
            print(f"⚠️ 无预处理规则: {chart_type}, 使用原始表格。")
            return table.copy()

        rule = self.PREPROCESS_RULES[chart_type]
        df = table.copy()

        # # 跳过指定行（按位置索引）
        # skip_rows = rule['skip_rows']
        # if skip_rows:
        #     # 使用 iloc 排除这些行
        #     rows_to_keep = [i for i in range(len(df)) if i not in skip_rows]
        #     if rows_to_keep:
        #         df = df.iloc[rows_to_keep].reset_index(drop=True)
        #     else:
        #         print(f"⚠️ 所有行都被跳过！返回空表: {chart_type}")
        #         df = pd.DataFrame()

        # 删除指定列（按位置索引）
        drop_cols = rule['drop_cols']
        if drop_cols and len(df.columns) > 0:
            # 检查列索引是否有效
            valid_drop_cols = [i for i in drop_cols if i < len(df.columns)]
            if valid_drop_cols:
                # 获取要保留的列标签
                cols_to_keep = [col for idx, col in enumerate(df.columns) if idx not in valid_drop_cols]
                df = df[cols_to_keep].reset_index(drop=True)
            else:
                print(f"⚠️ 指定删除的列索引无效: {drop_cols}, 列数: {len(df.columns)}")
                print(df)

        return df

    def process_all_charts(self):
        chart_types = [d for d in os.listdir(self.root_folder) if os.path.isdir(os.path.join(self.root_folder, d))]
        print(f"📊 Found {len(chart_types)} chart types: {chart_types}")

        for chart_type in tqdm(chart_types, desc="Chart Types"):
            # chart_type = "box"
            csv_dir = os.path.join(self.root_folder, chart_type, "csv")

            if chart_type == "parallel":
                continue
    
            if not os.path.exists(csv_dir):
                print(f"⚠️ Skipping {chart_type}, 'csv/' folder not found.")
                continue

            all_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
            if not all_files:
                print(f"⚠️ No CSV files in {csv_dir}.")
                continue

            selected_files = random.sample(all_files, min(self.sample_per_chart, len(all_files)))

            for file_name in selected_files:
                file_path = os.path.join(csv_dir, file_name)
                try:
                    rule = self.PREPROCESS_RULES[chart_type]

                    table = pd.read_csv(file_path, dtype=str, header=rule['skip_rows'][-1],skiprows=rule['skip_rows'][0:-1]).fillna("")  # 统一处理缺失值
                    # 根据实际列数，生成字符串列名：'0', '1', '2', '3', ...
                    # table.columns = [str(i) for i in range(table.shape[1])]
                    # print(table)
                    table = self.preprocess_table(table, chart_type)
                    embedding = self.embedder.embed_table(table)
                    relative_path = os.path.relpath(file_path, self.base_path).replace("\\", "/")
                    self.results.append({
                        "file_name": relative_path,
                        "embedding": embedding.tolist(),
                        "tsne": []  # 留空
                    })
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    print(table)
                # exit(0)

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ Embeddings saved (tsne empty): {self.save_path}")
        print(f"📦 Total: {len(self.results)}")


def main():
    # 示例：你可以为不同数据集运行多次，生成多个 JSON
    datasets = [
        {
            "root_folder": "X:/UniversityCourseData/Visualization/20250628TSne可视对比/MegaCQA",
            "save_path": "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/MegaCQA.json"
        }
    ]

    for dataset in datasets:
        processor = ChartEmbeddingExtractor(
            root_folder=dataset["root_folder"],
            save_path=dataset["save_path"],
            sample_per_chart=65,
            base_path=dataset["root_folder"]
        )
        processor.process_all_charts()
        processor.save()


if __name__ == "__main__":
    main()