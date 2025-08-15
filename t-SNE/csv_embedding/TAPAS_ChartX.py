# extract_embeddings_only.py

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12'

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
    def __init__(self, root_folder, save_path, sample_per_chart=50, base_path=None):
        self.root_folder = root_folder
        self.save_path = save_path
        self.sample_per_chart = sample_per_chart
        self.base_path = base_path or root_folder
        self.embedder = TableEmbedder()
        self.results = []

    def process_all_charts(self):
        chart_types = [d for d in os.listdir(self.root_folder) if os.path.isdir(os.path.join(self.root_folder, d))]
        print(f"📊 Found {len(chart_types)} chart types: {chart_types}")

        for chart_type in tqdm(chart_types, desc="Chart Types"):
            csv_dir = os.path.join(self.root_folder, chart_type, "csv")
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
                    table = pd.read_csv(file_path, skiprows=[], dtype=str)
                    embedding = self.embedder.embed_table(table)
                    relative_path = os.path.relpath(file_path, self.base_path).replace("\\", "/")
                    self.results.append({
                        "file_name": relative_path,
                        "embedding": embedding.tolist(),  # ✅ 保留 embedding
                        "tsne": []                       # ✅ tsne 留空
                    })
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

    def save(self):
        """保存结果，包含 embedding 和空的 tsne 字段"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ Embeddings saved with 'embedding' field: {self.save_path}")
        print(f"📦 Total embeddings: {len(self.results)}")


def main():
    # 设置 ChartX 路径
    root_folder = "X:/UniversityCourseData/Visualization/20250628TSne可视对比/ChartX"
    save_path = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/ChartX.json"

    processor = ChartEmbeddingExtractor(
        root_folder=root_folder,
        save_path=save_path,
        sample_per_chart=56,  # 每类采样 18 个文件
        base_path=root_folder
    )

    processor.process_all_charts()  # ✅ 只提取嵌入
    processor.save()               # ✅ 保存，tsne 留空，不保存 embedding


if __name__ == "__main__":
    main()