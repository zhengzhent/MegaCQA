# extract_embeddings_sampled_1000.py

import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from transformers import TapasTokenizer, TapasModel
import warnings
import random

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['LOKY_MAX_CPU_COUNT'] = '12'  # 防止 joblib 多进程警告


class TableEmbedder:
    """
    使用 TAPAS 模型对表格进行嵌入。
    """
    def __init__(self, model_name="google/tapas-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasModel.from_pretrained(model_name).to(self.device)

    def embed_table(self, table: pd.DataFrame):
        """
        将表格转换为嵌入向量。

        Args:
            table (pd.DataFrame): 输入表格。

        Returns:
            np.ndarray: 嵌入向量 (768,)
        """
        table = table.astype(str).fillna("")
        try:
            inputs = self.tokenizer(
                table=table,
                queries=["What is in the table?"],  # 固定查询
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            return embedding
        except Exception as e:
            print(f"❌ TAPAS encoding failed: {e}")
            return np.zeros(self.model.config.hidden_size)  # 返回零向量容错


class SampledCSVEmbedder:
    """
    专门用于处理 CSV 文件夹，随机采样 1000 个文件并提取嵌入。
    """
    def __init__(self, csv_folder: str, save_path: str, base_path: str = None, sample_size: int = 1000):
        """
        :param csv_folder: 包含所有 CSV 的文件夹路径
        :param save_path: 输出 JSON 保存路径
        :param base_path: 用于生成相对路径的根路径
        :param sample_size: 随机采样数量
        """
        self.csv_folder = csv_folder
        self.save_path = save_path
        self.base_path = base_path or csv_folder
        self.sample_size = sample_size
        self.embedder = TableEmbedder()
        self.results = []

    def process_all_csvs(self):
        """
        随机采样指定数量的 CSV 文件并提取嵌入。
        """
        if not os.path.exists(self.csv_folder):
            raise FileNotFoundError(f"CSV folder not found: {self.csv_folder}")

        all_files = [f for f in os.listdir(self.csv_folder) if f.endswith(".csv")]
        if len(all_files) == 0:
            print(f"⚠️ No CSV files found in {self.csv_folder}.")
            return

        # 🔍 随机采样最多 sample_size 个文件
        sampled_files = random.sample(all_files, min(self.sample_size, len(all_files)))
        print(f"📊 Found {len(all_files)} CSVs, sampling {len(sampled_files)} files...")

        for file_name in tqdm(sampled_files, desc="Processing sampled CSVs"):
            file_path = os.path.join(self.csv_folder, file_name)
            try:
                # 读取表格
                table = pd.read_csv(file_path, dtype=str).fillna("")  # 统一处理缺失值
                embedding = self.embedder.embed_table(table)
                relative_path = os.path.relpath(file_path, self.base_path).replace("\\", "/")

                self.results.append({
                    "file_name": relative_path,
                    "embedding": embedding.tolist(),
                    "tsne": []  # 留空，供后续 t-SNE 使用
                })
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

        print(f"✅ Successfully embedded {len(self.results)} sampled files.")

    def save(self):
        """
        保存结果到 JSON 文件。
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ Embeddings saved to: {self.save_path}")
        print(f"📦 Total embedded files: {len(self.results)}")


def main():
    # ================================
    # 🔧 用户配置区
    # ================================
    csv_folder = r"X:\UniversityCourseData\Visualization\20250628TSne可视对比\ChartBench\CSV"
    save_path = "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/CSV/ChartBench.json"
    sample_size = 800  # 随机采样 1000 个文件

    # ================================
    # ✅ 执行：采样 + 提取嵌入
    # ================================
    processor = SampledCSVEmbedder(
        csv_folder=csv_folder,
        save_path=save_path,
        base_path=csv_folder,
        sample_size=sample_size
    )
    processor.process_all_csvs()
    processor.save()


if __name__ == "__main__":
    main()