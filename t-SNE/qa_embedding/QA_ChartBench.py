import os
import json
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class GTETextEmbedder:
    def __init__(self, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name)
        print(f"✅ Loaded embedding model: {model_name}")

    def embed_text(self, text: str):
        try:
            return self.model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"❌ Text embedding error: {e}\nText: {text}")
            return None


class QAEmbeddingProcessor:
    def __init__(self, root_folder: str, save_dir: str, sample_per_chart_type: int = 40):
        self.root_folder = root_folder
        self.save_dir = save_dir
        self.sample_per_chart_type = sample_per_chart_type
        self.embedder = GTETextEmbedder()
        self.q_results = []  # 存储所有提取的问题嵌入

    def extract_questions(self, data):
        """从 JSON 数据中提取所有 Qr 和 query 字段"""
        questions = []

        # 提取 Acc+ 中的所有 Qr
        if "Acc+" in data and isinstance(data["Acc+"], dict):
            for key, item in data["Acc+"].items():
                if isinstance(item, dict) and "Qr" in item:
                    q_text = item["Qr"].strip()
                    if q_text:
                        questions.append({
                            "type": "Qr",
                            "source": f"Acc+.{key}"
                        })

        # 提取 NQA 中的 query
        if "NQA" in data and isinstance(data["NQA"], dict):
            q_text = data["NQA"].get("query", "").strip()
            if q_text:
                questions.append({
                    "type": "query",
                    "source": "NQA"
                })

        return questions

    def process_all(self):
        """
        处理 ChartBench 数据集：
        - 遍历每个图表类型下的 QA 文件夹
        - 随机选择 sample_per_chart_type 个 JSON 文件
        - 从每个文件中提取 Qr 和 query 字段
        - 对每个问题生成嵌入并保存
        """
        chart_types = [d for d in os.listdir(self.root_folder)
                       if os.path.isdir(os.path.join(self.root_folder, d))]

        total_files_processed = 0
        total_questions_extracted = 0

        for chart_type in tqdm(chart_types, desc="📊 Processing chart types"):
            if chart_type == "CSV":
                continue  # 跳过 CSV 类型
            print(f"🔍 Processing chart type: {chart_type}")
            qa_dir = os.path.join(self.root_folder, chart_type, "QA")
            if not os.path.exists(qa_dir):
                print(f"⚠️  No QA folder in {chart_type}, skipped.")
                continue

            json_files = [f for f in os.listdir(qa_dir) if f.endswith(".json")]
            if not json_files:
                print(f"⚠️  No JSON files in {qa_dir}")
                continue

            # 随机采样指定数量的文件
            selected_files = random.sample(json_files, min(self.sample_per_chart_type, len(json_files)))

            for filename in selected_files:
                file_path = os.path.join(qa_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    relative_path = os.path.relpath(file_path, self.root_folder).replace("\\", "/")

                    # 提取所有 Qr 和 query
                    question_entries = self.extract_questions(data)
                    for entry in question_entries:
                        # 获取原始文本
                        if entry["type"] == "Qr":
                            q_text = data["Acc+"][entry["source"].split('.')[-1]]["Qr"]
                        else:  # query
                            q_text = data["NQA"]["query"]

                        # 生成嵌入
                        q_vec = self.embedder.embed_text(q_text)
                        if q_vec:
                            self.q_results.append({
                                "file_name": relative_path,
                                "question_type": entry["type"],        # 区分 Qr 或 query
                                "source": entry["source"],             # 如 Acc+.CR 或 NQA
                                "embedding": q_vec,
                                "tsne": []
                            })
                            total_questions_extracted += 1

                    total_files_processed += 1

                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")

        print(f"✅ Finished processing. Total files: {total_files_processed}, Total questions extracted: {total_questions_extracted}")

    def save(self):
        """保存问题嵌入到文件，仅保留 file_name, embedding, tsne 字段"""
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "unsampled_ChartBench.json")

        # 构建精简版结果
        simplified_results = [
            {
                "file_name": item["file_name"],
                "embedding": item["embedding"],
                "tsne": []  # 明确留空
            }
            for item in self.q_results
        ]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(simplified_results)} question embeddings to {save_path} (only 'file_name', 'embedding', 'tsne')")


def main():
    # 修改为新的数据集路径
    root_folder = r"X:\UniversityCourseData\Visualization\20250628TSne可视对比\ChartBench"
    save_dir = r"X:\UniversityCourseData\Visualization\20250628TSne可视对比\t-SNE\output\QA"

    processor = QAEmbeddingProcessor(
        root_folder=root_folder,
        save_dir=save_dir,
        sample_per_chart_type=40  # 每个图表类型采样 40 个文件
    )
    processor.process_all()
    processor.save()


if __name__ == "__main__":
    main()