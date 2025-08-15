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
        self.q_results = []

    def extract_questions(self, data):
        """
        从 data 字典中提取所有 'normal_qa' 类型问题的 question 文本
        结构: data['json'] -> list of dicts -> metadata['question']
        """
        questions = []
        if "json" not in data or not isinstance(data["json"], list):
            return questions

        for item in data["json"]:
            if (
                isinstance(item, dict)
                and item.get("datatype") == "normal_qa"
                and "metadata" in item
                and isinstance(item["metadata"], dict)
                and "question" in item["metadata"]
                and isinstance(item["metadata"]["question"], str)
            ):
                q_text = item["metadata"]["question"].strip()
                if q_text:
                    questions.append(q_text)
        return questions

    def process_all(self):
        chart_types = [d for d in os.listdir(self.root_folder)
                       if os.path.isdir(os.path.join(self.root_folder, d))]

        total_files_processed = 0
        total_questions_extracted = 0

        for chart_type in tqdm(chart_types, desc="📊 Processing chart types"):
            type_dir = os.path.join(self.root_folder, chart_type)
            jsonl_files = [f for f in os.listdir(type_dir) if f.endswith(".jsonl")]

            if not jsonl_files:
                print(f"⚠️  No JSONL file found in {type_dir}")
                continue

            jsonl_file = jsonl_files[0]
            file_path = os.path.join(type_dir, jsonl_file)

            try:
                lines = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # 提取该行中所有 normal_qa 的问题
                            q_texts = self.extract_questions(data)
                            for q_text in q_texts:
                                lines.append({
                                    "file_name": jsonl_file,
                                    "question": q_text
                                })
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON decode error in {file_path}: {e}")
                            continue

                # 随机打乱并采样
                random.shuffle(lines)
                selected_lines = lines[:self.sample_per_chart_type]

                # 生成嵌入
                for item in selected_lines:
                    q_vec = self.embedder.embed_text(item["question"])
                    if q_vec:
                        self.q_results.append({
                            "file_name": item["file_name"],
                            "embedding": q_vec,
                            "tsne": []
                        })
                        total_questions_extracted += 1

                total_files_processed += 1

            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

        print(f"✅ Finished processing. Total chart types processed: {total_files_processed}, "
              f"Total questions extracted: {total_questions_extracted}")

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "unsampled_NovaChart.json")

        simplified_results = [
            {
                "file_name": item["file_name"],
                "embedding": item["embedding"],
                "tsne": []
            }
            for item in self.q_results
        ]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(simplified_results)} question embeddings to {save_path}")


def main():
    # ✅ 请根据实际路径修改
    root_folder = r"X:\UniversityCourseData\Visualization\20250628TSne可视对比\NovaChart\test"
    save_dir = r"X:\UniversityCourseData\Visualization\20250628TSne可视对比\t-SNE\output\QA"
    sample_per_chart_type = 120  # 每个图表类型抽取 120 个问题

    processor = QAEmbeddingProcessor(
        root_folder=root_folder,
        save_dir=save_dir,
        sample_per_chart_type=sample_per_chart_type
    )
    processor.process_all()
    processor.save()


if __name__ == "__main__":
    main()