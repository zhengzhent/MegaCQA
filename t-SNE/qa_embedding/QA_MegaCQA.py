import os
import json
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class GTETextEmbedder:
    def __init__(self, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        try:
            return self.model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"❌ Text embedding error: {e}\nText: {text}")
            return None


class QAEmbeddingProcessor:
    def __init__(self, root_folder: str, save_dir: str, sample_per_chart_type: int = 30):
        self.root_folder = root_folder
        self.save_dir = save_dir
        self.sample_per_chart_type = sample_per_chart_type
        self.embedder = GTETextEmbedder()
        self.q_results = []
        self.a_results = []

    def process_all(self):
        """
        对每个图表类型：
        - 随机选 sample_per_chart_type 个 QA 文件
        - 对每个 QA 文件中的每一个 qa_list（如 train/val/test），随机抽一个 QA
        - 提取 Q 和 A 的嵌入，分别加入 q_results 和 a_results
        """
        chart_types = [d for d in os.listdir(self.root_folder)
                       if os.path.isdir(os.path.join(self.root_folder, d))]

        for chart_type in tqdm(chart_types, desc="Processing chart types"):
            print(f"🔍 Processing chart type: {chart_type}")
            qa_dir = os.path.join(self.root_folder, chart_type, "QA")
            if not os.path.exists(qa_dir):
                print(f"⚠️ No QA folder in {chart_type}, skipped.")
                continue

            json_files = [f for f in os.listdir(qa_dir) if f.endswith(".json")]
            if not json_files:
                continue

            # 随机选择指定数量的文件
            selected_files = random.sample(json_files, min(self.sample_per_chart_type, len(json_files)))

            for filename in selected_files:
                file_path = os.path.join(qa_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)

                    relative_path = os.path.relpath(file_path, self.root_folder).replace("\\", "/")

                    # ✅ 遍历每一个 qa_list（如 train, val, test）
                    for list_name, qa_list in qa_data.items():
                        if not isinstance(qa_list, list) or len(qa_list) == 0:
                            continue  # 跳过非列表或空列表

                        # ✅ 从当前 qa_list 中随机抽一个 QA
                        sampled_qa = random.choice(qa_list)
                        q_text = sampled_qa.get("Q", "").strip()
                        a_text = sampled_qa.get("A", "").strip()

                        # 处理 Q
                        if q_text:
                            q_vec = self.embedder.embed_text(q_text)
                            if q_vec:
                                self.q_results.append({
                                    "file_name": relative_path,
                                    "embedding": q_vec,
                                    "tsne": []
                                })

                        # 处理 A
                        if a_text:
                            a_vec = self.embedder.embed_text(a_text)
                            if a_vec:
                                self.a_results.append({
                                    "file_name": relative_path,
                                    "embedding": a_vec,
                                    "tsne": []
                                })

                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")

    def save(self):
        """分别保存 Q 和 A 的嵌入到两个独立文件"""
        os.makedirs(self.save_dir, exist_ok=True)

        q_save_path = os.path.join(self.save_dir, "Q_MegaCQA.json")
        a_save_path = os.path.join(self.save_dir, "A_MegaCQA.json")

        # 保存 Q
        with open(q_save_path, "w", encoding="utf-8") as f:
            json.dump(self.q_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.q_results)} Q embeddings to {q_save_path}")

        # 保存 A
        with open(a_save_path, "w", encoding="utf-8") as f:
            json.dump(self.a_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.a_results)} A embeddings to {a_save_path}")


def main():
    root_folder = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/MegaCQA"
    save_dir = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA"

    processor = QAEmbeddingProcessor(
        root_folder=root_folder,
        save_dir=save_dir,
        sample_per_chart_type=40
    )
    processor.process_all()
    processor.save()


if __name__ == "__main__":
    main()