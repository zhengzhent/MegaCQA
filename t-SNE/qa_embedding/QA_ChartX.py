import json
import os
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
            print(f"❌ Embedding error: {e}\nText: {text}")
            return None


class ChartXQAEmbedder:
    def __init__(self, input_json_path: str, output_dir: str, sample_size: int = 50):
        self.input_json_path = input_json_path
        self.output_dir = output_dir  # 现在是目录
        self.sample_size = sample_size
        self.embedder = GTETextEmbedder()
        self.q_results = []  # 仅问题
        self.a_results = []  # 仅答案

    def process(self):
        """只做词嵌入，不进行 t-SNE"""
        with open(self.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📊 Loaded {len(data)} chart entries.")
        selected = random.sample(data, min(self.sample_size, len(data)))

        for item in tqdm(selected, desc="🔍 Processing QA pairs"):
            qa = item.get("QA", {})
            q = qa.get("input", "").strip()
            a = qa.get("output", "").strip()
            img_path = item.get("img", "")
            filename = os.path.basename(img_path) if img_path else "unknown.png"

            # 处理 Q
            if q:
                q_emb = self.embedder.embed_text(q)
                if q_emb:
                    self.q_results.append({
                        "filename": filename,
                        "embedding": q_emb,
                        "tsne": []
                    })

            # 处理 A
            if a:
                a_emb = self.embedder.embed_text(a)
                if a_emb:
                    self.a_results.append({
                        "filename": filename,
                        "embedding": a_emb,
                        "tsne": []
                    })

    def save(self):
        """分别保存 Q 和 A 的嵌入"""
        os.makedirs(self.output_dir, exist_ok=True)

        q_save_path = os.path.join(self.output_dir, "ChartX.json")
        a_save_path = os.path.join(self.output_dir, "A_ChartX.json")

        # 保存 Q
        with open(q_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.q_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.q_results)} Q embeddings to {q_save_path}")

        # 保存 A
        with open(a_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.a_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.a_results)} A embeddings to {a_save_path}")


def main():
    input_json = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/ChartX/ChartX_annotation.json"
    output_dir = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA"  # 注意：改为目录

    processor = ChartXQAEmbedder(
        input_json_path=input_json,
        output_dir=output_dir,
        sample_size=1500
    )
    processor.process()  # 只做嵌入
    processor.save()     # 分开保存


if __name__ == "__main__":
    main()