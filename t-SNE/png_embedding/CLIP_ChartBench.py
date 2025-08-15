import os
import json
import random
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import clip


class ClipImageEmbedder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def embed_image(self, image_path):
        try:
            image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image)
            return embedding.cpu().numpy().squeeze()
        except Exception as e:
            print(f"❌ Failed to embed image {image_path}: {e}")
            return None


class ImageEmbeddingProcessor:
    def __init__(self, root_folder, save_path, sample_per_chart=50, base_path=None):
        self.root_folder = root_folder
        self.save_path = save_path
        self.sample_per_chart = sample_per_chart
        self.base_path = base_path or root_folder
        self.embedder = ClipImageEmbedder()
        self.results = []

    def process_all_images(self):
        chart_types = [d for d in os.listdir(self.root_folder) if os.path.isdir(os.path.join(self.root_folder, d))]
        
        print(f"🖼️ Found {len(chart_types)} chart types: {chart_types}")

        for chart_type in tqdm(chart_types, desc="Chart Types"):
            if chart_type == "CSV":
                continue  # 跳过 CSV 类型
            png_dir = os.path.join(self.root_folder, chart_type, "PNG")
            if not os.path.exists(png_dir):
                print(f"⚠️  Skipping {chart_type}, 'PNG/' folder not found.")
                continue

            all_pngs = [f for f in os.listdir(png_dir) if f.endswith(".png")]
            if not all_pngs:
                print(f"⚠️  No images found in {png_dir}")
                continue

            selected_pngs = random.sample(all_pngs, min(self.sample_per_chart, len(all_pngs)))

            for file_name in selected_pngs:
                file_path = os.path.join(png_dir, file_name)
                embedding = self.embedder.embed_image(file_path)
                if embedding is not None:
                    relative_path = os.path.relpath(file_path, self.base_path).replace("\\", "/")
                    self.results.append({
                        "category": chart_type,
                        "file_name": relative_path,
                        "embedding": embedding.tolist(),
                        "tsne": []  # 留空，准备后续t-SNE处理
                    })

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        for item in self.results:
            item["embedding"] = json.loads(json.dumps(item["embedding"], separators=(",", ":")))

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ Embeddings saved to: {self.save_path}")
        print(f"📦 Total embeddings saved: {len(self.results)}")


def main():
    root_folder = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/ChartBench"
    save_path = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/ChartBench.json"

    processor = ImageEmbeddingProcessor(
        root_folder=root_folder,
        save_path=save_path,
        sample_per_chart=115,  # 可配置为你想采样的数量
        base_path=root_folder
    )
    processor.process_all_images()
    processor.save()


if __name__ == "__main__":
    main()