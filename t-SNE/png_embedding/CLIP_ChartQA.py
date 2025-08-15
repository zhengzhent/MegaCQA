import os
import json
from PIL import Image
import torch
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


def main(png_dir, save_path):
    embedder = ClipImageEmbedder()
    results = []

    # 确保目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 遍历png目录下的所有png文件
    for file_name in os.listdir(png_dir):
        if not file_name.endswith('.png'):
            continue

        file_path = os.path.join(png_dir, file_name)
        embedding = embedder.embed_image(file_path)
        if embedding is not None:
            results.append({
                "file_name": file_name,
                "embedding": embedding.tolist(),
                "tsne": []  # 此处保留供后续可能的t-SNE处理使用
            })

    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Embeddings saved to: {save_path}")
    print(f"📦 Total embeddings saved: {len(results)}")


if __name__ == "__main__":
    png_dir = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/ChartQA_sampled/png"
    save_path = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/PNG/ChartQA.json"

    main(png_dir, save_path)