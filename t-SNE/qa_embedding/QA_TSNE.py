import os
import json
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_all_embeddings(json_files):
    """
    从多个 JSON 文件加载嵌入，并保留文件索引映射。

    Returns:
        all_embeddings: 合并后的嵌入列表
        mapping: 列表，每个元素表示该嵌入属于哪个文件和索引
    """
    all_embeddings = []
    mapping = []  # (file_path, index_in_file, original_data) 的映射

    for json_path in json_files:
        if not os.path.exists(json_path):
            print(f"⚠️ File not found: {json_path}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            if "embedding" not in item or not item["embedding"]:
                continue
            all_embeddings.append(item["embedding"])
            mapping.append((json_path, idx, item))  # 保留原始数据用于写回

    print(f"✅ Loaded {len(all_embeddings)} embeddings from {len(json_files)} files.")
    return np.array(all_embeddings), mapping


def run_global_tsne(embeddings, random_state=42, perplexity=5):
    print("🔍 Running global t-SNE on all embeddings...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init='pca', n_iter=1200)
    tsne_results = tsne.fit_transform(embeddings)
    print("✅ t-SNE completed.")
    return tsne_results


def save_tsne_results(mapping, tsne_results):
    """
    将 t-SNE 结果写回对应的 JSON 文件
    """
    # 按文件分组更新
    file_updates = {}
    for (_, (file_path, idx, item)), tsne_vec in zip(enumerate(mapping), tsne_results):
        if file_path not in file_updates:
            file_updates[file_path] = []
        item_copy = item.copy()
        item_copy["tsne"] = tsne_vec.tolist()
        file_updates[file_path].append((idx, item_copy))

    # 写回每个文件
    for file_path, updates in file_updates.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for idx, updated_item in updates:
            data[idx]["tsne"] = updated_item["tsne"]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ t-SNE results saved to: {file_path}")


def main(json_files):
    # 配置：列出所有要合并进行 t-SNE 的 JSON 文件（这里假设是只包含词嵌入信息的文件）

    # 1. 加载所有嵌入
    embeddings, mapping = load_all_embeddings(json_files)

    # 2. 全局 t-SNE
    tsne_results = run_global_tsne(embeddings, perplexity=2.5)  # 根据需要调整perplexity值

    # 3. 写回结果
    save_tsne_results(mapping, tsne_results)


if __name__ == "__main__":
    Q_json_files = [
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/MegaCQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/ChartQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/ChartX.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/ChartBench.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/NovaChart.json",
        # 如果有更多文件，请在此处添加
    ]
    A_json_files = [
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/A_MegaCQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/A_ChartQA.json",
        "X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/A_ChartX.json",
        # 如果有更多文件，请在此处添加
    ]
    main(Q_json_files)
    # main(A_json_files)