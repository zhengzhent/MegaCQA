import json
import os
import random
from pathlib import Path

def sample_json_objects(input_path: str, output_path: str, sample_count: int, seed: int = 42):
    """
    从 JSON 文件中随机采样指定数量的对象并保存。

    :param input_path: 输入 JSON 文件路径（必须是对象列表）
    :param output_path: 输出 JSON 文件路径
    :param sample_count: 要采样的数量
    :param seed: 随机种子（保证可复现）
    """
    random.seed(seed)

    # 检查输入文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 输入文件不存在: {input_path}")

    # 读取 JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ JSON 解析错误: {e}")

    if not isinstance(data, list):
        raise ValueError("❌ JSON 数据必须是一个对象列表")

    total = len(data)
    if total == 0:
        print("⚠️ 警告：输入文件为空列表，无法采样。")
        return

    # 实际采样数量（不能超过总数）
    actual_sample = min(sample_count, total)

    # 随机采样
    sampled_data = random.sample(data, actual_sample)

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 成功采样 {actual_sample}/{total} 个对象")
    print(f"📁 保存至: {output_path}")


# ========================
#        使用示例
# ========================
if __name__ == "__main__":
    # ✏️ 修改这些参数即可
    # input_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/Q_MegaCQA.json"
    # output_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/MegaCQA.json"
    # input_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/unsampled_ChartBench.json"
    # output_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/ChartBench.json"
    input_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/unsampled_NovaChart.json"
    output_file = r"X:/UniversityCourseData/Visualization/20250628TSne可视对比/t-SNE/output/QA/NovaChart.json"
    num_samples = 1500  # 你想采样多少个？

    sample_json_objects(
        input_path=input_file,
        output_path=output_file,
        sample_count=num_samples,
        seed=42  # 可选：固定种子保证结果可复现
    )