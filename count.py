# import os
# import json
# from glob import glob

# # 设置 bar 类别路径
# script_dir = os.path.dirname(os.path.abspath(__file__))
# qa_dir = os.path.join(script_dir, "sample100", "bar", "QA")

# total_count = 0

# # 遍历 bar/QA/ 下所有 json 文件
# for qa_file in sorted(glob(os.path.join(qa_dir, "*.json"))):
#     filename = os.path.basename(qa_file)
#     try:
#         with open(qa_file, "r", encoding="utf-8") as f:
#             qa_data = json.load(f)
#     except Exception as e:
#         print(f"[跳过] 解析失败: {filename} ({e})")
#         continue

#     count = sum(len(qa_list) for qa_list in qa_data.values())
#     print(f"bar/{filename}: {count} QA pairs")
#     total_count += count

# print(f"\nbar 类别总计: {total_count} QA pairs")
import os
import json
from glob import glob
from collections import defaultdict

# 获取 sample100 路径（脚本同目录下）
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, "sample100")

# 存储每个类别的计数
qa_count_per_class = defaultdict(int)
total_count = 0

for chart_type in os.listdir(root_dir):
    chart_path = os.path.join(root_dir, chart_type)
    qa_dir = os.path.join(chart_path, "QA")

    if not os.path.isdir(qa_dir):
        continue

    for json_file in glob(os.path.join(qa_dir, "*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
        except Exception as e:
            print(f"[跳过] 无法解析 {json_file}: {e}")
            continue

        # 每个文件中所有类型（如CTR, VEC, SRP...）
        for qa_type in qa_data:
            qa_pairs = qa_data[qa_type]
            count = len(qa_pairs)
            qa_count_per_class[chart_type] += count
            total_count += count

# 输出结果
for chart_type, count in sorted(qa_count_per_class.items()):
    print(f"{chart_type}: {count} QA pairs")

print(f"\n总计: {total_count} QA pairs")

