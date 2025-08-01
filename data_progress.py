import os
import json
from glob import glob

# 你的数据主目录
root_dir = "sample100"
output = []

# 遍历22类图表目录
for chart_type in os.listdir(root_dir):
    chart_type_path = os.path.join(root_dir, chart_type)
    png_dir = os.path.join(chart_type_path, "png")
    qa_dir = os.path.join(chart_type_path, "QA")

    if not os.path.exists(png_dir) or not os.path.exists(qa_dir):
        continue

    for qa_file in glob(os.path.join(qa_dir, "*.json")):
        basename = os.path.splitext(os.path.basename(qa_file))[0]
        image_path = os.path.join(png_dir, basename + ".png")
        image_path = image_path.replace("\\", "/")  # 添加这一行

        if not os.path.exists(image_path):
            print(f"[警告] 缺失图片文件: {image_path}")
            continue

        try:
            with open(qa_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[错误] JSON解析失败: {qa_file}")
            continue

        for qa_type in qa_data:
            for pair in qa_data[qa_type]:
                question = pair.get("Q", "").strip()
                answer = pair.get("A", "").strip()
                if question and answer:
                    sample = {
                        "messages": [
                            {"role": "user", "content": f"<image>{question}"},
                            {"role": "assistant", "content": answer}
                        ],
                        "images": [image_path]
                    }
                    output.append(sample)

# 输出结果写入JSON
with open("visual_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"[完成] 共生成样本数: {len(output)}")
