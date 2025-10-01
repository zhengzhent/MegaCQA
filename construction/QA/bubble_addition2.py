import os
import json

def process_json_files():
    """
    Scans the current directory for JSON files and removes the first two
    QA pairs from the 'EVJ' list.
    """
    # 获取当前目录
    current_directory = './QA'
    
    # 查找所有.json文件
    json_files = [f for f in os.listdir(current_directory) if f.endswith('.json')]
    
    if not json_files:
        print("在当前目录中未找到任何 JSON 文件。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件。开始处理...")
    
    # 遍历每个JSON文件
    for filename in json_files:
        filepath = os.path.join(current_directory, filename)
        print(f"\n--- 正在处理: {filename} ---")
        
        try:
            # 用于标记文件是否被修改
            was_modified = False
            
            # 读取JSON文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查'EVJ'键是否存在，且其值为一个列表
            if 'EVJ' in data and isinstance(data.get('EVJ'), list):
                evj_list = data['EVJ']
                
                # 检查列表是否至少有两个元素可以删除
                if len(evj_list) >= 2:
                    # 删除前两个元素
                    # 通过切片保留从第三个元素到末尾的所有元素
                    data['EVJ'] = evj_list[2:]
                    was_modified = True
                    print(f"  成功: 已从 'EVJ' 部分删除前两个 QA 对。")
                else:
                    print(f"  信息: 'EVJ' 部分的 QA 对少于2个 ({len(evj_list)}个)，无需修改。")
            else:
                print("  信息: 文件中未找到 'EVJ' 键，或其内容不是列表，无需修改。")

            # 如果文件内容被修改，则写回文件
            if was_modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    # 使用 indent=4 保持JSON格式美观
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"  已保存: 更改已写回文件 {filename}。")

        except json.JSONDecodeError:
            print(f"  错误: 无法解析 {filename}。文件可能已损坏或不是有效的JSON格式。")
        except Exception as e:
            print(f"  错误: 处理 {filename} 时发生意外错误: {e}")

    print("\n--- 所有文件处理完毕。 ---")

# 运行主函数
if __name__ == "__main__":
    process_json_files()
