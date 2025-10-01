from dotenv import load_dotenv
from my_api import APIClient
import os


# 读取 Python 文件
def read_python_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def majority_voting_system(prompt):
    votes = {}
    deepseek_processor = APIClient("deepseek")
    doubao_processor = APIClient("doubao")
    qwen_processor = APIClient("qwen")
    gpt_processor = APIClient("gpt")
    gemini_processor = APIClient("gemini")
    voters = [doubao_processor, deepseek_processor, qwen_processor, gpt_processor, gemini_processor]
    for voter in voters:
        try:
            vote = voter.generate_text(prompt)
            # Save the vote result with the voter's name as the key
            votes[voter.model_name] = vote
        except Exception as e:
            print(f"Error during voting with {voter.model_name}: {str(e)}")
            votes[voter.model_name] = "Error"
    return votes


if __name__ == '__main__':
    # Load environment variables
    files_name = [ 'bubble',  'heatmap', 'line', 'node_link', 'parallel', 'radar', 'ridgeline', 'sankey', 'scatter', 'stack', 'stream', 'sunburst']

    # 创建目录
    os.makedirs('voting result', exist_ok=True)
    for file_name in files_name:
        description = f'principle/{file_name}.md'
        first_code = f"generation/data_generation/第一版/{file_name}.py"
        second_code = f"generation/data_generation/第二版/{file_name}.py"

        python_file_content = read_python_file(second_code)
        prompt_file_content = read_python_file(description)
        prompt = f'我将提供两个文件，一个是用于数据生成的Python代码，另一个是相关的描述说明（prompt）。请检查代码与描述的匹配： 根据代码内容和描述，判断它们是否一致。代码应该按照描述的要求生成数据。你需要对每个约束给出0或1（0代表不符合）,例如，约束1:0，约束2:1。并给出解释\n以下是python file：{python_file_content}, 以下是prompt:{prompt_file_content}'

        result = majority_voting_system(prompt)
        print(file_name)
        print(result)

        # Save the result in a dictionary-like format with UTF-8 encoding
        with open(f'voting result/first_result_{file_name}.txt', 'w', encoding='utf-8') as file:
            for key, value in result.items():
                file.write(f"{key}: {value}\n-----------------------\n")