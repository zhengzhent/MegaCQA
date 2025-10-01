import subprocess
import sys
from time import sleep


def modify_and_run_test(folder):
    # 修改 test.py 的命令行参数
    command = [
        'python', 'D:\\code\\MegaCQA\\QualityControl\\chart\\siliconflow_readability_test.py',  # test.py 文件路径
        rf'D:\code\MegaCQA\sample100\{folder}\png',  # 输入文件路径
    ]

    # 使用 subprocess.Popen 运行 test.py 并实时输出
    try:
        # 启动子进程，指定编码为 'utf-8'
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        # 实时输出 stdout 和 stderr
        for line in process.stdout:
            print(line, end='')  # 打印标准输出的每一行

        # 如果有错误输出，也打印错误信息
        for line in process.stderr:
            print(line, end='', file=sys.stderr)

        # 等待子进程完成
        process.wait()

        print("Test script executed successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"stderr: {e.stderr}")


# 示例：调用该函数来自动运行 test.py
if __name__ == "__main__":

    # folders = ['bubble']
    folders = ['nodelink']
    # folders = ['bar', 'box', 'bubble', 'chord', 'fill_bubble', 'funnel', 'heatmap', 'line', 'nodelink', 'parallel', 'pie', 'radar', 'ridgeline', 'sankey', 'scatter', 'stacked_area','stacked_bar', 'stream', 'sunburst', 'treemap', 'violin']
    for folder in folders:
        print(f"Running test for folder: {folder}")
        modify_and_run_test(folder)
        print(f"Finished running test for folder: {folder}\n")
