import pandas as pd
import os
import json
import random

import re
# todo:根据你的csv里首行有的信息进行修改
# 读取文件的第一行，依次返回大标题、子标题、单位、模式
def read_metadata(filepath):
    # header=None 表示不把任何行当成列名，nrows=1 只读第一行
    meta = pd.read_csv(filepath, header=None, nrows=1).iloc[0].tolist()
    meta = [element.strip() for element in meta]
    keys = ['title', 'subtitle', 'unit', 'orientation']
    return dict(zip(keys, meta))

def write_qa_to_json(csv_path: str, qa_type: str, qa_item: dict):

    # 找到 "csv" 在路径中的位置
    idx = csv_path.find('csv')
    # 截取 "/bar_chart/bar_chart_1.csv"
    rel = csv_path[idx + len('csv'):]
    # 取出子目录部分 "bar_chart"
    subdir = os.path.dirname(rel).lstrip(os.sep)

    # 前缀 "../"
    prefix = csv_path[:idx]
    # 构造 JSON 存放目录 "../QA/bar_chart"
    json_dir = "./QA/"
    os.makedirs(json_dir, exist_ok=True)

    # 构造 JSON 文件完整路径
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    json_path = os.path.join(json_dir, base_name + '.json')

    # 加载或初始化
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化各类空列表
        # data = {key: [] for key in ["CTR", "VEC", "SPR", "VPR", "VE", "EVJ", "SC", "NF", "NC", "MSR", "VA"]}
        data = {key: [] for key in ["CTR", "VEC", "VPR", "VE", "EVJ", "SC", "NF", "NC"]}

    # 追加 QA
    data.setdefault(qa_type, []).append(qa_item)

    # 写回
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def write_all_qa_to_json(csv_path: str, qa_dict: dict):
    # 找到 "csv" 在路径中的位置
    idx = csv_path.find('csv')
    rel = csv_path[idx + len('csv'):]
    subdir = os.path.dirname(rel).lstrip(os.sep)

    prefix = csv_path[:idx]
    json_dir = "./QA/"
    os.makedirs(json_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    json_path = os.path.join(json_dir, base_name + '.json')

    # 加载已有数据
    # if os.path.exists(json_path):
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    # else:
    data = {key: [] for key in ["CTR", "VEC", "SRP", "VPR", "VE", "EVJ", "SC", "NF", "NC", "MSR", "VA"]}
    # data = {key: [] for key in ["CTR", "VEC", "VPR", "VE", "EVJ", "SC", "NF", "NC"]}

    # 合并每一类 QA
    for k, v in qa_dict.items():
        data.setdefault(k, []).extend(v)

    # 写入
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def read_python_file(file_path):
    """
    以文本方式读取任意文件（包括 SVG）。
    返回文件内容的字符串形式。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    # todo 修改路径和任务类型
    csv_folder = './csv/'
    # svg_folder = './svg/bar_chart/'
    # 遍历文件夹下所有文件（全部都是 .csv）
    j = 1
    for fname in os.listdir(csv_folder):
    # for fname in [f'bar_Agriculture_and_Food_Production_{i}.csv' for i in range(1,9)]:
        # 构造完整路径
        csv_path = os.path.join(csv_folder, fname)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]

        # 读取首行信息
        meta = read_metadata(csv_path)

        # qa_dict = {key: [] for key in ["CTR", "VEC", "VPR", "VE", "EVJ", "SC", "NF", "NC"]}
        qa_dict = {key: [] for key in ["CTR", "VEC", "SRP", "VPR", "VE", "EVJ", "SC", "NF", "NC", "MSR", "VA"]}
        df = pd.read_csv(csv_path, skiprows=1, header=0)
        column_names = df.columns.tolist()
        category_name = df.iloc[:, 0].to_list()
        bar_count = df.iloc[:, 0].count()
        

    #CTR
        CTR_question = {'Q': "What type of chart is this?", 'A': "This chart is a {bar} chart."}
        qa_dict["CTR"] = [CTR_question]

    #VEC
        # 读取 CSV 文件：跳过第一行，第二行作为列名
        VEC_question = {'Q': "How many bars are in this bar chart?", 'A': f"There are {{{bar_count}}} bars in this bar chart."}
        qa_dict["VEC"] = [VEC_question]

    #SRP
        qa_dict['SRP'] = []

        Question = "Is this bar chart a horizontal bar chart or a vertical bar chart?"
        Answer = f"The bar chart is a {{{meta['orientation']}}} bar chart."
        qa_dict['SRP'].append({'Q': Question, 'A':Answer})

        if "horizontal" == meta['orientation']:
            Question = "Which category does the bottom bar represent?"
            Answer = f"The bottom bar represents {{{category_name[0]}}}."
        elif "vertical" == meta['orientation']:
            Question = "Which category does the leftmost bar represent?"
            Answer = f"The leftmost bar represents {{{category_name[0]}}}."
        else:
            raise ValueError("LLM的回答中没有显式的说明，柱状图是横向的还是竖向的。")
        qa_dict['SRP'].append({'Q': Question, 'A': Answer})


        if "horizontal" in meta["orientation"]:
            Question = "Which category does the topmost bar represent?"
            Answer = f"The topmost bar represents {{{category_name[-1]}}}."
        elif "vertical" in meta['orientation']:
            Question = "Which category does the rightmost bar represent?"
            Answer = f"The rightmost bar represents {{{category_name[-1]}}}."
        else:
            raise ValueError("LLM的回答中没有显式的说明，柱状图是横向的还是竖向的。")
        qa_dict['SRP'].append({'Q': Question, 'A':Answer})

        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        cat2 = df.iloc[idx2, 0]
        if "vertical" == meta["orientation"]:
            Question = f"What is the spatial relationship of bar {cat1} relative to the bar {cat2} in terms of horizontal (left/right) direction?"
            if( idx1 < idx2 ) : Answer = f"Bar {cat1} is to the {{left}} of the bar {cat2}."
            else: Answer = f"Bar {cat1} is to the {{right}} of the bar {cat2}."
        elif "horizontal" == meta["orientation"]:
            Question = f"What is the spatial relationship of bar {cat1} relative to the bar {cat2} in terms of vetical (above/below) direction??"
            if( idx1 < idx2 ) : Answer = f"Bar {cat1} is  {{below}} the bar {cat2}."
            else: Answer = f"Bar {cat1} is {{above}} the bar {cat2}."
        else:
            raise ValueError("LLM的回答中没有显式的说明，柱状图是横向的还是竖向的。")
        qa_dict['SRP'].append({'Q': Question, 'A':Answer})

    #VPR
        # 提取最大值对应的类别
        max_idx = df.iloc[:, 1].idxmax()  # 数值列中最大值所在的行索引
        max_category = df.iloc[max_idx, 0]  # 该行对应的类别名称
        # 提取最小值对应的类别
        min_idx = df.iloc[:, 1].idxmin()  # 数值列中最小值所在的行索引
        min_category = df.iloc[min_idx, 0]  # 该行对应的类别名称
        VPR_question_1 = {'Q': "Which category has the highest bar in this bar chart?", 'A': f"The category with the highest bar is {{{ str(max_category) }}}."}
        VPR_question_2 = {'Q': "Which category has the lowest bar in this bar chart?", 'A': f"The category with the lowest bar is {{{ str(min_category) }}}."}
        qa_dict["VPR"] = [VPR_question_1,VPR_question_2]

    #VE
        qa_dict['VE'] = []
        for i in random.sample(range(len(category_name)), random.randint(2,3)):
            VE_question = {'Q': f"What is the value of bar {category_name[i]}?",
                           'A': f"The value of { str(category_name[i]) } is {{{ str(df.iloc[i, 1]) }}} { meta['unit'] }."}
            qa_dict['VE'].append(VE_question)
        
    #EVJ
        VEJ_question1 = {'Q': f"What is the maximum {meta['unit']} in the bar chart?", 
                         'A': f"The maximum {meta['unit']} is {{{ str(df.iloc[:, 1].max()) }}} {meta['unit']}."}
        VEJ_question2 = {'Q': f"What is the minimum {meta['unit']} in the bar chart?", 
                         'A': f"The minimum {meta['unit']} is {{{ str(df.iloc[:, 1].min()) }}} {meta['unit']}."}
        qa_dict["EVJ"] = [VEJ_question1, VEJ_question2]

    #SC
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        val1 = df.iloc[idx1, 1]
        cat2 = df.iloc[idx2, 0]
        val2 = df.iloc[idx2, 1]
        # 计算指标
        total = val1 + val2
        SC_question1 = {'Q': f"What is the total value of {cat1} and {cat2}?", 
                        'A': f"The total value of {cat1} and {cat2} is {{{ str(total) }}} {meta['unit']}."}
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        val1 = df.iloc[idx1, 1]
        cat2 = df.iloc[idx2, 0]
        val2 = df.iloc[idx2, 1]
        average = (val1 + val2) / 2
        SC_question2 = {'Q': f"What is the average value of {cat1} and {cat2}?", 
                        'A': f"The average value of {cat1} and {cat2} is {{{ str(average) }}} {meta['unit']}."}
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        val1 = df.iloc[idx1, 1]
        cat2 = df.iloc[idx2, 0]
        val2 = df.iloc[idx2, 1]
        difference = abs(val1 - val2)
        SC_question3 = {'Q': f"What is the difference between {cat1} and {cat2}?", 
                        'A': f"The difference between {cat1} and {cat2} is {{{ str(difference) }}} {meta['unit']}."}
        qa_dict["SC"] = [SC_question1, SC_question2, SC_question3]
    #NF
        # 可以使用负数作为参数，最后一列为堆叠柱子的总和值
        total_value = df.sort_values(by = column_names[1], ascending=True, ignore_index=True) # 按照Total值进行升序排序，索引值不变
        avg_list = [ (total_value.iloc[i, 1] + total_value.iloc[i + 1, 1]) / 2 for i in range(len(total_value) - 1)]
        
        idx = max(random.randint( len(total_value) - 3 , len(total_value) - 2 ), 0)
        avg_value = avg_list[idx]
        NF_question_1 = {'Q': f"Which bars have { meta['unit'] } values exceed {avg_value:.2f} {meta['unit']}? Please list the bars and corresponding { meta['unit'] } values.",
                         'A':", ".join([f"{{{row.iloc[0]}}} has {{{row.iloc[1]}}} {meta['unit'] }" for _, row in total_value.iloc[idx + 1 :, ].iterrows()]) +'.'or "None"}
        
        idx = min(random.randint( 0 , 2 ), len(avg_list) - 1)
        avg_value = avg_list[idx]
        NF_question_2 = {'Q': f"Which bars have {{{meta['unit'] }}} values below {avg_value:.2f} {meta['unit']}? Please list the bars and corresponding {meta['unit'] } values.",
                         'A':", ".join([f"{{{row.iloc[0]}}} has {{{row.iloc[1]}}} {meta['unit'] }" for _, row in total_value.iloc[: idx + 1, ].iterrows()]) +'.'or "None"}
        qa_dict["NF"] = [NF_question_1, NF_question_2]

    #NC
        qa_dict["NC"] = []
        for num in range(2,min(len(df), 5)):
            idx = random.sample(range(len(df)), num)
            # 提取类别和对应的数值
            selected_rows = df.iloc[idx]
            names = selected_rows.iloc[:, 0].tolist()
            # names = ['{' + name + '}' for name in names]
            values = selected_rows.iloc[:, 1].tolist()
            # 找出最大值的索引
            max_idx = values.index(max(values))
            max_name = names[max_idx]
            NC_question = {'Q': f"Which is larger? {', '.join(names[:-1])} or {names[-1]}.", 'A': f"The value of the {{{ str(max_name) }}} is larger."}

            qa_dict["NC"].append(NC_question)


    #MSR
        qa_dict['MSR'] = []
        # 获取数值列
        values = df.iloc[:, 1]
        # 计算平均值
        average_value = values.mean()
        # 统计高于平均值的数量
        above_avg_count = (values > average_value).sum()
        Question = "How many bars in the bar chart have values above the overall average?"
        Answer = f"There are {{{ above_avg_count }}} bars with values above the average."
        qa_dict['MSR'].append({'Q': Question, 'A':Answer})

        # 计算相邻差值
        differences = values.diff().abs()  # 取绝对值
        # 找到最大差值的位置索引
        max_diff_index = differences.idxmax()
        # 获取这对相邻栏的标签和差值
        category_1 = df.iloc[max_diff_index - 1, 0]
        category_2 = df.iloc[max_diff_index, 0]
        Question = "Which pair of adjacent bars has the largest difference in values?"
        Answer = f"The pair of adjacent bars with the largest difference is between {{{ category_1 }}} and {{{ category_2 }}}."
        qa_dict['MSR'].append({'Q': Question, 'A':Answer})

        top_3 = df.sort_values(by=df.columns[1], ascending=False).head(3)
        category_1, category_2, category_3= top_3.iloc[:, 0].tolist()

        Question = "Which three categories have the highest combined values?"
        Answer = f"The three categories with the highest combined values are: {{{category_1}}}, {{{category_2}}}, and {{{category_3}}}."
        qa_dict['MSR'].append({'Q': Question, 'A':Answer})

    # #VA
        # 获取列名
        category_col = df.iloc[:, 0].name  # 第一列为类别名称
        value_col = df.iloc[:, 1].name     # 第二列为数值

        # 排序后的 DataFrame
        df_desc = df.sort_values(by=value_col, ascending=False)
        df_asc = df.sort_values(by=value_col, ascending=True)

        # 所有类别名称（按排序后）
        sorted_categories_desc = ['{' + col + '}' for col in df_desc[category_col].tolist()]
        sorted_categories_asc = ['{' + col + '}' for col in df_asc[category_col].tolist()]
        
        qa_dict['VA'] = []
        Question = f"Sort the bars by { meta['unit'] } values in descending order and list the labels from left to right."
        Answer = f"Sorted by {meta['unit']} value descending: {', '.join(sorted_categories_desc[:-1])}, and {sorted_categories_desc[-1]}."

        qa_dict['VA'].append({'Q': Question, 'A':Answer})

        Question = f"Sort the bars by { meta['unit'] } values in ascending order and list the labels from left to right."
        Answer = f"Sorted by {meta['unit']} value ascending: {', '.join(sorted_categories_asc[:-1])}, and {sorted_categories_asc[-1]}."
        qa_dict['VA'].append({'Q': Question, 'A':Answer})

        write_all_qa_to_json(csv_path=csv_path, qa_dict=qa_dict)
        if j%100 == 0:
            print(f"Now,{j} QAs have been generated!!!")
        j += 1


if __name__ == '__main__':
    main()