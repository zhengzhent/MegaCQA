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

def write_qa_to_json(json_path: str, qa_type: str, qa_item: dict):

    # 加载或初始化
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 初始化各类空列表
        data = {key: [] for key in ["CTR", "VEC", "SPR", "VPR", "VE", "EVJ", "SC", "NF", "NC", "MSR", "VA"]}
    # 追加 QA
    data.setdefault(qa_type, []).append(qa_item)

    # 写回
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def write_all_qa_to_json(json_path: str, qa_dict: dict):


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
    root_path = 'X:/UniversityCourseData/Visualization/2025.4.12可视化数据集构建/stacked_bar_chart'
    csv_folder = './csv'
    qa_folder = './QA'
    
    # 遍历文件夹下所有文件（全部都是 .csv）
    # for fname in os.listdir(csv_folder):
    for fname in os.listdir(csv_folder):
        # 构造完整路径
        csv_path = os.path.join(csv_folder, fname)
        # 将 .csv 后缀替换为 .json
        base_name = os.path.splitext(fname)[0]  # 去掉扩展名
        json_filename = base_name + '.json'
        json_path = os.path.join(qa_folder, json_filename)

        # 读取首行信息
        meta = read_metadata(csv_path)

        # qa_dict = {key: [] for key in ["CTR", "VEC", "VPR", "VE", "EVJ", "SC", "NF", "NC"]}
        qa_dict = {key: [] for key in ["CTR", "VEC", "SRP", "VPR", "VE", "EVJ", "SC", "NF", "NC", "MSR", "VA"]}

        df = pd.read_csv(csv_path, skiprows=1, header=0)
        column_names = df.columns.tolist()
        category_name = df.iloc[:, 0]
        bar_count = df.iloc[:, 0].count()
        
    #CTR
        CTR_question = {'Q': "What type of chart is this?", 'A': "This chart is a {stacked bar} chart."}
        qa_dict["CTR"] = [CTR_question]

    #VEC
        # 读取 CSV 文件：跳过第一行，第二行作为列名
        VEC_question1 = {'Q': "How many bars are in this stacked bar chart?", 'A': f"There are {{{bar_count}}} bars."}
        idx1 = random.sample(range(len(df)), 1)
        cat1 = df.iloc[idx1[0], 0]
        VEC_question3 = {'Q': f"How many stacked segments are in bar { str(cat1)} of this stacked bar chart?",
                         'A': f"There are {{{ len(column_names) - 1}}} stacked bars."}
        qa_dict["VEC"] = [VEC_question1,  VEC_question3]
    #SRP
        qa_dict['SRP'] = []

        Question = "Is this stacked bar chart a horizontal stacked bar chart or a vertical stacked bar chart?"
        Answer = f"The stacked bar chart is a {{{meta['orientation']}}} stacked bar chart."
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

        if "horizontal" == meta['orientation']:
            Question = "Which category does the topmost bar represent?"
            Answer = f"The bottom bar represents {{{category_name.to_list()[-1]}}}."
        elif "vertical" == meta['orientation']:
            Question = "Which category does the rightmost bar represent?"
            Answer = f"The leftmost bar represents {{{category_name.to_list()[-1]}}}."
        else:
            raise ValueError("LLM的回答中没有显式的说明，柱状图是横向的还是竖向的。")
        qa_dict['SRP'].append({'Q': Question, 'A':Answer})

        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        cat2 = df.iloc[idx2, 0]
        if "vertical" == meta['orientation']:
            Question = f"What is the spatial relationship of bar {cat1} relative to the bar {cat2} in terms of horizontal (left/right) direction?"
            if( idx1 < idx2 ) : Answer = f"Bar {cat1} is to the {{left}} of the bar {cat2}."
            else: Answer = f"Bar {cat1} is to the {{right}} of the bar {cat2}."
            qa_dict['SRP'].append({'Q': Question, 'A':Answer})

            idx1 = random.choice(range(len(df)))
            cat1 = df.iloc[idx1, 0]
            seg1, seg2 = random.sample(range(len(column_names) - 1), 2)
            Question = f"In the stacked bar of {cat1}, What is the spatial relationship of segment {column_names[seg1 + 1]} relative to the segment {column_names[seg2 + 1]} in terms of vertical (above/below) direction?"
            if(seg1 > seg2):
                Answer = f"{column_names[seg1 + 1]} segment is positioned {{above}} the {column_names[seg2 + 1]} segment."
            else:
                Answer = f"{column_names[seg1 + 1]} segment is positioned {{below}} the {column_names[seg2 + 1]} segment."
            qa_dict['SRP'].append({'Q': Question, 'A':Answer})


        elif "horizontal" == meta['orientation']:
            Question = f"What is the spatial relationship of bar {cat1} relative to the bar {cat2} in terms of vertical (below/above) direction?"
            if( idx1 < idx2 ) : 
                Answer = f"Bar {cat1} is {{below}} the bar {cat2}."
            else: 
                Answer = f"Bar {cat1} is {{above}} the bar {cat2}."

            qa_dict['SRP'].append({'Q': Question, 'A':Answer})

            idx1 = random.choice(range(len(df)))
            cat1 = df.iloc[idx1, 0]
            seg1, seg2 = random.sample(range(len(column_names) - 1), 2)
            Question = f"In the stacked bar of {cat1}, What is the spatial relationship of segment {column_names[seg1 + 1]} relative to the segment {column_names[seg2 + 1]} in terms of horizontal (left/right) direction?"
            if(seg1 > seg2):
                Answer = f"{column_names[seg1 + 1]} segment is positioned {{right}} the {column_names[seg2 + 1]} segment."
            else:
                Answer = f"{column_names[seg1 + 1]} segment is positioned {{left}} the {column_names[seg2 + 1]} segment."
            qa_dict['SRP'].append({'Q': Question, 'A':Answer})

        else:
            raise ValueError("LLM的回答中没有显式的说明，柱状图是横向的还是竖向的。")
        
        
    #VPR
        df['Total'] = df.iloc[:, 1 : ].sum(axis=1)
        # 找到最大  值和最小值对应的类别
        max_idx = df['Total'].idxmax()
        min_idx = df['Total'].idxmin()
        max_category = df.iloc[max_idx, 0]
        min_category = df.iloc[min_idx, 0]
        VPR_question_1 = {'Q': f"Which {column_names[0]} has the highest bar in this stacked bar chart?",
                          'A': f"The {column_names[0]} with the highest bar is {{{ str(max_category) }}}."}
        VPR_question_2 = {'Q': f"Which {column_names[0]} has the lowest bar in this stacked bar chart?", 
                          'A': f"The {column_names[0]} with the lowest bar is {{{ str(min_category) }}}."}
        
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        flatten_values = df.iloc[idx1, 1: len(df.columns) - 1].values.flatten()
        max_segment = df.columns[1 + flatten_values.argmax()]
        VPR_question_3 = {'Q': f"In the stacked bar representing {cat1}, which segment has the highest value?",
                          'A': f"The segment with the highest value is {{{max_segment}}}."}
        cat2 = df.iloc[idx2, 0]
        flatten_values = df.iloc[idx2, 1: len(df.columns) - 1].values.flatten()
        min_segment = df.columns[1 + flatten_values.argmin()]
        VPR_question_4 = {'Q': f"In the stacked bar representing {cat2}, which segment has the lowest value?",
                          'A': f"The segment with the lowest value is {{{min_segment}}}."}
        qa_dict["VPR"] = [VPR_question_1,VPR_question_2,VPR_question_3,VPR_question_4]

    #VE
        qa_dict["VE"] = []
        idx = random.sample(range(len(category_name)), 3)
        for i in idx:
            j = random.randint( 1, len( df.columns ) - 2 )
            segment = df.columns[j]
            VE_question = {'Q': f"What is the value of the { segment } segment within the { category_name.iloc[i] } bar in the stacked bar chart?", 
                            'A': f"The value of the {segment} segment is {{{df.iloc[i, j]:.2f}}} {meta['unit']}."}
            qa_dict["VE"].append(VE_question)

    #EVJ
        
        idx1 = random.randint(0, len(df) - 1)
        # print(df.iloc[idx1, 1: len( df.columns ) - 2].dtype)
        max_idx = df.iloc[idx1, 1: len( df.columns ) - 1].values.argmax() + 1
        
        # 柱子里最高的segment
        EVJ_question_1 = {'Q': f"What is the maximum {meta['unit']} among the stacked segments of bar {df.iloc[idx1, 0]}?",
                          'A': f"The maximum {meta['unit']} among the stacked segments of bar {df.iloc[idx1, 0]} is {{{df.iloc[idx1, max_idx]}}} {meta['unit']}."}
        idx1 = random.randint(0, len(df) - 1)
        # print(df.iloc[idx1, 1: len( df.columns ) - 2].dtype)
        min_idx = df.iloc[idx1, 1: len( df.columns ) - 1].values.argmin() + 1
        EVJ_question_2 = {'Q': f"What is the minimum {meta['unit']} among the stacked segments of bar {df.iloc[idx1, 0]}?",
                          'A': f"The minimum {meta['unit']} among the stacked segments of bar {df.iloc[idx1, 0]} is {{{df.iloc[idx1, min_idx]}}} {meta['unit']}."}
        
        #segment里最高的柱子
        idx1 = random.randint(1, len(column_names) - 1)
        max_idx = df[column_names[idx1]].idxmax()
        EVJ_question_3 = {'Q': f"In the stacked bar chart, what is the maximum {meta['unit']} of the segment {column_names[idx1]} among the bars?",
                          'A': f"The maximum {meta['unit']} among the bars of segment {column_names[idx1]} is {{{df.iloc[max_idx, idx1]}}} {meta['unit']}."}
        idx1 = random.randint(1, len(column_names) - 1)
        min_idx = df[column_names[idx1]].idxmin()
        EVJ_question_4 = {'Q': f"In the stacked bar chart, what is the minimum {meta['unit']} of the segment {column_names[idx1]} among the bars?",
                          'A': f"The minimum {meta['unit']} among the bars of segment {column_names[idx1]} is {{{df.iloc[min_idx, idx1]}}} {meta['unit']}."}
        qa_dict['EVJ'] = [EVJ_question_1, EVJ_question_2, EVJ_question_3, EVJ_question_4]

    #SC
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        val1 = df.iloc[idx1, len(df.columns) - 1]
        cat2 = df.iloc[idx2, 0]
        val2 = df.iloc[idx2, len(df.columns) - 1]
        SC_question1 = {'Q': f"What is the total value of the bar {cat1}?", 'A': f"The total value of bar {cat1} is {{{val1:.2f}}} {meta['unit']}."}
        SC_question2 = {'Q': f"What is the total value of the bar {cat2}?", 'A': f"The total value of bar {cat2} is {{{val2:.2f}}} {meta['unit']}."}
        SC_question3 = {'Q': f"What is the total value of all columns?", 'A': f"The total value of columns is {{{ df['Total'].sum(axis=0) }}} {meta['unit']}."}
        idx1, idx2 = random.sample(range(len(df)), 2)
        # 获取类别名称和对应的数值
        cat1 = df.iloc[idx1, 0]
        val1 = df.iloc[idx1, len(df.columns) - 1]
        cat2 = df.iloc[idx2, 0]
        val2 = df.iloc[idx2, len(df.columns) - 1]
        # 计算指标
        difference = abs(val1 - val2)
        SC_question4 = {'Q': f"What is the difference between total value of bar {cat1} and {cat2}?", 
                        'A': f"The difference between between total value of bar {cat1} and {cat2} is {{{ str(difference) }}} {meta['unit']}."}
        qa_dict["SC"] = [SC_question1, SC_question2, SC_question3, SC_question4]

    #NF
        # 可以使用负数作为参数，最后一列为堆叠柱子的总和值
        total_value = df.iloc[:, [ 0 , -1]].sort_values(by = 'Total', ascending=True, ignore_index=True) # 按照Total值进行升序排序，索引值不变
        avg_list = [ (total_value.iloc[i, 1] + total_value.iloc[i + 1, 1]) / 2 for i in range(len(total_value) - 1)]

        idx = min(random.randint( 0 , 2 ), len(avg_list) - 1)
        avg_value = avg_list[idx]

        NF_question_2 = {'Q': f"Which bars have total {meta['unit']} values below {avg_value:.2f} {meta['unit']}? Please list the bar and corresponding values.",
                         'A':", ".join([f"{{{row.iloc[0]}}} has {{{row.iloc[-1]}}} {meta['unit']}" for _, row in total_value.iloc[:idx + 1 , ].iterrows()]) + "."}
        
        idx = max(random.randint( len(total_value) - 3 , len(total_value) - 2 ), 0)
        avg_value = avg_list[idx]
        NF_question_1 = {'Q': f"Which bars have total {meta['unit']} values exceed {avg_value:.2f} {meta['unit']}? Please list the bar and corresponding values.",
                         'A':", ".join([f"{{{row.iloc[0]}}} has {{{row.iloc[-1]}}} {meta['unit']}" for _, row in total_value.iloc[idx + 1:, ].iterrows()]) + "."}
        qa_dict["NF"] = [NF_question_1, NF_question_2]
        
        idx = random.choice(range(len(df)))
        cat1 = df.iloc[idx, 0]
        # 最后一列是‘Total’，所以需要排除
        selected_rows = df.iloc[idx, 1 : len(column_names) ].sort_values()# 1️⃣ 对 selected_rows 按照值排序
        # 2️⃣ 生成相邻值之间的平均值
        avg_list = [(0 + selected_rows.iloc[0])/2]
        for i in range(len(selected_rows) - 1):
            avg = (selected_rows.iloc[i] + selected_rows.iloc[i + 1]) / 2
            avg_list.append(avg)
        avg_list.append(selected_rows.iloc[0] + selected_rows.iloc[len(selected_rows) - 1])
        
        if len(column_names) - 1 > 0:
            interval_length = min( random.randint(1, 3), len(column_names) - 1 )
            idx1 = random.randint(0, len(avg_list) - interval_length - 1)
            idx2 = idx1 + interval_length

            # 筛选出值在 avg_list[idx1] 到 avg_list[idx2] 之间的项
            filtered = selected_rows[
                (selected_rows > avg_list[idx1]) &
                (selected_rows < avg_list[idx2])
            ]
            NF_question_3 = {'Q': f"For bar {cat1} in the stacked bar chart, which segment have {meta['unit']} value between {avg_list[idx1]} and {avg_list[idx2]}? Please list the segment and corresponding values.",
                            #  'A': ", ".join([f"{{{selected_rows.index[i]}}} has {{{selected_rows.iloc[i]} {meta['unit']}}}" if selected_rows.iloc[i] > avg_list[idx1] and selected_rows.iloc[i] < avg_list[idx2] else for i in range(idx1, idx2)]) + '.'}
                            'A':", ".join([f"{{{index}}} has {{{value}}} {meta['unit']}" for index, value in filtered.items()]) + "."}
            qa_dict['NF'].append(NF_question_3)
        # qa_dict["NF"] = [NF_question_1, NF_question_2, NF_question_3]

    #NC
        qa_dict["NC"] = []
        for num in range(2,min(len(df) + 1, 4)):
            idx = random.sample(range(len(df)), num)
            # 提取类别和对应的数值
            selected_rows = df.iloc[idx]
            names = selected_rows.iloc[:, 0].tolist()
            names = [str(n) for n in names]
            values = selected_rows.iloc[:, -1].tolist()
            # 找出最大值的索引
            max_idx = values.index(max(values))
            max_name = names[max_idx]
            NC_question = {'Q': f"Which is larger? the total {meta['unit']} of {', '.join(names[:-1])} or {names[-1]}.",
                           'A': f"The total {meta['unit']} of the {{{ str(max_name) }}} is larger."}
            qa_dict["NC"].append(NC_question)
        
        # 柱子内部的比较
        if len(column_names) - 1 > 1:
            idx = random.randint(0, len(df) - 1)
            selected_col_list = random.sample(range(1, len(column_names)), 2)
            max_idx = max(selected_col_list, key= lambda x : df.iloc[idx, x])
            NC_question = {'Q': f"In the bar {category_name.iloc[idx]}, which has a larger {meta['unit']} value? {column_names[selected_col_list[0]]} or {column_names[selected_col_list[1]]}.",
                           'A': f"{{{column_names[max_idx]}}} has the larger {meta['unit']} value."}
            qa_dict["NC"].append(NC_question)

        if len(column_names) - 1 > 2:
            idx = random.randint(0, len(df) - 1)
            selected_col_list = random.sample(range(1, len(column_names)), 3)
            max_idx = max(selected_col_list, key= lambda x : df.iloc[idx, x])
            NC_question = {'Q': f"In the bar {category_name.iloc[idx]}, which has a larger {meta['unit']} value? {column_names[selected_col_list[0]]}, {column_names[selected_col_list[1]]} or {column_names[selected_col_list[2]]}.",
                           'A': f"{{{column_names[max_idx]}}} has the larger {meta['unit']} value."}
            qa_dict["NC"].append(NC_question)

        # if len(column_names) - 1 > 3:
        #     idx = random.randint(0, len(df) - 1)
        #     selected_col_list = random.sample(range(1, len(column_names)), 4)
        #     max_idx = max(selected_col_list, key= lambda x : df.iloc[idx, x])
        #     NC_question = {'Q': f"In the bar {{{category_name.iloc[idx]}}}, which has a larger {meta['unit']} value: {{{column_names[selected_col_list[0]]}}}, {{{column_names[selected_col_list[1]]}}}, {{{column_names[selected_col_list[2]]}}}or {{{column_names[selected_col_list[3]]}}}?",
        #                    'A': f"{{{column_names[max_idx]}}} has the larger {meta['unit']} value."}
        #     qa_dict["NC"].append(NC_question)

    
    #MSR
        # idx1, idx2 = random.sample(range(len(df)), 2)
        # # 获取类别名称和对应的数值
        # cat1 = df.iloc[idx1, 0]
        # cat2 = df.iloc[idx2, 0]
        # j = random.randint( 1, len( df.columns ) - 2 )
        # segment = df.columns[j]
        # difference = df.iloc[idx1,j] - df.iloc[idx2, j]
        # operater = 'decreased' if difference < 0 else 'increased'
        # difference = abs(difference)

        # MSR_question_1 = {'Q': f"Between bar {{{cat1}}} and bar {{{cat2}}} in the stacked bar chart, what is the increase or decrease in value of segment {{{segment}}}?",
        #                   'A': f"The value of segment {{{segment}}} has {operater} by {{{difference}}}."}
        # selected_line = df.iloc[ :, j].tolist()
        # max_idx = selected_line.index(max(selected_line))
        # max_name = category_name.tolist()[max_idx]
        # MSR_question_2 = {'Q': f"Which bar has the highest value for segment {{{segment}}} across all bars in the stacked bar chart?",
        #                   'A': f"The bar {{{max_name}}} with the highest value for segment {{{segment}}} is China."}
        
        # 将类别名与对应 segment 的值组合成 DataFrame 并排序
        j = random.randint( 1, len( df.columns ) - 2 )
        segment = df.columns[j]
        selected_line = df.iloc[ :, j].tolist()
        sorted_bars = pd.DataFrame({
            'Category': category_name,
            'Value': selected_line
        }).sort_values(by='Value', ascending=False)

        # 获取前三名的类别名
        top_three = sorted_bars.head(3)['Category'].tolist()

        if len(top_three) < 3:
            top_labels = ', '.join(top_three[:2]) + f", and {top_three[2] if len(top_three)==3 else ''}"
        top_three = [f"{{{col}}}" for col in top_three]
        top_labels = ', '.join( top_three) 
        
        MSR_question_3 = {'Q': f"Ranking bar by the value of segment {segment} in descending order, which are the top three countries(if the number of bars is less 3,list the existing bars in descending order)?",
                          'A': f"The top three bar ranked by the value of segment {segment} are: {top_labels}."}
        
        qa_dict['MSR'] = [ MSR_question_3]

        write_all_qa_to_json(json_path=json_path, qa_dict=qa_dict)


if __name__ == '__main__':
    main()