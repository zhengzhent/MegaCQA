      
import os
import pandas as pd
import json
from io import StringIO
from scipy.stats import gaussian_kde
import numpy as np
import random

input_dir = './csv'

output_dir = './QA'

def sum_dims_by_lab(data, index):
    sum = 0
    for i in range(0,len(dimensions)):
        sum+= data[index][i]
    return sum

def sum_labs_by_dim(data, index):
    sum = 0
    for i in range(0,len(labels)):
        sum+= data[i][index]
    return sum

def max_dims_by_lab(data):
    max = 0
    index=0
    for i in range(0,len(labels)):
        sum=sum_dims_by_lab(data,i)
        if sum>max:
            max =sum
            index=i
    return max,index

def get_max_min_indices(data, index):
    if index < 0 or index >= len(data):
        raise ValueError("行索引超出范围")
    row = data[index]
    max_index = row.index(max(row))
    min_index = row.index(min(row))
    
    return max_index, min_index

# def get_max_index_by_column(data, col_index):
#     if not data or col_index < 0 or col_index >= len(data[0]):
#         raise ValueError("列索引超出范围或数据为空")
    
#     column = [row[col_index] for row in data]
#     max_index = column.index(max(column))
    
#     return max_index

def get_max_index_by_column(data, col_index):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    column = [row[col_index] for row in data]
    max_value = max(column)
    max_indices = [i for i, v in enumerate(column) if v == max_value]
    return max_indices

def get_max_index_by_row(data, row_index):
    if not data or row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围或数据为空")
    row = data[row_index]
    max_index = row.index(max(row))
    return max_index

# def get_min_index_by_column(data, col_index):
#     if not data or col_index < 0 or col_index >= len(data[0]):
#         raise ValueError("列索引超出范围或数据为空")
    
#     column = [row[col_index] for row in data]
#     min_index = column.index(min(column))
    
#     return min_index

def get_min_index_by_column(data, col_index):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    column = [row[col_index] for row in data]
    min_value = min(column)
    min_indices = [i for i, v in enumerate(column) if v == min_value]
    return min_indices

def get_min_index_by_row(data, row_index):
    if not data or row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围或数据为空")
    row = data[row_index]
    min_index = row.index(min(row))
    return min_index

def get_index_greater_than(data, row_index, x):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    indices = [i for i, v in enumerate(data[row_index]) if v > x]
    return indices if indices else "None"

def get_index_less_than(data, row_index, x):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    indices = [i for i, v in enumerate(data[row_index]) if v < x]
    return indices if indices else "None"

def get_index_between(data, row_index, x1, x2):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    indices = [i for i, v in enumerate(data[row_index]) if x1 < v < x2]
    return indices if indices else "None"

def get_indices_greater_than_col(data, col_index, x):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    indices = [i for i, row in enumerate(data) if row[col_index] >= x]
    return indices if indices else "None"

def get_indices_less_than_col(data, col_index, x):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    indices = [i for i, row in enumerate(data) if row[col_index] <= x]
    return indices if indices else "None"

def get_indices_between_col(data, col_index, x1, x2):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    indices = [i for i, row in enumerate(data) if x1 >= row[col_index] >= x2]
    return indices if indices else "None"

def get_lab_over_result(data, row_index, dimensions, threshold):
    indices = get_index_greater_than(data, row_index, threshold)
    if indices == "None":
        return f"None is over {threshold}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{dimensions[idx]}}} is {{{data[row_index][idx]}}}")
        return ", ".join(result_list)
    
def get_lab_under_result(data, row_index, dimensions, threshold=50):
    indices = get_index_less_than(data, row_index, threshold)
    if indices == "None":
        return f"None is under {threshold}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{dimensions[idx]}}} is {{{data[row_index][idx]}}}")
        return ", ".join(result_list)

def get_lab_between_result(data, row_index, dimensions, x1, x2):
    indices = get_index_between(data, row_index, x1, x2)
    if indices == "None":
        return f"None is between {x1} and {x2}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{dimensions[idx]}}} is {{{data[row_index][idx]}}}")
        return ", ".join(result_list)

def get_dim_over_result(data, col_index, labels, threshold=50):
    indices = get_indices_greater_than_col(data, col_index, threshold)
    if indices == "None":
        return f"None is over {threshold}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{labels[idx]}}} is {{{data[idx][col_index]}}}")
        return ", ".join(result_list)

def get_dim_under_result(data, col_index, labels, threshold=50):
    indices = get_indices_less_than_col(data, col_index, threshold)
    if indices == "None":
        return f"None is under {threshold}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{labels[idx]}}} is {{{data[idx][col_index]}}}")
        return ", ".join(result_list)

def get_dim_between_result(data, col_index, labels, x1, x2):
    indices = get_indices_between_col(data, col_index, x1, x2)
    if indices == "None":
        return f"None is between {x1} and {x2}"
    else:
        result_list = []
        for idx in indices:
            result_list.append(f"{{{labels[idx]}}} is {{{data[idx][col_index]}}}")
        return ", ".join(result_list)

def compare_two_cols_in_row(data, row_index, col_index1, col_index2):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    if col_index1 < 0 or col_index1 >= len(data[0]) or col_index2 < 0 or col_index2 >= len(data[0]):
        raise ValueError("列索引超出范围")
    if data[row_index][col_index1] >= data[row_index][col_index2]:
        return col_index1,col_index2
    else:
        return col_index2,col_index1

def get_max_min_by_col(data, col_index):
    if not data or col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围或数据为空")
    column = [row[col_index] for row in data]
    return max(column), min(column)

def get_max_min_by_row(data, row_index):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    row = data[row_index]
    return max(row), min(row)

def compare_inside_outside(num1, num2):
    if num1 > num2:
        return "outside"
    else:
        return "inside"
#左大右小
def compare_two_rows_in_col(data, col_index, row_index1, row_index2):
    if col_index < 0 or col_index >= len(data[0]):
        raise ValueError("列索引超出范围")
    if row_index1 < 0 or row_index1 >= len(data) or row_index2 < 0 or row_index2 >= len(data):
        raise ValueError("行索引超出范围")
    if data[row_index1][col_index] >= data[row_index2][col_index]:
        return row_index1,row_index2
    else:
        return row_index2,row_index1

def get_max_sum_row_index(data):
    """
    返回每一行求和后，和最大的行索引
    """
    if not data or not data[0]:
        raise ValueError("数据为空")
    row_sums = [sum(row) for row in data]
    max_row_index = row_sums.index(max(row_sums))
    return max_row_index

def get_min_sum_row_index(data):
    """
    返回每一行求和后，和最小的行索引
    """
    if not data or not data[0]:
        raise ValueError("数据为空")
    row_sums = [sum(row) for row in data]
    min_row_index = row_sums.index(min(row_sums))
    return min_row_index

def max_of_three_cols_in_row(data, row_index, col_index1, col_index2, col_index3):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    for col in (col_index1, col_index2, col_index3):
        if col < 0 or col >= len(data[0]):
            raise ValueError("列索引超出范围")
    values = [
        (data[row_index][col_index1], col_index1),
        (data[row_index][col_index2], col_index2),
        (data[row_index][col_index3], col_index3)
    ]
    max_col = max(values, key=lambda x: x[0])[1]

    return max_col

def get_min_index_by_row(data, row_index):
    if row_index < 0 or row_index >= len(data):
        raise ValueError("行索引超出范围")
    row = data[row_index]
    min_index = row.index(min(row))
    return min_index

os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):

    csv_path = os.path.join(input_dir, filename)
    print(f"\n[INFO] 正在处理文件: {filename}")
    print(csv_path)
    if not os.path.exists(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}")
        continue
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            lines = f.readlines()
        
        #维度名
        dimensions=pd.read_csv(csv_path, header=None, skiprows=1, nrows=1).iloc[0,1:].tolist()
        #类别名
        labels=pd.read_csv(csv_path, header=None, skiprows=2,usecols=[0]).iloc[:,0].tolist()
        #数据
        data=pd.read_csv(csv_path, header=None, skiprows=2).iloc[:, 1:].values.tolist()
        #参数


        SRP_idx_1=random.randint(0,len(dimensions)-2)
        SRP_idx_2=random.randint(1,len(dimensions)-2)
        SRP_list=[                
                {"Q": f"In the radar chart, what is the next dimension in the clockwise direction from {dimensions[SRP_idx_1]}?",
                 "A": f"The next dimension clockwise from {dimensions[SRP_idx_1]} is {{{dimensions[SRP_idx_1+1]}}}."},
                {"Q": f"In the radar chart, what is the next dimension in the counterclockwise direction from {dimensions[SRP_idx_2]}?",
                 "A": f"The next dimension counterclockwise from {dimensions[SRP_idx_2]} is {{{dimensions[SRP_idx_1-1]}}}."},
                 ]
        if (len(labels)>1):
            SRP_idx_dim_2=random.randint(0,len(dimensions)-1)
            SRP_idx_lab_1=random.randint(0,len(labels)-1)
            SRP_idx_lab_2=random.randint(0,len(labels)-1)
            while SRP_idx_lab_1==SRP_idx_lab_2:
                SRP_idx_lab_2=random.randint(0,len(labels)-1)
            result=compare_inside_outside(data[SRP_idx_lab_1][SRP_idx_dim_2],data[SRP_idx_lab_2][SRP_idx_dim_2])
            SRP_list.append(
                {"Q": f"On the {dimensions[SRP_idx_dim_2]}, is the point for {labels[SRP_idx_lab_1]} inside or outside the point for {labels[SRP_idx_lab_2]}?",
                 "A": f"On the {dimensions[SRP_idx_dim_2]}, the point for {labels[SRP_idx_lab_1]} is {{{result}}} the point for {labels[SRP_idx_lab_2]}."
                }
            )


        VPR_idx_lab=random.randint(0,len(labels)-1)
        max_index, min_index=get_max_min_indices(data, VPR_idx_lab)
        max_lab_num,max_index_lab,=max_dims_by_lab(data)

        VPR_list=[                
                {"Q": f"Which label encloses the largest score in the radar chart?",
                 "A": f"label {{{labels[max_index_lab]}}} encloses the largest score."},
                {"Q": f"What are the strongest and weakest dimension of {labels[VPR_idx_lab]}?",
                 "A": f"The strongest aspect is {{{dimensions[max_index]}}}, while the weakest aspect is{{{dimensions[min_index]}}}."},
                 ]
        print(111)
        if len(labels)>1:
            VPR_idx_dim_1=random.randint(0,len(dimensions)-1)
            VPR_idx_dim_2=random.randint(0,len(dimensions)-1)
            VPR_idx_max=get_max_index_by_column(data, VPR_idx_dim_1)
            VPR_idx_min=get_min_index_by_column(data, VPR_idx_dim_2)
            if len(VPR_idx_max)==1:
                VPR_list.append(
                    {"Q": f"On dimension {dimensions[VPR_idx_dim_1]}, which label scores the highest?",
                    "A": f"On {dimensions[VPR_idx_dim_1]}, label {{{labels[VPR_idx_max[0]]}}} scores the highest."}
                )
            else:
                labels_str = ', '.join([f'{{{labels[idx]}}}' for idx in VPR_idx_max])
                VPR_list.append(
                    {"Q": f"On dimension {dimensions[VPR_idx_dim_1]}, which labels score the highest?",
                    "A": f"On {dimensions[VPR_idx_dim_1]}, labels {labels_str} all score the highest."}
                )
            if len(VPR_idx_min)==1:
                VPR_list.append(
                    {"Q": f"On dimension {dimensions[VPR_idx_dim_2]}, which label scores the lowest?",
                    "A": f"On {dimensions[VPR_idx_dim_2]}, label {{{labels[VPR_idx_min[0]]}}} scores the lowest."}
                )
            else:
                labels_str = ', '.join([f'{{{labels[idx]}}}' for idx in VPR_idx_min])
                VPR_list.append(
                    {"Q": f"On dimension {dimensions[VPR_idx_dim_2]}, which labels score the lowest?",
                    "A": f"On {dimensions[VPR_idx_dim_2]}, labels {labels_str} all score the lowest."}
                )
        print(222)
        VE_idx_lab_1=random.randint(0,len(labels)-1)
        VE_idx_lab_2=random.randint(0,len(labels)-1)
        VE_idx_dim_1=random.randint(0,len(dimensions)-1)
        VE_idx_dim_2=random.randint(0,len(dimensions)-1)
        while (VE_idx_lab_1==VE_idx_lab_2) and (VE_idx_dim_1==VE_idx_dim_2) :
            VE_idx_lab_2=random.randint(0,len(labels)-1)
            VE_idx_dim_2=random.randint(0,len(dimensions)-1)

                

        EVJ_idx_lab_1=random.randint(0,len(labels)-1)
        EVJ_idx_lab_2=random.randint(0,len(labels)-1)
        EVJ_idx_max=get_max_index_by_row(data, EVJ_idx_lab_1)
        EVJ_idx_min=get_min_index_by_row(data, EVJ_idx_lab_2)

        SC_idx_lab_1=random.randint(0,len(labels)-1)
        SC_idx_dim_1=random.randint(0,len(dimensions)-1)
        avg_value_1=round(sum_dims_by_lab(data, SC_idx_lab_1)/len(dimensions),2)
        avg_value_2=round(sum_labs_by_dim(data, SC_idx_dim_1)/len(labels),2)

        SC_list = [
        {"Q": f"What is the average value of {dimensions[SC_idx_dim_1]}?",
        "A": f"The average value of {dimensions[SC_idx_dim_1]} is {{{avg_value_2}}}."},
        {"Q": f"What is the average value of {labels[SC_idx_lab_1]}?",
        "A": f"The average value of {labels[SC_idx_lab_1]} is {{{avg_value_1}}}."},
        ]
        if len(labels) > 2:
            SC_num=random.randint(1,3)
            for i in range(SC_num):
                SC_idx_dim_2=random.randint(0,len(dimensions)-1)
                SC_idx_lab_2=random.randint(0,len(labels)-1)
                SC_idx_lab_3=random.randint(0,len(labels)-1)
                while SC_idx_lab_2==SC_idx_lab_3:
                    SC_idx_lab_3=random.randint(0,len(labels)-1)
                SC_list.append({
                    "Q": f"What is the difference in {dimensions[SC_idx_dim_2]} between {labels[SC_idx_lab_2]} and {labels[SC_idx_lab_3]}?",
                    "A": f"The difference in {dimensions[SC_idx_dim_2]} between {labels[SC_idx_lab_2]} and {labels[SC_idx_lab_3]} is {{{abs(data[SC_idx_lab_2][SC_idx_dim_2]-data[SC_idx_lab_3][SC_idx_dim_2])}}}."
                })

        NF_idx_lab_1=random.randint(0,len(labels)-1)
        NF_idx_lab_2=random.randint(0,len(labels)-1)
        NF_idx_lab_3=random.randint(0,len(labels)-1)
        NF_idx_dim_1=random.randint(0,len(dimensions)-1)
        NF_idx_dim_2=random.randint(0,len(dimensions)-1)
        NF_idx_dim_3=random.randint(0,len(dimensions)-1)


        max_num_by_dim,min_num_by_dim=get_max_min_by_row(data,NF_idx_lab_1)
        NF_uper_limit_lab=round((max_num_by_dim-min_num_by_dim)*0.8+min_num_by_dim)
        max_num_by_dim,min_num_by_dim=get_max_min_by_row(data,NF_idx_lab_2)
        NF_lower_limit_lab=round((max_num_by_dim-min_num_by_dim)*0.2+min_num_by_dim)
        max_num_by_dim,min_num_by_dim=get_max_min_by_col(data,NF_idx_dim_1)
        NF_uper_limit_dim=round((max_num_by_dim-min_num_by_dim)*0.8+min_num_by_dim)
        max_num_by_dim,min_num_by_dim=get_max_min_by_col(data,NF_idx_dim_2)
        NF_lower_limit_dim=round((max_num_by_dim-min_num_by_dim)*0.2+min_num_by_dim)

        result_lab_1=get_lab_over_result(data, NF_idx_lab_1, dimensions,NF_uper_limit_lab)
        result_lab_2=get_lab_under_result(data, NF_idx_lab_2, dimensions,NF_lower_limit_lab)
        result_lab_3=get_lab_between_result(data, NF_idx_lab_3, dimensions, NF_lower_limit_lab,NF_uper_limit_lab )
        result_dim_1=get_dim_over_result(data, NF_idx_dim_1, labels,NF_uper_limit_dim)
        result_dim_2=get_dim_under_result(data, NF_idx_dim_2, labels,NF_lower_limit_dim)
        result_dim_3=get_dim_between_result(data, NF_idx_dim_3, labels, NF_lower_limit_dim, NF_uper_limit_dim)

        NF_list=[
            {"Q": f"For {labels[NF_idx_lab_1]} in the radar chart, which dimensions have scores below {NF_lower_limit_lab}? Please list the dimensions and corresponding scores.",
                "A": f"{result_lab_2}."},
            {"Q": f"For {labels[NF_idx_lab_2]} in the radar chart, which dimensions have scores above {NF_uper_limit_lab}? Please list the dimensions and corresponding scores.",
                "A": f"{result_lab_1}."},
            {"Q": f"For {labels[NF_idx_lab_3]} in the radar chart, which dimensions have scores between {NF_lower_limit_lab} and {NF_uper_limit_lab}? Please list the dimensions and corresponding scores.",
                "A": f"{result_lab_3}."},
            {"Q": f"Which labels have scores above {NF_uper_limit_dim} for the {dimensions[NF_idx_dim_1]} dimension? Please list the labels and corresponding scores.",
                "A": f"{result_dim_1}."},
            {"Q": f"Which labels have scores below {NF_lower_limit_dim} for the {dimensions[NF_idx_dim_2]} dimension? Please list the labels and corresponding scores.",
                "A": f"{result_dim_2}."},
        ]
        NF_list = random.sample(NF_list, 3)
        NC_list=[]
        NC_num_1=random.randint(1,2)
        NC_num_2=random.randint(1,2)
        NC_num_3=4-NC_num_1-NC_num_2
        for i in range(NC_num_1):
            NC_idx_lab_1=random.randint(0,len(labels)-1)
            NC_idx_dim_1=random.randint(0,len(dimensions)-1)
            NC_idx_dim_2=random.randint(0,len(dimensions)-1)
            while NC_idx_dim_1==NC_idx_dim_2:
                NC_idx_dim_2=random.randint(0,len(dimensions)-1)
            NC_larger_idx,NC_less_idx=compare_two_cols_in_row(data,NC_idx_lab_1,NC_idx_dim_1,NC_idx_dim_2)
            NC_list.append({
                "Q" :f"For {labels[NC_idx_lab_1]}, which dimension has a higher score? {dimensions[NC_idx_dim_1]} or {dimensions[NC_idx_dim_2]}.",
                "A" :f"{{{dimensions[NC_larger_idx]}}} has a higher score than {dimensions[NC_less_idx]} for {labels[NC_idx_lab_1]}."
            })
        if len(labels)>1:
            for i in range(NC_num_2):
                NC_idx_lab_1=random.randint(0,len(labels)-1)
                NC_idx_lab_2=random.randint(0,len(labels)-1)
                NC_idx_dim_1=random.randint(0,len(dimensions)-1)
                while NC_idx_lab_1==NC_idx_lab_2:
                    NC_idx_lab_2=random.randint(0,len(labels)-1)
                NC_larger_idx,NC_less_idx=compare_two_rows_in_col(data,NC_idx_dim_1,NC_idx_lab_1,NC_idx_lab_2)
                NC_list.append({
                    "Q" :f"On {dimensions[NC_idx_dim_1]}, which label has a higher score? {labels[NC_idx_lab_1]} or {labels[NC_idx_lab_2]}.",
                    "A" :f"{{{labels[NC_larger_idx]}}} has a higher score on {dimensions[NC_idx_dim_1]} compared to {labels[NC_less_idx]}."
                })
        for i in range(NC_num_3):
            NC_idx_lab_1=random.randint(0,len(labels)-1)
            NC_idx_dim_1=random.randint(0,len(dimensions)-1)
            NC_idx_dim_2=random.randint(0,len(dimensions)-1)
            NC_idx_dim_3=random.randint(0,len(dimensions)-1)
            while NC_idx_dim_1==NC_idx_dim_2:
                NC_idx_dim_2=random.randint(0,len(dimensions)-1)
            while NC_idx_dim_1==NC_idx_dim_3 or NC_idx_dim_2==NC_idx_dim_3:
                NC_idx_dim_3=random.randint(0,len(dimensions)-1)
            NC_larger_idx=max_of_three_cols_in_row(data,NC_idx_lab_1,NC_idx_dim_1,NC_idx_dim_2,NC_idx_dim_3)
            NC_list.append({
                "Q" :f"For {labels[NC_idx_lab_1]}, which dimension has a higher score? {dimensions[NC_idx_dim_1]} or {dimensions[NC_idx_dim_2]} or {dimensions[NC_idx_dim_3]}.",
                "A" :f"{{{dimensions[NC_larger_idx]}}} has a higher score for {labels[NC_idx_lab_1]}."
            })
        MSR_idx_lab=random.randint(0,len(labels)-1)
        MSR_idx_min_dim=get_min_index_by_row(data,MSR_idx_lab)
        MSR_list=[
            {"Q": f"For {labels[MSR_idx_lab]}, based on the data shown in the radar chart, which dimension does it most need to improve?",
             "A": f"{labels[MSR_idx_lab]} most needs to improve its {{{dimensions[MSR_idx_min_dim]}}}."
            }
        ]
        if(len(labels)>1):
            MSR_idx_max_lab=get_max_sum_row_index(data)
            MSR_idx_min_lab=get_min_sum_row_index(data)
            MSR_list.append({
                "Q": f"In the radar chart, which label has the strongest overall capability across all labels?",
                "A": f"The label with the strongest overall capability is {{{labels[MSR_idx_max_lab]}}}."
                })
            MSR_list.append({
                "Q": f"In the radar chart, which label has the weaknest overall capability across all labels?",
                "A": f"The label with the weaknest overall capability is {{{labels[MSR_idx_min_lab]}}}."
            })

        qa_data = {
            "CTR": [
                {"Q": "What type of chart is this?", 
                 "A": "This chart is a {radar} chart."}
            ],
            "VEC": [
                {"Q": "How many labels are in this radar chart?",
                 "A": f"There are {{{len(labels)}}} labels."},
                {"Q": "How many dimensions are in this radar chart?",
                 "A": f"There are {{{len(dimensions)}}} dimensions."},
            ],
            "SRP": SRP_list,
            "VPR": VPR_list,
            "VE": [
                {"Q": f"What is the value of {labels[VE_idx_lab_1]} on the {dimensions[VE_idx_dim_1]} in the radar chart?",
                 "A": f"The value of {labels[VE_idx_lab_1]} on the Safety dimension is {{{data[VE_idx_lab_1][VE_idx_dim_1]}}}."},
                {"Q": f"What is the value of {labels[VE_idx_lab_2]} on the {dimensions[VE_idx_dim_2]} in the radar chart?",
                 "A": f"The value of {labels[VE_idx_lab_2]} on the Safety dimension is {{{data[VE_idx_lab_2][VE_idx_dim_2]}}}."},
            ],
            "EVJ": [
                {"Q": f"What is the maximum value of label {labels[EVJ_idx_lab_1]} across all dimensions in the radar chart?",
                 "A": f"The maximum value of {labels[EVJ_idx_lab_1]} across all dimensions is {{{data[VE_idx_lab_1][EVJ_idx_max]}}}."},
                {"Q": f"What is the minimum value of label {labels[EVJ_idx_lab_2]} across all dimensions in the radar chart?",
                 "A": f"The minimum value of {labels[EVJ_idx_lab_2]} across all dimensions is {{{data[VE_idx_lab_2][EVJ_idx_min]}}}."},
            ],
            "SC": SC_list,
            "NF": NF_list,
            "NC": NC_list,
            "MSR": MSR_list,
            "VA": []
        }

        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)

        print(f"[SUCCESS] 生成QA JSON: {json_path}")

    except Exception as e:
        print(f"[EXCEPTION] 处理文件 {filename} 时发生错误: {e}")

    