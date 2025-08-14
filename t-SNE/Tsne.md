# 总数据集
- MegaCQA 全部
- ChartQA test数据集
- ChartX 全部

# 采样数据集
- MegaCQV:每一个类别图表采样50,总数50 * 22 = 1100
- ChartQA: 1000个
- ChartX: 每一个类别图表采样56, 总数56 *18 = 1008
- ChartBench:每一个图表类型采样115, 总数112 * 9 = 1008
- NovaChart:每一个图表类型采样53, 总数53*19=1007

# CSV

# 采用模型与重要超参数
## 词嵌入模型
主模型:TAPAS
细分模型名称:google/tapas-base
## Tsne超参数
perplexity= 1.9

# PNG
## 词嵌入模型
主模型:CLIP
细分模型名称:ViT-B/32
## Tsne超参数
perplexity=4

# QA
## 词嵌入模型
主模型:SentenceTransformer
细分模型名称:thenlper/gte-base
## Tsne超参数
perplexity=3.5

# MegaCQA 数据集预处理
## 处理表格数据
1. bar
  跳过[0,1]行, 删除第[0]列
2. box
  跳过第[0,1]行
1. bubble
  跳过第[0,1]行
2. chord
  跳过第[0,1]行, 删除第[0]列
3. fill_bubble
  跳过第[0,1]行,删除第[3]列
4. funnel
  跳过第[0,1]行
5. heatmap
  跳过第[0]行, 删除第[0,1]列
6. line
  跳过第[0,1,2]行
7. node_link
  跳过第[0,1]行
8.  parallel
  跳过[0,1,2]行,删除第[0]列
9.  pie
  跳过[0,1]行,删除第[0]列
10. radar
    跳过[0,1]行
11. ridgeline
    跳过[0,1,2]行
12. sankey
    跳过[0,1]行,删除[0,1]列
13. scatter
  跳过[0,1]行
14. stacked_area
  跳过[0,1,2]行
15. stacked_bar
  跳过[0,1]行,删除[0]列
16. stream
  跳过[0,1,2]行
17. sunburst
  跳过[0,1]行,删除第[0,1]列
18. treemap
  跳过[0,1,2]行,删除[0,1]列
19. treemap_D3
  跳过[0,1,2]行,删除[0,1]列
20. violin
  跳过[0,1]行
# ChartQA 数据集预处理
## 原始QA数据
由于原始QA的信息包括了`train_augmented.json`和`train_human.json`两个文件,首先将其合并. 在网址`https://wejson.cn/join/`进行合并为`json_merge.json`.
## 采样
通过随机采样ChartQA/train数据集中的所有数据，保持原有文件结构
- ChartQA_sample
  - png
  - tables
  - json_merge.json

# ChartBench 数据集预处理
ChartBench数据集中拥有9个大图表类型, 每一个大图表类型中还有更细分的小图表类型
## 1. 重构文件目录结构
由于初始的文件目录结构为
- ChartBench
  - chart_type_1(e.g. area)
    - further_chart_type_1(e.g. area_stack)
      - chart_{i}, i 从0开始递增
        - image.png , 图片
        - meta.json , 元数据信息，可以忽略
        - QA.json , QA对,存储问答信息
        - table.json , 表格数据,待解析为csv数据
      - ...
    - further_chart_type_2
      - ...
    - ...
  - char_type_2
  - ...

希望将其重构为:
- ChartBench
  - chart_type_1
    - PNG
      - i.png , 将chart_type_1文件夹中的所有图片进行重命名,从1开始
      - ...
    - QA
      - QA_i.json
      - ...
    - TABLE
      - table_i.json
      - ...
  - chart_type_2
  - ...

文字描述:
将同一个chart_type的目录下的所有数据, 统一为从数字命名(从1开始), 简化文件数据结构, 同一数据的不同文件(png,json)的数字相同, 方便后续处理;

## table内容解析
通过读取json文件,将其转换成csv文件
当前每一个图表类型中的QA文件夹下为json文件,不适合进行词嵌入,将其中数据进行提取然后转换成csv文件(跳过图表类型为node_link和pie和radar)
原始json问价需要提取的键值对:
1. "x_data":该键对应的是一个列表,包含着图表x轴的元素,放置在csv文件的第一行
2. "y_data":该键对应的一个列表(可能为一维列表也可能是嵌套列表),每一个底层列表代表着一行csv的数据,如果是嵌套列表则代表着多行数据
# NavaChart
19个图表类型, 但是没有csv原始数据

## QA解析

