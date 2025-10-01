'''
对于核心答案是数值类型的，评估指标可选：
1. Relaxed Accuracy（放宽准确率）：设定一个容忍范围（例如±5%），如果预测值与实际值的差距在这个范围内，那么这个预测就被认为是“正确”。
   计算公式：Relaxed Accuracy = 预测值与真实值的差异在容忍范围内的次数 / 总样本数
2. Mean Squared Error (MSE, 均方误差): 一种常用的回归模型评估指标，用来度量模型预测值与真实值之间的差异。它计算的是预测值与真实值之间差异的平方的平均值。
   计算公式：MSE = 1/N * 所有样本(预测值-基准值)^2之和
'''

import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
# 下载停用词列表
nltk.download('stopwords')
# 获取英语停用词
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.remove('above')
STOPWORDS.remove('below')
STOPWORDS.remove('same')
STOPWORDS.remove('not')
STOPWORDS.remove('no')

from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer
from CLIPAttnWeb.CQA.qa_handler import QAHandler

class AnswerEvaluator:
    def __init__(self, data):
        self.data = data
    # ***************************************** 一、答案准确度指标
    # 1.1 被调函数：评估全局句子的相似度。可选 Sentence_BERT（默认），ROUGE，BLEU。 最好前两个一起使用，避免使用第二个
    def metric_global_nlp(self, baseline_answer, model_answer, metric='Sentence_BERT'):
        """
        评估两个答案句子（自然语言）句子之间的相似度
        :param baseline_answer: 基准答案
        :param model_answer: 模型答案
        :param metric: 评估指标，可选 Sentence_BERT（默认），ROUGE，BLEU
        :return:
        """
        if metric == 'Sentence_BERT':
            # 加载 Sentence-BERT 模型  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            # TODO: change this to actual local dir
            model = SentenceTransformer(r'D:\CQAVis\python_modules\all-MiniLM-L6-v2')
            # model = SentenceTransformer('/home/litong/DjangoWeb/CQAVis/sentence-transformers/all-MiniLM-L6-v2')
            # 获取句子嵌入向量
            baseline_embedding = model.encode(baseline_answer)
            model_embedding = model.encode(model_answer)
            # 计算余弦相似度 [0,1]
            cos_sim = cosine_similarity([baseline_embedding], [model_embedding])
            return cos_sim[0][0]

        elif metric == 'ROUGE': # 不考虑整体语义，只看表面匹配
            # 使用 ROUGE 计算指标
            # OUGE-1 衡量 unigrams（单字） 的重叠。ROUGE-L 计算 最长公共子序列（LCS），即保持词汇顺序的最大匹配。
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(baseline_answer, model_answer)
            # 提取并打印精确度、召回率 [0,1] 和 F1 分数
            precision_rouge1 = scores['rouge1'].precision
            recall_rouge1 = scores['rouge1'].recall
            fmeasure_rouge1 = scores['rouge1'].fmeasure
            precision_rougeL = scores['rougeL'].precision
            recall_rougeL = scores['rougeL'].recall
            fmeasure_rougeL = scores['rougeL'].fmeasure
            return [precision_rouge1, recall_rouge1, fmeasure_rouge1, precision_rougeL, recall_rougeL, fmeasure_rougeL]

        elif metric == 'BLEU': # 机器翻译早期阶段，简单，对词序不敏感，偏向短句。
            # 首先将文本转化为单词列表（分词）
            baseline_tokens = baseline_answer.lower().split()
            model_tokens = model_answer.lower().split()
            # 计算 BLEU 分数  [0,1]
            bleu_score = sentence_bleu([baseline_tokens], model_tokens)
            return bleu_score

    # 1.2 被调函数：评估关键字符串的准确度。可选 Jaccard（默认）和 Recall
    def metric_key_string(self, b_answer_key, m_answer_key, metric='Jaccard'):
        """
        评估两个答案的核心string答案列表之间的相似度（无关元素顺序）
        :param b_answer_key: 基准答案中的核心答案单词列表
        :param m_answer_key: 模型答案中的核心答案单词列表
        :param metric: 评估指标，可选 Jaccard（默认），Recall
        :return:
        """
        # def filter_stopwords(words):
        #     """过滤停用词"""
        #     return [word for word in words if word.lower() not in STOPWORDS]

        # 去停用词 + 小写
        set_b = set(word.lower() for word in b_answer_key if word.lower() not in STOPWORDS)
        set_m = set(word.lower() for word in m_answer_key if word.lower() not in STOPWORDS)
        # print(f"setb: {set_b}")
        # print(f"setm: {set_m}")

        # 完全匹配判断（只看两个集合是否一致）
        if set_b == set_m:
            return 1, 1.0

        # 如果模型答案包含所有基准关键词（不漏词），就算匹配
        if set_b.issubset(set_m):
            intersection = set_b & set_m
            #print(f"intersection: {intersection}")
            union = set_b | set_m
            #print(f"union: {union}")
            score = len(intersection) / len(union) if union else 0.0
            return 1, score

        # 否则不匹配（漏词）
        intersection = set_b & set_m
        union = set_b | set_m
        score = len(intersection) / len(union) if union else 0.0
        return 0, score

        # elif metric == 'F1':
        #     precision = len_inter / len(set_m)
        #     recall = len_inter / len(set_b)
        #     if precision + recall == 0:
        #         return 0.0
        #     return 2 * (precision * recall) / (precision + recall)

    def metric_key_string_sequence(self, b_answer_key, m_answer_key, metric='Jaccard'):
        """
        评估两个答案的核心string答案列表之间的相似度（有关元素顺序）
        :param b_answer_key: 基准答案中的核心答案单词列表
        :param m_answer_key: 模型答案中的核心答案单词列表
        :param metric: 评估指标，可选 Jaccard（默认），Recall
        :return:
        """
        # 去停用词 + 小写
        list_b = [word.lower() for word in b_answer_key if word.lower() not in STOPWORDS]
        list_m = [word.lower() for word in m_answer_key if word.lower() not in STOPWORDS]

        set_b = set(word.lower() for word in b_answer_key if word.lower() not in STOPWORDS)
        set_m = set(word.lower() for word in m_answer_key if word.lower() not in STOPWORDS)
        # print(f"setb: {set_b}")
        # print(f"setm: {set_m}")
        
        intersection = set_b & set_m
        union = set_b | set_m
        jaccard_score = len(intersection) / len(union) if union else 0.0

        # if labels are not matched
        if not set_b.issubset(set_m):
            return 0, jaccard_score
        
        # extract intersection list
        list_m_filtered = [w for w in list_m if w in list_b]

        # labels are matched, test sequence
        # here jaccard_score is always 1
        m_indices = []
        for word in list_b:
            m_indices.append(list_m_filtered.index(word))

        inv_count = 0
        n = len(m_indices)
        for i in range(n):
            for j in range(i + 1, n):
                if m_indices[i] > m_indices[j]:
                    inv_count += 1

        max_inv = n * (n - 1) / 2
        norm_inv = inv_count / max_inv if max_inv > 0 else 0.0
        #print(f"inv_count: {inv_count}")
        #print(f"normalized_inv: {norm_inv}")
        return 1 if norm_inv == 0 else 0, 1 - norm_inv

    def metric_key_string_with_relation(self, b_answer_key, m_answer_key, metric='Jaccard'):
        """
        评估两个答案的核心string答案列表之间的相似度（无关元素顺序）
        :param b_answer_key: 基准答案中的核心答案单词列表
        :param m_answer_key: 模型答案中的核心答案单词列表
        :param metric: 评估指标，可选 Jaccard（默认），Recall
        :return:
        """
        # def filter_stopwords(words):
        #     """过滤停用词"""
        #     return [word for word in words if word.lower() not in STOPWORDS]

        # 去停用词 + 小写
        set_b = set(word.lower() for word in b_answer_key if word.lower() not in STOPWORDS)
        set_m = set(word.lower() for word in m_answer_key if word.lower() not in STOPWORDS)
        # print(f"setb: {set_b}")
        # print(f"setm: {set_m}")
        
        trigger_words = {"trend", "pattern", "distribution", "correlation", "diagonal", "strong"}
        related_words = {"trend", "pattern", "distribution", "correlation", "diagonally", "positive"}

        if set_b & trigger_words:
            count = 0
            for w in m_answer_key:
                wl = w.lower()
                if wl not in set_b and wl in related_words:
                    count += 1
            if count > 0:
                return 1, 1.0
            else:
                return 0, 0.0
        
        # 完全匹配判断（只看两个集合是否一致）
        if set_b == set_m:
            return 1, 1.0

        # 如果模型答案包含所有基准关键词（不漏词），就算匹配
        if set_b.issubset(set_m):
            intersection = set_b & set_m
            #print(f"intersection: {intersection}")
            union = set_b | set_m
            #print(f"union: {union}")
            score = len(intersection) / len(union) if union else 0.0
            return 1, score

        # 否则不匹配（漏词）
        intersection = set_b & set_m
        union = set_b | set_m
        score = len(intersection) / len(union) if union else 0.0
        return 0, score

    # 1.3 被调函数：评估关键数值的准确度。可选 Relaxed Accuracy（默认），均方误差 MSE
    def metric_key_num(self, b_answer, m_answer, metric='Relax_Acc'):
        """
        评估两个答案的核心number答案列表之间的相似度
        :param b_answer: 基准答案中的数值
        :param m_answer: 模型答案中的数值
        :param metric: 评估指标，可选 Relaxed Accuracy（默认），均方误差 MSE
        :return:
        """
        if metric == 'Relax_Acc': # [0,1]
            # 容忍误差范围：上下 5% 的误差
            tolerance = 0.05 * b_answer  # 容忍误差为基准答案的 5%
            epsilon = 1e-8
            relaxed_accuracy = int(abs(b_answer - m_answer) <= tolerance) # 0或1
            relaxed_error = abs(b_answer - m_answer) / ((abs(b_answer) + epsilon)) #计算一下相对误差值
            return relaxed_accuracy, relaxed_error

        elif metric == 'MSE': # [0, +]
            # 计算 MSE：比较总和的均方误差
            mse = mean_squared_error([b_answer], [m_answer])
            return mse
        
    def metric_key_num_VEC(self, b_answer, m_answer, metric='Relax_Acc'):
        """
        评估两个答案的核心 number 答案列表之间的相似度（VEC 版本，使用固定 ±2 容忍区间）
        :param b_answer: 基准答案中的数值
        :param m_answer: 模型答案中的数值
        :param metric: 评估指标，可选 Relaxed Accuracy（默认），均方误差 MSE
        :return: 
            - 若 metric='Relax_Acc': (relaxed_accuracy: 0 或 1, relative_error: float)
            - 若 metric='MSE': mse: float
        """
        if metric == 'Relax_Acc':  # [0,1]
            # 容忍误差范围：上下固定 ±2
            tolerance = 2
            epsilon = 1e-8
            # 判断是否在容忍范围内
            relaxed_accuracy = int(abs(b_answer - m_answer) <= tolerance)  # 0 或 1
            # 计算相对误差（用于分析）
            relaxed_error = abs(b_answer - m_answer) / (abs(b_answer) + epsilon)
            return relaxed_accuracy, relaxed_error

        elif metric == 'MSE':  # [0, +∞]
            # 计算 MSE：比较单个数值的均方误差
            mse = (b_answer - m_answer) ** 2
            return mse

        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    # ***************************************** 二、推理指标
    # 2.1 被调函数：评估推理时间
    def metric_reasoning_time(self, time_cost):
        return time_cost

    # 2.2 被调函数：评估推理长度（以token为单位，一百个）
    def metric_reasoning_token(self, reasoning):
        '''
        :param reasoning: 推理过程 nlp段落
        :return: 推理总token数，以百为单位
        '''
        # TODO: change this to actual local dir
        # 分词器
        tokenizer = BertTokenizer.from_pretrained(r"D:\CQAVis\python_modules\bert-base-uncased")
        # Tokenize 文本
        tokens = tokenizer.tokenize(reasoning)
        return len(tokens)

    def is_year(self, text):
        # 判断文本是否为有效的年份（1900-2025）
        match = re.match(r'^(19\d{2}|20(0[0-9]|1[0-9]|20[0-5]))$', text)
        return bool(match)
    
    # ***********主函数
    def evaluate_answer(self):
        """
        抽取基准和模型的关键答案，写入样本
        Returns:
            List[json]: 新增属性项的data
        """
        # for item in self.data:
        # 使用 tqdm 包裹循环，显示进度条
        for item in tqdm(self.data, desc="Evaluating Answers", total=len(self.data)):
            # **************************** 抽取reasoning
            reasoning = item.get("reasoning", "")
            time_cost = item.get("time_cost", 0)
            # 1. 时间指标(以秒为单位)
            metric_time = self.metric_reasoning_time(time_cost)
            # 2. 推理token数指标
            metric_tokens = self.metric_reasoning_token(reasoning)
            item['metric_tokens'] = metric_tokens

            # ***************************  抽取答案
            chart_type = item.get("chart_type", "")
            qa_type = item.get("qa_type", "default")
            baseline_question = item.get("baseline_question", "")
            baseline_answer = item.get("baseline_answer", "")
            model_answer = item.get("model_answer", "")
            id = item.get("id","")
            #print(f'id: {id}, model_answer: {model_answer}')
            # 1. 全局nlp评估 （样本中已经有了 nlp_metric ）
            # if baseline_answer.strip() and model_answer.strip():
            #     nlp_eva = self.metric_global_nlp(baseline_answer=baseline_answer, model_answer=model_answer)
            # else:
            #     nlp_eva = None

            # 2. 局部关键词评估  正确为1，错误为0
            handler = QAHandler()
            if qa_type == 'CTR': # 函数返回baseline 和 model answer中查找到的phrase列表/字符串，或者None
                baseline_key, model_key = handler.handle_CTR(baseline_answer, model_answer, chart_type)
                #print(f'baseline key:{baseline_key}, model_key:{model_key}' )
                if baseline_key is not None and model_key is not None:
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = 1 # 正确1
                    item['metric_error'] = 0.0 # 误差 0.0
                else:
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0

            elif qa_type == 'VEC': # 函数返回在baseline 和 model answer中的数值
                baseline_key, model_key = handler.handle_VEC(baseline_answer, model_answer, baseline_question)
                #print(f'baseline key:{baseline_key}, model_key:{model_key}' )
                if baseline_key is not None and model_key is not None:
                    # relaxed_accuracy 0或1   relaxed_error为具体误差
                    relaxed_accuracy, relaxed_error = self.metric_key_num_VEC(baseline_key, model_key)
                    # print(f"metric_label: {relaxed_accuracy}")
                    # print(f"metric_error: {relaxed_error}")

                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = relaxed_accuracy  # 正确1 or 错误0
                    item['metric_error'] = relaxed_error  # 具体误差
                else:
                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0

            elif qa_type == 'SRP': # 函数返回在baseline 和 model answer中的单词列表
                baseline_key, model_key = handler.handle_SRP(baseline_answer, model_answer)
                #print(f'baseline key:{baseline_key}, model_key:{model_key}' )
                if baseline_key is not None and model_key is not None:
                    # phrase_acc 0或1   phrase_value 集合重叠度Jaccard 度量值 0-1(完全相同)
                    phrase_acc, phrase_value = self.metric_key_string(baseline_key, model_key)
                    #print(f'phrase_acc: {phrase_acc}, phrase_value: {phrase_value}')
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = phrase_acc  # 正确1 or 错误0
                    item['metric_error'] = 1-phrase_value  # 误差 1-Jaccard

                else:
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0

            elif qa_type == 'SC' : # 该类型只有数值类型的问答
                baseline_key, model_key = handler.handle_SC(baseline_answer, model_answer, chart_type) # 返回两个数值，或者两个None
                #print(f'baseline key:{baseline_key}, model_key:{model_key}' )
                if baseline_key is not None and model_key is not None:
                    # relaxed_accuracy 0或1   relaxed_error为具体误差
                    relaxed_accuracy, relaxed_error = self.metric_key_num(baseline_key, model_key)
                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = relaxed_accuracy  # 正确1 or 错误0
                    item['metric_error'] = relaxed_error  # 具体误差

                else:
                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0

            # elif qa_type == 'NC':  # 该类型只有单词类型的问答
            #     baseline_key, model_key = handler.handle_NC(baseline_answer, model_answer)  # 返回两个单词列表，或者两个None
            #     print(f'baseline key:{baseline_key}, model_key:{model_key}' )
            #     if baseline_key is not None and model_key is not None:
            #         # phrase_acc 0或1   phrase_value 集合重叠度Jaccard 度量值 0-1(完全相同)
            #         phrase_acc, phrase_value = self.metric_key_string(baseline_key, model_key)
            #         item['metric_name'] = 'phrase_acc'
            #         item['metric_label'] = phrase_acc  # 正确1 or 错误0
            #         item['metric_error'] = 1 - phrase_value  # 误差 1-Jaccard
            #     else:
            #         item['metric_name'] = 'phrase_acc'
            #         item['metric_label'] = 0  # 错误0
            #         item['metric_error'] = 1.0  # 误差 1.0
            
            elif qa_type == 'NF':

                baseline_key_words, model_key_words, baseline_key_num, model_key_num \
                    = handler.handle_NF(baseline_answer, model_answer, baseline_question)
                
                # print(f'baseline_key_words: {baseline_key_words}')
                # print(f'model_key_words: {model_key_words}')
                # print(f'baseline_key_num: {baseline_key_num}')
                # print(f'model_key_num: {model_key_num}')
                
                b_has_num = baseline_key_num is not None and model_key_num is not None
                b_has_words = baseline_key_words is not None and model_key_words is not None

                if chart_type == 'pie' and model_key_words is not None:
                    model_key_words = list(filter(lambda x: x != 'category', model_key_words))

                if b_has_num and b_has_words:
                    relaxed_accuracy, relaxed_error = self.metric_key_num(baseline_key_num, model_key_num)
                    phrase_acc, phrase_value = self.metric_key_string(baseline_key_words, model_key_words)

                    item['metric_name'] = 'mix_acc'
                    # only exactly match we assigned as positive
                    item['metric_label'] = relaxed_accuracy and phrase_acc  # 正确1 or 错误0
                    item['metric_error'] = relaxed_error + 1 - phrase_value  # 具体误差
                
                elif b_has_num:
                    relaxed_accuracy, relaxed_error = self.metric_key_num(baseline_key_num, model_key_num)

                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = relaxed_accuracy  # 正确1 or 错误0
                    item['metric_error'] = relaxed_error  # 具体误差
                
                elif b_has_words:
                    # phrase_acc 0或1   phrase_value 集合重叠度Jaccard 度量值 0-1(完全相同)
                    phrase_acc, phrase_value = self.metric_key_string(baseline_key_words, model_key_words)
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = phrase_acc  # 正确1 or 错误0
                    item['metric_error'] = 1 - phrase_value  # 误差 1-Jaccard

                else:
                    item['metric_name'] = 'mix_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0


            elif qa_type in ['VPR', 'NC', 'VE', 'EVJ', 'MSR', 'VA']:  # 该类既有数值类型的问答，又有单词型问答
                baseline_key, model_key = handler.handle_multi(baseline_answer, model_answer, chart_type)  # 返回两个数值，或者两个单词列表

                #print(f"baseline_key: {baseline_key}, model_key: {model_key}")
                # FIXED: Always examine model_key, for model may not brace key answer with '{}'
                # 数值型 VPR，进行数值型误差判断
                if isinstance(baseline_key, float) and model_key is not None:
                    # relaxed_accuracy 0或1   relaxed_error为具体误差
                        
                    relaxed_accuracy, relaxed_error = self.metric_key_num(baseline_key, model_key)
                    if self.is_year(str(baseline_key).rstrip('0').rstrip('.')):
                        relaxed_accuracy = int(baseline_key == model_key)
                    item['metric_name'] = 'relaxed_acc'
                    item['metric_label'] = relaxed_accuracy  # 正确1 or 错误0
                    item['metric_error'] = relaxed_error  # 具体误差

                elif isinstance(baseline_key, list) and model_key is not None:
                    # phrase_acc 0或1   phrase_value 集合重叠度Jaccard 度量值 0-1(完全相同)
                    if qa_type == 'VA' and chart_type in ['area', 'bar', 'chord']:
                        if chart_type == 'chord':
                            baseline_key = list(filter(lambda x: x != 'Arc', baseline_key))
                        phrase_acc, phrase_value = self.metric_key_string_sequence(baseline_key, model_key)
                    
                    elif qa_type == 'VPR':
                        phrase_acc, phrase_value = self.metric_key_string_with_relation(baseline_key, model_key)
                    
                    else:
                        phrase_acc, phrase_value = self.metric_key_string(baseline_key, model_key)
                    #print(f'baseline key:{baseline_key}, model_key:{model_key}' )

                    #print(f"metric_label: {phrase_acc}")
                    #print(f"metric_error: {1-phrase_value}")
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = phrase_acc  # 正确1 or 错误0
                    item['metric_error'] = 1 - phrase_value  # 误差 1-Jaccard
                else:
                    item['metric_name'] = 'phrase_acc'
                    item['metric_label'] = 0  # 错误0
                    item['metric_error'] = 1.0  # 误差 1.0
            #print(f"label: {item['metric_label']}, error: {item['metric_error']}")
            #print('------------------------------------------------')

        # TODO: write back to json
        # write to a temp file now just for sanity check
        output_file = r'D:\CQAVis\mllm_evaluation_data\fusion_metric\claude35HaikuLatest_png_svg_0.2k_processed_metric.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print('输出data len', len(data))


# ==============================
# ✅ 主程序入口（测试用）
# ==============================
if __name__ == "__main__":
    # TODO: load json data from file
    # 读取json数据
    json_file= r"D:\CQAVis\mllm_evaluation_data\fusion_eval\claude35HaikuLatest_png_svg_0.2k_processed.json"

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print('输入data len', len(data))
    # 统计model_answer为空的样本数
    empty_model_answer_count = sum(1 for item in data if item.get('model_answer') == "")
    print(f"Number of samples with empty model_answer: {empty_model_answer_count}")

    # 创建实例并执行评估
    evaluator = AnswerEvaluator(data)
    evaluation_results = evaluator.evaluate_answer()