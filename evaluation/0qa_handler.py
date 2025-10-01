# qa_handlers.py
import re
import json
import random
from word2number import w2n
import math

class QAHandler:
    def __init__(self):
        pass

    # 通用函数，提取各类数值情况（如果是科学计数法的，也会自动转换）
    def extract_all_numbers(self, text):
        """提取所有形式的数字（支持千分位、小数、百分比、负数、科学计数法）"""
        matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?', text)
        values = []
        for match in matches:
            cleaned = match.replace(',', '')
            is_percent = '%' in cleaned
            if is_percent:
                cleaned = cleaned.replace('%', '')
            try:
                value = float(cleaned)
                values.append(value)
            except ValueError:
                continue
        return values

    # 通用函数：判断数值是否是年份
    def is_year(self, text):
        # 判断文本是否为有效的年份（1900-2025）
        match = re.match(r'^(19\d{2}|20(0[0-9]|1[0-9]|20[0-5]))$', text)
        return bool(match)

    def is_number(self, s):
        # judge if the given value is number
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    # 通用函数：判断数值后是否有单位
    def extract_and_convert_value(self, number, text):
        """
        从文本中提取数值并处理单位（如千分位、负数、百分比、英文数字词等）。
        处理单位包括 'k', 'million', 'billion', 'ten thousand' 等。

        Args:
            number (float): 提取的数值
            text (str): 对应文本，用于检查单位

        Returns:
            float: 返回转换后的数值（如果有单位的话）
        """
        # 单位倍数字典（例如 'k' -> 1000, 'million' -> 1000000 等）
        unit_multipliers = {
            'k': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000,
            'thousand': 1000
        }

        if not math.isfinite(number):
            return None
        
        # 将输入的 number 转为归一化字符串（去掉末尾小数 0）
        if number == int(number):
            target_str = str(int(number))
        else:
            target_str = f"{number:.10f}".rstrip('0').rstrip('.')

        # 提取文本中的所有数字
        number_pattern = re.compile(r'(-?\d+(?:[.,]\d+)*)%?')
        for match in number_pattern.finditer(text):
            num_str_in_text = match.group(1)

            # 去除逗号、百分号、小数末尾 0，得到归一化字符串
            normalized = num_str_in_text.replace(',', '').rstrip('%')
            if '.' in normalized:
                normalized = normalized.rstrip('0').rstrip('.') if '.' in normalized else normalized

            # 如果匹配成功，检查后面是否有单位
            if normalized == target_str:
                after_match = text[match.end():].strip()
                # # 跳过可能的 "}"
                # if after_match.startswith("}"):
                #     after_match = after_match[1:].strip()

                if after_match:
                    after_match = after_match.replace(",", "").replace("}", "").strip()
                    after_match = after_match.rstrip('.').split()
                    if after_match:
                        first_word = after_match[0].lower()
                        if first_word in unit_multipliers:
                            return number * unit_multipliers[first_word]

        # 没有找到单位或匹配失败，返回原值
        return number


    def handle_CTR(self, baseline_answer, model_answer, chart_type):
        """
        处理图表类型识别类问答的评估逻辑。

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案
            chart_type (str): 图表类型

        Returns:
            str or None: 如果 model answer 中匹配到关键词，返回原始词形；否则返回 None
        """
        CHART_SYNONYMS = {
            "area": ["filled line", "filled-line"],
            "bar": ["column", "histogram"],
            "box": ["boxplot", "whisker"],
            "bubble": ["scatter", "dot"],
            "chord": ["flow", "arc"],
            # "fill-bubble": ["time series", "timeseries"],
            # "funnel": ["time series", "timeseries"],
            "heatmap": ["heat map", "heat"],
            "line": ["time series", "timeseries"],
            "node-link": ["nodelink", "network", "graph"],
            "parallel": ["parallel-coordinates", "coordinates"],
            # "pie": ["pie", "pie chart", "circle chart"],
            "radar": ["spider"],
            "ridgeline": ["ridge", 'mountain'],
            "sankey": ["flow"],
            "scatter": ["dot"],
            # "stacked-bar": ["stacked bar", "stacked bar chart", "stacked column chart"],
            "stream": ["streamgraph", "river", "rivergraph", "flow"],
            "sunburst": ["donut", "hierarchical pie"],
            "treemap": ["tree map", "hierarchical", "hierarchical rectangles", "nested", "nested rectangles"]
            #"violin": ["beeswarm"]
        }

        # Step 1: 提取 baseline 所有 {xxx} 中的内容
        placeholders = re.findall(r'\{(.*?)\}', baseline_answer)
        if not placeholders:
            return None, None  # 如果没有占位符，返回 None

        # Step 2: 对所有 baseline {} 生成所有关键词列表
        keywords = []
        for raw_chart_type in placeholders:
            cleaned_chart_type = raw_chart_type.strip()

            # Step 3: 去掉 "chart" 和 "plot"（不区分大小写）
            cleaned_chart_type = re.sub(
                r'\b(chart|plot)\b', '', cleaned_chart_type, flags=re.IGNORECASE
            ).strip()

            # Step 4: 替换下划线和连字符为空格，便于统一拆分
            cleaned_chart_type = re.sub(r'[_-]', ' ', cleaned_chart_type)

            # Step 5: 拆分关键词为列表（如 "node link"）
            keywords.extend(cleaned_chart_type.split())

        # Step 3: 在 model_answer 中模糊匹配（不区分大小写），查找是否有 baseline 关键词
        matched_keywords = []
        for keyword in keywords:
            direct_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            if direct_pattern.search(model_answer):
                matched_keywords.append(keyword)
        # 如果找到了匹配的关键词，返回
        if matched_keywords:
            return keywords, matched_keywords

        # Step 4: 如果没有直接找到，查找同义词（模型回答可能是同义词）
        # synonym_list = CHART_SYNONYMS.get(chart_type, [])
        # for synonym in synonym_list:
        #     synonym_pattern = re.compile(r'\b' + re.escape(synonym) + r'\b', re.IGNORECASE)
        #     if synonym_pattern.search(model_answer):
        #         return keywords, synonym  # 返回原始词形及所有关键词
        matched_synonyms = []
        synonym_list = CHART_SYNONYMS.get(chart_type, [])
        for synonym in synonym_list:
            synonym_pattern = re.compile(r'\b' + re.escape(synonym) + r'\b', re.IGNORECASE)
            if synonym_pattern.search(model_answer):
                matched_synonyms.append(synonym)  # 收集每一个匹配的 synonym
        if matched_synonyms:
            return keywords, matched_synonyms  # 返回原始关键词和所有匹配的 synonym 列表
        else:
            return keywords, None  # 或者返回空列表：matched_synonyms


    def handle_VEC(self, baseline_answer, model_answer, baseline_question):
        """
        处理视觉元素计数（VEC）类问答的评估逻辑。
        - Step 1: 提取 {数字}，直接求和返回。
        - Step 2: 如果提供了 question，只提取问题中“many 后面的对象”前的数字。
        -        否则，提取所有能转为数值的内容，并求和。

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案
            question (str or None): 可选的问题文本，用于上下文提取

        Returns:
            baseline_value, model_value (float or None)
        """

        def extract_target_word(q):
            """从问题中提取 'many' 后的目标词"""
            if not q:
                return None
            match = re.search(r'how many (\w+)', q, re.IGNORECASE)
            return match.group(1) if match else None

        def extract_number_from_placeholders(text):
            """提取 {} 中的所有数字（包括数字词如 one, two），支持千分位、负数、科学计数法、百分比"""
            placeholders = re.findall(r'\{([^\}]+)\}', text)
            values = []

            for placeholder in placeholders:
                # 先尝试提取常规数字（复用 extract_all_numbers）
                numbers = self.extract_all_numbers(placeholder)
                if numbers:
                    values.extend(numbers)
                    continue  # 跳过英文数字词的处理，已经提取到数字

                # 再尝试英文数字词
                try:
                    value = float(w2n.word_to_num(placeholder.strip()))
                    if value <= 150:  # 如果值小于150才计入
                        values.append(value)
                except ValueError:
                    pass  # 如果不是英文数字词，忽略

            return sum(values) if values else None

        def extract_number_before_target(text, target_word):
            """查找目标词前的数值，支持阿拉伯数字、百分比、千分位、英文数字词等"""
            tokens = re.findall(r'\b[\w\.,%\-]+\b', text.lower())  # 支持更多格式
            target_word = target_word.lower()

            for i, token in enumerate(tokens):
                if token == target_word:
                    if i > 0:
                        prev_token = tokens[i - 1]

                        # 先尝试作为英文数字词处理
                        try:
                            value = float(w2n.word_to_num(prev_token))
                            if value <= 150:
                                return value
                        except ValueError:
                            pass

                        # 再尝试用 extract_all_numbers 提取数字
                        numbers = self.extract_all_numbers(prev_token)
                        if numbers:
                            return numbers[0]  # 返回第一个匹配的数字

            return None

        def extract_and_sum(text):
            """暴力提取文本中的所有数字并求和，支持阿拉伯数字、百分比、千分位、英文数字词等"""
            tokens = re.findall(r'\b[\w\.,%\-]+\b', text.lower())  # 支持更多格式
            total = 0.0
            any_numbers_extracted = False  # 是否提取到了任何数字

            for token in tokens:
                # 检查是否为年份，如果是年份就跳过
                if self.is_year(token):
                    continue  # 跳过年份

                # 尝试英文数字词
                try:
                    # 如果是英文数字词且不是年份
                    value = float(w2n.word_to_num(token))
                    if value <= 150 and not self.is_year(str(value)):  # 如果不是年份且数值小于150
                        total += value
                        any_numbers_extracted = True
                    continue  # 如果是英文数字词，跳过后面的数字提取
                except ValueError:
                    pass

                # 尝试正则提取数字
                numbers = self.extract_all_numbers(token)
                if numbers:
                    total += sum(numbers)
                    any_numbers_extracted = True

            return total if any_numbers_extracted else None



        # Step 1: 提取目标词
        target_word = extract_target_word(baseline_question)

        # Step 2: 分别检查 baseline_answer 和 model_answer
        baseline_value = None
        model_value = None

        # 处理 baseline_answer：如果包含 {}，则提取并求和
        if bool(re.search(r'\{[^{}]*\}', baseline_answer)):
            baseline_value = extract_number_from_placeholders(baseline_answer)
        else:
            # 如果没有 {}, 查找目标词前的数值
            if target_word:
                baseline_value = extract_number_before_target(baseline_answer, target_word)

        # 处理 model_answer：如果包含 {}，则提取并求和
        if bool(re.search(r'\{[^{}]*\}', model_answer)):
            model_value = extract_number_from_placeholders(model_answer)
        else:
            # 如果没有 {}, 查找目标词前的数值
            if target_word:
                model_value = extract_number_before_target(model_answer, target_word)

        # Step 3: 如果没有找到目标词前的数字或 {数字}，暴力提取所有数字并求和
        if baseline_value is None:
            baseline_value = extract_and_sum(baseline_answer)

        if model_value is None:
            model_value = extract_and_sum(model_answer)


        return baseline_value, model_value

    def handle_SRP(self, baseline_answer, model_answer):
        """
        空间关系感知（SRP）类问答的评估逻辑。

        支持以下情况：
        - baseline 和 model 都有 {}: 各自提取并拆分关键词
        - baseline 有 {}: 用拆分后的词去 model 中模糊查找
        - model 没有 {}: 直接返回 None, None

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案

        Returns:
            baseline_words (List[str] or None): baseline 中提取并拆分后的关键词列表
            model_words (List[str] or None): 匹配到的 model 中的关键词列表，或 None
        """

        def extract_and_split_words(text):
            """从文本中提取并拆分出所有 { } 中的单词"""
            placeholders = re.findall(r'\{(.*?)\}', text)
            all_words = []

            for content in placeholders:
                # 按空格、逗号、括号等拆分，去除空白字符
                words = re.split(r'[(),\s]+', content)
                all_words.extend([w.strip() for w in words if w.strip()])

            return all_words

        srp_keywords = ['above', 'Above', 
                        'below', 'Below', 
                        'left', 'Left',
                        'right', 'Right',
                        'outside', 'Outside',
                        'inside','Inside',
                        'deeper', 'Deeper',
                        'shallower', 'Shallower',
                        'higher', 'Higher',
                        'lower', 'Lower',
                        'same', 'Same'
                        ]

        def is_spatial_question(baseline_words):
            return bool(set(baseline_words) & set(srp_keywords))

        def find_srp_keywords(text):
            """Find some keyword indicate that it is a pure srp question"""
            match = re.findall(r'\b(below|above|right|left|same|outside|inside|deeper|shallower|higher|lower)\b', text, re.IGNORECASE)
            return match
        

        # Step 1: 提取并拆分 baseline 和 model 的关键词
        baseline_words = extract_and_split_words(baseline_answer)
        b_is_spatial = is_spatial_question(baseline_words)

        if b_is_spatial:
            # try find keyword in raw answer
            baseline_words = find_srp_keywords(baseline_answer)
            model_words = find_srp_keywords(model_answer)
            return baseline_words, model_words
        

        model_words = extract_and_split_words(model_answer)

        # Step 2: 根据情况处理返回值
        if not baseline_words:
            # 如果 baseline 没有 {}，返回 None, None
            return None, None

        if baseline_words and model_words:
            # 如果 baseline 和 model 都有 {}
            return baseline_words, model_words

        if baseline_words and not model_words:
            # 如果 baseline 有 {} 而 model 没有，进行模糊查找
            matched_words = []
            for word in baseline_words:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                if pattern.search(model_answer):
                    matched_words.append(word)
            # 返回找到的匹配词和 baseline 的词
            return baseline_words, matched_words or None

    def handle_SC(self, baseline_answer, model_answer, chart_type):
        """
        处理 SC 类型问答（数值型单值比较），提取所有数值。

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案

        Returns:
            tuple: (baseline_values, model_values) if both have numeric values,
                   else (None, None)
        """

        def extract_values_from_placeholders(text):
            """从文本中的 { } 中提取数值并求和（支持千分位、负数、百分比、科学计数法等）"""
            placeholders = re.findall(r'\{([^\}]+)\}', text)
            total = 0.0
            has_numbers = False  # 是否提取到了任何数字
            processed_numbers = set()  # 用于存储已经处理过的数值

            for placeholder in placeholders:
                # 尝试提取阿拉伯数字
                numbers = self.extract_all_numbers(placeholder)
                for number in numbers:
                    if number not in processed_numbers:  # 如果该数字未被处理过
                        if not self.is_year(str(number)):  # 如果不是年份
                            processed_numbers.add(number)
                            # 在提取到的数字上调用单位转换函数
                            converted_value = self.extract_and_convert_value(number, text)
                            total += converted_value
                            has_numbers = True

                # 如果没有提取到数字，再尝试提取英文数字词
                if numbers:  # 如果已经提取了数字，跳过英文数字词处理
                    continue  # 跳过后面的英文数字词处理部分

                # 尝试提取英文数字词
                try:
                    value = float(w2n.word_to_num(placeholder.strip()))  # 尝试将英文数字词转换为数字
                    if value not in processed_numbers:  # 如果该数字未被处理过
                        if not self.is_year(str(value)):  # 如果不是年份
                            processed_numbers.add(value)
                            # 在提取到的英文数字词上调用单位转换函数
                            converted_value = self.extract_and_convert_value(value, text)
                            total += converted_value
                            has_numbers = True
                except ValueError:
                    pass  # 如果不是英文数字词，忽略

            return total if has_numbers else None

        def extract_values_from_text(text):
            """从文本中提取所有可能的数值并求和（支持千分位、负数、百分比、科学计数法等）"""
            # # 提取所有的数字
            # numbers = self.extract_all_numbers(text)
            #
            # # 过滤掉年份
            # filtered_numbers = [num for num in numbers if not self.is_year(str(num))]
            #
            # if filtered_numbers:
            #     return sum(filtered_numbers)  # 返回有效数值的总和
            #
            # # 如果没提取到有效数值，尝试英文数字词
            # try:
            #     # 先检查是否为年份，年份不进行转换
            #     if self.is_year(text.strip()):
            #         return None  # 如果是年份，跳过
            #
            #     value = float(w2n.word_to_num(text.strip()))  # 尝试将英文数字词转换为数字
            #     if not self.is_year(str(value)):  # 确保值不是年份
            #         return value  # 返回有效的单个数值
            # except:
            #     return None  # 如果无法转换英文数字词，则跳过
            #
            # return None  # 如果没有有效的数值，返回 None

            # 提取所有的数字
            numbers = self.extract_all_numbers(text)
            # 过滤掉年份
            filtered_numbers = [num for num in numbers if not self.is_year(str(num))]

            total = 0.0  # 初始化总和
            for number in filtered_numbers:
                # 在每个提取的数值上调用单位转换函数
                converted_value = self.extract_and_convert_value(number, text)
                if converted_value is not None: 
                    total += converted_value

            # 如果有有效数值，返回总和
            if total:
                return total

            # 如果没提取到有效数值，尝试英文数字词
            try:
                # 先检查是否为年份，年份不进行转换
                if self.is_year(text.strip()):
                    return None  # 如果是年份，跳过

                value = float(w2n.word_to_num(text.strip()))  # 尝试将英文数字词转换为数字
                if not self.is_year(str(value)):  # 确保值不是年份
                    # 在提取到的英文数字词上调用单位转换函数
                    converted_value = self.extract_and_convert_value(value, text)
                    return converted_value  # 返回有效的单个数值
            except:
                return None  # 如果无法转换英文数字词，则跳过

            return None  # 如果没有有效的数值，返回 None

        def get_baseline_value_list(text):
            """Return any values in {} as a list"""
            matches = re.findall(r'\{([^\}]+)\}', text)
            return matches
        
        def extract_all_numbers(text):
            """从extract all values from text"""
            num_matches = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
            return [float(n.replace(',', '')) for n in num_matches]
        
        # Special case
        baseline_value_list = get_baseline_value_list(baseline_answer)
        if len(baseline_value_list) == 1:
            baseline_str = baseline_value_list[0]
            baseline_value = float(re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', baseline_str).group(0).replace(',', ''))
            
            model_values = extract_all_numbers(model_answer)
            # model_values = extract_values_from_text(model_answer)

            # convert number unit
            filtered_model_values = [mv for mv in model_values if not self.is_year(str(model_values))]
            converted_model_values = []
            for mv in filtered_model_values:
                converted_value = self.extract_and_convert_value(mv, model_answer)
                converted_value = mv if chart_type == 'chord' else converted_value
                if converted_value is not None: 
                    converted_model_values.append(converted_value)

            if not converted_model_values:
                return baseline_value, None

            closest_val = min(converted_model_values, key=lambda x: abs(x - baseline_value))
            return baseline_value, closest_val

        # Step 1: 提取 baseline_answer 中所有 {} 中的数值，并返回求和
        baseline_values = extract_values_from_placeholders(baseline_answer)

        # 如果 baseline_answer 中没有 {}，直接返回 None, None
        if baseline_values is None:
            return None, None

        # Step 2: 如果 model_answer 中有 {}，从 model 中提取数值
        if '{' in model_answer:
            model_values = extract_values_from_placeholders(model_answer)
            # 返回 model_values 或 None（如果没有提取到有效值）
            return baseline_values, model_values if model_values is not None else None

        # Step 3: 如果 model_answer 中没有 {}，从 model 文本中提取数值
        model_values = extract_values_from_text(model_answer)

        return baseline_values, model_values if model_values is not None else None

    def handle_NC(self, baseline_answer, model_answer):
        """
        处理 NC 类型问答（字符型关键词匹配）

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案

        Returns:
            tuple: (baseline_words, matched_words) if match found,
                   else (None, None)
        """
        def extract_and_split_words(text):
            """提取 {} 内容并拆分成单词"""
            placeholders = re.findall(r'\{(.*?)\}', text)
            all_words = []

            for content in placeholders:
                words = re.split(r'[(),\s]+', content)
                all_words.extend([w.strip() for w in words if w.strip()])

            return all_words if all_words else None

        # Step 1: 提取 baseline 和 model 的关键词
        baseline_words = extract_and_split_words(baseline_answer)
        model_words = extract_and_split_words(model_answer)

        # Step 2: baseline 没有关键词，则返回 None, None
        if not baseline_words:
            return None, None  # baseline 没有关键词，无法判断

        # Step 3: model 有 {}，直接返回 baseline_words 和 model_words
        if model_words:
            return baseline_words, model_words

        # Step 4: model 没有 {}，使用 baseline 进行模糊匹配
        matched_words = []
        for word in baseline_words:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            if pattern.search(model_answer):
                matched_words.append(word)

        return baseline_words, matched_words if matched_words else None

    def handle_NF(self, baseline_answer, model_answer, baseline_question):
        """
        处理 'NF' 类型问答（数值提取类型和单词类型都存在的）

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案

        Returns:
            tuple: (baseline_words, model_words, baseline_value, model_value) if valid words/values found,
                   else return None at corresponding position
        """

        def extract_values_from_placeholders(text):
            """从文本中的 { } 中提取数值并返回列表（支持千分位、负数、百分比、科学计数法、英文数字词）"""
            placeholders = re.findall(r'\{([^\}]+)\}', text)
            values = []

            for placeholder in placeholders:
                # 先尝试提取阿拉伯数字（复用通用函数）
                numbers = self.extract_all_numbers(placeholder)
                for number in numbers:
                    if not self.is_year(str(number)):  # 如果不是年份
                        # 在提取到的数字上调用单位转换函数
                        converted_value = self.extract_and_convert_value(number, text)
                        values.append(converted_value)

                # 如果没有提取到数字，再尝试提取英文数字词
                if numbers:  # 如果已经提取了数字，跳过英文数字词处理
                    continue  # 跳过后面的英文数字词处理部分

                # 尝试提取英文数字词
                try:
                    value = float(w2n.word_to_num(placeholder))
                    if not self.is_year(str(value)):
                        # 在提取到的英文数字词上调用单位转换函数
                        converted_value = self.extract_and_convert_value(value, text)
                        values.append(converted_value)
                except ValueError:
                    pass  # 如果不是英文数字词，忽略

            return values if values else None


        def extract_values_from_text(text):
            """从文本中提取所有可能的数值并返回总和（支持千分位、负数、百分比、科学计数法、英文数字词）"""
            # 提取所有的数字
            numbers = self.extract_all_numbers(text)

            # 过滤掉年份
            filtered_numbers = [num for num in numbers if not self.is_year(str(num))]

            total = 0.0  # 初始化总和
            for number in filtered_numbers:
                # 在每个提取的数值上调用单位转换函数
                converted_value = self.extract_and_convert_value(number, text)
                if converted_value is not None:
                    total += converted_value

            # 如果有有效数值，返回总和
            if total:
                return total

            # 如果没提取到有效数值，尝试英文数字词
            try:
                # 先检查英文数字词是否是年份
                if self.is_year(text.strip()):
                    return None  # 如果是年份，跳过

                value = float(w2n.word_to_num(text.strip()))  # 尝试将英文数字词转换为数字
                if not self.is_year(str(value)):  # 确保值不是年份
                    # 在提取到的英文数字词上调用单位转换函数
                    converted_value = self.extract_and_convert_value(value, text)
                    return converted_value  # 返回有效的单个数值
            except:
                return None  # 如果无法转换英文数字词，则跳过

            return None  # 如果没有有效的数值，返回 None


        def extract_words_from_placeholders(text):
            """提取文本中的单词并返回列表（考虑连字符等符号，但保留非线性 non-linear 这类词）"""
            words = []
            placeholders = re.findall(r'\{([^\}]+)\}', text)

            for value in placeholders:
                if self.is_number(value):
                    continue
                # 特殊处理以 'non-' 开头的词
                tokens = re.split(r'[\s,():\'\._]+', value)  # 先按非 - 的符号拆分
                for token in tokens:
                    if self.is_number(token):
                        continue
                    # 如果包含 '-', 且不是以 'non-' 开头，就继续拆分
                    b_need_split = not (token.startswith('non-') or token.startswith('none-'))
                    if '-' in token and b_need_split:
                        parts = re.split(r'-', token)
                        words.extend(parts)
                    else:
                        words.append(token)
            return [word.strip() for word in words if word.strip()]

        def fuzzy_match_words(baseline_words, model_text):
            """模糊匹配 baseline_words 中的单词，在 model_text 中查找"""
            matched_words = []
            for word in baseline_words:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(model_text):
                    matched_words.append(word)
            return matched_words
        
        if re.search(r'list', baseline_question, re.IGNORECASE):
            if re.search(r'year', baseline_question, re.IGNORECASE):
                baseline_values = extract_values_from_placeholders(baseline_answer)
                baseline_year_list = []
                baseline_value_list = []

                for bv in baseline_values:
                    if self.is_year(str(int(bv))):
                        baseline_year_list.append(str(int(bv)))
                    else :
                        baseline_value_list.append(bv)
                baseline_value_sum = sum(baseline_value_list) 

                model_values = extract_values_from_placeholders(model_answer)
                model_year_list = []
                model_value_list = []

                if model_values is not None: 
                    for mv in model_values:
                        if self.is_year(str(int(mv))):
                            model_year_list.append(str(int(mv)))
                        else :
                            model_value_list.append(mv)
                
                    model_value_sum = sum(model_value_list) 
                else: 
                    model_value_sum = None

                return baseline_year_list, model_year_list, baseline_value_sum, model_value_sum
            else:
                baseline_label_list = extract_words_from_placeholders(baseline_answer)
                model_label_list = extract_words_from_placeholders(model_answer)
                baseline_value_list = extract_values_from_placeholders(baseline_answer)
                if baseline_value_list:
                    baseline_value_sum = sum(baseline_value_list)
                else:
                    baseline_value_sum = extract_values_from_text(baseline_answer)
                model_value_list = extract_values_from_placeholders(model_answer)
                if model_value_list:
                    model_value_sum = sum(model_value_list)
                else:
                    model_value_sum = extract_values_from_text(model_answer)

                if not model_label_list:
                    matched_words = fuzzy_match_words(baseline_label_list, model_answer)
                    model_label_list = matched_words if matched_words else None

                if not baseline_label_list:
                    baseline_label_list = None
                return baseline_label_list, model_label_list, baseline_value_sum, model_value_sum
                #   - 直接提取baseline_answer {}的所有单词，变成标签 bl（字符串列表）；直接提取 baseline_answer中的所有数值并求和 bn（求和）
                #   - 如果model_answer有{}，直接提取 model_answer {}的所有单词，变成标签 ml（字符串列表）；直接提取 model_answer中的所有数值并求和 mn（求和）
                #   - 如果model_answer无{}，模糊匹配 得到 ml（字符串列表）；直接提取 model_answer中的所有数值并求和 mn（求和）

        # Step 1: 提取数值和单词
        baseline_values = extract_values_from_placeholders(baseline_answer)
        model_values = extract_values_from_placeholders(model_answer)
        baseline_words = extract_words_from_placeholders(baseline_answer)
        model_words = extract_words_from_placeholders(model_answer)

        # dummy for output
        baseline_key_words, model_key_words, baseline_key_num, model_key_num = None, None, None, None

        baseline_key_num = sum(baseline_values) if baseline_values else None
        model_key_num = sum(model_values) if model_values else extract_values_from_text(model_answer)

        # Step 3: 处理单词型问题
        # 1. baseline 和 model 都有 {}，直接提取并返回拆解后的单词列表
        if baseline_words and model_words:
            baseline_key_words = baseline_words
            model_key_words = model_words

        # 2. baseline 有 {}，model 没有 {}，使用模糊匹配
        elif baseline_words and not model_words:
            matched_words = fuzzy_match_words(baseline_words, model_answer)
            baseline_key_words = baseline_words
            model_key_words = matched_words if matched_words else None

        return baseline_key_words, model_key_words, baseline_key_num, model_key_num

    def handle_multi(self, baseline_answer, model_answer, chart_type): # 'VPR', 'VE', 'EVJ', 'NF', 'MSR', 'VA'
        """
        处理 'VPR', 'VE', 'EVJ', 'NF', 'MSR', 'VA' 类型问答（数值提取类型和单词类型都存在的）

        Args:
            baseline_answer (str): 基准答案
            model_answer (str): 模型答案

        Returns:
            tuple: (baseline_value, model_value) if valid values found,
                   else (None, None)
        """

        def extract_values_from_placeholders(text, chart_type):
            """从文本中的 { } 中提取数值并返回列表（支持千分位、负数、百分比、科学计数法、英文数字词）"""
            placeholders = re.findall(r'\{([^\}]+)\}', text)
            values = []

            for placeholder in placeholders:
                # 先尝试提取阿拉伯数字（复用通用函数）
                numbers = self.extract_all_numbers(placeholder)
                for number in numbers:
                    if not self.is_year(str(number)):  # 如果不是年份
                        # 在提取到的数字上调用单位转换函数
                        converted_value = self.extract_and_convert_value(number, text)
                        converted_value = number if chart_type == 'chord' else converted_value
                        values.append(converted_value)

                # 如果没有提取到数字，再尝试提取英文数字词
                if numbers:  # 如果已经提取了数字，跳过英文数字词处理
                    continue  # 跳过后面的英文数字词处理部分

                # 尝试提取英文数字词
                try:
                    value = float(w2n.word_to_num(placeholder))
                    if not self.is_year(str(value)):
                        # 在提取到的英文数字词上调用单位转换函数
                        converted_value = self.extract_and_convert_value(value, text)
                        values.append(converted_value)
                except ValueError:
                    pass  # 如果不是英文数字词，忽略

            return values if values else None


        def extract_values_from_text(text):
            """从文本中提取所有可能的数值并返回总和（支持千分位、负数、百分比、科学计数法、英文数字词）"""
            # 提取所有的数字
            numbers = self.extract_all_numbers(text)

            # 过滤掉年份
            filtered_numbers = [num for num in numbers if not self.is_year(str(num))]

            total = 0.0  # 初始化总和
            for number in filtered_numbers:
                # 在每个提取的数值上调用单位转换函数
                converted_value = self.extract_and_convert_value(number, text)
                if converted_value is not None:
                    total += converted_value

            # 如果有有效数值，返回总和
            if total:
                return total

            # 如果没提取到有效数值，尝试英文数字词
            try:
                # 先检查英文数字词是否是年份
                if self.is_year(text.strip()):
                    return None  # 如果是年份，跳过

                value = float(w2n.word_to_num(text.strip()))  # 尝试将英文数字词转换为数字
                if not self.is_year(str(value)):  # 确保值不是年份
                    # 在提取到的英文数字词上调用单位转换函数
                    converted_value = self.extract_and_convert_value(value, text)
                    return converted_value  # 返回有效的单个数值
            except:
                return None  # 如果无法转换英文数字词，则跳过

            return None  # 如果没有有效的数值，返回 None


        def extract_words_from_placeholders(text):
            """提取文本中的单词并返回列表（考虑连字符等符号，但保留非线性 non-linear 这类词）"""
            words = []
            placeholders = re.findall(r'\{([^\}]+)\}', text)

            for value in placeholders:
                if self.is_number(value):
                    continue
                # 特殊处理以 'non-' 开头的词
                tokens = re.split(r'[\s,():\'\._]+', value)  # 先按非 - 的符号拆分
                for token in tokens:
                    if self.is_number(token):
                        continue
                    # 如果包含 '-', 且不是以 'non-' 开头，就继续拆分
                    b_need_split = not (token.startswith('non-') or token.startswith('none-'))
                    if '-' in token and b_need_split:
                        parts = re.split(r'-', token)
                        words.extend(parts)
                    else:
                        words.append(token)
            
            return [word.strip() for word in words if word.strip()]

        def fuzzy_match_words(baseline_words, model_text):
            """模糊匹配 baseline_words 中的单词，在 model_text 中查找"""
            matched_words = []
            for word in baseline_words:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(model_text):
                    matched_words.append(word)
            return matched_words

        # Step 1: 判断 baseline 和 model 中的数值型问题
        baseline_values = extract_values_from_placeholders(baseline_answer, chart_type)
        model_values = extract_values_from_placeholders(model_answer, chart_type)

        # 判断是数值型问题还是单词型问题
        is_numeric = False
        if baseline_values:
            is_numeric = True  # 如果baseline_values包含数值型内容，认为是数值型问题

        if is_numeric:
            # Step 2: 处理数值型问题
            if baseline_values and model_values:
                # 如果 baseline 和 model 都包含 {} 中的数值，直接计算总和
                baseline_value = sum(baseline_values)
                model_value = sum(model_values)
                return baseline_value, model_value

            # 如果 model 中没有 {}, 提取并返回 baseline 和 model 中的数值
            if not model_values:
                baseline_value = sum(baseline_values)
                model_value = extract_values_from_text(model_answer)  # 提取 model 中的数值
                return baseline_value, model_value

            return None, None  # 如果没有找到数值返回 None

        else:
            # Step 3: 处理单词型问题
            baseline_words = extract_words_from_placeholders(baseline_answer)
            model_words = extract_words_from_placeholders(model_answer)

            # 1. baseline 和 model 都有 {}，直接提取并返回拆解后的单词列表
            if baseline_words and model_words:
                return baseline_words, model_words

            # 2. baseline 有 {}，model 没有 {}，使用模糊匹配
            if baseline_words and not model_words:
                matched_words = fuzzy_match_words(baseline_words, model_answer)
                return baseline_words, matched_words if matched_words else None

        return None, None  # 默认返回 None，None


    # def handle_VPR(self, baseline_answer, model_answer):
    #     """
    #     视觉模式识别类问答的评估逻辑（极简版）。
    #
    #     仅返回三种情况：
    #     1. (baseline_sum, model_sum) → 数值型 VPR（都提取到了数值）
    #     2. (baseline_words, model_words) → 单词型 VPR（都提取到了关键词）
    #     3. (None, None) → 其他不匹配情况
    #
    #     Args:
    #         baseline_answer (str): 基准答案
    #         model_answer (str): 模型答案
    #
    #     Returns:
    #         tuple: (baseline_value, model_value)
    #     """
    #
    #     def extract_all_values(text):
    #         """从整个句子中提取所有数值（支持整数、小数）"""
    #         values = re.findall(r'\b\d+\.?\d*\b', text)
    #         if not values:
    #             return None
    #         return sum(map(float, values))
    #
    #     def extract_and_split_words(text):
    #         """提取 {} 内容并拆分成单词"""
    #         placeholders = re.findall(r'\{(.*?)\}', text)
    #         all_words = []
    #
    #         for content in placeholders:
    #             words = re.split(r'[(),\s]+', content)
    #             all_words.extend([w.strip() for w in words if w.strip()])
    #
    #         return all_words
    #
    #     # Step 1: 判断是否为数值型 VPR
    #     baseline_sum = extract_all_values(baseline_answer)
    #     model_sum = extract_all_values(model_answer)
    #     if baseline_sum is not None and model_sum is not None:
    #         return baseline_sum, model_sum  # 返回两个字段的数字之和
    #
    #     # Step 2: 判断是否为单词型 VPR
    #     baseline_words = extract_and_split_words(baseline_answer)
    #     model_words = extract_and_split_words(model_answer)
    #
    #     if not baseline_words:
    #         return None, None  # baseline 没有关键词，无法判断
    #
    #     # 如果 model 有关键词，返回两个关键词
    #     if model_words:
    #         return baseline_words, model_words
    #
    #     # 如果 model 没有 {}，尝试模糊查找关键词
    #     matched_words = []
    #     for word in baseline_words:
    #         pattern = re.compile(re.escape(word), re.IGNORECASE)
    #         if pattern.search(model_answer):
    #             matched_words.append(word)
    #
    #     if matched_words: # 如果能查到关键词
    #         return baseline_words, matched_words  # 单词型匹配成功（模糊匹配）
    #
    #     # 所有情况都不匹配
    #     return None, None





# 主程序逻辑
if __name__ == "__main__":
    # 读取 JSON 文件
    # cogvlm2-chat_png_0.2k.json, InternVL3-14b_png_0.2k.json,  llava-1.5-7b_png_0.2k.json
    with open(r"D:\CQAVis\mllm_evaluation_data\visual_eval\llava-1.5-7b_png_0.2k.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 筛选 qa_type 为 CTR 的样本
    ctr_samples = [item for item in data if item.get("qa_type") == "SC"]

    # 随机抽取最多 50 个样本
    samples = random.sample(ctr_samples, min(50, len(ctr_samples)))

    handler = QAHandler()

    # 遍历并测试
    for i, sample in enumerate(samples, 1):
        question = sample["baseline_question"]
        baseline = sample["baseline_answer"]
        model = sample["model_answer"]
        chart_type = sample["chart_type"]

        baseline_value, model_value = handler.handle_SC(baseline, model)

        print(f"Sample {i}:")
        print("  Baseline: ", baseline)
        print("  Model:    ", model)
        print("  baseline_value, model_value", baseline_value, model_value)
        print("-" * 60)