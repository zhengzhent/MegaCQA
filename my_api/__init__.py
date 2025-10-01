# 从各个API模块中导入相关类
from .deepseek_api import DeepseekAPI
from .doubao_api import DoubaoAPI
from .qwen_api import QwenAPI
from .gemini_api import GeminiAPI
from .gpt_api import GPTAPI

from .api import APIClient


# 可以选择提供某些默认接口
__all__ = [
    "DeepseekAPI",   # 用户可以直接使用 DeepseekAPI
    "GeminiAPI",     # 用户可以直接使用 GeminiAPI
    "GPTAPI",        # 用户可以直接使用 GPTAPI
    "DoubaoAPI",
    "QwenAPI",
    "APIClient",     # 用户可以通过 APIClient 统一访问各种模型
]