# from my_api.gemini_api import GeminiAPI
from my_api.qwen_api import QwenAPI
from my_api.deepseek_api import DeepseekAPI
from my_api.doubao_api import DoubaoAPI
from my_api.gpt_api import GPTAPI
from my_api.gemini_api import GeminiAPI

class APIClient:
    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == "deepseek":
            self.api = DeepseekAPI()
        elif self.model_name == "doubao":
            self.api = DoubaoAPI()
        elif self.model_name == "qwen":
            self.api = QwenAPI()
        elif self.model_name == "gpt":
            self.api = GPTAPI()
        elif self.model_name == "gemini":
            self.api = GeminiAPI()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def generate_text(self, prompt):
        return self.api.generate_text(prompt)

