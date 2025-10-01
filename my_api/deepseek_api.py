import os
from openai import OpenAI
import dotenv

class DeepseekAPI:
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("SILICON_API_KEY"), base_url="https://api.siliconflow.cn/v1")
        self.model="Pro/deepseek-ai/DeepSeek-R1"

    def generate_text(self, prompt, max_tokens=10000):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }

                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=max_tokens,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error during OpenAI API call: {str(e)}")