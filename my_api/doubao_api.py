import os
from openai import OpenAI
import dotenv

class DoubaoAPI:
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("Doubao_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3")
        self.model = "doubao-1-5-thinking-pro-m-250428"

    def generate_text(self, prompt, max_tokens=2000):
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