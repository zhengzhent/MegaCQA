import os
from openai import OpenAI
import dotenv

class GeminiAPI:
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("AIHUBMIX_API_KEY"), base_url="https://aihubmix.com/v1")
        self.model="gemini-2.0-flash"

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