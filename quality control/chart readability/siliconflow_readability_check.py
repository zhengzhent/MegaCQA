#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specially adapted SiliconFlow API readability check module
"""

import json
import sys
import warnings
import requests
import os
from typing import Dict, Tuple, Optional
import dotenv

INSTRUCTION = """Please evaluate this chart on the following 5 dimensions using a 1-5 scale: 1) Text Readability (font size, label overlap, font clarity), 2) Color & Contrast (graphic contrast, color differentiation), 3) Layout & Spacing (element spacing, alignment and neatness), 4) Graphic Clarity (line clarity, shape recognition, avoiding distracting visual effects), 5) Information Organization (legend clarity, axis labeling).
```
    {
        "Rationale": "Text Readability: X points - reason; Color & Contrast: X points - reason; Layout & Spacing: X points - reason; Graphic Clarity: X points - reason; Information Organization: X points - reason",
        "Score": 1-5
    }
```
where Score is the weighted average (Text Readability×0.2 + Color & Contrast×0.2 + Layout & Spacing×0.2 + Graphic Clarity×0.2 + Information Organization×0.2).
"""

# """Your task is to evaluate the readability of the visualization on a scale of 1 to 5, where 1 indicates very difficult to read and 5 indicates very easy to read. You will be given a visualization requirement and the corresponding visualization created based on that requirement. Additionally, reviews from others regarding this visualization will be provided for your reference. Please think carefully and provide your reasoning and score.
# ```
#     {
#         "Rationale": "a brief reason",
#         "Score": 1-5
#     }
# ```
#
# Examples:
# - If the visualization is clear and information can be easily interpreted, you might return:
# ```
#     {
#         "Rationale": "The chart is well-organized, and the use of contrasting colors helps in distinguishing different data sets effectively. The labels are legible, and the key insights can be understood at a glance.",
#         "Score": 5
#     }
# ```
# - Conversely, if the visualization is cluttered or confusing, you might return:
# ```
#     {
#         "Rationale": "While there is no overflow or overlap, the unconventional inverted y-axis and the use of decimal numbers for months on the x-axis deviate from the standard interpretation of bar charts, confusing readers and significantly affecting the chart's readability.",
#         "Score": 1
#     }
# ```
# """


def siliconflow_readability_check(context: dict, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct") -> Tuple[Optional[int], str, Optional[Dict]]:
    """
    Use SiliconFlow API for direct readability check
    """
    base64_image = context["base64"]
    
    reviews = ""
    if "reviews" in context and len(context["reviews"]) > 0:
        reviews = "Other Reviews:\n"
        reviews += "\n".join([
            f"""- {review["aspect"]}: {review["content"]}"""
            for review in context["reviews"]
        ])
        reviews += "\n\n"
    
    # Prepare request data
    api_key = os.getenv("SILICON_API_KEY")
    if not api_key:
        return None, "SILICON_API_KEY environment variable not set"
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build message - try different formats
    formats_to_try = [
        # Format 1: OpenAI standard format
        {
            "content": [
                {
                    "type": "text",
                    "text": f"{INSTRUCTION}\n\nPlease assess the readability of this visualization image:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            ]
        },
        # Format 2: Simplified image_url format
        {
            "content": [
                {
                    "type": "text",
                    "text": f"{INSTRUCTION}\n\nPlease assess the readability of this visualization image:"
                },
                {
                    "type": "image_url",
                    "image_url": base64_image
                }
            ]
        },
        # Format 3: Use image field
        {
            "content": [
                {
                    "type": "text",
                    "text": f"{INSTRUCTION}\n\nPlease assess the readability of this visualization image:"
                },
                {
                    "type": "image",
                    "image": base64_image
                }
            ]
        }
    ]
    
    for i, message_format in enumerate(formats_to_try, 1):
        try:
            # print(f"Trying format {i}...")
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a visualization readability expert. Respond only with valid JSON format."
                    },
                    {
                        "role": "user",
                        **message_format
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.0
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                try:
                    json_string = content.replace("```json\n", "").replace("```", "").strip()
                    parsed_result = json.loads(json_string)
                    score = parsed_result.get("Score")
                    rationale = parsed_result.get("Rationale", "")
                    
                    # print(f"✅ Format {i} succeeded!")
                    return score, rationale, result
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"❌ Format {i} JSON parse failed: {e}")
                    print(f"Raw response: {content[:200]}...")
                    continue
                    
            else:
                error_info = response.json() if response.text else {"error": "No response"}
                print(f"❌ Format {i} request failed (status code: {response.status_code})")
                print(f"Error info: {error_info}")
                continue
                
        except Exception as e:
            print(f"❌ format {i} error: {e}")
            continue
    
    return None, "All formats failed to get a valid response from SiliconFlow API"


def test_siliconflow_connection(model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct") -> bool:
    """
    test connection to SiliconFlow API and check if the specified model is available
    """
    dotenv.load_dotenv()
    api_key = os.getenv("SILICON_API_KEY")
    if not api_key:
        print("❌ not found SILICON_API_KEY environment variable")
        return False
    
    url = "https://api.siliconflow.cn/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            model_ids = [model.get("id") for model in models]
            
            if model_name in model_ids:
                print(f"✅ model {model_name} is available")
                return True
            else:
                print(f"❌ model {model_name} is not available")
                print(f"available models:")
                vision_models = [mid for mid in model_ids if any(keyword in mid.lower() for keyword in ["vision", "vl", "qwen", "cogvlm"])]
                for vm in vision_models[:5]:
                    print(f"  - {vm}")
                return False
        else:
            print(f"❌ api request failed with status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to SiliconFlow API: {e}")
        return False


if __name__ == "__main__":
    # test the SiliconFlow API connection
    if test_siliconflow_connection():
        print("SiliconFlow API connection successful")
    else:
        print("SiliconFlow API connection failed")