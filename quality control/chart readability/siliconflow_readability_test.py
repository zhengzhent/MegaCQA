#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiliconFlow API Readability Test Script
Specifically designed to solve API format compatibility issues
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import cairosvg
import dotenv
from PIL import Image
import io

# Import our SiliconFlow adapter module
from siliconflow_readability_check import siliconflow_readability_check, test_siliconflow_connection

class TokenBucket:
    def __init__(self, rate_per_sec, capacity):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.time()
        self.lock = __import__('threading').Lock()

    def consume(self, amount):
        with self.lock:
            now = time.time()
            # Refill tokens
            delta = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            self.timestamp = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def wait_for(self, amount):
        # Block until enough tokens are available
        while not self.consume(amount):
            time.sleep((amount - self.tokens) / self.rate)



class SiliconFlowReadabilityTester:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        """
        Initialize SiliconFlow readability tester
        
        Args:
            model_name: SiliconFlow vision model name
        """
        self.model_name = model_name
        
        # Test connection
        if not test_siliconflow_connection(model_name):
            raise ValueError(f"Cannot connect to SiliconFlow or model {model_name} is not available")

        # Initialize token bucket for rate limiting (20,000 tokens/minute)
        # Each API call is assumed to consume 4000 tokens
        self.token_bucket = TokenBucket(rate_per_sec=20000/60, capacity=20000)
    
    def convert_image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 encoding
        """
        image_path_obj = Path(image_path)
        
        
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path_obj}")
        
        # Supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.bmp'}
        
        if image_path_obj.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_path_obj.suffix}")
        
        if image_path_obj.suffix.lower() == '.svg':
            # SVG needs to be converted to PNG
            with open(image_path_obj, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            png_data = cairosvg.svg2png(bytestring=svg_content.encode())
            base64_encoded = base64.b64encode(png_data).decode('utf-8')
            return f"data:image/png;base64,{base64_encoded}"
        else:
            # Other formats encode directly
            with open(image_path_obj, 'rb') as f:
                image_data = f.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            
            # Determine MIME type
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg', 
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp'
            }.get(image_path_obj.suffix.lower(), 'image/png')
            
            return f"data:{mime_type};base64,{base64_encoded}"


    def save_result_to_txt(self, result: Dict[str, Any], result_folder: str):
        """
        Save single image test result to txt file
        """
        # Create result folder
        result_path = Path(result_folder)
        print(result_path)
        result_path.mkdir(exist_ok=True)
        
        # Get image filename (without extension)
        image_path = Path(result["image_path"])
        image_name = image_path.stem
        
        # Create txt filename
        txt_filename = f"{image_name}.txt"
        txt_path = result_path / txt_filename
        
        # Write results
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Image Name: {image_path.name}\n")
            f.write(f"Image Path: {result['image_path']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: Qwen/Qwen2.5-VL-72B-Instruct\n")
            if result["status"] == "success" and result["score"] is not None:
                f.write(f"Readability Score: {result['score']}/5\n")
                f.write(f"Score Reason: {result['rationale']}\n")
            else:
                f.write(f"Test Failed: {result['rationale']}\n")
        
        print(f"Result saved to: {txt_path}")
        return str(txt_path)
    
    def test_single_image(self, 
                         image_path: str,
                         result_folder: str,
                         save_to_txt: bool = True) -> Dict[str, Any]:
        """
        Test readability of a single image
        """
        try:
            # Convert image to base64
            base64_data = self.convert_image_to_base64(image_path)
            
            # Build context
            context = {
                "base64": base64_data,
            }

            # Execute readability check and capture full response (including usage)
            score, rationale, return_result = siliconflow_readability_check(context, self.model_name)
            # Extract total token usage from API response
            total_tokens = return_result.get("usage", {}).get("total_tokens", 0)
            print(total_tokens)
            # Rate limiting based on actual usage
            self.token_bucket.wait_for(total_tokens)

            result = {
                "image_path": str(image_path),
                "score": score,
                "rationale": rationale,
                "status": "success" if score is not None else "fail"
            }

            # Save to txt file
            if save_to_txt:
                txt_path = self.save_result_to_txt(result, result_folder)
                result["txt_path"] = txt_path
            
            return result
            
        except Exception as e:
            result = {
                "image_path": str(image_path),
                "score": None,
                "rationale": f"failure: {str(e)}",
                "status": "failure"
            }
            
            # Save to txt file
            if save_to_txt:
                txt_path = self.save_result_to_txt(result, result_folder)
                result["txt_path"] = txt_path

            return result
    
    def test_batch_images(self, 
                         image_folder: str,
                         result_folder: str,
                         output_file: Optional[str] = None
                         ) -> List[Dict[str, Any]]:
        """
        Recursively batch test images in folder and subfolders, saving results in corresponding subfolders under result_folder.
        """
        image_folder_obj = Path(image_folder)
        if not image_folder_obj.exists():
            raise FileNotFoundError(f"Folder does not exist: {image_folder_obj}")
        
        # Supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.bmp'}
        
        # Recursively find all image files
        image_files = [f for f in image_folder_obj.rglob("*") if f.is_file() and f.suffix.lower() in supported_formats]
        
        if not image_files:
            print(f"No supported image files found in folder {image_folder_obj}")
            return []
        
        print(f"Found {len(image_files)} image files, starting test...")
        # print(f"Using model: {self.model_name}")
        print(f"Results will be saved to: {result_folder}/")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"Testing ({i}/{len(image_files)}): {image_file.relative_to(image_folder_obj)}")

            # relative_path = image_file.relative_to(image_folder_obj)
            # result_subfolder = Path(result_folder) / relative_path.parent
            # result_subfolder.mkdir(parents=True, exist_ok=True)

            result = self.test_single_image(str(image_file), result_folder=str(result_folder))
            results.append(result)

            if result["status"] == "success":
                print(f"  score: {result['score']}/5")
                print(f"  reason: {result['rationale'][:100]}...")
            else:
                print(f" failure: {result['rationale']}")
            print()

        if output_file:
            # Get summary data
            summary = self.print_summary(results)
            
            # Create complete output data
            output_data = {
                "summary": summary,
                "results": results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Results and summary saved to: {output_file}")
        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """
        Print test result summary and return summary data
        """
        successful_results = [r for r in results if r["status"] == "success" and r["score"] is not None]
        
        if not successful_results:
            print("No successful test results")
            return {
                "total_images": len(results),
                "successful_tests": 0,
                "failed_tests": len(results),
                "average_score": None,
                "highest_score": None,
                "lowest_score": None,
                "score_distribution": {}
            }
        
        scores = [r["score"] for r in successful_results]
        
        print("=== Readability Test Summary ===")
        print(f"Using model: {self.model_name}")
        print(f"Total test images: {len(results)}")
        print(f"Successful tests: {len(successful_results)}")
        print(f"Failed tests: {len(results) - len(successful_results)}")
        print(f"Average readability score: {sum(scores) / len(scores):.2f}/5")
        print(f"Highest score: {max(scores)}")
        print(f"Lowest score: {min(scores)}")
        
        # Score distribution
        score_distribution = {}
        for score in scores:
            score_distribution[score] = score_distribution.get(score, 0) + 1
        
        print("\nScore distribution:")
        for score in sorted(score_distribution.keys()):
            count = score_distribution[score]
            print(f"  {score} points: {count} images")
        
        # Return summary data
        return {
            "model_name": self.model_name,
            "total_images": len(results),
            "successful_tests": len(successful_results),
            "failed_tests": len(results) - len(successful_results),
            "average_score": round(sum(scores) / len(scores), 2),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "score_distribution": score_distribution
        }


def main():
    dotenv.load_dotenv()
    """
    Main function - command line usage example
    """

    # Parse parameters
    input_path = sys.argv[1]
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"  # Default model
    
    # Check environment variables
    if not os.getenv("SILICON_API_KEY"):
        print("❌ Please set environment variable SILICON_API_KEY first")
        print("   export SILICON_API_KEY='your-siliconflow-api-key'")
        return
    
    # Create tester
    try:
        print(f"Initializing SiliconFlow readability tester...")
        print(f"Using model: {model_name}")
        tester = SiliconFlowReadabilityTester(model_name)
        print("✅ Initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    input_path_obj = Path(input_path)
    # print(f"Input path: {input_path_obj}")

    
    if input_path_obj.is_file():
        # Test single file
        print(f"Testing single image: {input_path}")
        # print(input_path_obj.parent.name)
        result = tester.test_single_image(input_path, rf"D:\code\MegaCQA\QualityControl\chart\result2\{input_path_obj.parent.name}")
        
        print(f"Result: {result['status']}")
        if result['score'] is not None:
            print(f"Score: {result['score']}/5")
            print(f"Reason: {result['rationale']}")
        else:
            print(f"Error: {result['rationale']}")
            
    elif input_path_obj.is_dir():
        # Batch test folder
        print(f"Batch testing folder: {input_path}")
        output_file = f"siliconflow_readability_results_{input_path_obj.parent.name}.json"
        results = tester.test_batch_images(input_path, rf"D:\code\MegaCQA\QualityControl\chart\result2\{input_path_obj.parent.name}", output_file)
        # Summary is already printed and saved in test_batch_images method
    else:
        print(f"Path does not exist: {input_path}")


if __name__ == "__main__":
    main() 