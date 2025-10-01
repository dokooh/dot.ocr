#!/usr/bin/env python3
"""
DotsOCR Processing Script
Applies DotsOCR transformer to page images and saves OCR results.
"""

import os
import sys
import json
import torch
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Optional imports with fallbacks
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not available, using fallback implementation")
    def process_vision_info(messages):
        """Fallback implementation for process_vision_info."""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                for content in message["content"]:
                    if isinstance(content, dict):
                        if content.get("type") == "image" and "image" in content:
                            image_inputs.append(content["image"])
                        elif content.get("type") == "video" and "video" in content:
                            video_inputs.append(content["video"])
        
        return image_inputs, video_inputs

try:
    from dots_ocr.utils import dict_promptmode_to_prompt
except ImportError:
    print("Warning: dots_ocr.utils not available, continuing without it")
    dict_promptmode_to_prompt = None


class DotsOCRProcessor:
    def __init__(self, model_path="./weights/DotsOCR"):
        """
        Initialize the DotsOCR processor.
        
        Args:
            model_path (str): Path to the DotsOCR model weights
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Standard prompt for OCR processing
        self.prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
    
    def load_model(self):
        """Load the DotsOCR model and processor."""
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                print(f"Model path not found: {self.model_path}")
                print("You need to download the DotsOCR model weights.")
                print("Please visit: https://huggingface.co/rednote-hilab/dots.ocr")
                return False
            
            # Load model with attention fallback
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else "cpu",
                "trust_remote_code": True
            }
            
            # Try FlashAttention2 first, fallback to eager attention
            if self.device == "cuda":
                flash_attn_available = False
                try:
                    import flash_attn
                    # Test if flash_attn actually works (not just importable)
                    _ = flash_attn.__version__
                    flash_attn_available = True
                    print("FlashAttention2 detected and working")
                except (ImportError, AttributeError, OSError, RuntimeError) as e:
                    print(f"FlashAttention2 not available or broken: {type(e).__name__}")
                    print("Using eager attention (this is fine)")
                    flash_attn_available = False
                
                if flash_attn_available:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using FlashAttention2 for faster inference")
                else:
                    model_kwargs["attn_implementation"] = "eager"
            else:
                model_kwargs["attn_implementation"] = "eager"
            
            # Try to load model with the chosen attention mechanism
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            except Exception as e:
                # If FlashAttention2 fails during model loading, fallback to eager
                if model_kwargs.get("attn_implementation") == "flash_attention_2":
                    print(f"FlashAttention2 failed during model loading: {type(e).__name__}")
                    print("Falling back to eager attention...")
                    model_kwargs["attn_implementation"] = "eager"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                else:
                    # Re-raise if it's not a FlashAttention2 issue
                    raise
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            print("Model and processor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def process_image(self, image_path):
        """
        Process a single image with DotsOCR.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: OCR result as text
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            
            print(f"Processing image: {image_path}")
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Inference: Generation of the output
            print("Generating OCR output...")
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else None
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def process_images_batch(self, image_paths, output_dir):
        """
        Process multiple images and save results.
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Directory to save results
            
        Returns:
            dict: Dictionary mapping image names to their results
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nProcessing image {i}/{len(image_paths)}")
            
            # Get image name for output file
            image_name = Path(image_path).stem
            
            # Process the image
            ocr_result = self.process_image(image_path)
            
            if ocr_result:
                # Save result as JSON file
                result_filename = f"{image_name}_ocr_result.json"
                result_path = os.path.join(output_dir, result_filename)
                
                try:
                    # Try to parse as JSON to validate
                    parsed_result = json.loads(ocr_result)
                    
                    # Save formatted JSON
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_result, f, indent=2, ensure_ascii=False)
                    
                    print(f"Saved result to: {result_path}")
                    results[image_name] = parsed_result
                    
                except json.JSONDecodeError:
                    # If not valid JSON, save as text
                    text_filename = f"{image_name}_ocr_result.txt"
                    text_path = os.path.join(output_dir, text_filename)
                    
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_result)
                    
                    print(f"Saved text result to: {text_path}")
                    results[image_name] = ocr_result
            else:
                print(f"Failed to process image: {image_path}")
        
        return results


def process_page_images(pages_dir, results_dir, model_path="./weights/DotsOCR"):
    """
    Process all page images in the pages directory.
    
    Args:
        pages_dir (str): Directory containing page images
        results_dir (str): Directory to save OCR results
        model_path (str): Path to DotsOCR model
    """
    processor = DotsOCRProcessor(model_path)
    
    # Load the model
    if not processor.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    image_files = []
    
    if os.path.exists(pages_dir):
        for root, dirs, files in os.walk(pages_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {pages_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process all images
    results = processor.process_images_batch(image_files, results_dir)
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {len(results)} images")
    print(f"Results saved in: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Process images with DotsOCR')
    parser.add_argument('--pages-dir', type=str, default='pages', help='Directory containing page images')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save OCR results')
    parser.add_argument('--model-path', type=str, default='./weights/DotsOCR', help='Path to DotsOCR model')
    parser.add_argument('--image', type=str, help='Process a single image file')
    
    args = parser.parse_args()
    
    if args.image:
        # Process single image
        processor = DotsOCRProcessor(args.model_path)
        if processor.load_model():
            result = processor.process_image(args.image)
            if result:
                print("OCR Result:")
                print(result)
    else:
        # Process all images in pages directory
        process_page_images(args.pages_dir, args.results_dir, args.model_path)


if __name__ == "__main__":
    main()