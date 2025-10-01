#!/usr/bin/env python3
"""
Simplified DotsOCR Processing Script
Works without qwen-vl-utils dependency using basic transformers functionality.
"""

import os
import sys
import json
import torch
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


class SimpleDotsOCRProcessor:
    def __init__(self, model_path="./weights/DotsOCR"):
        """
        Initialize the simplified DotsOCR processor.
        
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
                print("Run: python setup_model.py")
                return False
            
            # Load model with basic settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            if self.device == "cuda" and hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            print("Model and processor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you downloaded the model: python setup_model.py")
            print("2. Check if you have enough GPU/RAM memory")
            print("3. Try using CPU by setting CUDA_VISIBLE_DEVICES=\"\"")
            return False
    
    def process_image(self, image_path):
        """
        Process a single image with DotsOCR using simplified approach.
        
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
            
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare text prompt
            prompt = f"<|im_start|>user\n<image>\n{self.prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Process inputs using the processor
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            print("Generating OCR output...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4000,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return generated_text.strip()
            
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
                # Try to save as JSON first
                result_filename = f"{image_name}_ocr_result.json"
                result_path = os.path.join(output_dir, result_filename)
                
                try:
                    # Try to parse as JSON to validate
                    parsed_result = json.loads(ocr_result)
                    
                    # Save formatted JSON
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_result, f, indent=2, ensure_ascii=False)
                    
                    print(f"Saved JSON result to: {result_path}")
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


def process_page_images_simple(pages_dir, results_dir, model_path="./weights/DotsOCR"):
    """
    Process all page images using the simplified processor.
    """
    processor = SimpleDotsOCRProcessor(model_path)
    
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
    parser = argparse.ArgumentParser(description='Process images with simplified DotsOCR')
    parser.add_argument('--pages-dir', type=str, default='pages', help='Directory containing page images')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save OCR results')
    parser.add_argument('--model-path', type=str, default='./weights/DotsOCR', help='Path to DotsOCR model')
    parser.add_argument('--image', type=str, help='Process a single image file')
    
    args = parser.parse_args()
    
    if args.image:
        # Process single image
        processor = SimpleDotsOCRProcessor(args.model_path)
        if processor.load_model():
            result = processor.process_image(args.image)
            if result:
                print("OCR Result:")
                print(result)
    else:
        # Process all images in pages directory
        process_page_images_simple(args.pages_dir, args.results_dir, args.model_path)


if __name__ == "__main__":
    main()