#!/usr/bin/env python3
"""
DotsOCR Processing Script - Transformer-based Implementation
Applies DotsOCR transformer to page images and saves OCR results.
"""

import os
import sys
import json
import torch
from pathlib import Path
import argparse
from PIL import Image
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
        
        if messages is None:
            return image_inputs, video_inputs
            
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                content_list = message["content"]
                if content_list is None:
                    continue
                for content in content_list:
                    if isinstance(content, dict):
                        if content.get("type") == "image" and "image" in content:
                            # Load image as PIL Image for processing
                            image_path = content["image"]
                            try:
                                if isinstance(image_path, str):
                                    image = Image.open(image_path).convert('RGB')
                                    image_inputs.append(image)
                                else:
                                    image_inputs.append(image_path)
                            except Exception as e:
                                print(f"Error loading image {image_path}: {e}")
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
        
        # Check GPU memory and set device accordingly
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"Available GPU memory: {gpu_memory:.1f} GB")
            if gpu_memory < 8:  # Less than 8GB, use CPU to avoid OOM
                print("GPU memory insufficient for this large model, using CPU")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = "cpu"
        
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
        """Load the DotsOCR model and processor with comprehensive error handling."""
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                print(f"Model path not found: {self.model_path}")
                print("You need to download the DotsOCR model weights.")
                print("Please visit: https://huggingface.co/rednote-hilab/dots.ocr")
                return False
            
            # Determine model configuration based on device
            model_kwargs = {
                "trust_remote_code": True
            }
            
            # Configure dtype and device mapping
            if self.device == "cuda":
                # Try different configurations for GPU
                configurations = [
                    {
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": torch.bfloat16,
                        "device_map": "auto"
                    },
                    {
                        "attn_implementation": "eager",
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    },
                    {
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    }
                ]
            else:
                # CPU configurations
                configurations = [
                    {
                        "attn_implementation": "eager",
                        "torch_dtype": torch.float32,
                        "device_map": None
                    },
                    {
                        "torch_dtype": torch.float32,
                        "device_map": None
                    }
                ]
            
            # Try loading model with different configurations
            model_loaded = False
            for i, config in enumerate(configurations):
                try:
                    print(f"Trying model configuration {i+1}/{len(configurations)}...")
                    config.update(model_kwargs)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **config
                    )
                    
                    # For CPU, ensure all parameters are float32
                    if self.device == "cpu":
                        print("Converting model to float32 for CPU compatibility...")
                        self.model = self.model.float()
                    
                    model_loaded = True
                    print(f"✓ Model loaded successfully with configuration {i+1}")
                    break
                    
                except Exception as e:
                    print(f"Configuration {i+1} failed: {str(e)[:100]}...")
                    # Handle specific errors
                    if "video processor" in str(e).lower():
                        print("Video processor issue detected, trying alternative approach...")
                        continue
                    if i == len(configurations) - 1:
                        print("All configurations failed")
                        break
            
            if not model_loaded:
                print("Failed to load model with any configuration")
                return False
            
            # Load processor with error handling
            try:
                print("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                print("✓ Processor loaded successfully")
                
            except Exception as e:
                print(f"Error loading processor: {e}")
                print(f"Error type: {type(e).__name__}")
                
                # Try alternative processor loading approaches
                error_patterns = [
                    "video processor", "video_processor", "basevideoprocessor",
                    "unrecognized video processor", "video_processor_type",
                    "video_preprocessor_config.json", "model_type"
                ]
                
                if any(pattern in str(e).lower() for pattern in error_patterns):
                    print("Video processor issue detected, trying alternative approach...")
                    print(f"Matched error pattern in: {str(e)[:200]}...")
                    try:
                        # Try loading components separately
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        
                        # Create a minimal processor wrapper
                        class MinimalProcessor:
                            def __init__(self, tokenizer):
                                self.tokenizer = tokenizer
                            
                            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                                if hasattr(self.tokenizer, 'apply_chat_template'):
                                    try:
                                        return self.tokenizer.apply_chat_template(
                                            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
                                        )
                                    except Exception:
                                        pass
                                
                                # Fallback template
                                text = ""
                                for message in messages:
                                    if isinstance(message, dict) and "content" in message:
                                        for content in message["content"]:
                                            if isinstance(content, dict) and content.get("type") == "text":
                                                text += content.get("text", "") + " "
                                
                                if add_generation_prompt:
                                    text += "<|im_start|>assistant\n"
                                
                                return text.strip()
                            
                            def __call__(self, text=None, images=None, videos=None, **kwargs):
                                # Simple tokenization
                                if isinstance(text, list):
                                    text = text[0] if text else ""
                                
                                # Extract return_tensors to avoid duplicate argument
                                return_tensors = kwargs.pop('return_tensors', 'pt')
                                
                                return self.tokenizer(text, return_tensors=return_tensors, **kwargs)
                            
                            def batch_decode(self, sequences, **kwargs):
                                return self.tokenizer.batch_decode(sequences, **kwargs)
                        
                        self.processor = MinimalProcessor(tokenizer)
                        print("✓ Minimal processor created (video processor bypassed)")
                        
                    except Exception as e2:
                        print(f"Failed to create minimal processor: {e2}")
                        print("❌ Failed to load any processor")
                        return False
                else:
                    print("❌ Failed to load any processor")
                    return False
            
            print("Model and processor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def process_image(self, image_path):
        """
        Process a single image with DotsOCR using the transformer-based approach.
        
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
            
            # Prepare messages for the model (following the transformer example)
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
            
            # Preparation for inference (following the transformer example)
            try:
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print("✓ Chat template applied successfully")
            except Exception as e:
                print(f"Error in apply_chat_template: {e}")
                text = self.prompt  # Fallback to just the prompt
            
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                print(f"✓ Vision info processed: {len(image_inputs)} images, {len(video_inputs)} videos")
            except Exception as e:
                print(f"Error in process_vision_info: {e}")
                # Fallback: load image directly
                try:
                    image_inputs = [Image.open(image_path).convert('RGB')]
                    video_inputs = []
                except Exception as e2:
                    print(f"Error loading image directly: {e2}")
                    return None
            
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                print("✓ Inputs prepared successfully")
            except Exception as e:
                print(f"Error in processor call: {e}")
                # Try without videos and padding
                try:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        return_tensors="pt",
                    )
                    print("✓ Simplified inputs prepared successfully")
                except Exception as e2:
                    print(f"Error in simplified processor call: {e2}")
                    return None
            
            # Move inputs to device
            try:
                inputs = inputs.to(self.device)
                print(f"✓ Inputs moved to {self.device}")
            except Exception as e:
                print(f"Warning: Could not move inputs to device: {e}")
            
            # Inference: Generation of the output (following the transformer example)
            print("Generating OCR output...")
            try:
                with torch.no_grad():  # Save memory
                    generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
                print("✓ Generation completed successfully")
            except Exception as e:
                print(f"Error during generation: {e}")
                return None
            
            # Decode output (following the transformer example)
            try:
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                print("✓ Output decoded successfully")
                return output_text[0] if output_text else None
                
            except Exception as e:
                print(f"Error decoding output: {e}")
                return None
            
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
            print(f"\n--- Processing image {i}/{len(image_paths)} ---")
            
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
                    
                    print(f"✓ JSON result saved to: {result_path}")
                    results[image_name] = parsed_result
                    
                except json.JSONDecodeError:
                    # If not valid JSON, save as text
                    text_filename = f"{image_name}_ocr_result.txt"
                    text_path = os.path.join(output_dir, text_filename)
                    
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_result)
                    
                    print(f"✓ Text result saved to: {text_path}")
                    results[image_name] = ocr_result
            else:
                print(f"❌ Failed to process image: {image_path}")
        
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
    
    print(f"\n🎉 Processing completed!")
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
                print("\n🎉 OCR Result:")
                print(result)
            else:
                print("❌ Failed to process image")
    else:
        # Process all images in pages directory
        process_page_images(args.pages_dir, args.results_dir, args.model_path)


if __name__ == "__main__":
    main()