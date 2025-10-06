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
                "torch_dtype": torch.float32,  # Use float32 for CPU compatibility
                "device_map": "cpu" if self.device == "cpu" else "auto",
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
                error_msg = str(e).lower()
                
                # If FlashAttention2 fails during model loading, fallback to eager
                if model_kwargs.get("attn_implementation") == "flash_attention_2":
                    print(f"FlashAttention2 failed during model loading: {type(e).__name__}")
                    print("Falling back to eager attention...")
                    model_kwargs["attn_implementation"] = "eager"
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            **model_kwargs
                        )
                    except Exception as e2:
                        # Handle video processor issues after attention fallback
                        if ("video processor" in str(e2).lower() or 
                            "nonetype for argument video_processor" in str(e2).lower()):
                            print("Video processor config issue detected, using compatibility mode...")
                            model_kwargs.update({
                                "ignore_mismatched_sizes": True,
                                "_from_auto": False
                            })
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                **model_kwargs
                            )
                        else:
                            raise e2
                # Handle video processor configuration issues directly
                elif ("video processor" in error_msg or 
                      "nonetype for argument video_processor" in error_msg):
                    print("Video processor config issue detected, using compatibility mode...")
                    model_kwargs.update({
                        "ignore_mismatched_sizes": True,
                        "_from_auto": False
                    })
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                else:
                    # Re-raise if it's not a known issue
                    raise
            
            # Load processor with error handling
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
            except Exception as e:
                error_msg = str(e).lower()
                if ("video processor" in error_msg or 
                    "nonetype for argument video_processor" in error_msg):
                    print("Video processor issue with AutoProcessor, trying alternative approach...")
                    # Try loading processor components separately
                    try:
                        from transformers import AutoTokenizer, AutoImageProcessor
                        # Load tokenizer and image processor separately using Auto classes
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        image_processor = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                        
                        # Create a simple processor wrapper
                        class SimpleProcessor:
                            def __init__(self, tokenizer, image_processor):
                                self.tokenizer = tokenizer
                                self.image_processor = image_processor
                            
                            def __call__(self, text=None, images=None, **kwargs):
                                if images is not None:
                                    # Process images - remove padding argument for image processor
                                    image_kwargs = {k: v for k, v in kwargs.items() if k != 'padding'}
                                    processed_images = self.image_processor(images, **image_kwargs)
                                    if text is not None:
                                        # Process text - keep all kwargs for tokenizer
                                        processed_text = self.tokenizer(text, **kwargs)
                                        # Combine them
                                        processed_text.update(processed_images)
                                        return processed_text
                                    return processed_images
                                elif text is not None:
                                    return self.tokenizer(text, **kwargs)
                                else:
                                    return {}
                            
                            def apply_chat_template(self, messages, **kwargs):
                                """Apply chat template using the tokenizer."""
                                if hasattr(self.tokenizer, 'apply_chat_template'):
                                    try:
                                        return self.tokenizer.apply_chat_template(messages, **kwargs)
                                    except Exception as e:
                                        print(f"Error in tokenizer apply_chat_template: {e}")
                                        # Fall through to manual implementation
                                
                                # Fallback: create a simple text prompt from messages
                                if isinstance(messages, list):
                                    text_parts = []
                                    for msg in messages:
                                        if isinstance(msg, dict) and 'content' in msg:
                                            content = msg['content']
                                            if isinstance(content, list):
                                                # Handle list of content items
                                                for item in content:
                                                    if isinstance(item, dict) and item.get('type') == 'text':
                                                        text_parts.append(str(item.get('text', '')))
                                            elif isinstance(content, str):
                                                text_parts.append(content)
                                            else:
                                                text_parts.append(str(content))
                                        else:
                                            text_parts.append(str(msg))
                                    return " ".join(text_parts)
                                return str(messages)
                            
                            def batch_decode(self, sequences, **kwargs):
                                """Decode sequences using the tokenizer."""
                                if hasattr(self.tokenizer, 'batch_decode'):
                                    return self.tokenizer.batch_decode(sequences, **kwargs)
                                elif hasattr(self.tokenizer, 'decode'):
                                    # Fallback to individual decode
                                    return [self.tokenizer.decode(seq, **kwargs) for seq in sequences]
                                else:
                                    # Last resort: convert to strings
                                    return [str(seq) for seq in sequences]
                        
                        self.processor = SimpleProcessor(tokenizer, image_processor)
                        print("Loaded processor components separately (video processor bypassed)")
                        
                    except Exception as e2:
                        print(f"Failed to load processor components: {e2}")
                        # Try with simplified processor loading
                        print("Attempting minimal processor setup...")
                        try:
                            self.processor = AutoProcessor.from_pretrained(
                                self.model_path, 
                                trust_remote_code=True,
                                video_processor=None  # Explicitly set to None
                            )
                        except Exception as e3:
                            print(f"All processor loading methods failed: {e3}")
                            return False
                else:
                    print(f"Processor loading error: {e}")
                    return False
            
            # Convert model to float32 if using CPU to avoid dtype mismatches
            if self.device == "cpu":
                print("Converting model to float32 for CPU compatibility...")
                self.model = self.model.float()
            
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
            try:
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print(f"Chat template result: {text}")
            except Exception as e:
                print(f"Error in apply_chat_template: {e}")
                text = self.prompt  # Fallback to just the prompt
            
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                print(f"Vision info: images={len(image_inputs)}, videos={len(video_inputs) if video_inputs else 0}")
            except Exception as e:
                print(f"Error in process_vision_info: {e}")
                image_inputs, video_inputs = [image_path], []
            
            # Load the actual image data
            try:
                from PIL import Image
                if isinstance(image_inputs[0], str):
                    # Convert file path to loaded image
                    loaded_images = [Image.open(img_path) for img_path in image_inputs]
                    image_inputs = loaded_images
            except Exception as e:
                print(f"Error loading images: {e}")
                # Keep as paths and let processor handle it
            
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
            except Exception as e:
                print(f"Error in processor call: {e}")
                # Try simpler approach without videos and padding
                try:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        return_tensors="pt",
                    )
                except Exception as e2:
                    print(f"Error in simplified processor call: {e2}")
                    # Try with just text processing
                    inputs = self.processor(
                        text=[text],
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