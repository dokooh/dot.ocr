#!/usr/bin/env python3
"""
DotsOCR Processing Script - Updated with improved error handling
"""

import os
import json
import argparse
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Optional imports with fallbacks
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not available, using fallback implementation")
    def process_vision_info(messages):
        """Fallback implementation for process_vision_info"""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                for content in message["content"]:
                    if isinstance(content, dict):
                        if content.get("type") == "image":
                            image_path = content.get("image")
                            if image_path:
                                try:
                                    image = Image.open(image_path)
                                    image_inputs.append(image)
                                except Exception as e:
                                    print(f"Error loading image {image_path}: {e}")
        
        return image_inputs, video_inputs


class ImprovedSimpleProcessor:
    """Improved processor wrapper with better error handling"""
    
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.chat_template = None
        
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Apply chat template with improved error handling"""
        try:
            # Try the tokenizer's chat template first
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    return self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=tokenize, 
                        add_generation_prompt=add_generation_prompt
                    )
                except Exception as e:
                    print(f"Tokenizer chat template failed: {e}")
            
            # Fallback to simple template
            return self._simple_chat_template(messages, add_generation_prompt)
            
        except Exception as e:
            print(f"Error in apply_chat_template: {e}")
            # Ultimate fallback - just extract text
            text_content = ""
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    for content in message["content"]:
                        if isinstance(content, dict) and content.get("type") == "text":
                            text_content += content.get("text", "") + " "
            return text_content.strip()
    
    def _simple_chat_template(self, messages, add_generation_prompt=True):
        """Simple fallback chat template"""
        formatted_text = ""
        
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", [])
                
                formatted_text += f"<|im_start|>{role}\n"
                
                # Process content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            formatted_text += item.get("text", "") + "\n"
                elif isinstance(content, str):
                    formatted_text += content + "\n"
                
                formatted_text += "<|im_end|>\n"
        
        if add_generation_prompt:
            formatted_text += "<|im_start|>assistant\n"
        
        return formatted_text
    
    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt"):
        """Process inputs with improved error handling"""
        try:
            # Remove unsupported parameters and use only supported ones
            kwargs = {}
            
            if text is not None:
                kwargs["text"] = text
            if images is not None:
                kwargs["images"] = images
            # Skip videos parameter as it's not supported
            if return_tensors:
                kwargs["return_tensors"] = return_tensors
                
            # Try without padding first, then add if supported
            try:
                result = self.processor(**kwargs)
            except TypeError as e:
                if "padding" in str(e):
                    # Remove padding and try again
                    print("Note: Processor doesn't support padding parameter, proceeding without it")
                    result = self.processor(**kwargs)
                else:
                    raise e
            
            return result
            
        except Exception as e:
            print(f"Error in processor call: {e}")
            # Return a minimal structure
            if text and isinstance(text, list):
                text_str = text[0] if text else ""
            else:
                text_str = str(text) if text else ""
                
            encoded = self.tokenizer(text_str, return_tensors=return_tensors)
            return encoded
    
    def batch_decode(self, sequences, **kwargs):
        """Decode token sequences"""
        return self.tokenizer.batch_decode(sequences, **kwargs)


class DotsOCRProcessor:
    def __init__(self, model_path="./weights/DotsOCR"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load DotsOCR model with comprehensive error handling"""
        try:
            print("Loading DotsOCR model...")
            
            # Determine the best dtype and device settings
            if self.device == "cuda":
                torch_dtype = torch.float16  # Use float16 for GPU
                device_map = "auto"
            else:
                torch_dtype = torch.float32  # Use float32 for CPU
                device_map = None
            
            # Try loading with different attention implementations
            attention_configs = [
                {"attn_implementation": "eager", "torch_dtype": torch_dtype},
                {"torch_dtype": torch_dtype},  # No attention implementation specified
                {"attn_implementation": "sdpa", "torch_dtype": torch_dtype},
            ]
            
            model_loaded = False
            
            for i, config in enumerate(attention_configs):
                try:
                    print(f"Trying model loading configuration {i+1}/3...")
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map=device_map,
                        trust_remote_code=True,
                        **config
                    )
                    
                    # For CPU, ensure all parameters are float32
                    if self.device == "cpu":
                        print("Converting model to float32 for CPU compatibility...")
                        self.model = self.model.float()
                    
                    model_loaded = True
                    print(f"✓ Model loaded successfully with configuration {i+1}")
                    break;
                    
                except Exception as e:
                    print(f"Configuration {i+1} failed: {str(e)[:100]}...")
                    if i == len(attention_configs) - 1:
                        raise e
            
            if not model_loaded:
                raise Exception("Failed to load model with any configuration")
            
            # Load processor with error handling
            try:
                print("Loading processor...")
                processor = AutoProcessor.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                
                # Try to load tokenizer separately
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True
                    )
                except:
                    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                
                # Use improved wrapper
                self.processor = ImprovedSimpleProcessor(processor, tokenizer)
                print("✓ Processor loaded successfully")
                
            except Exception as e:
                print(f"Error loading processor: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def process_image(self, image_path):
        """Process a single image with DotsOCR"""
        if not self.model or not self.processor:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            print(f"Processing image: {image_path}")
            
            # Define the prompt
            prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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

            # Prepare messages
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            try:
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print("✓ Chat template applied successfully")
            except Exception as e:
                print(f"Error in apply_chat_template: {e}")
                text = prompt  # Fallback to just the prompt
            
            # Process vision info
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                print("✓ Vision info processed successfully")
            except Exception as e:
                print(f"Error in process_vision_info: {e}")
                # Fallback: load image directly
                try:
                    image_inputs = [Image.open(image_path)]
                    video_inputs = []
                except Exception as e2:
                    print(f"Error loading image directly: {e2}")
                    return None
            
            # Prepare inputs for the model
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    return_tensors="pt",
                )
                print("✓ Inputs prepared successfully")
            except Exception as e:
                print(f"Error in processor call: {e}")
                # Try simplified approach
                try:
                    inputs = self.processor(
                        text=[text] if isinstance(text, str) else text,
                        images=image_inputs,
                        return_tensors="pt"
                    )
                    print("✓ Simplified processor call succeeded")
                except Exception as e2:
                    print(f"Error in simplified processor call: {e2}")
                    return None
            
            # Move to device
            try:
                inputs = inputs.to(self.device)
                print(f"✓ Inputs moved to {self.device}")
            except Exception as e:
                print(f"Warning: Could not move inputs to {self.device}: {e}")
            
            # Generate
            print("Generating OCR output...")
            try:
                with torch.no_grad():  # Save memory
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=24000,
                        do_sample=False,  # Use greedy decoding for consistency
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                print("✓ Generation completed successfully")
            except Exception as e:
                print(f"Error during generation: {e}")
                return None
            
            # Decode output
            try:
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_images_batch(self, image_paths, output_dir):
        """Process multiple images and save results"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n--- Processing image {i}/{len(image_paths)} ---")
            
            result = self.process_image(image_path)
            
            if result:
                # Save result
                image_name = Path(image_path).stem
                output_file = os.path.join(output_dir, f"{image_name}_ocr.json")
                
                try:
                    # Try to parse as JSON first
                    try:
                        json_result = json.loads(result)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(json_result, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # Save as text if not valid JSON
                        output_file = output_file.replace('.json', '.txt')
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(result)
                    
                    print(f"✓ Result saved: {output_file}")
                    results.append(output_file)
                    
                except Exception as e:
                    print(f"Error saving result: {e}")
            else:
                print(f"Failed to process image: {image_path}")
        
        return results


def process_page_images(pages_dir, results_dir, model_path="./weights/DotsOCR"):
    """
    Process all page images in the pages directory.
    """
    processor = DotsOCRProcessor(model_path)
    
    # Load the model
    if not processor.load_model():
        print("Failed to load model. Exiting.")
        return []
    
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
        return []
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process all images
    results = processor.process_images_batch(image_files, results_dir)
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {len(results)} images")
    print(f"Results saved in: {results_dir}")
    
    return results


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
                print("Failed to process image")
    else:
        # Process all images in directory
        process_page_images(args.pages_dir, args.results_dir, args.model_path)


if __name__ == "__main__":
    main()