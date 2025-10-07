#!/usr/bin/env python3
"""
Test script to reproduce the video processor configuration error
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

def test_processor_loading():
    model_path = "./weights/DotsOCR"
    
    print("Testing processor loading...")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    try:
        print("\n1. Attempting AutoProcessor loading...")
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print("✓ AutoProcessor loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ AutoProcessor failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check for video processor related errors
        error_str = str(e).lower()
        video_errors = [
            "video processor", "video_processor", "basevideoprocessor",
            "unrecognized video processor", "video_processor_type",
            "video_preprocessor_config.json", "model_type"
        ]
        
        matched_patterns = [pattern for pattern in video_errors if pattern in error_str]
        if matched_patterns:
            print(f"✓ Video processor error detected (patterns: {matched_patterns})")
            
            try:
                print("\n2. Attempting tokenizer-only fallback...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print("✓ Tokenizer loaded successfully!")
                print("✓ Fallback approach should work")
                return True
            except Exception as e2:
                print(f"❌ Tokenizer fallback also failed: {e2}")
                return False
        else:
            print("❌ Non-video processor error, fallback may not help")
            return False

if __name__ == "__main__":
    success = test_processor_loading()
    if success:
        print("\n🎉 Processor loading test completed - pipeline should work!")
    else:
        print("\n❌ Processor loading test failed - pipeline may have issues")
    
    sys.exit(0 if success else 1)