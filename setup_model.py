#!/usr/bin/env python3
"""
Setup script to download DotsOCR model weights.
"""

import os
import sys
from pathlib import Path

def download_model():
    """Download the DotsOCR model weights."""
    model_dir = "./weights/DotsOCR"
    
    if os.path.exists(model_dir):
        print(f"Model directory already exists: {model_dir}")
        
        # Check if model files exist
        model_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing_files = [f for f in model_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if not missing_files:
            print("Model appears to be already downloaded.")
            return True
        else:
            print(f"Model directory exists but some files are missing: {missing_files}")
    
    print("Downloading DotsOCR model weights...")
    print("This may take a while (model is ~10GB)")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Create weights directory if it doesn't exist
        os.makedirs("./weights", exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id="rednote-hilab/dots.ocr",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded successfully to: {model_dir}")
        return True
        
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system("pip install huggingface_hub")
        
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="rednote-hilab/dots.ocr",
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"✓ Model downloaded successfully to: {model_dir}")
            return True
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("\nAlternative method:")
        print("You can also download using git:")
        print("  git lfs install")
        print("  git clone https://huggingface.co/rednote-hilab/dots.ocr ./weights/DotsOCR")
        return False

def main():
    print("DotsOCR Model Setup")
    print("="*50)
    
    if download_model():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now run the pipeline:")
        print("  python main.py")
    else:
        print("\n❌ Setup failed!")
        print("Please check the error messages above and try manual installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()