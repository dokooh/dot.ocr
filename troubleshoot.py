#!/usr/bin/env python3
"""
Troubleshooting script for DotsOCR pipeline.
Checks dependencies and provides fixes for common issues.
"""

import sys
import os
import importlib
import subprocess


def check_package(package_name, optional=False):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True, "✓"
    except ImportError:
        return False, "❌" if not optional else "⚠️"


def check_dependencies():
    """Check all dependencies."""
    print("🔍 Checking Dependencies")
    print("="*40)
    
    # Core dependencies
    core_deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fitz", "PyMuPDF"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("huggingface_hub", "Hugging Face Hub")
    ]
    
    # Optional dependencies
    optional_deps = [
        ("qwen_vl_utils", "Qwen VL Utils"),
        ("dots_ocr", "DotsOCR Utils")
    ]
    
    missing_core = []
    
    print("\nCore Dependencies:")
    for module, name in core_deps:
        available, status = check_package(module)
        print(f"  {status} {name}")
        if not available:
            missing_core.append((module, name))
    
    print("\nOptional Dependencies:")
    for module, name in optional_deps:
        available, status = check_package(module, optional=True)
        note = "" if available else " (fallback available)"
        print(f"  {status} {name}{note}")
    
    return missing_core


def check_files():
    """Check if required files exist."""
    print("\n🔍 Checking Files")
    print("="*40)
    
    files_to_check = [
        ("main.py", "Main pipeline script"),
        ("pdf_to_images.py", "PDF converter"),
        ("process_ocr.py", "OCR processor"),
        ("process_ocr_simple.py", "Simple OCR processor"),
        ("setup_model.py", "Model downloader"),
        ("pdfs/", "PDFs directory"),
        ("pages/", "Pages directory"),
        ("results/", "Results directory"),
        ("weights/", "Weights directory")
    ]
    
    for filepath, description in files_to_check:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "❌"
        print(f"  {status} {description}: {filepath}")


def check_model():
    """Check if the DotsOCR model is available."""
    print("\n🔍 Checking Model")
    print("="*40)
    
    model_path = "./weights/DotsOCR"
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        if files:
            print(f"  ✓ Model directory exists with {len(files)} files")
            
            # Check for key model files
            key_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            for key_file in key_files:
                exists = os.path.exists(os.path.join(model_path, key_file))
                status = "✓" if exists else "❌"
                print(f"    {status} {key_file}")
                
        else:
            print("  ❌ Model directory is empty")
    else:
        print("  ❌ Model directory does not exist")


def check_gpu():
    """Check GPU availability."""
    print("\n🔍 Checking GPU")
    print("="*40)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"  ✓ CUDA available: {gpu_count} GPU(s)")
            print(f"  ✓ Primary GPU: {gpu_name}")
        else:
            print("  ⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("  ❌ PyTorch not installed")


def provide_solutions(missing_core):
    """Provide solutions for common issues."""
    print("\n🛠️  Solutions")
    print("="*40)
    
    if missing_core:
        print("\n📦 Missing Core Dependencies:")
        print("Run one of these commands:")
        print("  python install_deps.py")
        print("  pip install -r requirements.txt")
        print("  pip install torch transformers PyMuPDF Pillow numpy huggingface-hub")
    
    print("\n🤖 Model Download:")
    print("If model is missing:")
    print("  python setup_model.py")
    
    print("\n🔧 Common Fixes:")
    print("1. Module import errors:")
    print("   - Use: python process_ocr_simple.py (fallback version)")
    print("   - Or: python install_deps.py")
    
    print("\n2. Memory issues:")
    print("   - Force CPU: set CUDA_VISIBLE_DEVICES=\"\"")
    print("   - Or add: export CUDA_VISIBLE_DEVICES=\"\"")
    
    print("\n3. Kaggle/Colab issues:")
    print("   - Use simplified processor: process_ocr_simple.py")
    print("   - Install core deps only: pip install torch transformers PyMuPDF Pillow")


def main():
    print("🔧 DotsOCR Pipeline Troubleshooting")
    print("="*50)
    
    missing_core = check_dependencies()
    check_files()
    check_model()
    check_gpu()
    provide_solutions(missing_core)
    
    print("\n" + "="*50)
    if missing_core:
        print("❌ Issues found - see solutions above")
    else:
        print("✅ Core setup looks good!")
        print("\nNext steps:")
        print("1. Download model: python setup_model.py")
        print("2. Add PDFs to pdfs/ directory")
        print("3. Run pipeline: python main.py")


if __name__ == "__main__":
    main()