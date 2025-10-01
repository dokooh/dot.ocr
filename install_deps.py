#!/usr/bin/env python3
"""
Install dependencies with fallback handling for optional packages.
"""

import subprocess
import sys

def install_package(package_name, optional=False):
    """Install a Python package with error handling."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"⚠️  Optional package {package_name} failed to install (this is OK)")
            return False
        else:
            print(f"❌ Failed to install required package {package_name}: {e}")
            return False

def main():
    print("Installing DotsOCR Pipeline Dependencies")
    print("="*50)
    
    # Required packages
    required_packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "transformers>=4.30.0",
        "PyMuPDF>=1.23.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "huggingface-hub>=0.16.0",
        "reportlab"  # For creating test PDFs
    ]
    
    # Optional packages that might not be available
    optional_packages = [
        "qwen-vl-utils",
        "dots-ocr"
    ]
    
    failed_required = []
    
    print("\nInstalling required packages...")
    for package in required_packages:
        if not install_package(package):
            failed_required.append(package)
    
    print("\nInstalling optional packages...")
    for package in optional_packages:
        install_package(package, optional=True)
    
    print("\n" + "="*50)
    if failed_required:
        print("❌ Some required packages failed to install:")
        for package in failed_required:
            print(f"  - {package}")
        print("\nPlease install these manually or check your Python environment.")
        return False
    else:
        print("✅ All required packages installed successfully!")
        print("\nOptional packages may have failed - this is normal.")
        print("The pipeline will work with fallback implementations.")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)