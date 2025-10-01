#!/usr/bin/env python3
"""
Demo script to showcase the DotsOCR pipeline setup and usage.
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Check if the setup is complete and ready to use."""
    print("🔍 Checking DotsOCR Pipeline Setup")
    print("="*50)
    
    # Check directories
    dirs_to_check = ['pdfs', 'pages', 'results', 'weights']
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
    
    # Check scripts
    scripts_to_check = ['main.py', 'pdf_to_images.py', 'process_ocr.py', 'setup_model.py']
    for script in scripts_to_check:
        if os.path.exists(script):
            print(f"✓ Script exists: {script}")
        else:
            print(f"❌ Script missing: {script}")
    
    # Check if model is downloaded
    model_path = "./weights/DotsOCR"
    if os.path.exists(model_path):
        model_files = os.listdir(model_path)
        if model_files:
            print(f"✓ Model directory exists with {len(model_files)} files")
        else:
            print(f"❌ Model directory is empty")
    else:
        print(f"❌ Model not downloaded: {model_path}")
    
    # Check for test PDFs
    if os.path.exists('pdfs'):
        pdf_files = [f for f in os.listdir('pdfs') if f.endswith('.pdf')]
        if pdf_files:
            print(f"✓ Found {len(pdf_files)} PDF files: {pdf_files}")
        else:
            print(f"ℹ️  No PDF files found in pdfs/ directory")
    
    print("\n" + "="*50)

def show_usage():
    """Show usage instructions."""
    print("\n📖 Usage Instructions")
    print("="*50)
    
    print("\n🚀 Quick Start:")
    print("1. Download the DotsOCR model:")
    print("   python setup_model.py")
    print("\n2. Place PDF files in the pdfs/ directory")
    print("\n3. Run the complete pipeline:")
    print("   python main.py")
    
    print("\n🔧 Individual Components:")
    print("• Convert PDFs to images only:")
    print("  python pdf_to_images.py")
    print("\n• Process images with OCR only:")
    print("  python process_ocr.py")
    print("\n• Process a single PDF:")
    print("  python main.py --pdf path/to/document.pdf")
    
    print("\n📁 Output:")
    print("• Page images: pages/document_name/")
    print("• OCR results: results/document_name/")
    
    print("\n⚙️  Configuration:")
    print("• Change DPI: --dpi 600")
    print("• Custom directories: --pdfs-dir, --pages-dir, --results-dir")
    print("• Custom model path: --model-path")

def show_next_steps():
    """Show what to do next."""
    print("\n🎯 Next Steps")
    print("="*50)
    
    model_exists = os.path.exists("./weights/DotsOCR")
    pdfs_exist = os.path.exists("pdfs") and any(f.endswith('.pdf') for f in os.listdir('pdfs'))
    
    if not model_exists:
        print("1. 📥 Download the DotsOCR model:")
        print("   python setup_model.py")
        print("   (This will download ~10GB of model weights)")
    else:
        print("✓ DotsOCR model is ready")
    
    if not pdfs_exist:
        print("2. 📄 Add PDF files:")
        print("   • Place your PDF files in the pdfs/ directory")
        print("   • Or create a test PDF: python create_test_pdf.py")
    else:
        print("✓ PDF files are ready")
    
    if model_exists and pdfs_exist:
        print("3. 🚀 Run the pipeline:")
        print("   python main.py")
        print("\n   This will:")
        print("   • Convert PDFs to high-resolution images")
        print("   • Apply DotsOCR for structured text extraction")
        print("   • Save results as JSON files with layout information")
    
    print("\n💡 Tips:")
    print("• Use higher DPI (600-1200) for better OCR accuracy")
    print("• GPU processing is much faster than CPU")
    print("• Results include bounding boxes, categories, and formatted text")

def main():
    print("🔥 DotsOCR Pipeline Demo")
    print("="*60)
    print("Advanced OCR for PDF documents using transformer models")
    print("Model: rednote-hilab/dots.ocr")
    print("="*60)
    
    check_setup()
    show_usage()
    show_next_steps()
    
    print("\n🎉 Setup Complete!")
    print("The DotsOCR pipeline is ready for PDF processing.")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()