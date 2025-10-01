#!/usr/bin/env python3
"""
DotsOCR Pipeline Main Script
Orchestrates the complete pipeline from PDF to OCR results.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Import our custom modules
from pdf_to_images import process_all_pdfs, convert_pdf_to_images
from process_ocr import process_page_images


def run_complete_pipeline(pdfs_dir="pdfs", pages_dir="pages", results_dir="results", 
                         model_path="./weights/DotsOCR", dpi=300):
    """
    Run the complete PDF to OCR pipeline.
    
    Args:
        pdfs_dir (str): Directory containing PDF files
        pages_dir (str): Directory to save page images
        results_dir (str): Directory to save OCR results
        model_path (str): Path to DotsOCR model
        dpi (int): DPI for image conversion
    """
    print("="*60)
    print("DotsOCR Pipeline - PDF to OCR Processing")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Convert PDFs to images
    print("\n" + "="*40)
    print("Step 1: Converting PDFs to Images")
    print("="*40)
    
    if not os.path.exists(pdfs_dir):
        print(f"Error: PDFs directory does not exist: {pdfs_dir}")
        print("Please create the directory and place your PDF files there.")
        return False
    
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in {pdfs_dir}")
        print("Please place your PDF files in the pdfs directory.")
        return False
    
    pdf_to_images = process_all_pdfs(pdfs_dir, pages_dir, dpi)
    
    if not pdf_to_images:
        print("Error: Failed to convert any PDFs to images")
        return False
    
    total_pages = sum(len(images) for images in pdf_to_images.values())
    print(f"\n✓ Successfully converted {len(pdf_to_images)} PDFs to {total_pages} page images")
    
    # Step 2: Process images with DotsOCR
    print("\n" + "="*40)
    print("Step 2: Processing Images with DotsOCR")
    print("="*40)
    
    if not os.path.exists(model_path):
        print(f"\nWarning: DotsOCR model not found at: {model_path}")
        print("You need to download the DotsOCR model weights.")
        print("Please visit: https://huggingface.co/rednote-hilab/dots.ocr")
        print("\nTo download the model, you can use:")
        print("git lfs install")
        print("git clone https://huggingface.co/rednote-hilab/dots.ocr ./weights/DotsOCR")
        return False
    
    try:
        process_page_images(pages_dir, results_dir, model_path)
        print(f"\n✓ OCR processing completed successfully")
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        return False
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Pipeline Completed Successfully!")
    print("="*60)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"PDFs processed: {len(pdf_to_images)}")
    print(f"Pages processed: {total_pages}")
    print(f"Results saved in: {results_dir}")
    print("\nFiles created:")
    print(f"  - Page images: {pages_dir}/")
    print(f"  - OCR results: {results_dir}/")
    
    return True


def run_single_pdf_pipeline(pdf_path, pages_dir="pages", results_dir="results", 
                           model_path="./weights/DotsOCR", dpi=300):
    """
    Run the pipeline for a single PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        pages_dir (str): Directory to save page images
        results_dir (str): Directory to save OCR results
        model_path (str): Path to DotsOCR model
        dpi (int): DPI for image conversion
    """
    print("="*60)
    print("DotsOCR Pipeline - Single PDF Processing")
    print("="*60)
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    start_time = time.time()
    
    # Step 1: Convert PDF to images
    print("\n" + "="*40)
    print("Step 1: Converting PDF to Images")
    print("="*40)
    
    pdf_name = Path(pdf_path).stem
    pdf_pages_dir = os.path.join(pages_dir, pdf_name)
    
    images = convert_pdf_to_images(pdf_path, pdf_pages_dir, dpi)
    
    if not images:
        print("Error: Failed to convert PDF to images")
        return False
    
    print(f"✓ Successfully converted PDF to {len(images)} page images")
    
    # Step 2: Process images with DotsOCR
    print("\n" + "="*40)
    print("Step 2: Processing Images with DotsOCR")
    print("="*40)
    
    if not os.path.exists(model_path):
        print(f"\nWarning: DotsOCR model not found at: {model_path}")
        print("You need to download the DotsOCR model weights.")
        print("Please visit: https://huggingface.co/rednote-hilab/dots.ocr")
        return False
    
    try:
        pdf_results_dir = os.path.join(results_dir, pdf_name)
        process_page_images(pdf_pages_dir, pdf_results_dir, model_path)
        print(f"✓ OCR processing completed successfully")
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        return False
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Single PDF Pipeline Completed Successfully!")
    print("="*60)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"PDF: {pdf_path}")
    print(f"Pages processed: {len(images)}")
    print(f"Results saved in: {os.path.join(results_dir, pdf_name)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='DotsOCR Pipeline - Complete PDF to OCR processing')
    parser.add_argument('--pdf', type=str, help='Process a single PDF file')
    parser.add_argument('--pdfs-dir', type=str, default='pdfs', help='Directory containing PDF files')
    parser.add_argument('--pages-dir', type=str, default='pages', help='Directory to save page images')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save OCR results')
    parser.add_argument('--model-path', type=str, default='./weights/DotsOCR', help='Path to DotsOCR model')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for image conversion (default: 300)')
    
    args = parser.parse_args()
    
    if args.pdf:
        # Process single PDF
        success = run_single_pdf_pipeline(
            args.pdf, args.pages_dir, args.results_dir, args.model_path, args.dpi
        )
    else:
        # Process all PDFs in directory
        success = run_complete_pipeline(
            args.pdfs_dir, args.pages_dir, args.results_dir, args.model_path, args.dpi
        )
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()