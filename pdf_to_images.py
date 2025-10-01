#!/usr/bin/env python3
"""
PDF to Images Converter
Converts PDF documents to individual page images for OCR processing.
"""

import os
import sys
import fitz  # PyMuPDF
from pathlib import Path
import argparse

# Check PyMuPDF version for compatibility
def get_fitz_version():
    """Get PyMuPDF version info."""
    try:
        return fitz.version[0] if hasattr(fitz, 'version') else "unknown"
    except:
        return "unknown"

def open_pdf_document(pdf_path):
    """Open PDF document with version compatibility."""
    try:
        # Use fitz.Document as primary method (works with all versions)
        return fitz.Document(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with fitz.Document: {e}")
        try:
            # Fallback to fitz.open if Document fails
            return fitz.open(pdf_path)
        except Exception as e2:
            print(f"Error opening PDF with fitz.open: {e2}")
            raise e2


def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save page images
        dpi (int): Resolution for the images (default: 300)
    
    Returns:
        list: List of saved image file paths
    """
    try:
        # Open PDF document with version compatibility
        pdf_document = open_pdf_document(pdf_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_images = []
        
        # Get PDF filename without extension for naming
        pdf_name = Path(pdf_path).stem
        
        print(f"Converting PDF: {pdf_path}")
        print(f"Number of pages: {len(pdf_document)}")
        
        # Convert each page to image
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Create transformation matrix for the desired DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            
            # Render page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Generate output filename
            image_filename = f"{pdf_name}_page_{page_num + 1}.png"
            image_path = os.path.join(output_dir, image_filename)
            
            # Save image
            pix.save(image_path)
            saved_images.append(image_path)
            
            print(f"Saved page {page_num + 1} to: {image_path}")
        
        pdf_document.close()
        print(f"Successfully converted {len(saved_images)} pages")
        
        return saved_images
        
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []


def process_all_pdfs(pdfs_dir, pages_dir, dpi=300):
    """
    Process all PDF files in the pdfs directory.
    
    Args:
        pdfs_dir (str): Directory containing PDF files
        pages_dir (str): Directory to save page images
        dpi (int): Resolution for the images
    
    Returns:
        dict: Dictionary mapping PDF names to their page image paths
    """
    pdf_to_images = {}
    
    if not os.path.exists(pdfs_dir):
        print(f"PDFs directory does not exist: {pdfs_dir}")
        return pdf_to_images
    
    # Find all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdfs_dir}")
        return pdf_to_images
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        pdf_name = Path(pdf_file).stem
        
        # Create subdirectory for this PDF's pages
        pdf_pages_dir = os.path.join(pages_dir, pdf_name)
        
        print(f"\nProcessing: {pdf_file}")
        images = convert_pdf_to_images(pdf_path, pdf_pages_dir, dpi)
        
        if images:
            pdf_to_images[pdf_name] = images
    
    return pdf_to_images


def main():
    parser = argparse.ArgumentParser(description='Convert PDF documents to images')
    parser.add_argument('--pdf', type=str, help='Path to a single PDF file')
    parser.add_argument('--pdfs-dir', type=str, default='pdfs', help='Directory containing PDF files')
    parser.add_argument('--output-dir', type=str, default='pages', help='Output directory for images')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images (default: 300)')
    
    args = parser.parse_args()
    
    if args.pdf:
        # Process single PDF file
        if not os.path.exists(args.pdf):
            print(f"PDF file not found: {args.pdf}")
            sys.exit(1)
        
        images = convert_pdf_to_images(args.pdf, args.output_dir, args.dpi)
        print(f"\nConverted {len(images)} pages from {args.pdf}")
        
    else:
        # Process all PDFs in directory
        pdf_to_images = process_all_pdfs(args.pdfs_dir, args.output_dir, args.dpi)
        
        total_pages = sum(len(images) for images in pdf_to_images.values())
        print(f"\nTotal processed: {len(pdf_to_images)} PDFs, {total_pages} pages")


if __name__ == "__main__":
    main()