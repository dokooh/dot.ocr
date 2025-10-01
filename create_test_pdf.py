#!/usr/bin/env python3
"""
Create a simple test PDF for demonstration.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_test_pdf():
    """Create a simple test PDF with some text content."""
    pdf_path = "pdfs/test_document.pdf"
    
    # Create pdfs directory if it doesn't exist
    os.makedirs("pdfs", exist_ok=True)
    
    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Test Document for DotsOCR")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "This is a test document to demonstrate the DotsOCR pipeline.")
    c.drawString(100, 680, "It contains multiple text elements including:")
    
    c.drawString(120, 650, "• Headers and titles")
    c.drawString(120, 630, "• Regular text paragraphs")
    c.drawString(120, 610, "• Lists and bullet points")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 570, "Section Header")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 540, "This section contains sample text that should be extracted")
    c.drawString(100, 520, "by the DotsOCR model and formatted as structured output.")
    
    # Add a simple table-like structure
    c.drawString(100, 480, "Sample Table:")
    c.drawString(120, 460, "Column 1       Column 2       Column 3")
    c.drawString(120, 440, "Row 1 Data     Value A        123")
    c.drawString(120, 420, "Row 2 Data     Value B        456")
    
    # Add a formula-like text
    c.drawString(100, 380, "Formula: E = mc²")
    
    c.showPage()
    
    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Page 2: Additional Content")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "This is the second page of the test document.")
    c.drawString(100, 680, "It demonstrates multi-page PDF processing.")
    
    c.drawString(100, 640, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    c.drawString(100, 620, "Sed do eiusmod tempor incididunt ut labore et dolore magna")
    c.drawString(100, 600, "aliqua. Ut enim ad minim veniam, quis nostrud exercitation.")
    
    c.save()
    
    print(f"Test PDF created: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    try:
        import reportlab
        create_test_pdf()
    except ImportError:
        print("reportlab not installed. Install with: pip install reportlab")
        print("Alternatively, place your own PDF files in the pdfs/ directory.")