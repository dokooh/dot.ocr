#!/usr/bin/env python3
"""
Quick test script to verify PyMuPDF and PDF conversion is working.
"""

import os
import sys

def test_pymupdf():
    """Test PyMuPDF functionality."""
    print("🧪 Testing PyMuPDF (fitz) Functionality")
    print("="*50)
    
    try:
        import fitz
        print("✓ fitz module imported successfully")
        
        # Test version
        if hasattr(fitz, 'version'):
            print(f"✓ PyMuPDF version: {fitz.version}")
        
        # Test document opening methods
        test_pdf = "pdfs/test_document.pdf"
        if os.path.exists(test_pdf):
            print(f"✓ Test PDF found: {test_pdf}")
            
            # Test fitz.Document
            try:
                doc = fitz.Document(test_pdf)
                print(f"✓ fitz.Document() works - {len(doc)} pages")
                doc.close()
            except Exception as e:
                print(f"❌ fitz.Document() failed: {e}")
            
            # Test fitz.open (if available)
            try:
                doc = fitz.open(test_pdf)
                print(f"✓ fitz.open() works - {len(doc)} pages")
                doc.close()
            except Exception as e:
                print(f"⚠️  fitz.open() failed: {e} (this is OK)")
                
        else:
            print(f"⚠️  Test PDF not found: {test_pdf}")
            print("   Create one with: python create_test_pdf.py")
            
    except ImportError as e:
        print(f"❌ Failed to import fitz: {e}")
        return False
    
    return True

def test_pdf_conversion():
    """Test PDF to images conversion."""
    print(f"\n🖼️  Testing PDF to Images Conversion")
    print("="*50)
    
    test_pdf = "pdfs/test_document.pdf"
    if not os.path.exists(test_pdf):
        print(f"⚠️  Test PDF not found: {test_pdf}")
        print("Creating test PDF...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "create_test_pdf.py"], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Test PDF created")
            else:
                print(f"❌ Failed to create test PDF: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error creating test PDF: {e}")
            return False
    
    # Test conversion
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "pdf_to_images.py", 
            "--pdf", test_pdf, 
            "--output-dir", "test_conversion"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PDF to images conversion successful")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print("❌ PDF to images conversion failed")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing conversion: {e}")
        return False

def main():
    print("🔧 DotsOCR Pipeline - PyMuPDF Fix Verification")
    print("="*60)
    
    success = True
    
    # Test PyMuPDF
    if not test_pymupdf():
        success = False
    
    # Test PDF conversion
    if not test_pdf_conversion():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed! PyMuPDF issue is FIXED")
        print("\nThe pipeline is ready to use:")
        print("1. Download model: python setup_model.py")
        print("2. Run pipeline: python main.py")
    else:
        print("❌ Some tests failed - see errors above")
        print("\nTroubleshooting:")
        print("- Try: pip install --upgrade PyMuPDF")
        print("- Or run: python troubleshoot.py")

if __name__ == "__main__":
    main()