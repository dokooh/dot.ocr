#!/usr/bin/env python3
"""
Debug PyMuPDF (fitz) module to check available methods.
"""

import fitz

def debug_fitz():
    """Debug fitz module to see available methods."""
    print("🔍 PyMuPDF (fitz) Module Debug")
    print("="*40)
    
    # Check version
    try:
        if hasattr(fitz, 'version'):
            version = fitz.version
            print(f"PyMuPDF version: {version}")
        else:
            print("Version info not available")
    except Exception as e:
        print(f"Error getting version: {e}")
    
    # Check available attributes
    print(f"\nAvailable attributes in fitz module:")
    attrs = [attr for attr in dir(fitz) if not attr.startswith('_')]
    for attr in sorted(attrs):
        print(f"  - {attr}")
    
    # Test different ways to open documents
    print(f"\nTesting document opening methods:")
    
    methods_to_test = [
        ('fitz.open', lambda p: fitz.open(p) if hasattr(fitz, 'open') else None),
        ('fitz.Document', lambda p: fitz.Document(p) if hasattr(fitz, 'Document') else None),
        ('fitz.PyMuPDFDocument', lambda p: fitz.PyMuPDFDocument(p) if hasattr(fitz, 'PyMuPDFDocument') else None),
    ]
    
    test_path = "pdfs/test_document.pdf"
    
    for method_name, method_func in methods_to_test:
        try:
            if hasattr(fitz, method_name.split('.')[1]):
                print(f"  ✓ {method_name} - Available")
                if os.path.exists(test_path):
                    try:
                        doc = method_func(test_path)
                        if doc:
                            print(f"    ✓ Successfully opened test PDF with {method_name}")
                            doc.close()
                        else:
                            print(f"    ❌ {method_name} returned None")
                    except Exception as e:
                        print(f"    ❌ Error with {method_name}: {e}")
                else:
                    print(f"    ⚠️  Test PDF not found: {test_path}")
            else:
                print(f"  ❌ {method_name} - Not available")
        except Exception as e:
            print(f"  ❌ {method_name} - Error: {e}")

if __name__ == "__main__":
    import os
    debug_fitz()