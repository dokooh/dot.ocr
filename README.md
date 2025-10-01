# DotsOCR PDF Processing Pipeline

This repository provides a complete pipeline for processing PDF documents using the DotsOCR transformer model. It converts PDF pages to images and applies OCR to extract structured layout information.

## Features

- Convert PDF documents to high-quality page images
- Apply DotsOCR transformer for advanced OCR processing
- Extract structured layout information including bounding boxes, categories, and formatted text
- Support for multiple output formats (JSON, HTML, LaTeX, Markdown)
- Batch processing of multiple PDF files
- Configurable image resolution and processing parameters

## Setup

### 1. Virtual Environment
The virtual environment `dot.ocr` has been created and configured.

To activate it manually:
```bash
# Windows
dot.ocr\Scripts\activate.bat

# Linux/Mac
source dot.ocr/bin/activate
```

### 2. Required Packages

**Core Dependencies** (always required):
- torch, torchvision, torchaudio
- transformers
- PyMuPDF (for PDF processing)
- Pillow (for image processing)
- numpy
- huggingface-hub

**Optional Dependencies** (may not be available on all systems):
- qwen-vl-utils (fallback implementation provided)
- dots-ocr (fallback implementation provided)

**Installation Options:**
```bash
# Option 1: Install with automatic fallback handling
python install_deps.py

# Option 2: Manual installation
pip install -r requirements.txt

# Option 3: Core packages only
pip install torch transformers PyMuPDF Pillow numpy huggingface-hub
```

### 3. Download DotsOCR Model

You need to download the DotsOCR model weights:

```bash
# Make sure git-lfs is installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/rednote-hilab/dots.ocr ./weights/DotsOCR
```

Alternatively, you can download the model using the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="rednote-hilab/dots.ocr", local_dir="./weights/DotsOCR")
```

## Directory Structure

```
dot.ocr/
├── pdfs/          # Place your PDF files here
├── pages/         # Generated page images will be saved here
├── results/       # OCR results will be saved here
├── weights/       # DotsOCR model weights
├── main.py        # Main pipeline script
├── pdf_to_images.py  # PDF to images converter
├── process_ocr.py    # DotsOCR processing script
└── README.md      # This file
```

## Usage

### Process All PDFs in the pdfs/ Directory

1. Place your PDF files in the `pdfs/` directory
2. Run the complete pipeline:

```bash
python main.py
```

### Process a Single PDF File

```bash
python main.py --pdf path/to/your/document.pdf
```

### Custom Options

```bash
python main.py \
  --pdfs-dir custom_pdfs \
  --pages-dir custom_pages \
  --results-dir custom_results \
  --model-path custom_model_path \
  --dpi 600
```

### Individual Components

#### Convert PDFs to Images Only
```bash
python pdf_to_images.py --pdfs-dir pdfs --output-dir pages --dpi 300
```

#### Process Images with OCR Only
```bash
python process_ocr.py --pages-dir pages --results-dir results
```

## Output Format

The OCR results are saved as JSON files containing structured layout information:

```json
{
  "elements": [
    {
      "bbox": [x1, y1, x2, y2],
      "category": "Text|Title|Table|Formula|Picture|etc.",
      "text": "Extracted and formatted content"
    }
  ]
}
```

### Text Formatting Rules

- **Text/Title**: Formatted as Markdown
- **Table**: Formatted as HTML
- **Formula**: Formatted as LaTeX
- **Picture**: Text field omitted
- **Others**: Formatted as Markdown

## Model Information

This pipeline uses the DotsOCR model from Hugging Face:
- **Model**: rednote-hilab/dots.ocr
- **Paper**: [DotsOCR Paper](https://huggingface.co/rednote-hilab/dots.ocr)
- **License**: Check the model repository for licensing information

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Sufficient RAM for model loading (8GB+ recommended)
- Storage space for model weights (~10GB)

## Troubleshooting

### Common Issues

1. **PyMuPDF "fitz has no attribute 'open'" error**:
   - ✅ **FIXED** - Updated to use `fitz.Document()` instead
   - This was a compatibility issue with newer PyMuPDF versions
   - If still occurring, update PyMuPDF: `pip install --upgrade PyMuPDF`

2. **FlashAttention2 errors** (installation, symbol, CUDA issues):
   - ✅ **FIXED** - Automatic detection and fallback to eager attention
   - Handles: "flash_attn not installed", "undefined symbol", CUDA loading errors
   - FlashAttention2 is optional (provides faster inference but can be problematic)
   - Pipeline works perfectly without it
   - Diagnostic tool: `python fix_flash_attention.py`
   - Optional setup: `python setup_flash_attention.py`

3. **Module not found errors** (qwen_vl_utils, dots_ocr):
   - These are known issues with some package dependencies
   - The pipeline includes fallback implementations
   - Run `python install_deps.py` for automatic handling
   - If issues persist, use the simplified processor: `python process_ocr_simple.py`

2. **Model not found**: Make sure you've downloaded the DotsOCR model weights:
   ```bash
   python setup_model.py
   ```

3. **CUDA out of memory**: Use CPU processing or reduce memory usage:
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   python main.py
   ```

4. **No PDFs found**: Ensure PDF files are placed in the `pdfs/` directory

5. **Permission errors**: Make sure the script has write permissions for `pages/` and `results/` directories

6. **Import errors on Kaggle/Colab**: Use the simplified version:
   ```bash
   python process_ocr_simple.py --pages-dir pages --results-dir results
   ```

### Performance Tips

- Use GPU processing for faster inference
- Higher DPI (600-1200) for better OCR accuracy on small text
- Lower DPI (150-300) for faster processing of large documents
- Process PDFs individually for better memory management with large files

## Example Workflow

1. **Setup** (one-time):
   ```bash
   # Download model weights
   git clone https://huggingface.co/rednote-hilab/dots.ocr ./weights/DotsOCR
   ```

2. **Process documents**:
   ```bash
   # Place PDFs in pdfs/ directory
   cp /path/to/documents/*.pdf pdfs/
   
   # Run pipeline
   python main.py
   
   # Check results
   ls results/
   ```

3. **View results**:
   - Page images: `pages/document_name/`
   - OCR results: `results/document_name/`

## Contributing

Feel free to submit issues and pull requests to improve this pipeline.

## License

This project is provided as-is. Please check the DotsOCR model license for usage restrictions.