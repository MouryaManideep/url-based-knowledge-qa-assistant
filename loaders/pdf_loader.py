import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import os
from utils.logger import setup_logger

# File: loaders/pdf_loader.py
# Purpose: Read PDF files and turn each page into a simple text record for the RAG pipeline.

logger = setup_logger("pdf_loader")

def load_pdf_content(file_source: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file path or file-like object (bytes).
    Returns list of dicts: {"page_num": int, "text": str, "title": str, "source": str}
    """
    pages_data = []
    try:
        # Support three common PDF input types:
        # 1) raw bytes from an uploaded file
        # 2) a file path on disk
        # 3) a file-like object with a read() method
        if isinstance(file_source, bytes):
            doc = fitz.open(stream=file_source, filetype="pdf")
        elif isinstance(file_source, str) and os.path.exists(file_source):
            doc = fitz.open(file_source)
        else:
            if hasattr(file_source, "read"):
                doc = fitz.open(stream=file_source.read(), filetype="pdf")
            else:
                raise ValueError("Unsupported PDF file source type.")

        # Use the document title from PDF metadata when available; otherwise fall back to the uploaded file name.
        title = doc.metadata.get("title") or file_name
        if not title:
            title = "Uploaded PDF"

        logger.info(f"Loading PDF with {len(doc)} pages: {file_name}")
        # Extract text from each page so the chunking step can build retrievable sections.
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text()
            pages_data.append({
                "page_num": page_idx + 1,
                "text": text,
                "title": title,
                "source": file_name
            })
            
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF {file_name}: {str(e)}")
        
    return pages_data
