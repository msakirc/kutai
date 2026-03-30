"""Extract text from PDF files."""

import asyncio
from functools import partial
from pathlib import Path

from src.infra.logging_config import get_logger

logger = get_logger("tools.pdf_extract")


def _extract_pdf_text(file_path: str, max_pages: int = 50) -> str:
    """Extract text from a PDF file. Tries multiple backends."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"
    if not path.suffix.lower() == ".pdf":
        return f"Error: not a PDF file: {file_path}"

    # Try PyMuPDF (fitz) first — fastest, best quality
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                pages.append(f"\n...(truncated at {max_pages} pages)")
                break
            pages.append(page.get_text())
        doc.close()
        text = "\n\n".join(pages)
        if len(text) > 50000:
            text = text[:50000] + "\n...(truncated)"
        return text
    except ImportError:
        pass

    # Try pdfplumber — good for tables
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    pages.append(f"\n...(truncated at {max_pages} pages)")
                    break
                text = page.extract_text() or ""
                pages.append(text)
        text = "\n\n".join(pages)
        if len(text) > 50000:
            text = text[:50000] + "\n...(truncated)"
        return text
    except ImportError:
        pass

    # Try PyPDF2 — widely installed
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                pages.append(f"\n...(truncated at {max_pages} pages)")
                break
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)
        if len(text) > 50000:
            text = text[:50000] + "\n...(truncated)"
        return text
    except ImportError:
        pass

    return "Error: no PDF extraction library available. Install: pip install PyMuPDF OR pdfplumber OR PyPDF2"


async def extract_pdf(file_path: str, max_pages: int = 50) -> str:
    """Async wrapper for PDF text extraction."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_extract_pdf_text, file_path, max_pages))
