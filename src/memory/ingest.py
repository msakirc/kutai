# memory/ingest.py
"""
Phase 11.5 — Document Ingestion

Processes documents (URLs, PDF, DOCX, Markdown, plain text) into
chunks and stores them in the semantic vector collection.

Public API:
    result = await ingest_document(source, source_type="auto")
    result = await ingest_url(url)
    result = await ingest_file(filepath)

Result dict: {"chunks": int, "source": str, "status": "ok"|"error"}
"""
import logging
import os
from typing import Optional

from vector_store import embed_and_store, is_ready

logger = logging.getLogger(__name__)


# ─── Text Chunking ───────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size tokens.

    Uses word boundaries to avoid splitting mid-word.
    ~4 chars per token estimate.
    """
    if not text or not text.strip():
        return []

    max_chars = chunk_size * 4  # rough token-to-char
    overlap_chars = overlap * 4

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_len + word_len > max_chars and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_words = []
            overlap_len = 0
            for w in reversed(current):
                if overlap_len + len(w) + 1 > overlap_chars:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current = overlap_words
            current_len = overlap_len
        current.append(word)
        current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ─── URL Extraction ──────────────────────────────────────────────────────────

async def _extract_url_text(url: str) -> Optional[str]:
    """
    Download and extract clean text from a URL.

    Tries trafilatura first (best for articles), then falls back to
    basic HTML tag stripping.
    """
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Orchestrator/1.0)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"URL fetch failed: {resp.status_code} for {url}")
                return None
            html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return None

    # Try trafilatura
    try:
        import trafilatura
        text = trafilatura.extract(html, include_links=False, include_tables=True)
        if text and len(text) > 100:
            return text
    except ImportError:
        pass

    # Fallback: basic HTML stripping
    import re
    # Remove script/style
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode entities
    try:
        import html as html_lib
        text = html_lib.unescape(text)
    except Exception:
        pass

    return text if len(text) > 50 else None


# ─── File Extraction ─────────────────────────────────────────────────────────

def _extract_file_text(filepath: str) -> Optional[str]:
    """
    Extract text from a file based on its extension.

    Supports: .txt, .md, .py, .js, .ts, .go, .rs, .java, .c, .cpp,
              .pdf, .docx
    """
    if not os.path.isfile(filepath):
        logger.error(f"File not found: {filepath}")
        return None

    ext = os.path.splitext(filepath)[1].lower()

    # Plain text / code files
    text_exts = {
        ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
        ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
        ".rb", ".php", ".sh", ".bash", ".zsh", ".yml", ".yaml",
        ".json", ".toml", ".ini", ".cfg", ".env", ".csv",
    }

    if ext in text_exts:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return None

    # PDF
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyMuPDF not installed — can't read PDF. pip install pymupdf")
            return None
        except Exception as e:
            logger.error(f"Failed to read PDF {filepath}: {e}")
            return None

    # DOCX
    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            logger.warning("python-docx not installed — can't read DOCX. pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"Failed to read DOCX {filepath}: {e}")
            return None

    logger.warning(f"Unsupported file type: {ext}")
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

async def ingest_url(url: str) -> dict:
    """
    Ingest a URL: download, extract text, chunk, embed, store.

    Returns:
        {"chunks": int, "source": url, "status": "ok"|"error", "error": str?}
    """
    if not is_ready():
        return {"chunks": 0, "source": url, "status": "error",
                "error": "Vector store not initialized"}

    text = await _extract_url_text(url)
    if not text:
        return {"chunks": 0, "source": url, "status": "error",
                "error": "Failed to extract text from URL"}

    chunks = _chunk_text(text)
    stored = 0

    for i, chunk in enumerate(chunks):
        doc_id = await embed_and_store(
            text=chunk,
            metadata={
                "source": url,
                "source_type": "url",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "ingested_document",
                "importance": 6,
            },
            collection="semantic",
        )
        if doc_id:
            stored += 1

    logger.info(f"Ingested URL {url}: {stored}/{len(chunks)} chunks stored")
    return {"chunks": stored, "source": url, "status": "ok"}


async def ingest_file(filepath: str) -> dict:
    """
    Ingest a file: read, extract text, chunk, embed, store.

    Returns:
        {"chunks": int, "source": filepath, "status": "ok"|"error", "error": str?}
    """
    if not is_ready():
        return {"chunks": 0, "source": filepath, "status": "error",
                "error": "Vector store not initialized"}

    text = _extract_file_text(filepath)
    if not text:
        return {"chunks": 0, "source": filepath, "status": "error",
                "error": f"Failed to extract text from {filepath}"}

    chunks = _chunk_text(text)
    stored = 0
    filename = os.path.basename(filepath)

    for i, chunk in enumerate(chunks):
        doc_id = await embed_and_store(
            text=chunk,
            metadata={
                "source": filepath,
                "filename": filename,
                "source_type": "file",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "ingested_document",
                "importance": 6,
            },
            collection="semantic",
        )
        if doc_id:
            stored += 1

    logger.info(f"Ingested file {filepath}: {stored}/{len(chunks)} chunks stored")
    return {"chunks": stored, "source": filepath, "status": "ok"}


async def ingest_document(
    source: str,
    source_type: str = "auto",
) -> dict:
    """
    Auto-detect source type and ingest.

    Args:
        source:      URL or file path.
        source_type: "url", "file", or "auto" (detect).

    Returns:
        Result dict with chunks, source, status.
    """
    if source_type == "auto":
        if source.startswith("http://") or source.startswith("https://"):
            source_type = "url"
        else:
            source_type = "file"

    if source_type == "url":
        return await ingest_url(source)
    else:
        return await ingest_file(source)
