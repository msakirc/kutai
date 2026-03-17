# web_extract.py — URL content extraction
# Uses trafilatura (primary) with readability-lxml fallback

import asyncio
from src.infra.logging_config import get_logger
logger = get_logger("tools.web_extract")

async def extract_url(url: str) -> str:
    """Extract clean text content from a URL. Returns extracted text or error message."""
    try:
        import trafilatura
        import httpx
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            html = resp.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if text:
            logger.debug("extracted via trafilatura", url=url, chars=len(text))
            return text
    except Exception as e:
        logger.warning("trafilatura failed, trying readability", url=url, error=str(e))
    # Fallback: readability-lxml
    try:
        from readability import Document
        import httpx
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        doc = Document(resp.text)
        import re
        text = re.sub(r'<[^>]+>', ' ', doc.summary())
        text = re.sub(r'\s+', ' ', text).strip()
        logger.debug("extracted via readability", url=url, chars=len(text))
        return text
    except Exception as e:
        logger.error("all extraction methods failed", url=url, error=str(e))
        return f"Error extracting URL: {e}"
