# src/tools/content_extract.py
"""Content extraction using Trafilatura with BeautifulSoup fallback.

Standalone module — usable by web_search pipeline or directly by agents.
"""

import re
from dataclasses import dataclass

from src.infra.logging_config import get_logger

logger = get_logger("tools.content_extract")

_PRICE_PATTERNS = [
    re.compile(r"\d[\d.,]+\s*(?:TL|₺|USD|\$|EUR|€|GBP|£)", re.IGNORECASE),
    re.compile(r"(?:TL|₺|\$|€|£)\s*\d[\d.,]+", re.IGNORECASE),
    re.compile(r"\d[\d.,]+\s*(?:lira|dolar|euro|pound)", re.IGNORECASE),
]

_REVIEW_PATTERNS = [
    re.compile(r"\d+\.?\d*\s*(?:out of|/)\s*[5-9]\d*\s*(?:stars?)?", re.IGNORECASE),
    re.compile(r"(?:user\s*)?(?:rating|review|score)\s*[:=]\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:pros?|cons?)\s*:", re.IGNORECASE),
    re.compile(r"(?:highly\s+)?recommend", re.IGNORECASE),
    re.compile(r"\d+\s*reviews?\b", re.IGNORECASE),
]


@dataclass
class ExtractedContent:
    text: str = ""
    title: str = ""
    url: str = ""
    word_count: int = 0
    has_prices: bool = False
    has_reviews: bool = False


def extract_content(html: str, url: str = "") -> ExtractedContent:
    if not html or not html.strip():
        return ExtractedContent(url=url)

    text = ""
    title = ""

    try:
        import trafilatura
        text = trafilatura.extract(
            html, include_tables=True, include_comments=True,
            include_links=False, favor_recall=True, url=url or None,
        ) or ""
        meta = trafilatura.extract_metadata(html, default_url=url or None)
        if meta and meta.title:
            title = meta.title
    except Exception as e:
        logger.debug("trafilatura extraction failed, trying fallback", error=str(e)[:100])

    if not text or len(text) < 50:
        try:
            from src.tools.page_fetch import extract_main_text
            text = extract_main_text(html, max_chars=30000)
        except Exception as e:
            logger.debug("beautifulsoup fallback also failed", error=str(e)[:100])

    if not title:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
        except Exception:
            pass

    word_count = len(text.split()) if text else 0
    has_prices = any(p.search(text) for p in _PRICE_PATTERNS) if text else False
    has_reviews = any(p.search(text) for p in _REVIEW_PATTERNS) if text else False

    logger.debug("content extracted", url=url[:80] if url else "", word_count=word_count,
                 has_prices=has_prices, has_reviews=has_reviews)

    return ExtractedContent(text=text, title=title, url=url, word_count=word_count,
                            has_prices=has_prices, has_reviews=has_reviews)
