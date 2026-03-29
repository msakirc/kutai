# src/tools/page_fetch.py
"""Async page fetcher with HTML content extraction using BeautifulSoup."""

import asyncio
import re

import aiohttp
from bs4 import BeautifulSoup

from src.infra.logging_config import get_logger

logger = get_logger("tools.page_fetch")

# Standard browser User-Agent to avoid bot blocks
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Tags that contain non-content elements
_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"]


def extract_main_text(html: str, max_chars: int = 1500) -> str:
    """Extract main text content from HTML, stripping boilerplate.

    Priority: <article> → <main> → <body>.
    Strips script, style, nav, header, footer, aside.
    Collapses whitespace. Truncates to max_chars on word boundary.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove non-content tags
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # Find main content container
    content = soup.find("article") or soup.find("main") or soup.find("body")
    if not content:
        return ""

    # Get text, collapse whitespace
    text = content.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = text.strip()

    # Truncate on word boundary
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return text


async def fetch_page_content(
    session: aiohttp.ClientSession,
    url: str,
    max_chars: int = 1500,
    timeout: float = 8.0,
) -> str | None:
    """Fetch a single page and extract main text.

    Returns extracted text, or None on any error (timeout, non-HTML, HTTP error).
    """
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
            max_redirects=3,
        ) as resp:
            # Skip non-HTML responses
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/xhtml" not in content_type:
                logger.debug("page_fetch: skipping non-HTML", url=url[:80], ct=content_type[:50])
                return None

            if resp.status != 200:
                logger.debug("page_fetch: HTTP error", url=url[:80], status=resp.status)
                return None

            html = await resp.text(encoding=None)  # let aiohttp detect encoding
            text = extract_main_text(html, max_chars=max_chars)
            if not text or len(text) < 50:
                logger.debug("page_fetch: too little content", url=url[:80], text_len=len(text) if text else 0)
                return None

            logger.debug("page_fetch: ok", url=url[:80], text_len=len(text))
            return text

    except asyncio.TimeoutError:
        logger.debug("page_fetch: timeout", url=url[:80])
        return None
    except Exception as e:
        logger.debug("page_fetch: error", url=url[:80], error=str(e)[:100])
        return None


async def fetch_pages(
    urls: list[str],
    max_pages: int = 3,
    max_chars: int = 1500,
    total_timeout: float = 12.0,
) -> dict[str, str]:
    """Fetch multiple pages in parallel, returning {url: extracted_text}.

    Only fetches http/https URLs. Limits to max_pages.
    Returns dict of successfully fetched pages (may be empty).
    """
    # Filter to http(s) only and limit count
    valid_urls = [u for u in urls if u.startswith(("http://", "https://"))][:max_pages]

    if not valid_urls:
        return {}

    results = {}
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_page_content(session, url, max_chars=max_chars)
                for url in valid_urls
            ]
            fetched = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout,
            )

            for url, content in zip(valid_urls, fetched):
                if isinstance(content, str) and content:
                    results[url] = content

    except asyncio.TimeoutError:
        logger.debug("page_fetch: total timeout reached", fetched=len(results))
    except Exception as e:
        logger.debug("page_fetch: batch error", error=str(e)[:100])

    return results
