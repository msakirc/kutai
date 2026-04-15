"""Eksi Sozluk scraper -- Turkey's largest collaborative dictionary / forum.

Eksi Sozluk is a valuable source of user opinions on products, brands, and
services.  Entries are text-based with author, date, and favorite count
metadata.

Rate limiting: **strict 5-second minimum** between requests to respect the
site's expectations for automated access.
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import cache_reviews, get_cached_reviews
from ..models import Product
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.eksisozluk")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- Eksi Sozluk scraper disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://eksisozluk.com"

# Minimum delay is enforced at 5 seconds (overrides config if lower)
_MIN_DELAY_SECONDS = 5


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("eksisozluk")
class EksiSozlukScraper(BaseScraper):
    """Scrape user entries from Eksi Sozluk.

    Entries are treated as review-like content with text, author, date,
    and favorite count.  Short entries (< 50 chars) are filtered out.
    """

    def __init__(self) -> None:
        super().__init__(domain="eksisozluk")
        # Enforce minimum 5s delay
        if self._rate_cfg.get("delay_seconds", 0) < _MIN_DELAY_SECONDS:
            self._rate_cfg["delay_seconds"] = _MIN_DELAY_SECONDS

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Eksi Sozluk for topics matching *query*.

        Returns Product objects where each represents a topic (baslik).
        """
        if not _BS4_AVAILABLE:
            return []

        encoded = urllib.parse.quote_plus(query)
        url = f"{_BASE_URL}/basliklar/ara?searchForm.Keywords={encoded}"

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        return self._parse_search(response.text, max_results)

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        """Parse Eksi Sozluk search/autocomplete results."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        # Topic links from search results — filter out site navigation
        topic_links = (
            soup.select("ul.topic-list li a")
            or soup.select("ul#search-result-list li a")
            or [
                a for a in soup.select("a[href*='/--']")
                if "sozluk-kural" not in a.get("href", "")
                and "is-ilan" not in a.get("href", "")
            ]
        )

        seen_urls: set[str] = set()

        for link in topic_links[:max_results]:
            try:
                title = link.get_text(strip=True)
                if not title:
                    continue

                href = link.get("href", "")
                topic_url = (
                    href if href.startswith("http") else f"{_BASE_URL}{href}"
                )

                if topic_url in seen_urls:
                    continue
                seen_urls.add(topic_url)

                # Entry count from small text like "(123)"
                entry_count: int | None = None
                small = link.select_one("small")
                if small:
                    m = re.search(r"(\d+)", small.get_text())
                    if m:
                        entry_count = int(m.group(1))
                    # Remove the count text from title
                    title = title.replace(small.get_text(), "").strip()

                specs: dict[str, Any] = {
                    "type": "eksisozluk_topic",
                }
                if entry_count is not None:
                    specs["entry_count"] = entry_count

                products.append(
                    Product(
                        name=title,
                        url=topic_url,
                        source="eksisozluk",
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("topic parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    # ------------------------------------------------------------------
    # get_product (not applicable)
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Eksi Sozluk does not have product pages. Returns None."""
        return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract entries from an Eksi Sozluk topic.

        Filters out entries shorter than 50 characters.  Results are
        sorted by favorite_count descending.
        """
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_reviews(url, "eksisozluk")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        all_entries: list[dict] = []

        for page in range(1, max_pages + 1):
            page_url = f"{url}?p={page}" if page > 1 else url

            try:
                response = await self._fetch(page_url)
            except Exception as exc:
                logger.error(
                    "entries fetch failed", url=url, page=page, error=str(exc)
                )
                break

            if response.status_code != 200:
                break

            page_entries = self._parse_entries(response.text)
            if not page_entries:
                break

            all_entries.extend(page_entries)

        # Filter short entries
        all_entries = [e for e in all_entries if len(e.get("text", "")) >= 50]

        # Sort by favorite count (descending)
        all_entries.sort(key=lambda e: e.get("favorite_count", 0), reverse=True)

        # Cache
        if all_entries:
            try:
                await cache_reviews(url, all_entries, "eksisozluk")
            except Exception:
                pass

        logger.info("entries fetched", url=url, count=len(all_entries))
        return all_entries

    def _parse_entries(self, html: str) -> list[dict]:
        """Parse entries from a single page."""
        entries: list[dict] = []

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        entry_els = (
            soup.select("ul#entry-item-list li")
            or soup.select("div[data-id]")
        )

        for el in entry_els:
            try:
                # Entry text
                content_el = el.select_one("div.content")
                if content_el is None:
                    continue
                text = content_el.get_text(separator="\n", strip=True)
                if not text:
                    continue

                # Author
                author_el = el.select_one("a.entry-author")
                author = author_el.get_text(strip=True) if author_el else None

                # Date
                date_el = el.select_one("a.entry-date")
                date_str = date_el.get_text(strip=True) if date_el else None

                # Favorite count
                favorite_count = 0
                fav_el = el.select_one("span.entry-favorite-count")
                if fav_el:
                    try:
                        favorite_count = int(fav_el.get_text(strip=True))
                    except (ValueError, TypeError):
                        pass

                # Entry ID
                entry_id = el.get("data-id")

                entry: dict[str, Any] = {
                    "text": text,
                    "source": "eksisozluk",
                    "author": author,
                    "date": date_str,
                    "favorite_count": favorite_count,
                    "helpful_count": favorite_count,  # map to common field
                }
                if entry_id:
                    entry["entry_id"] = entry_id

                entries.append(entry)
            except Exception as exc:
                logger.debug("entry parse error", error=str(exc))
                continue

        return entries

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("eksisozluk", "entry-item", "topic-list", "content")
        return any(marker in text.lower() for marker in markers)
