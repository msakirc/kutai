"""Sikayetvar scraper -- Turkey's largest consumer complaint platform.

Extracts complaint data including title, brand, text, date, and
resolution status.  Calculates per-brand resolution rates.

This data source is treated as **warnings** -- complaints and
resolution information rather than product reviews.
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

logger = get_logger("shopping.scrapers.sikayetvar")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- Sikayetvar scraper disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.sikayetvar.com"
_SEARCH_URL = "https://www.sikayetvar.com/arama"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("sikayetvar")
class SikayetvarScraper(BaseScraper):
    """Scrape consumer complaints from sikayetvar.com.

    Data is treated as a **warnings** source -- complaints,
    resolution status, and brand reputation indicators.
    """

    data_type: str = "warnings"

    def __init__(self) -> None:
        super().__init__(domain="sikayetvar")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Sikayetvar for complaints matching *query*."""
        if not _BS4_AVAILABLE:
            return []

        params = {"q": query}

        try:
            response = await self._fetch(_SEARCH_URL, params=params)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        return self._parse_search(response.text, max_results)

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        """Parse Sikayetvar search results into Product-like objects."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        # Complaint cards
        cards = (
            soup.select("article.complaint-card")
            or soup.select("div.complaint-item")
            or soup.select("div.card-v2")
        )

        for card in cards[:max_results]:
            try:
                # Title
                title_el = (
                    card.select_one("h2.complaint-title a")
                    or card.select_one("a.complaint-title")
                    or card.select_one("h3 a")
                )
                if title_el is None:
                    continue
                title = title_el.get_text(strip=True)
                if not title:
                    continue

                href = title_el.get("href", "")
                complaint_url = (
                    href if href.startswith("http") else f"{_BASE_URL}{href}"
                )

                # Brand
                brand_el = (
                    card.select_one("a.complaint-brand")
                    or card.select_one("span.brand-name")
                    or card.select_one("a[href*='/firma/']")
                )
                brand = brand_el.get_text(strip=True) if brand_el else None

                # Date
                date_el = card.select_one("time") or card.select_one("span.date")
                date_str: str | None = None
                if date_el:
                    date_str = date_el.get("datetime", date_el.get_text(strip=True))

                # Resolution status
                resolved = False
                status_el = (
                    card.select_one("span.badge-resolved")
                    or card.select_one("span.cozuldu")
                    or card.select_one("span.status")
                )
                if status_el:
                    status_text = status_el.get_text(strip=True).lower()
                    resolved = "çözüldü" in status_text or "resolved" in status_text

                specs: dict[str, Any] = {
                    "type": "complaint",
                    "data_type": "warnings",
                    "resolved": resolved,
                }
                if brand:
                    specs["brand"] = brand
                if date_str:
                    specs["complaint_date"] = date_str

                products.append(
                    Product(
                        name=title,
                        url=complaint_url,
                        source="sikayetvar",
                        seller_name=brand,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("card parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    # ------------------------------------------------------------------
    # get_product (not applicable -- treated as warnings source)
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Sikayetvar is a complaint platform, not a product source."""
        return None

    # ------------------------------------------------------------------
    # get_reviews (complaints)
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract complaint details from Sikayetvar.

        Returns complaint entries with text, brand, date, and resolution
        status.  Also calculates a brand resolution rate when multiple
        complaints for the same brand are found.
        """
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_reviews(url, "sikayetvar")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        all_complaints: list[dict] = []

        for page in range(1, max_pages + 1):
            page_url = f"{url}?page={page}" if page > 1 else url

            try:
                response = await self._fetch(page_url)
            except Exception as exc:
                logger.error(
                    "complaints fetch failed", url=url, page=page, error=str(exc)
                )
                break

            if response.status_code != 200:
                break

            page_complaints = self._parse_complaint_page(response.text)
            if not page_complaints:
                break

            all_complaints.extend(page_complaints)

        # Calculate resolution rate per brand
        all_complaints = self._enrich_resolution_rates(all_complaints)

        # Cache
        if all_complaints:
            try:
                await cache_reviews(url, all_complaints, "sikayetvar")
            except Exception:
                pass

        logger.info("complaints fetched", url=url, count=len(all_complaints))
        return all_complaints

    def _parse_complaint_page(self, html: str) -> list[dict]:
        """Parse complaints from a single page."""
        complaints: list[dict] = []

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        # Single complaint detail page
        detail_el = (
            soup.select_one("div.complaint-detail-body")
            or soup.select_one("article.complaint-detail")
        )
        if detail_el:
            try:
                text = detail_el.get_text(separator="\n", strip=True)
                if text:
                    # Title
                    title_el = soup.select_one("h1.complaint-title") or soup.select_one("h1")
                    title = title_el.get_text(strip=True) if title_el else ""

                    # Brand
                    brand_el = (
                        soup.select_one("a.brand-name")
                        or soup.select_one("span.brand-name")
                    )
                    brand = brand_el.get_text(strip=True) if brand_el else None

                    # Date
                    date_el = soup.select_one("time") or soup.select_one("span.complaint-date")
                    date_str = (
                        date_el.get("datetime", date_el.get_text(strip=True))
                        if date_el
                        else None
                    )

                    # Resolution
                    resolved = bool(
                        soup.select_one("span.badge-resolved")
                        or soup.select_one("div.cozuldu")
                    )

                    complaints.append({
                        "text": text,
                        "title": title,
                        "source": "sikayetvar",
                        "data_type": "warnings",
                        "brand": brand,
                        "date": date_str,
                        "resolved": resolved,
                    })
            except Exception as exc:
                logger.debug("detail parse error", error=str(exc))

            return complaints

        # Listing page with multiple complaints
        cards = (
            soup.select("article.complaint-card")
            or soup.select("div.complaint-item")
        )

        for card in cards:
            try:
                title_el = card.select_one("h2 a") or card.select_one("a.complaint-title")
                title = title_el.get_text(strip=True) if title_el else ""

                # Snippet text
                snippet_el = card.select_one("p.complaint-text") or card.select_one("div.description")
                text = snippet_el.get_text(strip=True) if snippet_el else title

                brand_el = card.select_one("a.complaint-brand") or card.select_one("span.brand-name")
                brand = brand_el.get_text(strip=True) if brand_el else None

                date_el = card.select_one("time") or card.select_one("span.date")
                date_str = (
                    date_el.get("datetime", date_el.get_text(strip=True))
                    if date_el
                    else None
                )

                resolved = bool(card.select_one("span.badge-resolved") or card.select_one("span.cozuldu"))

                complaints.append({
                    "text": text,
                    "title": title,
                    "source": "sikayetvar",
                    "data_type": "warnings",
                    "brand": brand,
                    "date": date_str,
                    "resolved": resolved,
                })
            except Exception as exc:
                logger.debug("card parse error", error=str(exc))
                continue

        return complaints

    @staticmethod
    def _enrich_resolution_rates(complaints: list[dict]) -> list[dict]:
        """Add resolution_rate to each complaint based on its brand."""
        brand_stats: dict[str, dict[str, int]] = {}

        for c in complaints:
            brand = c.get("brand")
            if not brand:
                continue
            if brand not in brand_stats:
                brand_stats[brand] = {"total": 0, "resolved": 0}
            brand_stats[brand]["total"] += 1
            if c.get("resolved"):
                brand_stats[brand]["resolved"] += 1

        for c in complaints:
            brand = c.get("brand")
            if brand and brand in brand_stats:
                stats = brand_stats[brand]
                if stats["total"] > 0:
                    c["resolution_rate"] = round(
                        stats["resolved"] / stats["total"], 2
                    )

        return complaints

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("sikayetvar", "complaint", "sikayet", "şikayet")
        return any(marker in text.lower() for marker in markers)
