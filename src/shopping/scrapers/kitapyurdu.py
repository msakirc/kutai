"""kitapyurdu.com scraper — Turkey's leading book marketplace.

Specialises in:
  - Books (Türkçe and foreign language)
  - E-books
  - Stationery

Architecture notes:
  - Search: GET /index.php?route=product/search&filter_name=QUERY
  - Product items: div.ky-product
  - Name: span.ky-product-title
  - Author: span.ky-product-author (may be inside .ky-product-labels)
  - Publisher: span.ky-product-publisher
  - URL: a.ky-product-cover[href] (absolute)
  - Price: span.ky-product-price (format: "429,00TL")
  - Rating: count selected stars (ky-product-rating-star--selected)
  - Image: img[src] inside a.ky-product-cover
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.kitapyurdu")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- kitapyurdu scraper disabled")

_BASE_URL = "https://www.kitapyurdu.com"


@register_scraper("kitapyurdu")
class KitapyurduScraper(BaseScraper):
    """kitapyurdu.com book and stationery scraper.

    kitapyurdu.com is Turkey's largest dedicated book marketplace with
    a wide catalogue of Turkish and foreign language titles.  Listing
    pages are server-side rendered and fully scrapable without JavaScript.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/index.php"

    def __init__(self) -> None:
        super().__init__(domain="kitapyurdu")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search kitapyurdu.com for books matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?route=product/search&filter_name={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("kitapyurdu search failed", query=query, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product page."""
        if not _BS4:
            return None
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return None
            return self._parse_product_page(url, resp.text)
        except Exception as exc:
            logger.debug("kitapyurdu get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Kitapyurdu does not expose user reviews in structured form."""
        return []

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: Any) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 500:
            return False
        markers = ("kitapyurdu", "ky-product", "ky-product-title")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(".ky-product")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("kitapyurdu item parse error", error=str(exc))

        logger.info("kitapyurdu listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        # --- name & URL ---
        cover_el = item.select_one("a.ky-product-cover")
        title_el = item.select_one("span.ky-product-title")

        if not title_el:
            return None

        name = normalize_product_name(title_el.get_text(strip=True))
        if len(name) < 2:
            return None

        url = cover_el.get("href", "") if cover_el else ""
        if url and not url.startswith("http"):
            url = f"{self._BASE_URL}{url}"

        # --- author ---
        author_el = item.select_one("span.ky-product-author")
        author = author_el.get_text(strip=True) if author_el else None

        # --- publisher ---
        publisher_el = item.select_one("span.ky-product-publisher")
        publisher = publisher_el.get_text(strip=True) if publisher_el else None

        # --- price ---
        price: float | None = None
        price_el = item.select_one("span.ky-product-price")
        if price_el:
            # Text is like "429,00TL" — remove TL suffix first
            price_text = price_el.get_text(strip=True)
            price = parse_turkish_price(price_text)

        # --- rating (count selected stars, max 5) ---
        rating: float | None = None
        selected_stars = item.select(".ky-product-rating-star--selected")
        if selected_stars:
            rating = float(len(selected_stars))

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("a.ky-product-cover img")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        # Build specs with book-specific fields
        specs: dict[str, Any] = {}
        if author:
            specs["author"] = author
        if publisher:
            specs["publisher"] = publisher

        return Product(
            name=name,
            url=url,
            source="kitapyurdu",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating,
            fetched_at=now_iso,
        )

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse a product detail page."""
        soup = BeautifulSoup(html, "lxml")
        now_iso = datetime.now(timezone.utc).isoformat()

        name_el = soup.select_one("h1")
        if not name_el:
            return None
        name = normalize_product_name(name_el.get_text(strip=True))

        price: float | None = None
        price_el = soup.select_one("[class*=sell-price], [class*=price]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        image_url: str | None = None
        img_el = soup.select_one(".product-cover img, [class*=cover] img")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        return Product(
            name=name,
            url=url,
            source="kitapyurdu",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
