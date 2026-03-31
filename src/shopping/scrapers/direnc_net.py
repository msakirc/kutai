"""direnc.net scraper — Turkey's leading electronics components & robotics marketplace.

Specialises in:
  - Arduino, Raspberry Pi, ESP32, STM32 development boards
  - Sensors, LEDs, resistors, capacitors
  - 3D printer parts, robotics components
  - Soldering tools, multimeters, oscilloscopes
  - Maker / hobbyist electronics

Architecture notes:
  - Search: GET /arama?q=QUERY — returns HTML with product grid
  - Product items: div.productItem
  - Name: a.productDescription[title] or a.productDescription text
  - URL: a.image-wrapper[href] or a.productDescription[href] (relative, prepend base)
  - Price: .currentPrice text (format: "137,61 TL")
  - Old price: .oldPrice (when discounted)
  - Stock status: .out-of-stock span (absent when in stock)
  - Image: img[data-src] inside .imgInner
  - No ratings/reviews on listing page
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.direnc_net")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 (BeautifulSoup) is not installed -- direnc.net scraper disabled")

_BASE_URL = "https://www.direnc.net"


@register_scraper("direnc")
class DirencNetScraper(BaseScraper):
    """direnc.net electronics components scraper.

    direnc.net is Turkey's leading online marketplace for electronic
    components, development boards (Arduino / Raspberry Pi / ESP32 / STM32),
    sensors, 3D printer parts, and maker hobbyist supplies.

    The listing page is rendered server-side and is accessible via plain TLS
    without JS — making it reliably scrapable via the tiered scraper.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/arama"

    def __init__(self) -> None:
        super().__init__(domain="direnc")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search direnc.net for products matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?q={query}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("direnc search failed", query=query, error=str(exc))
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
            logger.debug("direnc get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews  (not available on listing pages)
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """direnc.net does not expose user reviews. Returns empty list."""
        return []

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: Any) -> bool:
        """Return True if the response looks like real direnc.net content."""
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 500:
            return False
        markers = ("direnc.net", "productItem", "productDescription", "productPrice")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        """Parse the search results page and return a list of Products."""
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(".productItem")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("direnc item parse error", error=str(exc))

        logger.info("direnc listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        """Parse a single .productItem div into a Product."""
        # --- name & URL ---
        name_el = item.select_one("a.productDescription")
        if not name_el:
            return None

        name = (
            name_el.get("title", "").strip()
            or name_el.get_text(strip=True)
        )
        if len(name) < 3:
            return None
        name = normalize_product_name(name)

        href = name_el.get("href", "")
        if not href:
            # Try the image-wrapper link
            img_link = item.select_one("a.image-wrapper")
            if img_link:
                href = img_link.get("href", "")
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # --- price ---
        price: float | None = None
        original_price: float | None = None

        curr_price_el = item.select_one(".currentPrice")
        if curr_price_el:
            price_text = curr_price_el.get_text(strip=True)
            price = parse_turkish_price(price_text)

        old_price_el = item.select_one(".oldPrice")
        if old_price_el:
            old_text = old_price_el.get_text(strip=True)
            original_price = parse_turkish_price(old_text)

        # --- stock status ---
        out_of_stock_el = item.select_one(".out-of-stock")
        in_stock = out_of_stock_el is None

        availability = "in_stock" if in_stock else "out_of_stock"

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img[data-src]")
        if img_el:
            src = img_el.get("data-src", "")
            if src and not src.startswith("data:"):
                image_url = f"https:{src}" if src.startswith("//") else src

        return Product(
            name=name,
            url=url,
            source="direnc",
            original_price=original_price,
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            availability=availability,
            fetched_at=now_iso,
        )

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse a product detail page."""
        soup = BeautifulSoup(html, "lxml")
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- name ---
        name_el = soup.select_one("h1, .product-name, .productName")
        if not name_el:
            og_title = soup.select_one('meta[property="og:title"]')
            if og_title:
                name = normalize_product_name(og_title.get("content", "").strip())
            else:
                return None
        else:
            name = normalize_product_name(name_el.get_text(strip=True))

        # --- price ---
        price: float | None = None
        original_price: float | None = None

        price_el = soup.select_one(".currentPrice, .product-price, .price")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        old_price_el = soup.select_one(".oldPrice, .old-price")
        if old_price_el:
            original_price = parse_turkish_price(old_price_el.get_text(strip=True))

        # --- image ---
        image_url: str | None = None
        og_img = soup.select_one('meta[property="og:image"]')
        if og_img:
            image_url = og_img.get("content")
        if not image_url:
            img_el = soup.select_one(".product-image img, .main-image img")
            if img_el:
                src = img_el.get("data-src") or img_el.get("src") or ""
                if src and not src.startswith("data:"):
                    image_url = f"https:{src}" if src.startswith("//") else src

        return Product(
            name=name,
            url=url,
            source="direnc",
            original_price=original_price,
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
