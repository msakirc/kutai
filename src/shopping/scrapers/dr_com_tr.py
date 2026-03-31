"""dr.com.tr scraper — D&R Turkey's book, music, game and electronics retailer.

Specialises in:
  - Books (Turkish and foreign language)
  - Music CDs, vinyl
  - Movies, games
  - Small electronics, stationery

Architecture notes:
  - Search: GET /search?q=QUERY
  - Product items: div.product-card (.js-prd-item)
  - Data source: data-gtm JSON attribute on each card — contains:
      item_name, author, publisher, price, discount_rate, item_rating,
      number_of_comments, item_stock, item_category, item_id
  - URL: a.js-search-prd-item[href] (relative, prepend base)
  - Price: .prd-price text (format: "479,40 TL")
  - Image: img.lazyload[data-src]
  - Rating: item_rating (0–10 scale) from GTM data
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.dr_com_tr")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- dr.com.tr scraper disabled")

_BASE_URL = "https://www.dr.com.tr"


@register_scraper("dr")
class DrComTrScraper(BaseScraper):
    """D&R (dr.com.tr) book and entertainment scraper.

    D&R is one of Turkey's largest retail chains for books, music, movies,
    and electronics.  Listing pages embed GTM JSON data directly on product
    card elements, providing clean structured data without JS rendering.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/search"

    def __init__(self) -> None:
        super().__init__(domain="dr")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search dr.com.tr for products matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?q={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("dr.com.tr search failed", query=query, error=str(exc))
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
            logger.debug("dr.com.tr get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """D&R does not expose structured reviews. Returns empty list."""
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
        markers = ("dr.com.tr", "product-card", "js-prd-item", "prd-price")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(".product-card")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("dr.com.tr item parse error", error=str(exc))

        logger.info("dr.com.tr listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        # --- GTM data (primary data source) ---
        gtm_str = item.get("data-gtm", "")
        gtm: dict = {}
        if gtm_str:
            try:
                gtm = json.loads(gtm_str)
            except json.JSONDecodeError:
                pass

        # --- name ---
        name_raw = gtm.get("item_name", "").strip()
        if not name_raw:
            return None
        name = normalize_product_name(name_raw)

        # --- URL ---
        link_el = item.select_one("a.js-search-prd-item, a[href*='/urunno=']")
        href = link_el.get("href", "") if link_el else ""
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # --- price ---
        price: float | None = None
        price_el = item.select_one(".prd-price, [class*=price]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        # Reconstruct original price from GTM discount_rate
        original_price: float | None = None
        discount_rate = gtm.get("discount_rate", 0)
        if price is not None and discount_rate and discount_rate > 0:
            try:
                dr = float(discount_rate)
                original_price = round(price / (1 - dr / 100), 2)
            except (ZeroDivisionError, ValueError):
                pass

        # --- rating (GTM gives 0–10 scale, normalise to 0–5) ---
        rating: float | None = None
        item_rating = gtm.get("item_rating")
        if item_rating is not None:
            try:
                rating = round(float(item_rating) / 2.0, 1)
            except (ValueError, TypeError):
                pass

        review_count: int | None = None
        n_comments = gtm.get("number_of_comments")
        if n_comments is not None:
            try:
                review_count = int(n_comments)
            except (ValueError, TypeError):
                pass

        # --- stock ---
        in_stock_str = gtm.get("item_stock", "Yes")
        availability = "in_stock" if str(in_stock_str).lower() in ("yes", "true", "1") else "out_of_stock"

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img.lazyload, img[data-src]")
        if img_el:
            src = img_el.get("data-src") or img_el.get("src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        # --- specs from GTM ---
        specs: dict[str, Any] = {}
        if gtm.get("author"):
            specs["author"] = gtm["author"]
        if gtm.get("publisher"):
            specs["publisher"] = gtm["publisher"]
        if gtm.get("item_category"):
            specs["category"] = gtm["item_category"]
        if gtm.get("item_variant"):
            specs["variant"] = gtm["item_variant"]
        if gtm.get("item_id"):
            specs["item_id"] = gtm["item_id"]

        return Product(
            name=name,
            url=url,
            source="dr",
            original_price=original_price,
            discounted_price=price,
            discount_percentage=float(discount_rate) if discount_rate else None,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating,
            review_count=review_count,
            availability=availability,
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
        price_el = soup.select_one(".prd-price, [class*=sell-price], [class*=price]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        image_url: str | None = None
        img_el = soup.select_one(".product-image img, [class*=product-img] img")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        return Product(
            name=name,
            url=url,
            source="dr",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
