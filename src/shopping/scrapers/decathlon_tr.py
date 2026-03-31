"""decathlon.com.tr scraper — Decathlon Turkey's sports goods retailer.

Specialises in:
  - Bicycles (mountain, road, city, kids)
  - Camping & outdoor equipment
  - Fitness & gym gear
  - Swimming, running, football, basketball equipment
  - Hiking boots, sports apparel

Architecture notes:
  - Search: GET /search?Ntt=QUERY
  - Product items: div[data-supermodelid]
  - Name: span.vh (visually hidden full name inside the product link)
  - URL: a.dpb-product-model-link[href] (relative, prepend base)
  - Price: span.vtmn-price text (format: "₺20.990")
  - Brand: 3rd pipe-separated text segment in the item's full text
  - Review count: span.vtmn-rating_comment--secondary (format: "(1760)")
  - Image: img[src] or img[srcset] inside the item
  - No numeric rating score on listing page (only review count)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.decathlon_tr")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- decathlon.com.tr scraper disabled")

_BASE_URL = "https://www.decathlon.com.tr"

# Decathlon prices use ₺ symbol with Turkish number format (. as thousand sep)
_PRICE_RE = re.compile(r"[₺\s]*([\d.,]+)")


def _parse_decathlon_price(text: str) -> float | None:
    """Parse Decathlon price strings like '₺20.990' or '₺1.299,90'."""
    if not text:
        return None
    m = _PRICE_RE.search(text.strip())
    if not m:
        return None
    raw = m.group(1)
    # Decathlon uses period as thousands separator, comma as decimal
    # e.g. "20.990" -> 20990, "1.299,90" -> 1299.90
    if "," in raw:
        # Has decimal: remove thousands dots, replace comma decimal
        raw = raw.replace(".", "").replace(",", ".")
    else:
        # No comma: period is thousands separator
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


@register_scraper("decathlon")
class DecathlonTrScraper(BaseScraper):
    """decathlon.com.tr sports equipment scraper.

    Decathlon Turkey renders search results server-side with Svelte components.
    The HTML is fully present in the initial response without JS execution,
    making it scrapable via TLS tier.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/search"

    def __init__(self) -> None:
        super().__init__(domain="decathlon")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search decathlon.com.tr for products matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?Ntt={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("decathlon search failed", query=query, error=str(exc))
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
            logger.debug("decathlon get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Decathlon does not expose structured reviews on listing pages."""
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
        markers = ("decathlon", "dpb-product-model-link", "vtmn-price", "data-supermodelid")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select("[data-supermodelid]")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("decathlon item parse error", error=str(exc))

        logger.info("decathlon listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        # --- name & URL ---
        link_el = item.select_one("a.dpb-product-model-link")
        if not link_el:
            return None

        href = link_el.get("href", "")
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # Name is in the visually-hidden span inside the link
        name_el = link_el.select_one("span.vh")
        name = normalize_product_name(
            name_el.get_text(strip=True) if name_el else link_el.get_text(strip=True)
        )
        if len(name) < 3:
            return None

        # --- price ---
        price: float | None = None
        price_el = item.select_one("span.vtmn-price")
        if price_el:
            price = _parse_decathlon_price(price_el.get_text(strip=True))

        # --- brand (extracted from pipe-separated full text) ---
        # Pattern: "ProductName|Price|BrandName|ProductName again|(ReviewCount)|..."
        full_text = item.get_text(separator="|", strip=True)
        parts = [p.strip() for p in full_text.split("|") if p.strip()]
        brand: str | None = None
        # Brand is usually after the price and before the repeated product name
        price_idx = -1
        for i, part in enumerate(parts):
            if "₺" in part or "TL" in part:
                price_idx = i
                break
        if price_idx >= 0 and price_idx + 1 < len(parts):
            candidate = parts[price_idx + 1]
            # Brand is typically short (1–3 words) and not a product sentence
            if len(candidate.split()) <= 4 and len(candidate) < 40:
                brand = candidate

        # --- review count ---
        review_count: int | None = None
        rating_el = item.select_one(".vtmn-rating_comment--secondary")
        if rating_el:
            rating_text = rating_el.get_text(strip=True)
            m = re.search(r"(\d+)", rating_text)
            if m:
                review_count = int(m.group(1))

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img[src]")
        if img_el:
            src = img_el.get("src", "")
            # Prefer higher-quality srcset variant
            srcset = img_el.get("srcset", "")
            if srcset:
                # Take the last (largest) srcset entry
                last_entry = srcset.strip().split(",")[-1].strip()
                src_candidate = last_entry.split(" ")[0]
                if src_candidate:
                    src = src_candidate
            if src and not src.startswith("data:"):
                image_url = src

        specs: dict[str, Any] = {}
        if brand:
            specs["brand"] = brand
        product_id = item.get("data-supermodelid")
        if product_id:
            specs["model_id"] = product_id

        return Product(
            name=name,
            url=url,
            source="decathlon",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            review_count=review_count,
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
        price_el = soup.select_one("span.vtmn-price, [class*=price]")
        if price_el:
            price = _parse_decathlon_price(price_el.get_text(strip=True))

        image_url: str | None = None
        og_img = soup.select_one('meta[property="og:image"]')
        if og_img:
            image_url = og_img.get("content")

        return Product(
            name=name,
            url=url,
            source="decathlon",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
