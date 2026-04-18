"""arabam.com scraper — Turkey's leading second-hand car marketplace.

Specialises in:
  - Second-hand (ikinci el) cars
  - New car listings from dealers
  - Car specs: year, km, color, location

Architecture notes:
  - Search: GET /ikinci-el/otomobil?query=QUERY
  - Listing items: tr.listing-list-item
  - Data available in table cells (TD indices):
      TD[1]: model name (div.listing-text-new.color-blackYEAR inside h2)
      TD[2]: listing title / description
      TD[3]: year
      TD[4]: km
      TD[5]: color
      TD[6]: price (format: "1.169.000 TL")
      TD[7]: date
      TD[8]: city
  - URL: a.link-overlay[href] (relative, prepend base)
  - Image: img.listing-image[src]
  - Item ID: data-imp-id on the tr element
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

logger = get_logger("shopping.scrapers.arabam")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- arabam.com scraper disabled")

_BASE_URL = "https://www.arabam.com"


@register_scraper("arabam")
class ArabamScraper(BaseScraper):
    """arabam.com second-hand car listing scraper.

    arabam.com is Turkey's most popular platform for buying and selling
    used cars.  Listings include full specs: year, km, color, location,
    and price.  The listing page is server-side rendered.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/ikinci-el/otomobil"

    def __init__(self) -> None:
        super().__init__(domain="arabam")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search arabam.com for car listings matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?query={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("arabam search failed", query=query, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single listing page."""
        if not _BS4:
            return None
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return None
            return self._parse_product_page(url, resp.text)
        except Exception as exc:
            logger.debug("arabam get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """arabam.com is a classified-ad marketplace and does not host
        per-model user reviews.

        Verified 2026-04-18 against /ikinci-el/otomobil-volkswagen-golf,
        /yeni/<brand>/<model> (404), and /inceleme (404).  The only
        editorial content lives at /blog/category/otomobil-inceleme/ —
        these are long-form articles, not user-generated reviews, and
        are out of scope for get_reviews.
        """
        logger.debug("get_reviews called on arabam (no reviews)", url=url)
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
        markers = ("arabam.com", "listing-list-item", "listing-table", "otomobil")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select("tr.listing-list-item")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("arabam item parse error", error=str(exc))

        logger.info("arabam listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        tds = item.find_all("td")
        if len(tds) < 7:
            return None

        # --- name from TD[1] ---
        model_el = tds[1].select_one(".listing-text-new, .listing-modelname div")
        if not model_el:
            model_el = tds[1]
        name = normalize_product_name(model_el.get_text(strip=True))
        if len(name) < 3:
            return None

        # --- URL ---
        link_el = item.select_one("a.link-overlay")
        href = link_el.get("href", "") if link_el else ""
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # --- price from TD[6] ---
        price: float | None = None
        price_text = tds[6].get_text(strip=True) if len(tds) > 6 else ""
        if price_text:
            price = parse_turkish_price(price_text)

        # --- specs from cells ---
        specs: dict[str, Any] = {}

        year_text = tds[3].get_text(strip=True) if len(tds) > 3 else ""
        if year_text and year_text.isdigit():
            specs["year"] = int(year_text)

        km_text = tds[4].get_text(strip=True) if len(tds) > 4 else ""
        if km_text:
            km_clean = re.sub(r"[^\d]", "", km_text)
            if km_clean:
                specs["km"] = int(km_clean)

        color_text = tds[5].get_text(strip=True) if len(tds) > 5 else ""
        if color_text:
            specs["color"] = color_text

        city_text = tds[8].get_text(strip=True) if len(tds) > 8 else ""
        if city_text:
            # City is first line of TD[8] before action buttons
            city_lines = [l.strip() for l in city_text.split("\n") if l.strip()]
            if city_lines:
                specs["city"] = city_lines[0]

        date_text = tds[7].get_text(strip=True) if len(tds) > 7 else ""
        if date_text:
            specs["listing_date"] = date_text

        # Listing title (description)
        title_el = tds[2].select_one(".listing-text-new, .listing-title-lines") if len(tds) > 2 else None
        if title_el:
            specs["listing_title"] = title_el.get_text(strip=True)[:200]

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img.listing-image")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        # Listing ID
        listing_id = item.get("data-imp-id")
        if listing_id:
            specs["listing_id"] = listing_id

        return Product(
            name=name,
            url=url,
            source="arabam",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            category_path="Otomobil",
            fetched_at=now_iso,
        )

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse a single car listing detail page."""
        soup = BeautifulSoup(html, "lxml")
        now_iso = datetime.now(timezone.utc).isoformat()

        name_el = soup.select_one("h1, .heading-title")
        if not name_el:
            return None
        name = normalize_product_name(name_el.get_text(strip=True))

        price: float | None = None
        price_el = soup.select_one(".price, [class*=price], [class*=fiyat]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        image_url: str | None = None
        og_img = soup.select_one('meta[property="og:image"]')
        if og_img:
            image_url = og_img.get("content")

        return Product(
            name=name,
            url=url,
            source="arabam",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            category_path="Otomobil",
            fetched_at=now_iso,
        )
