"""Sahibinden scraper -- Turkey's largest classifieds platform.

STATUS: DISABLED (2026-03-31)

Sahibinden uses aggressive multi-layer protection that blocks all
automated access from datacenter IPs:

Tested and failed (2026-03-31):
  - curl_cffi TLS (chrome131): 302 → forced login redirect
  - nodriver headless Chrome: Cloudflare challenge page (12K chars, no content)
  - nodriver visible Chrome: CF challenge "Yükleniyor" (18K, no content)
  - Scrapling StealthyFetcher (Camoufox): 403 "Olağan dışı erişim tespit ettik"
  - Mobile API endpoints (api/m-api/gw/rest.sahibinden.com): all 404 or timeout
  - RSS/Atom feeds: all blocked
  - Session cookie accumulation (homepage → category): still redirected to login

Protection stack:
  - Cloudflare (TLS fingerprinting + Turnstile JS challenge)
  - Custom session tokens (vid, cdid, csid — mandatory for browsing)
  - IP reputation check (datacenter IPs auto-blocked regardless of fingerprint)
  - Mandatory login redirect for listing/category pages

Only known working approach: residential proxy + real browser with aged cookies.
GitHub refs: 0Baris/sahibinden-scraper (uses nodriver + residential IP).

Re-enable when: residential proxy available, or sahibinden relaxes protection.

Original features (preserved for future use):
  - Electronics, furniture, appliances categories
  - Rate limiting: strict 15-second minimum between requests
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import cache_search, get_cached_search
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.sahibinden")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- Sahibinden scraper disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.sahibinden.com"
_SEARCH_URL = "https://www.sahibinden.com/arama"

# Strict minimum delay to avoid IP bans
_MIN_DELAY_SECONDS = 15

# Focus categories
_CATEGORY_KEYWORDS = {
    "elektronik": "elektronik",
    "bilgisayar": "bilgisayar",
    "telefon": "cep-telefonu",
    "tablet": "tablet",
    "mobilya": "mobilya",
    "beyaz esya": "beyaz-esya",
    "küçük ev aletleri": "kucuk-ev-aletleri",
}


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


# DISABLED: sahibinden blocks all datacenter IPs (see docstring for details)
# @register_scraper("sahibinden")
class SahibindenScraper(BaseScraper):
    """Scrape classified listings from sahibinden.com.

    Focus: electronics, furniture, appliances.
    Strict 15+ second delay between requests.

    NOTE: Sahibinden employs strong bot detection (Cloudflare + proprietary
    fingerprinting).  Even the TLS-bypass tier consistently returns a bot
    challenge page.  ``requires_auth = True`` signals to callers that this
    scraper is not reliably usable without a real browser session.
    """

    # Strong bot detection — requires a real browser session (Playwright/etc.)
    requires_auth: bool = True

    def __init__(self) -> None:
        super().__init__(domain="sahibinden")
        # Enforce minimum 15s delay
        if self._rate_cfg.get("delay_seconds", 0) < _MIN_DELAY_SECONDS:
            self._rate_cfg["delay_seconds"] = _MIN_DELAY_SECONDS

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Sahibinden classifieds for *query*."""
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_search(query, "sahibinden")
            if cached is not None:
                logger.debug("search cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        params: dict[str, str] = {"query_text": query}

        try:
            response = await self._fetch(_SEARCH_URL, params=params)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        products = self._parse_search(response.text, max_results)

        if products:
            try:
                await cache_search(
                    query, "sahibinden",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception:
                pass

        return products

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        """Parse Sahibinden search results."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        # Listing rows
        rows = (
            soup.select("tr.searchResultsItem")
            or soup.select("div.classified-list-item")
            or soup.select("tbody.searchResultsRowClass tr")
        )

        for row in rows[:max_results]:
            try:
                # Title
                title_el = (
                    row.select_one("a.classifiedTitle")
                    or row.select_one("td.searchResultsTitleValue a")
                    or row.select_one("a[title]")
                )
                if title_el is None:
                    continue
                title = (
                    title_el.get("title", "")
                    or title_el.get_text(strip=True)
                )
                if not title:
                    continue
                title = normalize_product_name(title)

                # URL
                href = title_el.get("href", "")
                listing_url = (
                    href if href.startswith("http") else f"{_BASE_URL}{href}"
                )

                # Price
                price_el = (
                    row.select_one("td.searchResultsPriceValue span")
                    or row.select_one("span.searchResultsPrice")
                    or row.select_one("div.price")
                )
                price: float | None = None
                if price_el:
                    price = parse_turkish_price(price_el.get_text(strip=True))

                # Location
                loc_el = (
                    row.select_one("td.searchResultsLocationValue")
                    or row.select_one("span.location")
                )
                location: str | None = None
                if loc_el:
                    location = loc_el.get_text(separator=" ", strip=True)

                # Date
                date_el = (
                    row.select_one("td.searchResultsDateValue span")
                    or row.select_one("span.date")
                )
                date_str: str | None = None
                if date_el:
                    date_str = date_el.get_text(strip=True)

                # Image / photo count
                img_el = row.select_one("img")
                image_url = None
                if img_el:
                    image_url = img_el.get("data-src") or img_el.get("src")

                photo_count: int | None = None
                photo_el = row.select_one("span.photo-count") or row.select_one("span.resim-sayisi")
                if photo_el:
                    m = re.search(r"(\d+)", photo_el.get_text())
                    if m:
                        photo_count = int(m.group(1))

                # Condition (new/used)
                condition: str | None = None
                attrs_el = row.select("td.searchResultsAttributeValue")
                for attr_el in attrs_el:
                    attr_text = attr_el.get_text(strip=True).lower()
                    if "sıfır" in attr_text or "yeni" in attr_text:
                        condition = "new"
                    elif "ikinci el" in attr_text or "kullanılmış" in attr_text:
                        condition = "used"

                # Seller type
                seller_type: str | None = None
                seller_el = row.select_one("td.searchResultsSmallText")
                if seller_el:
                    seller_text = seller_el.get_text(strip=True).lower()
                    if "mağaza" in seller_text:
                        seller_type = "store"
                    elif "bireysel" in seller_text:
                        seller_type = "individual"

                specs: dict[str, Any] = {"type": "classified"}
                if location:
                    specs["location"] = location
                if date_str:
                    specs["listing_date"] = date_str
                if condition:
                    specs["condition"] = condition
                if seller_type:
                    specs["seller_type"] = seller_type
                if photo_count is not None:
                    specs["photo_count"] = photo_count

                products.append(
                    Product(
                        name=title,
                        url=listing_url,
                        source="sahibinden",
                        discounted_price=price,
                        currency="TRY",
                        image_url=image_url,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("row parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single listing from Sahibinden."""
        if not _BS4_AVAILABLE:
            return None

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("product fetch failed", url=url, error=str(exc))
            return None

        if response.status_code != 200:
            return None

        return self._parse_listing(url, response.text)

    def _parse_listing(self, url: str, html: str) -> Product | None:
        """Parse a single Sahibinden listing page."""
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

        # Title
        title_el = soup.select_one("h1.classifiedDetailTitle") or soup.select_one("h1")
        if not title_el:
            return None
        title = normalize_product_name(title_el.get_text(strip=True))

        # Price
        price_el = soup.select_one("div.classifiedInfo h3") or soup.select_one("span.price")
        price: float | None = None
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        # Specs from info list
        specs: dict[str, Any] = {"type": "classified"}
        try:
            info_items = soup.select("ul.classifiedInfoList li")
            for item in info_items:
                label_el = item.select_one("strong")
                value_el = item.select_one("span")
                if label_el and value_el:
                    k = label_el.get_text(strip=True).rstrip(":")
                    v = value_el.get_text(strip=True)
                    if k and v:
                        specs[k] = v
        except Exception:
            pass

        # Image
        img_el = soup.select_one("img.classifiedDetailMainPhoto") or soup.select_one("div.classifiedDetailPhoto img")
        image_url = None
        if img_el:
            image_url = img_el.get("data-src") or img_el.get("src")

        return Product(
            name=title,
            url=url,
            source="sahibinden",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            fetched_at=now_iso,
        )

    # ------------------------------------------------------------------
    # get_reviews (not applicable for classifieds)
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Sahibinden is a classifieds platform -- no product reviews."""
        return []

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 500:
            return False

        # Bot detection
        block_markers = ("captcha", "robot", "güvenlik doğrulaması")
        for marker in block_markers:
            if marker in text.lower():
                logger.warning("possible bot detection", domain=self.domain)
                return False

        markers = ("sahibinden", "searchResults", "classifiedDetail")
        return any(marker in text for marker in markers)
