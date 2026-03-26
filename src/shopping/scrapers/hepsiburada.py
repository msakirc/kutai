"""Hepsiburada scraper -- Turkey's second-largest e-commerce platform.

Marked as a **high-risk** scraper: Hepsiburada employs aggressive bot
detection (Akamai, Datadome).  Every parsing step is wrapped in
try/except so failures degrade gracefully to empty results rather than
crashing the pipeline.

Strategy:
  - search: try ``__NEXT_DATA__`` JSON extraction first, then fall back
    to HTML parsing with BeautifulSoup.
  - get_product: try ``__NEXT_DATA__`` first, then look for known API
    endpoints embedded in the page.
  - get_reviews: hit the Hepsiburada review API endpoint directly.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import (
    cache_product,
    cache_search,
    cache_reviews,
    get_cached_product,
    get_cached_search,
    get_cached_reviews,
)
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.hepsiburada")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- Hepsiburada HTML fallback disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.hepsiburada.com"
_SEARCH_URL = "https://www.hepsiburada.com/ara"
_REVIEW_API = "https://user-content-gw-hermes.hepsiburada.com/queryapi/v2/ApprovedUserContents"

# Flag: this scraper is high-risk for bot detection
HIGH_RISK = True


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("hepsiburada")
class HepsiburadaScraper(BaseScraper):
    """Scrape product data from Hepsiburada.

    High-risk scraper -- aggressive bot detection.  All parsing is
    defensive; failures return empty results.
    """

    high_risk: bool = True

    def __init__(self) -> None:
        super().__init__(domain="hepsiburada")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Hepsiburada for *query*."""
        # Cache
        try:
            cached = await get_cached_search(query, "hepsiburada")
            if cached is not None:
                logger.debug("search cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception as exc:
            logger.debug("search cache lookup failed", error=str(exc))

        url = f"{_SEARCH_URL}?q={urllib.parse.quote(query, safe='')}"

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        html = response.text

        # Strategy 1: __NEXT_DATA__
        products = self._parse_search_next_data(html, max_results)

        # Strategy 2: HTML fallback
        if not products and _BS4_AVAILABLE:
            products = self._parse_search_html(html, max_results)

        # Cache results
        if products:
            try:
                await cache_search(
                    query,
                    "hepsiburada",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception as exc:
                logger.debug("search cache write failed", error=str(exc))

        return products

    def _parse_search_next_data(
        self, html: str, max_results: int
    ) -> list[Product]:
        """Try to extract search results from embedded __NEXT_DATA__."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            m = re.search(
                r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            if not m:
                return []

            data = json.loads(m.group(1))
            props = data.get("props", {}).get("pageProps", {})

            # Hepsiburada nests search results in various keys
            items = (
                props.get("products")
                or props.get("searchResult", {}).get("products", [])
                or props.get("productList", {}).get("products", [])
                or []
            )

            for item in items[:max_results]:
                try:
                    product = self._item_to_product(item, now_iso)
                    if product is not None:
                        products.append(product)
                except Exception as exc:
                    logger.debug("next_data item parse error", error=str(exc))
                    continue

        except Exception as exc:
            logger.debug("__NEXT_DATA__ search parse failed", error=str(exc))

        return products

    def _parse_search_html(self, html: str, max_results: int) -> list[Product]:
        """Fallback: parse search results from raw HTML."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        cards = (
            soup.select("li[data-productid]")
            or soup.select("div.product-card")
            or soup.select("div[data-test-id='product-card']")
        )

        for card in cards[:max_results]:
            try:
                # Name
                name_el = (
                    card.select_one("h3")
                    or card.select_one("a[title]")
                    or card.select_one("span.product-title")
                )
                if name_el is None:
                    continue
                name = normalize_product_name(
                    name_el.get("title", "") or name_el.get_text(strip=True)
                )
                if not name:
                    continue

                # URL
                link_el = card.select_one("a[href]")
                href = link_el["href"] if link_el else ""
                product_url = (
                    href
                    if href.startswith("http")
                    else f"{_BASE_URL}{href}"
                )

                # Price
                price_el = (
                    card.select_one("div[data-test-id='price-current-price']")
                    or card.select_one("span.product-price")
                    or card.select_one("span[data-bind*='price']")
                )
                discounted_price: float | None = None
                if price_el:
                    discounted_price = parse_turkish_price(
                        price_el.get_text(strip=True)
                    )

                # Rating
                rating: float | None = None
                rating_el = card.select_one("span.rating-value")
                if rating_el:
                    try:
                        rating = float(rating_el.get_text(strip=True))
                    except (ValueError, TypeError):
                        pass

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="hepsiburada",
                        discounted_price=discounted_price,
                        currency="TRY",
                        rating=rating,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("html card parse error", error=str(exc))
                continue

        logger.info("search HTML parsed", count=len(products))
        return products

    def _item_to_product(self, item: dict, now_iso: str) -> Product | None:
        """Convert a single JSON item to a Product."""
        try:
            name = item.get("name") or item.get("productName", "")
            if not name:
                return None
            name = normalize_product_name(name)

            # URL
            slug = item.get("url") or item.get("slug", "")
            product_url = (
                slug if slug.startswith("http") else f"{_BASE_URL}{slug}"
            )

            # Prices
            original_price = _safe_float(
                item.get("originalPrice") or item.get("listPrice")
            )
            discounted_price = _safe_float(
                item.get("price")
                or item.get("sellingPrice")
                or item.get("salePrice")
            )

            discount_pct: float | None = None
            if (
                original_price
                and discounted_price
                and original_price > discounted_price
            ):
                discount_pct = round(
                    (1 - discounted_price / original_price) * 100, 1
                )

            # Rating
            rating = _safe_float(item.get("rating") or item.get("averageRating"))
            review_count = _safe_int(
                item.get("reviewCount") or item.get("ratingCount")
            )

            # Image
            image_url = item.get("imageUrl") or item.get("image")

            # Seller
            seller_name = item.get("merchantName") or item.get("seller")

            # Free shipping
            free_shipping = bool(
                item.get("freeCargo")
                or item.get("freeShipping")
                or item.get("isFreeShipping")
            )

            return Product(
                name=name,
                url=product_url,
                source="hepsiburada",
                original_price=original_price,
                discounted_price=discounted_price,
                discount_percentage=discount_pct,
                currency="TRY",
                image_url=image_url,
                rating=rating,
                review_count=review_count,
                seller_name=seller_name,
                free_shipping=free_shipping,
                fetched_at=now_iso,
            )
        except Exception as exc:
            logger.debug("item_to_product failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product page from Hepsiburada."""
        # Cache
        try:
            cached = await get_cached_product(url)
            if cached is not None:
                logger.debug("product cache hit", url=url)
                return self._dict_to_product(cached)
        except Exception as exc:
            logger.debug("product cache lookup failed", error=str(exc))

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("product fetch failed", url=url, error=str(exc))
            return None

        if response.status_code != 200:
            logger.warning("product non-200", url=url, status=response.status_code)
            return None

        html = response.text
        now_iso = datetime.now(timezone.utc).isoformat()
        product: Product | None = None

        # Strategy 1: __NEXT_DATA__
        try:
            m = re.search(
                r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            if m:
                data = json.loads(m.group(1))
                props = data.get("props", {}).get("pageProps", {})
                pd = (
                    props.get("product")
                    or props.get("productDetail")
                    or props.get("data", {}).get("product", {})
                    or {}
                )
                if pd:
                    product = self._item_to_product(pd, now_iso)
                    if product:
                        product.url = url
        except Exception as exc:
            logger.debug("product __NEXT_DATA__ parse failed", error=str(exc))

        # Strategy 2: structured data fallback
        if product is None:
            try:
                structured = self.extract_structured_data(html)
                ld = structured.get("json_ld", {})
                og = structured.get("opengraph", {})

                pname = ""
                if isinstance(ld, dict):
                    pname = ld.get("name", "")
                if not pname:
                    pname = og.get("title", "")
                if not pname:
                    return None

                pname = normalize_product_name(pname)

                price: float | None = None
                if isinstance(ld, dict):
                    offers = ld.get("offers", {})
                    if isinstance(offers, dict):
                        price = _safe_float(
                            offers.get("lowPrice") or offers.get("price")
                        )

                product = Product(
                    name=pname,
                    url=url,
                    source="hepsiburada",
                    discounted_price=price,
                    currency="TRY",
                    image_url=og.get("image"),
                    fetched_at=now_iso,
                )
            except Exception as exc:
                logger.debug("product structured parse failed", error=str(exc))

        # Cache on success
        if product is not None:
            try:
                await cache_product(
                    url, self._product_to_dict(product), "hepsiburada", "prices"
                )
            except Exception as exc:
                logger.debug("product cache write failed", error=str(exc))

        return product

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Fetch reviews via the Hepsiburada review API."""
        # Cache
        try:
            cached = await get_cached_reviews(url, "hepsiburada")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception as exc:
            logger.debug("reviews cache lookup failed", error=str(exc))

        sku = self._extract_sku(url)
        if not sku:
            logger.warning("could not extract SKU for reviews", url=url)
            return []

        all_reviews: list[dict] = []

        for page in range(1, max_pages + 1):
            try:
                api_url = (
                    f"{_REVIEW_API}?skuId={sku}"
                    f"&page={page}&pageSize=20&orderBy=MostRecent"
                )
                response = await self._fetch(api_url)
            except Exception as exc:
                logger.error(
                    "review fetch failed", url=url, page=page, error=str(exc)
                )
                break

            if response.status_code != 200:
                logger.warning(
                    "review non-200", url=url, page=page, status=response.status_code
                )
                break

            page_reviews = self._parse_reviews(response)
            if not page_reviews:
                break

            all_reviews.extend(page_reviews)

        # Cache
        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "hepsiburada")
            except Exception as exc:
                logger.debug("reviews cache write failed", error=str(exc))

        logger.info("reviews fetched", url=url, count=len(all_reviews))
        return all_reviews

    def _parse_reviews(self, response: httpx.Response) -> list[dict]:
        """Parse a single page of reviews from the API response."""
        reviews: list[dict] = []
        try:
            data = response.json()
        except Exception as exc:
            logger.debug("review JSON decode failed", error=str(exc))
            return []

        items = data.get("data", {}).get("approvedUserContents", [])
        if not items:
            items = data.get("approvedUserContents", [])

        for item in items:
            try:
                text = item.get("review") or item.get("comment", "")
                if not text:
                    continue

                review: dict[str, Any] = {
                    "text": text,
                    "source": "hepsiburada",
                    "rating": _safe_float(item.get("star") or item.get("rate")),
                    "date": item.get("createdAt") or item.get("reviewDate"),
                    "author": item.get("nickname") or item.get("userName"),
                    "helpful_count": _safe_int(item.get("likeCount")) or 0,
                    "verified_purchase": bool(item.get("isPurchaseVerified")),
                }

                # Seller info if available
                seller = item.get("sellerName") or item.get("merchantName")
                if seller:
                    review["seller_name"] = seller

                reviews.append(review)
            except Exception as exc:
                logger.debug("review item parse error", error=str(exc))
                continue

        return reviews

    @staticmethod
    def _extract_sku(url: str) -> str | None:
        """Extract the product SKU / ID from a Hepsiburada URL."""
        # Pattern: ...-p-HBCV000... or ...-pm-HBCV000...
        m = re.search(r"-p[m]?-([A-Za-z0-9]+)", url)
        if m:
            return m.group(1)
        # Query param fallback
        m = re.search(r"[?&]sku=([^&]+)", url)
        if m:
            return m.group(1)
        return None

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        """Check that the response is real Hepsiburada content."""
        if response.status_code >= 400:
            return False

        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                data = response.json()
                return isinstance(data, dict)
            except Exception:
                return False

        text = response.text
        if not text or len(text) < 500:
            return False

        # Check for bot-detection pages
        block_markers = ("captcha", "challenge-platform", "are you human", "robot")
        for marker in block_markers:
            if marker in text.lower():
                logger.warning("possible bot detection page", domain=self.domain)
                return False

        markers = ("hepsiburada", "__NEXT_DATA__", "product", "listing")
        return any(marker in text.lower() for marker in markers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
