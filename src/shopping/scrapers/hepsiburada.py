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
# Working parameter: sku= (not skuId=). Response nests items under
# data.approvedUserContent.approvedUserContentList with 'review.content' and 'star'.

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
        """Parse search results from Hepsiburada HTML.

        Hepsiburada uses CSS modules (hashed class names) so we rely on
        ``data-test-id`` attributes and element structure instead of class names.

        Card structure (as of 2026-03):
          <article class="productCard-module_article__*">
            <h2 data-test-id="title-N"><a title="Product Name" href="/slug-p-HBCV...">
            <div data-test-id="final-price-N">PRICE TL</div>
        """
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        import re as _re

        # Cards are <article> elements whose class contains "productCard"
        cards = soup.find_all("article", class_=_re.compile("productCard"))

        # Fallback: old-style selectors
        if not cards:
            cards = (
                soup.select("li[data-productid]")
                or soup.select("div[data-test-id='product-card']")
            )

        for card in cards[:max_results]:
            try:
                # Name + URL: <h2 data-test-id="title-N"><a title="..." href="...">
                title_el = card.find(
                    attrs={"data-test-id": _re.compile(r"^title-\d+$")}
                )
                if title_el is None:
                    # old-style fallback
                    title_el = (
                        card.find("h3")
                        or card.find("a", title=True)
                    )
                if title_el is None:
                    continue

                name_link = title_el.find("a", href=True) if title_el.name != "a" else title_el
                if name_link is None:
                    continue

                name = normalize_product_name(
                    name_link.get("title", "") or name_link.get_text(strip=True)
                )
                if not name:
                    continue

                href = name_link.get("href", "")
                product_url = (
                    href if href.startswith("http") else f"{_BASE_URL}{href}"
                )

                # Price: <div data-test-id="final-price-N">
                price_el = card.find(
                    attrs={"data-test-id": _re.compile(r"final-price-\d+")}
                ) or card.find(
                    attrs={"data-test-id": _re.compile(r"price-current")}
                )
                discounted_price: float | None = None
                if price_el:
                    discounted_price = parse_turkish_price(
                        price_el.get_text(strip=True)
                    )

                # Original price (crossed out)
                orig_el = card.find(
                    attrs={"data-test-id": _re.compile(r"original-price|list-price")}
                )
                original_price: float | None = None
                if orig_el:
                    original_price = parse_turkish_price(orig_el.get_text(strip=True))

                discount_pct: float | None = None
                if original_price and discounted_price and original_price > discounted_price:
                    discount_pct = round(
                        (1 - discounted_price / original_price) * 100, 1
                    )

                # SKU: extract HB product ID from URL (-p-HBCV... or -pm-HBV...)
                sku: str | None = None
                sku_m = _re.search(r"-p[m]?-([A-Za-z0-9]+)", href)
                if sku_m:
                    sku = sku_m.group(1)

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="hepsiburada",
                        original_price=original_price,
                        discounted_price=discounted_price,
                        discount_percentage=discount_pct,
                        currency="TRY",
                        sku=sku,
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

    async def get_reviews(self, url: str, *, max_pages: int = 5) -> list[dict]:
        """Fetch reviews via the Hepsiburada review API.

        API endpoint: ``/queryapi/v2/ApprovedUserContents?sku=<SKU>&page=N&pageSize=20``
        Response nests items under ``data.approvedUserContent.approvedUserContentList``.
        Each item has ``review.content`` (text) and ``star`` (rating).

        Returns partial results if later pages time out (pages 1..N-1 kept).
        """
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

        # --- Strategy 1: review API ---
        # Try the URL-derived SKU first, then alternative param names if no rows.
        sku_candidates = [sku]
        # Some Hepsiburada products have SKU and merchantSku; try both with/without prefix.
        if sku.startswith("HBV") or sku.startswith("HBC"):
            sku_candidates.append(sku)  # already canonical

        for page in range(1, max_pages + 1):
            try:
                # Note: working param is 'sku=' (not 'skuId=' which returns 400)
                api_url = (
                    f"{_REVIEW_API}?sku={sku}"
                    f"&page={page}&pageSize=20&orderBy=MostRecent"
                )
                response = await self._fetch(api_url)
            except Exception as exc:
                logger.error(
                    "review fetch failed", url=url, page=page, error=str(exc)
                )
                break  # Return partial results

            if response.status_code != 200:
                logger.warning(
                    "review non-200", url=url, page=page, status=response.status_code
                )
                break

            page_reviews = self._parse_reviews(response)
            if not page_reviews:
                break

            all_reviews.extend(page_reviews)

            if page < max_pages:
                import asyncio as _asyncio
                await _asyncio.sleep(2.0)

        # --- Strategy 2: product-page JSON fallback ---
        # If the API returned nothing (blocked, sku mismatch, etc.), parse
        # embedded reviews from the product page itself.
        if not all_reviews:
            try:
                page_reviews = await self._reviews_from_product_page(url)
                if page_reviews:
                    all_reviews.extend(page_reviews)
                    logger.info("hb reviews via product-page fallback", count=len(page_reviews))
            except Exception as exc:
                logger.debug("hb product-page review fallback failed", error=str(exc))

        # Cache
        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "hepsiburada")
            except Exception as exc:
                logger.debug("reviews cache write failed", error=str(exc))

        logger.info("reviews fetched", url=url, count=len(all_reviews))
        return all_reviews

    def _parse_reviews(self, response: httpx.Response) -> list[dict]:
        """Parse a single page of reviews from the API response.

        Handles two response shapes:
        - New (v2): ``data.approvedUserContent.approvedUserContentList``
          with item fields ``review.content``, ``star``, ``createdAt``,
          ``customer.displayName``, ``reactions.likeCount``.
        - Legacy: ``data.approvedUserContents`` with ``review``/``comment``, ``star``.
        """
        reviews: list[dict] = []
        try:
            data = response.json()
        except Exception as exc:
            logger.debug("review JSON decode failed", error=str(exc))
            return []

        # New response structure (v2 with sku= param)
        items = (
            data.get("data", {})
            .get("approvedUserContent", {})
            .get("approvedUserContentList", [])
        )

        # Legacy fallback paths
        if not items:
            items = (
                data.get("data", {}).get("approvedUserContents", [])
                or data.get("approvedUserContents", [])
            )

        for item in items:
            try:
                # New structure: review is nested under 'review' dict
                review_obj = item.get("review", {})
                if isinstance(review_obj, dict):
                    text = review_obj.get("content", "")
                else:
                    text = ""

                # Legacy structure: review is a plain string field
                if not text:
                    text = item.get("review") or item.get("comment", "")

                if not text:
                    continue

                # Star rating
                rating = _safe_float(item.get("star") or item.get("rate"))

                # Date
                date = item.get("createdAt") or item.get("reviewDate") or item.get("contentUpdatedAt")

                # Author: new structure has nested 'customer'
                customer = item.get("customer", {})
                if isinstance(customer, dict):
                    author = (
                        customer.get("displayName")
                        or customer.get("name")
                        or item.get("displayCustomerName")
                    )
                else:
                    author = item.get("nickname") or item.get("userName") or item.get("displayCustomerName")

                # Helpful count: nested 'reactions'
                reactions = item.get("reactions", {})
                helpful_count = 0
                if isinstance(reactions, dict):
                    helpful_count = _safe_int(reactions.get("likeCount")) or 0
                else:
                    helpful_count = _safe_int(item.get("likeCount")) or 0

                # Seller name: nested 'order'
                order = item.get("order", {})
                seller = None
                if isinstance(order, dict):
                    seller = order.get("merchantName")
                if not seller:
                    seller = item.get("sellerName") or item.get("merchantName")

                review_dict: dict[str, Any] = {
                    "text": text,
                    "source": "hepsiburada",
                    "rating": rating,
                    "date": date,
                    "author": author,
                    "helpful_count": helpful_count,
                    "verified_purchase": bool(item.get("isPurchaseVerified")),
                }

                if seller:
                    review_dict["seller_name"] = seller

                reviews.append(review_dict)
            except Exception as exc:
                logger.debug("review item parse error", error=str(exc))
                continue

        return reviews

    async def _reviews_from_product_page(self, url: str) -> list[dict]:
        """Fallback: parse reviews embedded in the product page HTML.

        Hepsiburada product pages embed a Redux store under
        ``<script id="reduxStore">`` with reviews inside
        ``ProductReviews`` / ``approvedUserContents`` keys, plus a
        JSON-LD ``Product`` block with ``review[]``.
        """
        try:
            response = await self._fetch(url)
        except Exception:
            return []
        if response.status_code != 200:
            return []
        html = response.text

        reviews: list[dict] = []

        # Try JSON-LD Product.review first (cheap regex)
        try:
            for m in re.finditer(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                html,
                re.DOTALL | re.IGNORECASE,
            ):
                try:
                    blob = json.loads(m.group(1))
                except Exception:
                    continue
                blocks = blob if isinstance(blob, list) else [blob]
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    rs = b.get("review") or b.get("reviews") or []
                    if isinstance(rs, dict):
                        rs = [rs]
                    for r in rs:
                        if not isinstance(r, dict):
                            continue
                        body = r.get("reviewBody") or r.get("description") or ""
                        if not body:
                            continue
                        rating_obj = r.get("reviewRating") or {}
                        rating = None
                        if isinstance(rating_obj, dict):
                            rating = _safe_float(rating_obj.get("ratingValue"))
                        author_obj = r.get("author") or {}
                        author = author_obj.get("name") if isinstance(author_obj, dict) else author_obj
                        reviews.append({
                            "text": body,
                            "source": "hepsiburada",
                            "rating": rating,
                            "date": r.get("datePublished"),
                            "author": author,
                            "helpful_count": 0,
                            "verified_purchase": False,
                        })
        except Exception as exc:
            logger.debug("hb JSON-LD review parse error", error=str(exc))

        if reviews:
            return reviews

        # Try embedded approvedUserContents in inline scripts
        try:
            for m in re.finditer(
                r'"approvedUserContentList"\s*:\s*(\[.*?\])\s*,',
                html,
                re.DOTALL,
            ):
                try:
                    arr = json.loads(m.group(1))
                except Exception:
                    continue
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    rev_obj = item.get("review", {})
                    text = rev_obj.get("content") if isinstance(rev_obj, dict) else None
                    if not text:
                        continue
                    customer = item.get("customer", {})
                    author = customer.get("displayName") if isinstance(customer, dict) else None
                    reviews.append({
                        "text": text,
                        "source": "hepsiburada",
                        "rating": _safe_float(item.get("star")),
                        "date": item.get("createdAt"),
                        "author": author,
                        "helpful_count": 0,
                        "verified_purchase": bool(item.get("isPurchaseVerified")),
                    })
                if reviews:
                    break
        except Exception as exc:
            logger.debug("hb inline-script review parse error", error=str(exc))

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
        block_markers = ("captcha", "challenge-platform", "are you human", "are you a robot")
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
