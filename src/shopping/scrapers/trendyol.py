"""Trendyol scraper -- Turkey's largest e-commerce marketplace.

Primary: HTML scraping of ``www.trendyol.com/sr`` (TLS tier required).
Fallback: ``public.trendyol.com`` JSON API (may be unavailable - DNS issues
observed as of 2026-03).  Product detail pages use ``__NEXT_DATA__`` embedded
JSON first, then structured data extraction.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import (
    cache_product,
    cache_search,
    get_cached_product,
    get_cached_search,
    get_cached_reviews,
    cache_reviews,
)
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.trendyol")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

# NOTE: public.trendyol.com has DNS failures as of 2026-03 -- kept as fallback
_SEARCH_API = (
    "https://public.trendyol.com/discovery-web-searchgw-service"
    "/v2/api/infinite-scroll/sr"
)
# Primary HTML search endpoint (requires TLS tier, returns 403 on plain HTTP)
_SEARCH_HTML_URL = "https://www.trendyol.com/sr"
_REVIEW_API = "https://public-mdc.trendyol.com/discovery-web-socialgw-service/api"
# Fallback review source: product's /yorumlar HTML page (STEALTH tier required)
_REVIEW_HTML_BASE = "https://www.trendyol.com"
_PRODUCT_BASE = "https://www.trendyol.com"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("trendyol")
class TrendyolScraper(BaseScraper):
    """Scrape product data from Trendyol via public APIs."""

    def __init__(self) -> None:
        super().__init__(domain="trendyol")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Trendyol.

        Strategy 1: scrape ``www.trendyol.com/sr`` HTML (TLS bypass, primary).
        Strategy 2: ``public.trendyol.com`` JSON API (fallback; DNS may fail).
        """
        # Cache
        try:
            cached = await get_cached_search(query, "trendyol")
            if cached is not None:
                logger.debug("search cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception as exc:
            logger.debug("search cache lookup failed", error=str(exc))

        products: list[Product] = []

        # --- Strategy 1: HTML scraping ---
        try:
            import urllib.parse as _up
            html_url = f"{_SEARCH_HTML_URL}?q={_up.quote(query, safe='')}"
            response = await self._fetch(html_url)
            if response.status_code == 200:
                products = self._parse_search_html(response.text, max_results)
                if products:
                    logger.info("search HTML parsed", count=len(products))
        except Exception as exc:
            logger.warning("search HTML fetch failed", query=query, error=str(exc))

        # --- Strategy 2: JSON API fallback ---
        if not products:
            params = {
                "q": query,
                "qt": query,
                "st": query,
                "os": "1",
                "pi": "1",
                "culture": "tr-TR",
                "userGenderId": "0",
                "pId": "0",
                "scoringAlgorithmId": "2",
                "categoryRelevancyEnabled": "false",
                "isLegalRequirementConfirmed": "false",
                "searchStrategyType": "DEFAULT",
                "productStampType": "TypeA",
            }
            try:
                response = await self._fetch(_SEARCH_API, params=params)
                if response.status_code == 200:
                    products = self._parse_search_response(response, max_results)
                else:
                    logger.warning(
                        "search API non-200", query=query, status=response.status_code
                    )
            except Exception as exc:
                logger.error("search API fetch failed", query=query, error=str(exc))

        # Cache
        if products:
            try:
                await cache_search(
                    query, "trendyol", [self._product_to_dict(p) for p in products]
                )
            except Exception as exc:
                logger.debug("search cache write failed", error=str(exc))

        return products

    def _parse_search_response(
        self, response: httpx.Response, max_results: int
    ) -> list[Product]:
        """Parse products from the Trendyol search API JSON response."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            data = response.json()
        except Exception as exc:
            logger.error("search JSON decode failed", error=str(exc))
            return []

        # The response nests products under "result.products"
        result = data.get("result", data)
        items = result.get("products", [])

        for item in items[:max_results]:
            try:
                product = self._parse_search_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("search item parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    def _parse_search_html(self, html: str, max_results: int) -> list[Product]:
        """Parse search results from Trendyol's HTML search page.

        Trendyol renders ``<a class="product-card">`` elements server-side.
        Each card contains ``span.product-brand``, ``span.product-name``,
        ``div.price-section``, and an ``<img>`` tag.
        """
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("bs4 not available -- Trendyol HTML parsing disabled")
            return []

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("HTML parse failed", error=str(exc))
            return []

        cards = soup.find_all("a", class_="product-card")
        logger.debug("trendyol HTML cards found", count=len(cards))

        for card in cards[:max_results]:
            try:
                href = card.get("href", "")
                if not href:
                    continue
                product_url = (
                    href if href.startswith("http") else f"{_PRODUCT_BASE}{href}"
                )

                # Brand + product name
                brand_el = card.select_one("span.product-brand")
                name_el = card.select_one("span.product-name")
                brand = brand_el.get_text(strip=True) if brand_el else ""
                pname = name_el.get_text(strip=True) if name_el else ""
                if not pname and not brand:
                    continue
                full_name = f"{brand} {pname}".strip() if brand else pname
                full_name = normalize_product_name(full_name)

                # Price (single or discounted)
                # Trendyol shows: div.price-section (current) and
                # optionally div.original-price (crossed out)
                price_el = card.select_one(
                    "div.price-section, [data-testid='price-section']"
                )
                discounted_price: float | None = None
                if price_el:
                    discounted_price = parse_turkish_price(
                        price_el.get_text(strip=True)
                    )

                orig_price_el = card.select_one(
                    "div.original-price, [data-testid='original-price']"
                )
                original_price: float | None = None
                if orig_price_el:
                    original_price = parse_turkish_price(
                        orig_price_el.get_text(strip=True)
                    )

                discount_pct: float | None = None
                if original_price and discounted_price and original_price > discounted_price:
                    discount_pct = round(
                        (1 - discounted_price / original_price) * 100, 1
                    )

                # Image
                img_el = card.select_one("img")
                image_url: str | None = None
                if img_el:
                    src = img_el.get("src") or img_el.get("data-src", "")
                    if src and not src.startswith("data:"):
                        image_url = src

                products.append(
                    Product(
                        name=full_name,
                        url=product_url,
                        source="trendyol",
                        original_price=original_price,
                        discounted_price=discounted_price,
                        discount_percentage=discount_pct,
                        currency="TRY",
                        image_url=image_url,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("HTML card parse error", error=str(exc))
                continue

        return products

    def _parse_search_item(self, item: dict, now_iso: str) -> Product | None:
        """Convert a single search API item to a Product."""
        name = item.get("name") or item.get("brandName", "")
        if not name:
            return None

        brand = item.get("brandName", "")
        if brand and not name.lower().startswith(brand.lower()):
            name = f"{brand} {name}"

        name = normalize_product_name(name)

        # URL
        url_fragment = item.get("url", "")
        product_url = (
            url_fragment
            if url_fragment.startswith("http")
            else f"{_PRODUCT_BASE}{url_fragment}"
        )

        # Prices
        price_info = item.get("price", {})
        original_price: float | None = None
        discounted_price: float | None = None
        discount_pct: float | None = None

        if isinstance(price_info, dict):
            original_price = _safe_float(
                price_info.get("originalPrice")
                or price_info.get("sellingPrice")
            )
            discounted_price = _safe_float(
                price_info.get("sellingPrice")
                or price_info.get("discountedPrice")
            )
            dp = price_info.get("discountedPrice")
            if dp is not None:
                discounted_price = _safe_float(dp)
            discount_pct = _safe_float(price_info.get("discountRatio"))
        elif isinstance(price_info, (int, float)):
            discounted_price = float(price_info)

        # If no explicit discount percentage, calculate it
        if (
            discount_pct is None
            and original_price
            and discounted_price
            and original_price > discounted_price
        ):
            discount_pct = round((1 - discounted_price / original_price) * 100, 1)

        # Rating
        rating_score = _safe_float(
            item.get("ratingScore", {}).get("averageRating")
            if isinstance(item.get("ratingScore"), dict)
            else None
        )
        review_count = _safe_int(
            item.get("ratingScore", {}).get("totalRatingCount")
            if isinstance(item.get("ratingScore"), dict)
            else None
        )

        # Image
        images = item.get("images", [])
        image_url: str | None = None
        if images:
            first = images[0] if isinstance(images[0], str) else str(images[0])
            if first.startswith("http"):
                image_url = first
            else:
                image_url = f"https://cdn.dsmcdn.com/{first}"

        # Seller
        merchant = item.get("merchantName") or item.get("merchant", {})
        seller_name: str | None = None
        if isinstance(merchant, str):
            seller_name = merchant
        elif isinstance(merchant, dict):
            seller_name = merchant.get("name")

        # Promotions / campaign badges
        promotions: list[str] = []
        for promo in item.get("promotions", []):
            if isinstance(promo, dict):
                text = promo.get("text") or promo.get("name", "")
                if text:
                    promotions.append(text)
            elif isinstance(promo, str):
                promotions.append(promo)

        # Free shipping
        free_ship = item.get("freeCargo", False) or item.get("freeShipping", False)
        rush_delivery = item.get("rushDeliveryDuration")

        # Specs dict
        specs: dict[str, Any] = {}
        if promotions:
            specs["promotions"] = promotions
        if rush_delivery:
            specs["rush_delivery"] = rush_delivery

        category_id = item.get("categoryId")
        category_name = item.get("categoryName")
        category_path = (
            f"{category_id}:{category_name}"
            if category_id and category_name
            else category_name
        )

        return Product(
            name=name,
            url=product_url,
            source="trendyol",
            original_price=original_price,
            discounted_price=discounted_price,
            discount_percentage=discount_pct,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating_score,
            review_count=review_count,
            seller_name=seller_name,
            free_shipping=bool(free_ship),
            category_path=category_path,
            fetched_at=now_iso,
        )

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product page from Trendyol.

        Tries the embedded ``__NEXT_DATA__`` JSON first; falls back to
        structured data extraction.
        """
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

        product = self._parse_product_page(url, response.text)

        if product is not None:
            try:
                await cache_product(
                    url, self._product_to_dict(product), "trendyol", "prices"
                )
            except Exception as exc:
                logger.debug("product cache write failed", error=str(exc))

        return product

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse product data from the Trendyol product page."""
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Try __NEXT_DATA__ ---
        next_data = self._extract_next_data(html)
        if next_data:
            product = self._product_from_next_data(url, next_data, now_iso)
            if product is not None:
                return product

        # --- Fallback to structured data (JSON-LD / OpenGraph) ---
        structured = self.extract_structured_data(html)
        return self._product_from_structured(url, structured, html, now_iso)

    @staticmethod
    def _extract_next_data(html: str) -> dict | None:
        """Extract the ``__NEXT_DATA__`` JSON blob from page HTML."""
        try:
            m = re.search(
                r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            if m:
                return json.loads(m.group(1))
        except Exception as exc:
            logger.debug("__NEXT_DATA__ extraction failed", error=str(exc))
        return None

    def _product_from_next_data(
        self, url: str, data: dict, now_iso: str
    ) -> Product | None:
        """Build a Product from Trendyol __NEXT_DATA__."""
        try:
            props = data.get("props", {}).get("pageProps", {})
            product_data = props.get("product", props.get("productDetail", {}))
            if not product_data:
                return None

            name = product_data.get("name", "")
            brand = product_data.get("brand", {})
            brand_name = brand.get("name", "") if isinstance(brand, dict) else str(brand)
            if brand_name and not name.lower().startswith(brand_name.lower()):
                name = f"{brand_name} {name}"

            name = normalize_product_name(name)

            # Prices
            original_price = _safe_float(product_data.get("originalPrice"))
            selling_price = _safe_float(
                product_data.get("price", {}).get("sellingPrice")
                if isinstance(product_data.get("price"), dict)
                else product_data.get("price")
            )
            discounted_price = selling_price or _safe_float(
                product_data.get("discountedPrice")
            )

            discount_pct: float | None = None
            if original_price and discounted_price and original_price > discounted_price:
                discount_pct = round(
                    (1 - discounted_price / original_price) * 100, 1
                )

            # Rating
            rating_data = product_data.get("ratingScore", {})
            rating: float | None = None
            review_count: int | None = None
            if isinstance(rating_data, dict):
                rating = _safe_float(rating_data.get("averageRating"))
                review_count = _safe_int(rating_data.get("totalRatingCount"))

            # Specs / attributes
            specs: dict[str, Any] = {}
            for attr in product_data.get("attributes", []):
                if isinstance(attr, dict):
                    k = attr.get("key", {})
                    v = attr.get("value", {})
                    key_name = k.get("name", "") if isinstance(k, dict) else str(k)
                    val_name = v.get("name", "") if isinstance(v, dict) else str(v)
                    if key_name and val_name:
                        specs[key_name] = val_name

            # Seller
            merchant = product_data.get("merchant", {})
            seller_name: str | None = None
            seller_rating: float | None = None
            seller_review_count: int | None = None
            if isinstance(merchant, dict):
                seller_name = merchant.get("name")
                seller_rating = _safe_float(merchant.get("sellerScore"))
                seller_review_count = _safe_int(merchant.get("reviewCount"))

            # Promotions
            promotions: list[str] = []
            for promo in product_data.get("promotions", []):
                if isinstance(promo, dict):
                    text = promo.get("text") or promo.get("name", "")
                    if text:
                        promotions.append(text)
            if promotions:
                specs["promotions"] = promotions

            # Installment info
            installment = product_data.get("installment")
            installment_info: dict | None = None
            if isinstance(installment, dict):
                installment_info = {
                    "count": installment.get("installmentCount"),
                    "amount": _safe_float(installment.get("installmentPrice")),
                    "total": _safe_float(installment.get("totalPrice")),
                }

            # Cargo / shipping
            cargo = product_data.get("cargo", {})
            free_shipping = False
            shipping_time_days: int | None = None
            if isinstance(cargo, dict):
                free_shipping = bool(cargo.get("isFreeShipping", False))
                delivery_date = cargo.get("deliveryDate")
                if delivery_date:
                    specs["estimated_delivery"] = delivery_date
                rush = cargo.get("rushDeliveryDuration")
                if rush:
                    specs["rush_delivery"] = rush

            # Image
            images = product_data.get("images", [])
            image_url: str | None = None
            if images:
                first = images[0] if isinstance(images[0], str) else str(images[0])
                if first.startswith("http"):
                    image_url = first
                else:
                    image_url = f"https://cdn.dsmcdn.com/{first}"

            # Category
            category = product_data.get("category", {})
            category_path: str | None = None
            if isinstance(category, dict):
                hierarchy = category.get("hierarchy", "")
                category_path = hierarchy if hierarchy else category.get("name")

            return Product(
                name=name,
                url=url,
                source="trendyol",
                original_price=original_price,
                discounted_price=discounted_price,
                discount_percentage=discount_pct,
                currency="TRY",
                image_url=image_url,
                specs=specs,
                rating=rating,
                review_count=review_count,
                seller_name=seller_name,
                seller_rating=seller_rating,
                seller_review_count=seller_review_count,
                free_shipping=free_shipping,
                shipping_time_days=shipping_time_days,
                installment_info=installment_info,
                category_path=category_path,
                fetched_at=now_iso,
            )

        except Exception as exc:
            logger.error("__NEXT_DATA__ product parse failed", error=str(exc))
            return None

    def _product_from_structured(
        self,
        url: str,
        structured: dict,
        html: str,
        now_iso: str,
    ) -> Product | None:
        """Build a Product from JSON-LD / OpenGraph as a last resort."""
        try:
            ld = structured.get("json_ld", {})
            og = structured.get("opengraph", {})

            name = ""
            if isinstance(ld, dict):
                name = ld.get("name", "")
            if not name:
                name = og.get("title", "")
            if not name:
                # Try <title> tag
                m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()

            if not name:
                return None

            name = normalize_product_name(name)

            # Price from JSON-LD offers
            discounted_price: float | None = None
            original_price: float | None = None
            if isinstance(ld, dict):
                offers = ld.get("offers", {})
                if isinstance(offers, dict):
                    discounted_price = _safe_float(
                        offers.get("lowPrice") or offers.get("price")
                    )
                    original_price = _safe_float(offers.get("highPrice"))
                elif isinstance(offers, list) and offers:
                    discounted_price = _safe_float(offers[0].get("price"))

            # Image from OG
            image_url = og.get("image")

            return Product(
                name=name,
                url=url,
                source="trendyol",
                original_price=original_price,
                discounted_price=discounted_price,
                currency="TRY",
                image_url=image_url,
                fetched_at=now_iso,
            )
        except Exception as exc:
            logger.error("structured data product parse failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 5) -> list[dict]:
        """Fetch product reviews from Trendyol.

        Strategy 1: ``public-mdc.trendyol.com`` review API (fast, JSON).
        Strategy 2: ``/yorumlar`` HTML page scrape (STEALTH tier, slower but
          reliable when the API subdomain has DNS issues -- observed 2026-03).

        Returns partial results if later pages time out (pages 1-N-1 kept).
        """
        # Cache
        try:
            cached = await get_cached_reviews(url, "trendyol")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception as exc:
            logger.debug("reviews cache lookup failed", error=str(exc))

        # Extract content ID from URL
        content_id = self._extract_content_id(url)
        if content_id is None:
            logger.warning("could not extract content ID from URL", url=url)
            return []

        all_reviews: list[dict] = []
        api_failed = False

        # --- Strategy 1: JSON API ---
        for page in range(1, max_pages + 1):
            try:
                reviews_url = (
                    f"{_REVIEW_API}/reviews/{content_id}"
                    f"?page={page}&culture=tr-TR&order=MOST_RECENT"
                )
                response = await self._fetch(reviews_url)
            except Exception as exc:
                logger.warning(
                    "review API fetch failed, will fall back to HTML",
                    url=url, page=page, error=str(exc),
                )
                api_failed = True
                break

            if response.status_code != 200:
                logger.warning(
                    "review API non-200, will fall back to HTML",
                    url=url, page=page, status=response.status_code,
                )
                api_failed = True
                break

            page_reviews = self._parse_review_response(response)
            if not page_reviews:
                break  # No more reviews

            all_reviews.extend(page_reviews)
            import asyncio as _asyncio
            await _asyncio.sleep(2.0)

        # --- Strategy 2: HTML /yorumlar fallback ---
        if api_failed or not all_reviews:
            logger.info("falling back to /yorumlar HTML scraping", url=url)
            html_reviews = await self._get_reviews_from_html(
                url, content_id, max_pages=max_pages
            )
            if html_reviews:
                all_reviews = html_reviews

        # Cache
        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "trendyol")
            except Exception as exc:
                logger.debug("reviews cache write failed", error=str(exc))

        logger.info("reviews fetched", url=url, count=len(all_reviews))
        return all_reviews

    async def _get_reviews_from_html(
        self, url: str, content_id: str, *, max_pages: int = 5
    ) -> list[dict]:
        """Scrape reviews from the ``/yorumlar`` HTML page (STEALTH tier).

        The page embeds reviews in inline JS as JSON-like objects.
        Returns partial results if a later page times out.
        """
        import asyncio as _asyncio

        # Build the base /yorumlar URL from the product URL
        base_url = url.split("?")[0].rstrip("/")
        if not base_url.endswith("/yorumlar"):
            base_url = f"{base_url}/yorumlar"

        all_reviews: list[dict] = []

        try:
            from src.tools.scraper import scrape_url, ScrapeTier
        except ImportError:
            logger.warning("scrape_url not available for /yorumlar fallback")
            return []

        for page in range(1, max_pages + 1):
            page_url = f"{base_url}?page={page}"
            try:
                result = await scrape_url(page_url, max_tier=ScrapeTier.STEALTH, timeout=25.0)
            except Exception as exc:
                logger.warning(
                    "yorumlar page fetch failed",
                    page=page, error=str(exc),
                )
                break  # Return what we have so far

            if not result.ok:
                logger.warning(
                    "yorumlar page non-OK",
                    page=page, status=result.status,
                )
                break

            page_reviews = self._parse_yorumlar_html(result.html)
            if not page_reviews:
                break  # No more reviews

            all_reviews.extend(page_reviews)
            logger.debug("yorumlar page parsed", page=page, count=len(page_reviews))

            if page < max_pages:
                await _asyncio.sleep(2.5)

        return all_reviews

    @staticmethod
    def _parse_yorumlar_html(html: str) -> list[dict]:
        """Parse reviews from the Trendyol /yorumlar HTML page.

        Reviews are embedded in inline JS as JSON fragments with fields:
        ``comment``, ``rate``, ``userFullName``, ``createdDate``, ``appealCount``.
        """
        reviews: list[dict] = []

        # Match review blocks: each starts with "comment":"..." and has rate/userFullName
        # Pattern: {"comment":"...","rate":N,...,"userFullName":"...","createdDate":EPOCH,...}
        block_pattern = re.compile(
            r'"comment"\s*:\s*"(?P<comment>[^"]+)"'
            r'.*?"rate"\s*:\s*(?P<rate>\d+)'
            r'(?:.*?"userFullName"\s*:\s*"(?P<author>[^"]*)")?'
            r'(?:.*?"createdDate"\s*:\s*(?P<ts>\d{13}))?'
            r'(?:.*?"appealCount"\s*:\s*(?P<helpful>\d+))?',
            re.DOTALL,
        )

        # Scan blocks in smaller windows to avoid massive backtracking
        chunk_size = 4000
        step = 3500
        seen: set[str] = set()

        for start in range(0, len(html), step):
            chunk = html[start : start + chunk_size]
            for m in block_pattern.finditer(chunk):
                comment = m.group("comment")
                if not comment or comment in seen:
                    continue
                seen.add(comment)

                rate = _safe_float(m.group("rate"))
                author = m.group("author") or None
                ts = m.group("ts")
                date_str: str | None = None
                if ts:
                    try:
                        date_str = datetime.fromtimestamp(
                            int(ts) / 1000, tz=timezone.utc
                        ).isoformat()
                    except Exception:
                        pass
                helpful = _safe_int(m.group("helpful"))

                reviews.append({
                    "text": comment,
                    "source": "trendyol",
                    "rating": rate,
                    "date": date_str,
                    "author": author,
                    "helpful_count": helpful or 0,
                })

        return reviews

    @staticmethod
    def _extract_content_id(url: str) -> str | None:
        """Extract the numeric product/content ID from a Trendyol URL.

        Trendyol URLs typically end with ``-p-<id>`` or contain ``contentId=<id>``.
        """
        # Pattern: ...-p-123456 or ...-p-123456?...
        m = re.search(r"-p-(\d+)", url)
        if m:
            return m.group(1)
        # Query param fallback
        m = re.search(r"contentId=(\d+)", url)
        if m:
            return m.group(1)
        return None

    def _parse_review_response(self, response: httpx.Response) -> list[dict]:
        """Parse a single page of reviews from the Trendyol review API."""
        reviews: list[dict] = []

        try:
            data = response.json()
        except Exception as exc:
            logger.error("review JSON decode failed", error=str(exc))
            return []

        result = data.get("result", data)
        items = result.get("productReviews", result.get("reviews", []))

        for item in items:
            try:
                review = self._parse_single_review(item)
                if review:
                    reviews.append(review)
            except Exception as exc:
                logger.debug("review item parse error", error=str(exc))
                continue

        return reviews

    @staticmethod
    def _parse_single_review(item: dict) -> dict | None:
        """Convert a single review API item to a dict."""
        comment = item.get("comment", "")
        if not comment:
            return None

        rating = _safe_float(item.get("rate") or item.get("star"))

        # Date
        created = item.get("createdDate") or item.get("lastModifiedDate")
        date_str: str | None = None
        if isinstance(created, (int, float)):
            try:
                date_str = datetime.fromtimestamp(
                    created / 1000, tz=timezone.utc
                ).isoformat()
            except Exception:
                pass
        elif isinstance(created, str):
            date_str = created

        author = item.get("userFullName") or item.get("nickName")
        seller_name = item.get("sellerName")

        helpful = _safe_int(item.get("appealCount") or item.get("likeCount"))

        review: dict[str, Any] = {
            "text": comment,
            "source": "trendyol",
            "rating": rating,
            "date": date_str,
            "author": author,
            "helpful_count": helpful or 0,
        }

        if seller_name:
            review["seller_name"] = seller_name

        # Seller reply
        reply = item.get("sellerReply")
        if reply and isinstance(reply, dict):
            review["seller_reply"] = reply.get("comment", "")

        return review

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        """Check that the response looks like real Trendyol content."""
        if response.status_code >= 400:
            return False

        content_type = response.headers.get("content-type", "")

        # API responses are JSON
        if "application/json" in content_type:
            try:
                data = response.json()
                # Trendyol API always wraps in "result"
                return isinstance(data, dict) and (
                    "result" in data or "products" in data or "productReviews" in data
                )
            except Exception:
                return False

        # HTML pages
        text = response.text
        if not text or len(text) < 500:
            return False

        markers = ("trendyol", "__NEXT_DATA__", "product", "p-card")
        return any(marker in text.lower() for marker in markers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Try to convert *value* to float; return ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Try to convert *value* to int; return ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
