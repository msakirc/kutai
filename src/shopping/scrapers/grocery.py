"""Grocery scrapers -- AktuelKatalog, Getir, Migros.

These scrapers focus on grocery and FMCG product data with an emphasis
on unit price calculation (price per kg/L) for comparison shopping.

All three extract: product name, unit price, price per kg/L, campaign
badge, and stock status.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import cache_search, get_cached_search, cache_reviews, get_cached_reviews
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.grocery")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- grocery scrapers HTML parsing disabled")


# ---------------------------------------------------------------------------
# Unit price helpers
# ---------------------------------------------------------------------------


def _calculate_unit_price(
    price: float | None,
    quantity_text: str | None,
) -> dict[str, Any] | None:
    """Try to calculate price per kg or per litre from product text.

    Returns a dict like ``{"unit_price": 12.50, "unit": "kg"}`` or None.
    """
    if price is None or not quantity_text:
        return None

    quantity_text = quantity_text.lower().strip()

    # Patterns: "500 g", "1.5 kg", "1 lt", "500 ml", "2 L", "750ml"
    patterns = [
        (r"([\d.,]+)\s*kg", "kg", 1.0),
        (r"([\d.,]+)\s*g(?:r)?(?:\b|$)", "kg", 0.001),
        (r"([\d.,]+)\s*l(?:t|itre)?(?:\b|$)", "L", 1.0),
        (r"([\d.,]+)\s*ml", "L", 0.001),
        (r"([\d.,]+)\s*cl", "L", 0.01),
        (r"([\d.,]+)\s*adet", "adet", 1.0),
    ]

    for pattern, unit, multiplier in patterns:
        m = re.search(pattern, quantity_text)
        if m:
            try:
                raw_val = m.group(1).replace(",", ".")
                quantity = float(raw_val) * multiplier
                if quantity > 0:
                    return {
                        "unit_price": round(price / quantity, 2),
                        "unit": unit,
                        "quantity": round(quantity, 3),
                    }
            except (ValueError, ZeroDivisionError):
                continue

    return None


# ---------------------------------------------------------------------------
# AktuelKatalog
# ---------------------------------------------------------------------------


@register_scraper("aktuelkatalog")
class AktuelKatalogScraper(BaseScraper):
    """Scrape weekly promotion catalogs from aktuelkatalog.com.tr.

    NOTE: As of 2026-03 the domain resolves with a DNS failure
    (getaddrinfo failed).  ``is_available = False`` disables search
    calls until the domain is restored.
    """

    _BASE_URL = "https://www.aktuelkatalog.com.tr"
    _SEARCH_URL = "https://www.aktuelkatalog.com.tr/ara"
    # Domain DNS failure — disable until restored
    is_available: bool = False

    def __init__(self) -> None:
        super().__init__(domain="aktuelkatalog")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        if not self.is_available:
            logger.debug("aktuelkatalog search skipped: is_available=False (DNS failure)")
            return []
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_search(query, "aktuelkatalog")
            if cached is not None:
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        params = {"q": query}

        try:
            response = await self._fetch(self._SEARCH_URL, params=params)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            return []

        products = self._parse_search(response.text, max_results)

        if products:
            try:
                await cache_search(
                    query, "aktuelkatalog",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception:
                pass

        return products

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        cards = (
            soup.select("div.product-card")
            or soup.select("div.urun-kart")
            or soup.select("article.product")
        )

        for card in cards[:max_results]:
            try:
                name_el = card.select_one("h3") or card.select_one("a.product-title")
                if not name_el:
                    continue
                name = normalize_product_name(name_el.get_text(strip=True))
                if not name:
                    continue

                link_el = card.select_one("a[href]")
                href = link_el["href"] if link_el else ""
                product_url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

                # Price
                price_el = card.select_one("span.price") or card.select_one("span.fiyat")
                discounted_price = parse_turkish_price(price_el.get_text(strip=True)) if price_el else None

                # Campaign badge
                badge_el = card.select_one("span.badge") or card.select_one("span.kampanya")
                campaign_badge = badge_el.get_text(strip=True) if badge_el else None

                # Store / market name
                store_el = card.select_one("span.market") or card.select_one("img[alt]")
                store_name = None
                if store_el:
                    store_name = store_el.get("alt", "") or store_el.get_text(strip=True)

                # Unit price calculation
                specs: dict[str, Any] = {"type": "grocery_promo"}
                if campaign_badge:
                    specs["campaign_badge"] = campaign_badge
                if store_name:
                    specs["store"] = store_name

                unit_info = _calculate_unit_price(discounted_price, name)
                if unit_info:
                    specs["unit_price"] = unit_info["unit_price"]
                    specs["unit"] = unit_info["unit"]
                    specs["quantity"] = unit_info["quantity"]

                # Image
                img_el = card.select_one("img")
                image_url = None
                if img_el:
                    image_url = img_el.get("data-src") or img_el.get("src")

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="aktuelkatalog",
                        discounted_price=discounted_price,
                        currency="TRY",
                        image_url=image_url,
                        seller_name=store_name,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("card parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    async def get_product(self, url: str) -> Product | None:
        """Catalog items don't have individual product pages."""
        return None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Catalog source has no reviews."""
        return []

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        return "aktuelkatalog" in text.lower() or "katalog" in text.lower()


# ---------------------------------------------------------------------------
# Getir
# ---------------------------------------------------------------------------


@register_scraper("getir")
class GetirScraper(BaseScraper):
    """Scrape grocery product data from Getir's web/API.

    NOTE: As of 2026-03 all Getir search endpoints return 403 or 404:
    - /arama/?q=* → 404
    - /api/search → 403
    - /_next/data/{buildId}/search.json → 404
    - /_next/static/{buildId}/_buildManifest.js → 403

    The site is a Next.js SPA with no publicly accessible search API.
    ``is_available = False`` disables search calls until a working
    endpoint is identified.
    """

    _BASE_URL = "https://getir.com"
    _API_URL = "https://getir.com/api"
    # All known search endpoints return 403/404 — disable until fixed
    is_available: bool = False

    def __init__(self) -> None:
        super().__init__(domain="getir")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        if not self.is_available:
            logger.debug("getir search skipped: is_available=False (all endpoints dead)")
            return []

        # Cache
        try:
            cached = await get_cached_search(query, "getir")
            if cached is not None:
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        # Try API endpoint first
        products = await self._search_api(query, max_results)

        # Fallback to web search
        if not products and _BS4_AVAILABLE:
            products = await self._search_web(query, max_results)

        if products:
            try:
                await cache_search(
                    query, "getir",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception:
                pass

        return products

    def _parse_search_response(self, data: dict | list) -> list[Product]:
        """Parse a Getir API search response into Products.

        Accepts either the full JSON envelope (dict with ``data.products`` or
        ``products`` key) or a bare list of product dicts.
        """
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", {}).get("products", data.get("products", []))

        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items:
            try:
                name = item.get("name") or item.get("shortDescription", "")
                if not name:
                    continue
                name = normalize_product_name(name)

                slug = item.get("slug") or item.get("url", "")
                product_url = (
                    slug if slug.startswith("http") else f"{self._BASE_URL}/{slug}"
                )

                price = _safe_float(item.get("price") or item.get("priceText"))
                original_price = _safe_float(item.get("originalPrice"))

                # SKU
                raw_id = item.get("id") or item.get("_id") or item.get("productId")
                sku = f"gt-{raw_id}" if raw_id is not None else None

                # Stock status
                in_stock = item.get("inStock", True)
                availability = "in_stock" if in_stock else "out_of_stock"

                # Unit price
                specs: dict[str, Any] = {"type": "grocery"}
                unit_text = item.get("unitText") or item.get("shortDescription") or name
                unit_info = _calculate_unit_price(price, unit_text)
                if unit_info:
                    specs.update(unit_info)

                # Campaign
                campaign = item.get("campaign") or item.get("badge")
                if campaign:
                    if isinstance(campaign, dict):
                        specs["campaign_badge"] = campaign.get("name", str(campaign))
                    else:
                        specs["campaign_badge"] = str(campaign)

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="getir",
                        original_price=original_price,
                        discounted_price=price,
                        currency="TRY",
                        image_url=item.get("imageUrl") or item.get("image"),
                        availability=availability,
                        sku=sku,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception:
                continue

        return products

    async def _search_api(self, query: str, max_results: int) -> list[Product]:
        """Try to search via Getir's internal API."""
        try:
            search_url = f"{self._API_URL}/search?q={urllib.parse.quote(query)}"
            response = await self._fetch(search_url)

            if response.status_code != 200:
                return []

            data = response.json()
            products = self._parse_search_response(data)
            return products[:max_results]

        except Exception as exc:
            logger.debug("Getir API search failed", error=str(exc))

        return []

    async def _search_web(self, query: str, max_results: int) -> list[Product]:
        """Fallback: parse Getir web pages."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            url = f"{self._BASE_URL}/search?q={urllib.parse.quote(query)}"
            response = await self._fetch(url)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            cards = soup.select("div[data-testid='product-card']") or soup.select("article.product")

            for card in cards[:max_results]:
                try:
                    name_el = card.select_one("span.product-name") or card.select_one("h3")
                    if not name_el:
                        continue
                    name = normalize_product_name(name_el.get_text(strip=True))

                    price_el = card.select_one("span.product-price") or card.select_one("span.price")
                    price = parse_turkish_price(price_el.get_text(strip=True)) if price_el else None

                    specs: dict[str, Any] = {"type": "grocery"}
                    unit_info = _calculate_unit_price(price, name)
                    if unit_info:
                        specs.update(unit_info)

                    products.append(
                        Product(
                            name=name,
                            url=url,
                            source="getir",
                            discounted_price=price,
                            currency="TRY",
                            specs=specs,
                            fetched_at=now_iso,
                        )
                    )
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Getir web search failed", error=str(exc))

        return products

    async def get_product(self, url: str) -> Product | None:
        return None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        # Getir is a quick-commerce app — product detail pages do not host
        # user reviews/ratings.  The web SPA is also gated (403/404 across
        # all known endpoints; ``is_available = False``).  No reviews to
        # scrape.  Verified 2026-04-18 via snapshot of getir.com URLs.
        return []

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        ct = response.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                return isinstance(response.json(), dict)
            except Exception:
                return False
        text = response.text
        return bool(text) and len(text) > 200


# ---------------------------------------------------------------------------
# Migros
# ---------------------------------------------------------------------------


@register_scraper("migros")
class MigrosScraper(BaseScraper):
    """Scrape grocery products from Migros (migros.com.tr).

    Note: Migros is an Angular SPA. The REST API (``/rest/products/search``)
    returns results only when a store is selected in the session.  Without
    a ``storeId`` cookie the API returns hitCount=0.  The ``/rest/search``
    path is a 404 (deprecated).  HTML scraping also fails since all pages
    return the SPA shell.  Currently this scraper returns empty results
    unless a valid store session is configured.
    """

    _BASE_URL = "https://www.migros.com.tr"
    # Updated endpoint: /rest/search → 404; /rest/products/search is live
    # but requires a store session cookie to return results.
    _API_URL = "https://www.migros.com.tr/rest/products/search"

    def __init__(self) -> None:
        super().__init__(domain="migros")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        # Cache
        try:
            cached = await get_cached_search(query, "migros")
            if cached is not None:
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        # Try REST API
        products = await self._search_api(query, max_results)

        if not products and _BS4_AVAILABLE:
            products = await self._search_web(query, max_results)

        if products:
            try:
                await cache_search(
                    query, "migros",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception:
                pass

        return products

    def _parse_search_response(self, data: dict | list) -> list[Product]:
        """Parse a Migros REST API search response into Products.

        Accepts either the full JSON envelope (dict with ``data.storeProductInfos``
        or ``products`` key) or a bare list of product dicts.
        """
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", {}).get("storeProductInfos", data.get("products", []))

        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items:
            try:
                name = item.get("name") or item.get("productName", "")
                if not name:
                    continue
                name = normalize_product_name(name)

                slug = item.get("prettyName") or item.get("url", "")
                product_url = (
                    slug if slug.startswith("http")
                    else f"{self._BASE_URL}/{slug}"
                )

                price = _safe_float(item.get("shownPrice") or item.get("price"))
                original_price = _safe_float(item.get("regularPrice") or item.get("strikeThroughPrice"))

                # SKU
                raw_id = item.get("id") or item.get("productId") or item.get("sku")
                sku = f"mg-{raw_id}" if raw_id is not None else None

                # Stock
                in_stock = item.get("inStock", True)
                availability = "in_stock" if in_stock else "out_of_stock"

                # Unit price from API
                specs: dict[str, Any] = {"type": "grocery"}
                unit_price_text = item.get("unitPrice") or item.get("birimFiyat")
                if unit_price_text:
                    specs["unit_price_text"] = str(unit_price_text)

                # Calculate unit price from name
                unit_info = _calculate_unit_price(price, name)
                if unit_info:
                    specs.update(unit_info)

                # Campaign badge
                badges = item.get("badges") or item.get("campaignBadges", [])
                if badges:
                    if isinstance(badges, list):
                        badge_texts = []
                        for b in badges:
                            if isinstance(b, dict):
                                badge_texts.append(b.get("name", str(b)))
                            else:
                                badge_texts.append(str(b))
                        if badge_texts:
                            specs["campaign_badge"] = ", ".join(badge_texts)
                    elif isinstance(badges, str):
                        specs["campaign_badge"] = badges

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="migros",
                        original_price=original_price,
                        discounted_price=price,
                        currency="TRY",
                        image_url=item.get("imageUrl") or item.get("image"),
                        availability=availability,
                        sku=sku,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception:
                continue

        return products

    async def _search_api(self, query: str, max_results: int) -> list[Product]:
        """Search via Migros REST API."""
        try:
            # param name changed from "q" to "term" in new endpoint
            params = {"term": query, "sayfa": "1", "siralamaTipi": "1"}
            response = await self._fetch(self._API_URL, params=params)

            if response.status_code != 200:
                return []

            data = response.json()
            products = self._parse_search_response(data)
            return products[:max_results]

        except Exception as exc:
            logger.debug("Migros API search failed", error=str(exc))

        return []

    async def _search_web(self, query: str, max_results: int) -> list[Product]:
        """Fallback: parse Migros web page."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            url = f"{self._BASE_URL}/arama?q={urllib.parse.quote(query)}"
            response = await self._fetch(url)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            cards = soup.select("div.product-card") or soup.select("sm-list-page-item")

            for card in cards[:max_results]:
                try:
                    name_el = card.select_one("a.product-name") or card.select_one("h5")
                    if not name_el:
                        continue
                    name = normalize_product_name(name_el.get_text(strip=True))

                    price_el = card.select_one("span.price") or card.select_one("span.amount")
                    price = parse_turkish_price(price_el.get_text(strip=True)) if price_el else None

                    specs: dict[str, Any] = {"type": "grocery"}
                    unit_info = _calculate_unit_price(price, name)
                    if unit_info:
                        specs.update(unit_info)

                    products.append(
                        Product(
                            name=name,
                            url=url,
                            source="migros",
                            discounted_price=price,
                            currency="TRY",
                            specs=specs,
                            fetched_at=now_iso,
                        )
                    )
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Migros web search failed", error=str(exc))

        return products

    async def get_product(self, url: str) -> Product | None:
        return None

    @staticmethod
    def _extract_migros_pid(url: str) -> str | None:
        """Extract the short product id from a Migros sanalmarket URL.

        URL pattern: ``/<slug>-p-<id>`` where id is a hex hash (e.g. ``d2c30b``).
        """
        m = re.search(r"-p-([a-f0-9]+)", url, re.IGNORECASE)
        return m.group(1) if m else None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Fetch product comments/ratings from Migros sanalmarket.

        Migros is an Angular SPA and reviews live behind store-session
        gated REST endpoints.  We probe the documented review paths; any
        ``200 + application/json`` response is parsed.  When all paths
        return SPA-shell HTML or 404 (the common case without a store
        cookie) we return [].

        Verified 2026-04-18: without a store session every endpoint either
        404s or returns the 5KB Angular shell.  Comment text is therefore
        not retrievable in the current configuration.
        """
        # Cache
        try:
            cached = await get_cached_reviews(url, "migros")
            if cached is not None:
                return cached
        except Exception:
            pass

        pid = self._extract_migros_pid(url)
        if not pid:
            return []

        # Strategy 1: probe documented REST review endpoints.  Migros has
        # rotated these several times; we try the known set in order.
        candidate_paths = [
            f"/rest/products/{pid}/comments",
            f"/rest/comment-rate/product-comments/{pid}",
            f"/rest/comment/list?productId={pid}",
            f"/sm/api/comment-rate/product-comments/{pid}",
            f"/sm/api/products/{pid}/comments",
        ]

        all_reviews: list[dict] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for path in candidate_paths:
            try:
                response = await self._fetch(self._BASE_URL + path)
            except Exception as exc:
                logger.debug("migros review probe failed", path=path, error=str(exc))
                continue
            if response.status_code != 200:
                continue
            ct = response.headers.get("content-type", "").lower()
            if "application/json" not in ct:
                continue
            try:
                data = response.json()
            except Exception:
                continue
            items = (
                (data.get("data") or {}).get("comments")
                or data.get("comments")
                or data.get("productComments")
                or data.get("reviews")
                or []
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (
                    item.get("comment")
                    or item.get("commentText")
                    or item.get("text")
                    or ""
                )
                if not text:
                    continue
                rating = _safe_float(item.get("rate") or item.get("rating") or item.get("star"))
                author = item.get("userName") or item.get("nickName") or item.get("author")
                date = item.get("commentDate") or item.get("createdDate") or item.get("date")
                helpful = item.get("helpfulCount") or item.get("likeCount") or 0
                all_reviews.append({
                    "text": text,
                    "source": "migros",
                    "rating": rating,
                    "date": str(date) if date else None,
                    "author": author,
                    "helpful_count": int(helpful) if isinstance(helpful, (int, float)) else 0,
                })
            if all_reviews:
                break  # First working endpoint wins

        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "migros")
            except Exception:
                pass
        else:
            logger.debug(
                "migros reviews unavailable (SPA shell / store-session gated)",
                url=url, pid=pid,
            )

        # NOTE: a JSON-LD/aggregateRating fallback on the product page was
        # tested but the Angular SPA renders an empty 84KB shell -- no
        # structured data hydrated server-side.  Skipped to keep the call
        # path fast.

        return all_reviews

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        ct = response.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                return isinstance(response.json(), dict)
            except Exception:
                return False
        text = response.text
        return bool(text) and len(text) > 200


# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
