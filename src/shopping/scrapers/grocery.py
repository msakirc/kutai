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
from ..cache import cache_search, get_cached_search
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
    """Scrape weekly promotion catalogs from aktuelkatalog.com.tr."""

    _BASE_URL = "https://www.aktuelkatalog.com.tr"
    _SEARCH_URL = "https://www.aktuelkatalog.com.tr/ara"

    def __init__(self) -> None:
        super().__init__(domain="aktuelkatalog")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
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
    """Scrape grocery product data from Getir's web/API."""

    _BASE_URL = "https://getir.com"
    _API_URL = "https://getir.com/api"

    def __init__(self) -> None:
        super().__init__(domain="getir")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
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

    async def _search_api(self, query: str, max_results: int) -> list[Product]:
        """Try to search via Getir's internal API."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            search_url = f"{self._API_URL}/search?q={urllib.parse.quote(query)}"
            response = await self._fetch(search_url)

            if response.status_code != 200:
                return []

            data = response.json()
            items = data.get("data", {}).get("products", data.get("products", []))

            for item in items[:max_results]:
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
                            specs=specs,
                            fetched_at=now_iso,
                        )
                    )
                except Exception:
                    continue

        except Exception as exc:
            logger.debug("Getir API search failed", error=str(exc))

        return products

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
    """Scrape grocery products from Migros (migros.com.tr)."""

    _BASE_URL = "https://www.migros.com.tr"
    _API_URL = "https://www.migros.com.tr/rest/search"

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

    async def _search_api(self, query: str, max_results: int) -> list[Product]:
        """Search via Migros REST API."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            params = {"q": query, "sayfa": "1", "sirpieces": str(max_results)}
            response = await self._fetch(self._API_URL, params=params)

            if response.status_code != 200:
                return []

            data = response.json()
            items = data.get("data", {}).get("storeProductInfos", data.get("products", []))

            for item in items[:max_results]:
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
                            specs=specs,
                            fetched_at=now_iso,
                        )
                    )
                except Exception:
                    continue

        except Exception as exc:
            logger.debug("Migros API search failed", error=str(exc))

        return products

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

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
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
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
