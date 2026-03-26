"""Home improvement scrapers -- Koctas and IKEA Turkey.

These scrapers focus on dimensional data in product specifications
(height, width, depth, weight) which is critical for home improvement
and furniture shopping decisions.

IKEA's scraper uses ``extract_structured_data`` to pull Schema.org
data from the page.
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
    get_cached_product,
    get_cached_search,
)
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.home_improvement")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- home improvement scrapers disabled")


# ---------------------------------------------------------------------------
# Dimension extraction helper
# ---------------------------------------------------------------------------


def _extract_dimensions(specs: dict[str, str]) -> dict[str, Any]:
    """Extract dimensional data from a specs dict.

    Looks for width, height, depth, weight keys in various Turkish
    and English forms and normalises them.
    """
    dims: dict[str, Any] = {}

    dimension_keys = {
        "genişlik": "width_cm",
        "en": "width_cm",
        "width": "width_cm",
        "yükseklik": "height_cm",
        "boy": "height_cm",
        "height": "height_cm",
        "derinlik": "depth_cm",
        "depth": "depth_cm",
        "ağırlık": "weight_kg",
        "weight": "weight_kg",
        "uzunluk": "length_cm",
        "length": "length_cm",
    }

    for key, dim_name in dimension_keys.items():
        for spec_key, spec_val in specs.items():
            if key in spec_key.lower():
                # Try to extract numeric value
                m = re.search(r"([\d.,]+)", str(spec_val))
                if m:
                    try:
                        val = float(m.group(1).replace(",", "."))
                        dims[dim_name] = val
                    except (ValueError, TypeError):
                        pass
                break

    return dims


# ---------------------------------------------------------------------------
# Koctas
# ---------------------------------------------------------------------------


@register_scraper("koctas")
class KoctasScraper(BaseScraper):
    """Scrape home improvement products from koctas.com.tr."""

    _BASE_URL = "https://www.koctas.com.tr"
    _SEARCH_URL = "https://www.koctas.com.tr/search"

    def __init__(self) -> None:
        super().__init__(domain="koctas")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_search(query, "koctas")
            if cached is not None:
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        params = {"q": query, "text": query}

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
                    query, "koctas",
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
            or soup.select("div.product-list-item")
            or soup.select("li.product-item")
        )

        for card in cards[:max_results]:
            try:
                name_el = (
                    card.select_one("a.product-title")
                    or card.select_one("h3.product-name")
                    or card.select_one("a[title]")
                )
                if not name_el:
                    continue
                name = normalize_product_name(
                    name_el.get("title", "") or name_el.get_text(strip=True)
                )
                if not name:
                    continue

                href = name_el.get("href", "")
                product_url = (
                    href if href.startswith("http") else f"{self._BASE_URL}{href}"
                )

                # Price
                price_el = card.select_one("span.price") or card.select_one("div.product-price")
                discounted_price = parse_turkish_price(price_el.get_text(strip=True)) if price_el else None

                # Old price
                old_price_el = card.select_one("span.old-price") or card.select_one("del")
                original_price = parse_turkish_price(old_price_el.get_text(strip=True)) if old_price_el else None

                # Image
                img_el = card.select_one("img")
                image_url = None
                if img_el:
                    image_url = img_el.get("data-src") or img_el.get("src")

                # Rating
                rating: float | None = None
                rating_el = card.select_one("span.rating-value")
                if rating_el:
                    try:
                        rating = float(rating_el.get_text(strip=True).replace(",", "."))
                    except (ValueError, TypeError):
                        pass

                specs: dict[str, Any] = {"type": "home_improvement"}

                discount_pct: float | None = None
                if original_price and discounted_price and original_price > discounted_price:
                    discount_pct = round((1 - discounted_price / original_price) * 100, 1)

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="koctas",
                        original_price=original_price,
                        discounted_price=discounted_price,
                        discount_percentage=discount_pct,
                        currency="TRY",
                        image_url=image_url,
                        rating=rating,
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
        """Fetch a Koctas product page with dimensional data."""
        if not _BS4_AVAILABLE:
            return None

        try:
            cached = await get_cached_product(url)
            if cached is not None:
                return self._dict_to_product(cached)
        except Exception:
            pass

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("product fetch failed", url=url, error=str(exc))
            return None

        if response.status_code != 200:
            return None

        product = self._parse_product(url, response.text)

        if product is not None:
            try:
                await cache_product(url, self._product_to_dict(product), "koctas", "prices")
            except Exception:
                pass

        return product

    def _parse_product(self, url: str, html: str) -> Product | None:
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

        title_el = soup.select_one("h1.product-title") or soup.select_one("h1")
        if not title_el:
            return None
        name = normalize_product_name(title_el.get_text(strip=True))

        # Price
        price_el = soup.select_one("span.product-price") or soup.select_one("span.price")
        price = parse_turkish_price(price_el.get_text(strip=True)) if price_el else None

        # Specs table with dimensional focus
        specs: dict[str, Any] = {"type": "home_improvement"}
        raw_specs: dict[str, str] = {}
        try:
            spec_rows = (
                soup.select("table.product-spec tr")
                or soup.select("div.product-features li")
                or soup.select("ul.spec-list li")
            )
            for row in spec_rows:
                if row.name == "tr":
                    cells = row.select("td")
                    if len(cells) >= 2:
                        k = cells[0].get_text(strip=True)
                        v = cells[1].get_text(strip=True)
                        if k and v:
                            raw_specs[k] = v
                else:
                    text = row.get_text(strip=True)
                    if ":" in text:
                        parts = text.split(":", 1)
                        raw_specs[parts[0].strip()] = parts[1].strip()
        except Exception:
            pass

        specs.update(raw_specs)

        # Extract and add dimensions
        dims = _extract_dimensions(raw_specs)
        if dims:
            specs["dimensions"] = dims

        # Structured data fallback
        structured = self.extract_structured_data(html)
        if structured.get("json_ld"):
            ld = structured["json_ld"]
            if isinstance(ld, dict) and not name:
                name = ld.get("name", name)

        return Product(
            name=name,
            url=url,
            source="koctas",
            discounted_price=price,
            currency="TRY",
            specs=specs,
            fetched_at=now_iso,
        )

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        return []

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("koctas", "koçtaş", "product", "ürün")
        return any(marker in text.lower() for marker in markers)


# ---------------------------------------------------------------------------
# IKEA Turkey
# ---------------------------------------------------------------------------


@register_scraper("ikea")
class IKEAScraper(BaseScraper):
    """Scrape furniture and home products from IKEA Turkey.

    Uses ``extract_structured_data`` for Schema.org data extraction,
    with focus on dimensional specifications.
    """

    _BASE_URL = "https://www.ikea.com.tr"
    _SEARCH_URL = "https://www.ikea.com.tr/arama"

    def __init__(self) -> None:
        super().__init__(domain="ikea")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_search(query, "ikea")
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
                    query, "ikea",
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
            soup.select("div.product-compact")
            or soup.select("div.range-revamp-product-compact")
            or soup.select("div.serp-grid__item")
        )

        for card in cards[:max_results]:
            try:
                name_el = (
                    card.select_one("span.range-revamp-header-section__title--small")
                    or card.select_one("span.product-compact__name")
                    or card.select_one("h3")
                )
                desc_el = (
                    card.select_one("span.range-revamp-header-section__description-text")
                    or card.select_one("span.product-compact__type")
                )

                if not name_el:
                    continue

                name = name_el.get_text(strip=True)
                description = desc_el.get_text(strip=True) if desc_el else ""
                if description:
                    name = f"{name} - {description}"
                name = normalize_product_name(name)

                # URL
                link_el = card.select_one("a[href]")
                href = link_el["href"] if link_el else ""
                product_url = (
                    href if href.startswith("http") else f"{self._BASE_URL}{href}"
                )

                # Price
                price_el = (
                    card.select_one("span.range-revamp-price__integer")
                    or card.select_one("span.product-compact__price")
                )
                price: float | None = None
                if price_el:
                    price = parse_turkish_price(price_el.get_text(strip=True))

                # Dimension hint from description
                specs: dict[str, Any] = {"type": "home_improvement", "store": "ikea"}
                dim_text = description or name
                dims = _extract_dimensions({"description": dim_text})
                if dims:
                    specs["dimensions"] = dims

                # Image
                img_el = card.select_one("img")
                image_url = None
                if img_el:
                    image_url = img_el.get("data-src") or img_el.get("src")

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="ikea",
                        discounted_price=price,
                        currency="TRY",
                        image_url=image_url,
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
        """Fetch an IKEA product page with Schema.org data extraction."""
        if not _BS4_AVAILABLE:
            return None

        try:
            cached = await get_cached_product(url)
            if cached is not None:
                return self._dict_to_product(cached)
        except Exception:
            pass

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("product fetch failed", url=url, error=str(exc))
            return None

        if response.status_code != 200:
            return None

        product = self._parse_product(url, response.text)

        if product is not None:
            try:
                await cache_product(url, self._product_to_dict(product), "ikea", "prices")
            except Exception:
                pass

        return product

    def _parse_product(self, url: str, html: str) -> Product | None:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Use extract_structured_data for Schema.org
        structured = self.extract_structured_data(html)
        ld = structured.get("json_ld", {})
        og = structured.get("opengraph", {})

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

        # Name from structured data or HTML
        name = ""
        if isinstance(ld, dict):
            name = ld.get("name", "")
        if not name:
            name = og.get("title", "")
        if not name:
            title_el = soup.select_one("h1") or soup.select_one("span.product-name")
            if title_el:
                name = title_el.get_text(strip=True)
        if not name:
            return None
        name = normalize_product_name(name)

        # Price from structured data
        price: float | None = None
        if isinstance(ld, dict):
            offers = ld.get("offers", {})
            if isinstance(offers, dict):
                price = _safe_float(offers.get("price") or offers.get("lowPrice"))

        if price is None:
            price_el = soup.select_one("span.range-revamp-price__integer") or soup.select_one("span.price")
            if price_el:
                price = parse_turkish_price(price_el.get_text(strip=True))

        # Specs with dimensional focus
        specs: dict[str, Any] = {"type": "home_improvement", "store": "ikea"}
        raw_specs: dict[str, str] = {}

        # From Schema.org microdata
        for item in structured.get("schema_org", []):
            if isinstance(item, dict):
                prop = item.get("property", "")
                val = item.get("value", "")
                if prop and val:
                    raw_specs[prop] = val

        # From HTML spec tables
        try:
            spec_rows = (
                soup.select("div.range-revamp-product-details__container dl")
                or soup.select("table.product-details tr")
            )
            for row in spec_rows:
                if row.name == "dl":
                    dts = row.select("dt")
                    dds = row.select("dd")
                    for dt, dd in zip(dts, dds):
                        k = dt.get_text(strip=True)
                        v = dd.get_text(strip=True)
                        if k and v:
                            raw_specs[k] = v
                elif row.name == "tr":
                    cells = row.select("td")
                    if len(cells) >= 2:
                        raw_specs[cells[0].get_text(strip=True)] = cells[1].get_text(strip=True)
        except Exception:
            pass

        specs.update(raw_specs)

        # Extract dimensions
        dims = _extract_dimensions(raw_specs)
        if dims:
            specs["dimensions"] = dims

        # Image
        image_url = og.get("image")
        if not image_url:
            img_el = soup.select_one("img.range-revamp-aspect-ratio-image__image")
            if img_el:
                image_url = img_el.get("src")

        return Product(
            name=name,
            url=url,
            source="ikea",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            fetched_at=now_iso,
        )

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        return []

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("ikea", "product", "range-revamp")
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
