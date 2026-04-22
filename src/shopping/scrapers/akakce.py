"""Akakce scraper -- Turkish price aggregator.

Akakce.com compares prices across many Turkish retailers.  It has
relatively low anti-bot measures, so plain httpx + BeautifulSoup is
sufficient.  If ``bs4`` is not installed the module logs a warning
and all methods return empty results gracefully.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import (
    cache_product,
    cache_reviews,
    cache_search,
    get_cached_product,
    get_cached_reviews,
    get_cached_search,
)
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.akakce")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 (BeautifulSoup) is not installed -- Akakce scraper disabled")


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("akakce")
class AkakceScraper(BaseScraper):
    """Scrape product and price data from akakce.com."""

    _BASE_URL = "https://www.akakce.com"

    def __init__(self) -> None:
        super().__init__(domain="akakce")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Akakce for *query* and return product cards."""
        if not _BS4_AVAILABLE:
            return []

        # Check cache first
        try:
            cached = await get_cached_search(query, "akakce")
            if cached is not None:
                logger.debug("search cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception as exc:
            logger.debug("search cache lookup failed", error=str(exc))

        encoded = urllib.parse.quote(query, safe="")
        url = f"{self._BASE_URL}/arama/?q={encoded}"

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning(
                "search non-200",
                query=query,
                status=response.status_code,
            )
            return []

        products = self._parse_search_results(response.text, max_results)

        # Cache the results
        try:
            await cache_search(
                query, "akakce", [self._product_to_dict(p) for p in products]
            )
        except Exception as exc:
            logger.debug("search cache write failed", error=str(exc))

        return products

    def _parse_search_results(self, html: str, max_results: int) -> list[Product]:
        """Parse product cards from the Akakce search results page."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        # Akakce uses list items / divs with product info.  The exact
        # selectors may change -- we try multiple patterns defensively.
        cards = (
            soup.select("li.p")           # common pattern
            or soup.select("div.p")       # alternate
            or soup.select("[data-pr]")   # data-attribute variant
        )

        for card in cards[:max_results]:
            try:
                product = self._parse_card(card, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("card parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    def _parse_search_html(self, html: str) -> list[Product]:
        """Alias for testability — delegates to _parse_search_results."""
        return self._parse_search_results(html, max_results=100)

    def _parse_card(self, card: Any, now_iso: str) -> Product | None:
        """Extract a single Product from a search-result card element."""
        # Product name — class changes with redesigns (pn_7 → iC, etc.)
        name_el = (
            card.select_one("a.pn_7")
            or card.select_one("a.iC")
            or card.select_one("a[title]")
        )
        if name_el is None:
            return None

        raw_name = name_el.get("title", "") or name_el.get_text(strip=True)
        if not raw_name:
            return None

        name = normalize_product_name(raw_name)

        # URL
        href = name_el.get("href", "")
        product_url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # Prices — class changes with redesigns (pt_v8 → pt_v9, etc.)
        price_el = (
            card.select_one("span.pt_v9")
            or card.select_one("span.pt_v8")
            or card.select_one("span.pb_v8")
            or card.select_one("span.fiyat")
        )
        original_price: float | None = None
        discounted_price: float | None = None

        if price_el:
            price_text = price_el.get_text(strip=True)
            parsed = parse_turkish_price(price_text)
            if parsed is not None:
                discounted_price = parsed

        # Second price (original / list price if crossed out)
        old_price_el = (
            card.select_one("span.pt_v9.old")
            or card.select_one("span.pt_v8.old")
            or card.select_one("del")
        )
        if old_price_el:
            old_text = old_price_el.get_text(strip=True)
            original_price = parse_turkish_price(old_text)

        # Store count — may be in a dedicated span or embedded in price text
        store_el = card.select_one("span.mc") or card.select_one("span.dt_v8")
        store_count: int | None = None
        if store_el:
            m = re.search(r"(\d+)", store_el.get_text())
            if m:
                store_count = int(m.group(1))
        elif price_el:
            # Akakce embeds store count in price text: "25.499,00TL+16 FİYAT"
            m = re.search(r"\+(\d+)\s*(?:FİYAT|fiyat)", price_el.get_text(strip=True))
            if m:
                store_count = int(m.group(1))

        # Image
        img_el = card.select_one("img")
        image_url: str | None = None
        if img_el:
            image_url = img_el.get("data-src") or img_el.get("src")
            if image_url and not image_url.startswith("http"):
                image_url = f"https:{image_url}" if image_url.startswith("//") else None

        specs: dict[str, Any] = {}
        if store_count is not None:
            specs["store_count"] = store_count

        discount_pct: float | None = None
        if original_price and discounted_price and original_price > discounted_price:
            discount_pct = round(
                (1 - discounted_price / original_price) * 100, 1
            )

        # SKU: derive from URL trailing numeric id  ,1234567890.html
        sku: str | None = None
        m_sku = re.search(r",(\d+)\.html", product_url)
        if m_sku:
            sku = f"ak-{m_sku.group(1)}"

        return Product(
            name=name,
            url=product_url,
            source="akakce",
            original_price=original_price,
            discounted_price=discounted_price,
            discount_percentage=discount_pct,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            sku=sku,
            fetched_at=now_iso,
        )

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product detail page from Akakce."""
        if not _BS4_AVAILABLE:
            return None

        # Cache check
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
                    url, self._product_to_dict(product), "akakce", "prices"
                )
            except Exception as exc:
                logger.debug("product cache write failed", error=str(exc))

        return product

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse the Akakce product detail page."""
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("product HTML parse failed", error=str(exc))
            return None

        # Product name
        name_el = soup.select_one("h1") or soup.select_one("span.pn")
        if name_el is None:
            logger.warning("no product name found", url=url)
            return None

        name = normalize_product_name(name_el.get_text(strip=True))

        # Price comparison table
        prices: list[dict[str, Any]] = []
        try:
            rows = soup.select("ul.p_w li") or soup.select("div.p_w tr")
            for row in rows:
                store_el = row.select_one("span.v_v8") or row.select_one("a.m_v8")
                price_el = row.select_one("span.pt_v8") or row.select_one("span.fiyat")
                if store_el and price_el:
                    store_name = store_el.get_text(strip=True)
                    price_val = parse_turkish_price(price_el.get_text(strip=True))
                    if price_val is not None:
                        link_el = row.select_one("a[href]")
                        store_url = link_el["href"] if link_el else None
                        prices.append({
                            "store": store_name,
                            "price": price_val,
                            "url": store_url,
                        })
        except Exception as exc:
            logger.debug("price table parse error", error=str(exc))

        # Best price
        best_price: float | None = None
        original_price: float | None = None
        if prices:
            sorted_prices = sorted(prices, key=lambda p: p["price"])
            best_price = sorted_prices[0]["price"]
            if len(sorted_prices) > 1:
                original_price = sorted_prices[-1]["price"]

        # Spec table
        specs: dict[str, Any] = {}
        try:
            spec_rows = soup.select("li.p_s_r") or soup.select("div.p_s_r tr")
            for sr in spec_rows:
                key_el = sr.select_one("span.p_s_n") or sr.select_one("td.p_s_n")
                val_el = sr.select_one("span.p_s_v") or sr.select_one("td.p_s_v")
                if key_el and val_el:
                    key = key_el.get_text(strip=True)
                    val = val_el.get_text(strip=True)
                    if key and val:
                        specs[key] = val
        except Exception as exc:
            logger.debug("spec table parse error", error=str(exc))

        if prices:
            specs["price_comparison"] = prices

        # Price history from embedded JS variable
        price_history = self._extract_price_history(html)
        if price_history:
            specs["price_history"] = price_history

        # Image
        img_el = soup.select_one("img.p_i") or soup.select_one("div.p_i img")
        image_url: str | None = None
        if img_el:
            image_url = img_el.get("data-src") or img_el.get("src")
            if image_url and not image_url.startswith("http"):
                image_url = f"https:{image_url}" if image_url.startswith("//") else None

        # Structured data fallback
        structured = self.extract_structured_data(html)
        if structured.get("json_ld"):
            ld = structured["json_ld"]
            if isinstance(ld, dict):
                if not name:
                    name = ld.get("name", "")
                if best_price is None:
                    offers = ld.get("offers", {})
                    if isinstance(offers, dict):
                        p = offers.get("lowPrice") or offers.get("price")
                        if p is not None:
                            try:
                                best_price = float(p)
                            except (ValueError, TypeError):
                                pass

        discount_pct: float | None = None
        if original_price and best_price and original_price > best_price:
            discount_pct = round((1 - best_price / original_price) * 100, 1)

        return Product(
            name=name,
            url=url,
            source="akakce",
            original_price=original_price,
            discounted_price=best_price,
            discount_percentage=discount_pct,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            fetched_at=now_iso,
        )

    @staticmethod
    def _extract_price_history(html: str) -> list[dict[str, Any]]:
        """Try to pull the price history data from an embedded JS variable.

        Akakce sometimes embeds a JS array like ``var defined = [{...}, ...]``
        or ``priceHistory = [...]``.  We regex for it and parse as JSON.
        """
        history: list[dict[str, Any]] = []
        try:
            # Common patterns: "var defined = [...]" or "priceHistory = [...]"
            for pattern in (
                r"(?:priceHistory|defined|ph)\s*=\s*(\[.*?\])\s*;",
                r'"priceHistory"\s*:\s*(\[.*?\])',
            ):
                m = re.search(pattern, html, re.DOTALL)
                if m:
                    raw = m.group(1)
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        for entry in parsed:
                            if isinstance(entry, dict):
                                history.append(entry)
                            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                                history.append({"date": entry[0], "price": entry[1]})
                    break
        except Exception as exc:
            logger.debug("price history extraction failed", error=str(exc))

        return history

    # ------------------------------------------------------------------
    # get_reviews  (Akakce is a price aggregator -- reviews are minimal)
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract user comments from Akakce.

        Each Akakce product page links to a reviews page at
        ``/yorum/?p=PRODUCT_ID``.  If *url* is already a reviews page
        (``/yorum/?p=...``) we fetch directly, otherwise we extract the
        product ID from the product URL (``...,PRODUCT_ID.html``) and
        build the reviews URL.

        Pagination is *not* exposed as numbered pages on Akakce — the
        reviews list is rendered inline.  Sites with very many comments
        load more via a JS endpoint we do not call; max_pages is left in
        the signature for API parity but only the first page is fetched.
        """
        if not _BS4_AVAILABLE:
            return []

        # Cache check
        try:
            cached = await get_cached_reviews(url, "akakce")
            if cached is not None:
                logger.debug("akakce reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        review_url = self._build_review_url(url)
        if not review_url:
            logger.debug("akakce: could not derive review URL", url=url)
            return []

        try:
            resp = await self._fetch(review_url)
        except Exception as exc:
            logger.debug("akakce reviews fetch failed", url=review_url, error=str(exc))
            return []
        if resp.status_code != 200:
            return []

        reviews = self._parse_reviews_html(resp.text)

        # Sort by helpful_count descending
        reviews.sort(key=lambda r: r.get("helpful_count", 0), reverse=True)

        if reviews:
            try:
                await cache_reviews(url, reviews, "akakce")
            except Exception:
                pass

        logger.info("akakce reviews fetched", url=url, count=len(reviews))
        return reviews

    @staticmethod
    def _build_review_url(url: str) -> str | None:
        """Derive ``/yorum/?p=ID`` from any Akakce URL."""
        if "/yorum/" in url and "p=" in url:
            return url
        # Product URL pattern: .../en-ucuz-name-fiyati,PRODUCTID.html
        m = re.search(r",(\d+)\.html", url)
        if m:
            return f"https://www.akakce.com/yorum/?p={m.group(1)}"
        return None

    def _parse_reviews_html(self, html: str) -> list[dict]:
        """Parse review entries from /yorum/?p=ID page.

        Each review is ``<li data-c="ID" data-d="DISLIKES" data-l="LIKES">``
        with author in ``.h b a``, date in ``.h .d[title]``, optional rating
        in ``.sc_v9[data-sc]`` (0-20 scale, mapped to 0-5 stars), and the
        comment text in ``.cm``.
        """
        reviews: list[dict] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        review_lis = (
            soup.select("li[data-c]")
            or soup.select("ul.cms li")
            or soup.select("article.review")
        )

        for li in review_lis:
            try:
                cm_el = li.select_one(".cm") or li.select_one("div.review-text")
                if cm_el is None:
                    continue
                text = cm_el.get_text(separator=" ", strip=True)
                if not text or len(text) < 3:
                    continue

                # Author
                author_el = (
                    li.select_one(".h b a")
                    or li.select_one(".h b")
                    or li.select_one("a.upr")
                )
                author = author_el.get_text(strip=True) if author_el else None

                # Date — title attribute holds full timestamp
                date_str: str | None = None
                d_el = li.select_one(".h .d") or li.select_one("span.d")
                if d_el:
                    date_str = d_el.get("title") or d_el.get_text(strip=True)
                    # Strip "Eklenme tarihi: " prefix if present
                    if date_str:
                        date_str = re.sub(r"^Eklenme tarihi:\s*", "", date_str).strip()

                # Rating: 0-20 scale -> 0-5 stars
                rating: float | None = None
                sc_el = li.select_one(".sc_v9") or li.select_one("[data-sc]")
                if sc_el:
                    raw = sc_el.get("data-sc")
                    try:
                        if raw is not None:
                            rating = round(float(raw) / 4.0, 1)
                    except (ValueError, TypeError):
                        pass

                # Likes / dislikes from data attributes
                likes = 0
                dislikes = 0
                try:
                    likes = int(li.get("data-l", 0) or 0)
                    dislikes = int(li.get("data-d", 0) or 0)
                except (ValueError, TypeError):
                    pass

                # City — italics inside .h
                city: str | None = None
                city_el = li.select_one(".h i")
                if city_el:
                    city = city_el.get_text(strip=True) or None

                review: dict[str, Any] = {
                    "text": text,
                    "source": "akakce",
                    "author": author,
                    "date": date_str,
                    "rating": rating,
                    "helpful_count": likes,
                    "unhelpful_count": dislikes,
                }
                comment_id = li.get("data-c")
                if comment_id:
                    review["entry_id"] = comment_id
                if city:
                    review["city"] = city

                reviews.append(review)
            except Exception as exc:
                logger.debug("akakce review parse error", error=str(exc))
                continue

        return reviews

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        """Check that the response looks like real Akakce content."""
        if response.status_code >= 400:
            return False

        text = response.text
        if not text or len(text) < 500:
            return False

        # Quick heuristic: real pages contain the site branding or product data
        markers = ("akakce", "p_w", "pn_7", "fiyat", "ara/")
        return any(marker in text.lower() for marker in markers)
