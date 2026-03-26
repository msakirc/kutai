"""Amazon Turkey scraper -- amazon.com.tr

Scrapes product data from Amazon's Turkish marketplace.  HTML parsing is
the primary strategy; embedded JSON from ``<script>`` tags is tried first
where available.

.. note::

    The **preferred** method for production use is the Amazon Product
    Advertising API (PA-API 5.0) with credentials stored in the project's
    credential store.  This scraper exists as a fallback for when PA-API
    credentials are unavailable or the quota is exhausted.
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

logger = get_logger("shopping.scrapers.amazon_tr")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- Amazon TR HTML parsing disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.amazon.com.tr"
_SEARCH_URL = "https://www.amazon.com.tr/s"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("amazon_tr")
class AmazonTrScraper(BaseScraper):
    """Scrape product data from amazon.com.tr.

    .. note::
        PA-API (Product Advertising API) is the preferred method when
        credentials are configured in the credential store.  This HTML
        scraper is a fallback.
    """

    def __init__(self) -> None:
        super().__init__(domain="amazon_tr")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Amazon TR for *query*."""
        # Cache
        try:
            cached = await get_cached_search(query, "amazon_tr")
            if cached is not None:
                logger.debug("search cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception as exc:
            logger.debug("search cache lookup failed", error=str(exc))

        params = {"k": query, "language": "tr_TR"}

        try:
            response = await self._fetch(_SEARCH_URL, params=params)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        html = response.text

        # Strategy 1: embedded JSON from script tags
        products = self._parse_search_json(html, max_results)

        # Strategy 2: HTML parsing
        if not products and _BS4_AVAILABLE:
            products = self._parse_search_html(html, max_results)

        # Cache
        if products:
            try:
                await cache_search(
                    query,
                    "amazon_tr",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception as exc:
                logger.debug("search cache write failed", error=str(exc))

        return products

    def _parse_search_json(self, html: str, max_results: int) -> list[Product]:
        """Try to extract search results from embedded script-tag JSON."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            # Amazon sometimes embeds search result data in script tags
            for pattern in (
                r'data-a-state=["\']{"key":"search-results"}["\']>(.*?)</script>',
                r'"searchResults"\s*:\s*(\[.*?\])\s*[,}]',
                r'window\.__SEARCH_RESULTS__\s*=\s*({.*?});',
            ):
                m = re.search(pattern, html, re.DOTALL)
                if not m:
                    continue

                data = json.loads(m.group(1))
                items = data if isinstance(data, list) else data.get("results", [])

                for item in items[:max_results]:
                    try:
                        product = self._json_item_to_product(item, now_iso)
                        if product:
                            products.append(product)
                    except Exception:
                        continue

                if products:
                    break

        except Exception as exc:
            logger.debug("search JSON extraction failed", error=str(exc))

        return products

    def _parse_search_html(self, html: str, max_results: int) -> list[Product]:
        """Parse search results from HTML using BeautifulSoup."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        # Amazon search results use data-component-type="s-search-result"
        cards = (
            soup.select("div[data-component-type='s-search-result']")
            or soup.select("div.s-result-item[data-asin]")
        )

        for card in cards[:max_results]:
            try:
                asin = card.get("data-asin", "")
                if not asin:
                    continue

                # Name
                name_el = (
                    card.select_one("h2 a span")
                    or card.select_one("h2 span")
                    or card.select_one("span.a-text-normal")
                )
                if name_el is None:
                    continue
                name = normalize_product_name(name_el.get_text(strip=True))
                if not name:
                    continue

                # URL
                link_el = card.select_one("h2 a[href]")
                href = link_el["href"] if link_el else f"/dp/{asin}"
                product_url = (
                    href if href.startswith("http") else f"{_BASE_URL}{href}"
                )

                # Price
                price_whole = card.select_one("span.a-price-whole")
                price_frac = card.select_one("span.a-price-fraction")
                discounted_price: float | None = None
                if price_whole:
                    try:
                        whole = price_whole.get_text(strip=True).replace(".", "").replace(",", "")
                        frac = price_frac.get_text(strip=True) if price_frac else "00"
                        discounted_price = float(f"{whole}.{frac}")
                    except (ValueError, TypeError):
                        pass

                # Original price (strikethrough)
                original_price: float | None = None
                old_price_el = card.select_one("span.a-price[data-a-strike='true'] span.a-offscreen")
                if old_price_el:
                    original_price = parse_turkish_price(old_price_el.get_text(strip=True))

                # Rating
                rating: float | None = None
                rating_el = card.select_one("span.a-icon-alt")
                if rating_el:
                    m = re.search(r"([\d,]+)", rating_el.get_text())
                    if m:
                        try:
                            rating = float(m.group(1).replace(",", "."))
                        except (ValueError, TypeError):
                            pass

                # Review count
                review_count: int | None = None
                review_el = card.select_one("span.a-size-base.s-underline-text")
                if review_el:
                    rc_text = review_el.get_text(strip=True).replace(".", "").replace(",", "")
                    m = re.search(r"(\d+)", rc_text)
                    if m:
                        review_count = int(m.group(1))

                # Image
                img_el = card.select_one("img.s-image")
                image_url = img_el.get("src") if img_el else None

                # Prime badge -> free shipping
                free_shipping = bool(card.select_one("i.a-icon-prime"))

                discount_pct: float | None = None
                if (
                    original_price
                    and discounted_price
                    and original_price > discounted_price
                ):
                    discount_pct = round(
                        (1 - discounted_price / original_price) * 100, 1
                    )

                products.append(
                    Product(
                        name=name,
                        url=product_url,
                        source="amazon_tr",
                        original_price=original_price,
                        discounted_price=discounted_price,
                        discount_percentage=discount_pct,
                        currency="TRY",
                        image_url=image_url,
                        rating=rating,
                        review_count=review_count,
                        free_shipping=free_shipping,
                        specs={"asin": asin} if asin else {},
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("html card parse error", error=str(exc))
                continue

        logger.info("search HTML parsed", count=len(products))
        return products

    def _json_item_to_product(self, item: dict, now_iso: str) -> Product | None:
        """Convert embedded JSON search item to Product."""
        try:
            name = item.get("title") or item.get("name", "")
            if not name:
                return None
            name = normalize_product_name(name)

            asin = item.get("asin", "")
            url_path = item.get("url") or item.get("detailPageUrl") or f"/dp/{asin}"
            product_url = (
                url_path if url_path.startswith("http") else f"{_BASE_URL}{url_path}"
            )

            price_info = item.get("price", {})
            discounted_price: float | None = None
            original_price: float | None = None

            if isinstance(price_info, dict):
                discounted_price = _safe_float(
                    price_info.get("current") or price_info.get("value")
                )
                original_price = _safe_float(price_info.get("previous"))
            elif isinstance(price_info, (int, float)):
                discounted_price = float(price_info)

            return Product(
                name=name,
                url=product_url,
                source="amazon_tr",
                original_price=original_price,
                discounted_price=discounted_price,
                currency="TRY",
                image_url=item.get("image") or item.get("imageUrl"),
                rating=_safe_float(item.get("rating")),
                review_count=_safe_int(item.get("reviewCount")),
                specs={"asin": asin} if asin else {},
                fetched_at=now_iso,
            )
        except Exception as exc:
            logger.debug("json_item_to_product failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product page from Amazon TR."""
        # Cache
        try:
            cached = await get_cached_product(url)
            if cached is not None:
                logger.debug("product cache hit", url=url)
                return self._dict_to_product(cached)
        except Exception as exc:
            logger.debug("product cache lookup failed", error=str(exc))

        if not _BS4_AVAILABLE:
            logger.warning("bs4 not available, cannot parse product page")
            return None

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
                    url, self._product_to_dict(product), "amazon_tr", "prices"
                )
            except Exception as exc:
                logger.debug("product cache write failed", error=str(exc))

        return product

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse an Amazon TR product detail page."""
        now_iso = datetime.now(timezone.utc).isoformat()

        # Try structured data first
        structured = self.extract_structured_data(html)
        ld = structured.get("json_ld", {})

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("product HTML parse failed", error=str(exc))
            return None

        # Name
        name = ""
        if isinstance(ld, dict):
            name = ld.get("name", "")
        if not name:
            title_el = soup.select_one("#productTitle") or soup.select_one("h1 span")
            if title_el:
                name = title_el.get_text(strip=True)
        if not name:
            return None
        name = normalize_product_name(name)

        # Price
        discounted_price: float | None = None
        original_price: float | None = None

        if isinstance(ld, dict):
            offers = ld.get("offers", {})
            if isinstance(offers, dict):
                discounted_price = _safe_float(
                    offers.get("lowPrice") or offers.get("price")
                )

        if discounted_price is None:
            price_el = (
                soup.select_one("span.a-price span.a-offscreen")
                or soup.select_one("#priceblock_ourprice")
                or soup.select_one("#priceblock_dealprice")
            )
            if price_el:
                discounted_price = parse_turkish_price(price_el.get_text(strip=True))

        old_price_el = soup.select_one("span.a-price[data-a-strike='true'] span.a-offscreen")
        if old_price_el:
            original_price = parse_turkish_price(old_price_el.get_text(strip=True))

        # Rating
        rating: float | None = None
        rating_el = soup.select_one("#acrPopover span.a-icon-alt")
        if rating_el:
            m = re.search(r"([\d,]+)", rating_el.get_text())
            if m:
                try:
                    rating = float(m.group(1).replace(",", "."))
                except (ValueError, TypeError):
                    pass

        # Review count
        review_count: int | None = None
        rc_el = soup.select_one("#acrCustomerReviewText")
        if rc_el:
            m = re.search(r"([\d.]+)", rc_el.get_text().replace(".", ""))
            if m:
                review_count = _safe_int(m.group(1))

        # Image
        og = structured.get("opengraph", {})
        image_url = og.get("image")
        if not image_url:
            img_el = soup.select_one("#landingImage") or soup.select_one("#imgBlkFront")
            if img_el:
                image_url = img_el.get("data-old-hires") or img_el.get("src")

        # Specs
        specs: dict[str, Any] = {}
        try:
            detail_rows = soup.select("#productDetails_techSpec_section_1 tr")
            for row in detail_rows:
                key_el = row.select_one("th")
                val_el = row.select_one("td")
                if key_el and val_el:
                    k = key_el.get_text(strip=True)
                    v = val_el.get_text(strip=True)
                    if k and v:
                        specs[k] = v
        except Exception:
            pass

        # ASIN
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
        if asin_match:
            specs["asin"] = asin_match.group(1)

        # Availability
        availability = "in_stock"
        avail_el = soup.select_one("#availability span")
        if avail_el:
            avail_text = avail_el.get_text(strip=True).lower()
            if "stokta yok" in avail_text or "unavailable" in avail_text:
                availability = "out_of_stock"
            elif "az kaldı" in avail_text or "only" in avail_text:
                availability = "low_stock"

        discount_pct: float | None = None
        if original_price and discounted_price and original_price > discounted_price:
            discount_pct = round(
                (1 - discounted_price / original_price) * 100, 1
            )

        return Product(
            name=name,
            url=url,
            source="amazon_tr",
            original_price=original_price,
            discounted_price=discounted_price,
            discount_percentage=discount_pct,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating,
            review_count=review_count,
            availability=availability,
            fetched_at=now_iso,
        )

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Parse reviews from Amazon TR product review pages."""
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_reviews(url, "amazon_tr")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception as exc:
            logger.debug("reviews cache lookup failed", error=str(exc))

        # Build review URL from product URL
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
        if not asin_match:
            logger.warning("could not extract ASIN for reviews", url=url)
            return []

        asin = asin_match.group(1)
        all_reviews: list[dict] = []

        for page in range(1, max_pages + 1):
            review_url = (
                f"{_BASE_URL}/product-reviews/{asin}"
                f"?pageNumber={page}&language=tr_TR"
            )

            try:
                response = await self._fetch(review_url)
            except Exception as exc:
                logger.error(
                    "review fetch failed", url=url, page=page, error=str(exc)
                )
                break

            if response.status_code != 200:
                break

            page_reviews = self._parse_review_page(response.text)
            if not page_reviews:
                break

            all_reviews.extend(page_reviews)

        # Cache
        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "amazon_tr")
            except Exception as exc:
                logger.debug("reviews cache write failed", error=str(exc))

        logger.info("reviews fetched", url=url, count=len(all_reviews))
        return all_reviews

    def _parse_review_page(self, html: str) -> list[dict]:
        """Parse a single page of Amazon reviews."""
        reviews: list[dict] = []

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        review_divs = soup.select("div[data-hook='review']")

        for div in review_divs:
            try:
                # Text
                body_el = div.select_one("span[data-hook='review-body'] span")
                if not body_el:
                    continue
                text = body_el.get_text(strip=True)
                if not text:
                    continue

                # Rating
                rating: float | None = None
                star_el = div.select_one("i[data-hook='review-star-rating'] span")
                if star_el:
                    m = re.search(r"([\d,]+)", star_el.get_text())
                    if m:
                        try:
                            rating = float(m.group(1).replace(",", "."))
                        except (ValueError, TypeError):
                            pass

                # Date
                date_el = div.select_one("span[data-hook='review-date']")
                date_str = date_el.get_text(strip=True) if date_el else None

                # Author
                author_el = div.select_one("span.a-profile-name")
                author = author_el.get_text(strip=True) if author_el else None

                # Verified
                verified = bool(
                    div.select_one("span[data-hook='avp-badge']")
                )

                # Helpful count
                helpful = 0
                helpful_el = div.select_one("span[data-hook='helpful-vote-statement']")
                if helpful_el:
                    m = re.search(r"(\d+)", helpful_el.get_text())
                    if m:
                        helpful = int(m.group(1))

                reviews.append({
                    "text": text,
                    "source": "amazon_tr",
                    "rating": rating,
                    "date": date_str,
                    "author": author,
                    "verified_purchase": verified,
                    "helpful_count": helpful,
                })
            except Exception as exc:
                logger.debug("review item parse error", error=str(exc))
                continue

        return reviews

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        """Check that the response is real Amazon content."""
        if response.status_code >= 400:
            return False

        text = response.text
        if not text or len(text) < 500:
            return False

        # Bot detection
        block_markers = ("captcha", "robot check", "automated access")
        for marker in block_markers:
            if marker in text.lower():
                logger.warning("possible bot detection", domain=self.domain)
                return False

        markers = ("amazon", "a-price", "s-result", "productTitle")
        return any(marker in text for marker in markers)


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
