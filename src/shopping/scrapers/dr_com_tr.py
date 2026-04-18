"""dr.com.tr scraper — D&R Turkey's book, music, game and electronics retailer.

Specialises in:
  - Books (Turkish and foreign language)
  - Music CDs, vinyl
  - Movies, games
  - Small electronics, stationery

Architecture notes:
  - Search: GET /search?q=QUERY
  - Product items: div.product-card (.js-prd-item)
  - Data source: data-gtm JSON attribute on each card — contains:
      item_name, author, publisher, price, discount_rate, item_rating,
      number_of_comments, item_stock, item_category, item_id
  - URL: a.js-search-prd-item[href] (relative, prepend base)
  - Price: .prd-price text (format: "479,40 TL")
  - Image: img.lazyload[data-src]
  - Rating: item_rating (0–10 scale) from GTM data
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.dr_com_tr")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- dr.com.tr scraper disabled")

_BASE_URL = "https://www.dr.com.tr"


@register_scraper("dr")
class DrComTrScraper(BaseScraper):
    """D&R (dr.com.tr) book and entertainment scraper.

    D&R is one of Turkey's largest retail chains for books, music, movies,
    and electronics.  Listing pages embed GTM JSON data directly on product
    card elements, providing clean structured data without JS rendering.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/search"

    def __init__(self) -> None:
        super().__init__(domain="dr")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search dr.com.tr for products matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?q={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("dr.com.tr search failed", query=query, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a single product page."""
        if not _BS4:
            return None
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return None
            return self._parse_product_page(url, resp.text)
        except Exception as exc:
            logger.debug("dr.com.tr get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Fetch reviews via D&R's NewDrProductReviewsAsync JSON endpoint.

        Strategy:
          1. Cache check (``shopping.cache.get_cached_reviews``).
          2. Resolve the internal numeric ``prdId`` (different from the SKU /
             ``urunno``). Fetch the product page and grep ``prdId: <int>`` from
             the page-level ``var page = { ... }`` block.
          3. POST ``/Product/NewDrProductReviewsAsync?productid=<prdId>`` —
             returns ``{success: true, result: [...]}`` with the full review list.
          4. Map fields: ``ReviewText -> text``, ``Rating`` (0–10 → 0–5),
             ``WrittenOn`` (".NET Date" /Date(ms)/) → ISO date, etc.

        The ebook variant of the endpoint (``...AsyncEbook``) is tried as a
        fallback when the standard endpoint returns ``{success:false}``.
        """
        from ..cache import get_cached_reviews, cache_reviews

        if not _BS4:
            return []

        cached = await get_cached_reviews(url, "dr")
        if cached is not None:
            return cached

        try:
            prd_id = await self._resolve_prd_id(url)
            if not prd_id:
                logger.debug("dr.com.tr reviews: no prd_id", url=url)
                return []

            raw_reviews = await self._fetch_dr_reviews_api(prd_id, referer=url)
            reviews: list[dict] = []
            for r in raw_reviews:
                parsed = self._parse_dr_review(r)
                if parsed and parsed.get("text"):
                    reviews.append(parsed)

            await cache_reviews(url, reviews, "dr")
            logger.info(
                "dr.com.tr reviews fetched",
                url=url,
                prd_id=prd_id,
                count=len(reviews),
            )
            return reviews
        except Exception as exc:
            logger.debug("dr.com.tr get_reviews failed", url=url, error=str(exc))
            return []

    async def _resolve_prd_id(self, url: str) -> str | None:
        """Find the internal numeric prdId by fetching the product page."""
        try:
            resp = await self._fetch(url)
            if not resp or not resp.text:
                return None
            html = resp.text
            # var page = { prdId: 1048558, ... } (or "prdId": 1048558)
            m = re.search(r"prdId\s*[:=]\s*['\"]?(\d+)", html)
            if m:
                return m.group(1)
            # Fallback: data-id attribute on <main> or wrapper
            m = re.search(r'data-id=["\'](\d+)["\']', html)
            if m:
                return m.group(1)
        except Exception:
            return None
        return None

    async def _fetch_dr_reviews_api(self, prd_id: str, referer: str) -> list[dict]:
        """Hit DR's review JSON endpoint directly via httpx (POST)."""
        import httpx as _httpx

        endpoints = [
            f"{self._BASE_URL}/Product/NewDrProductReviewsAsync?productid={prd_id}",
            f"{self._BASE_URL}/Product/NewDrProductReviewsAsyncEbook?productid={prd_id}",
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": referer or self._BASE_URL,
        }
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for ep in endpoints:
                try:
                    r = await client.post(ep, headers=headers)
                    if r.status_code != 200:
                        continue
                    j = r.json()
                    if not isinstance(j, dict):
                        continue
                    if not j.get("success"):
                        continue
                    result = j.get("result")
                    if isinstance(result, list) and result:
                        return result
                except Exception:
                    continue
        return []

    def _parse_dr_review(self, r: dict) -> dict | None:
        text = (r.get("ReviewText") or "").strip()
        if not text:
            return None

        # Author from CustomerName + CustomerLastName
        author_parts = [
            (r.get("CustomerName") or "").strip(),
            (r.get("CustomerLastName") or "").strip(),
        ]
        author = " ".join(p for p in author_parts if p) or None

        # Rating: D&R reviews use 0–10 scale; many reviews carry Rating=0 even
        # when the review is positive (legacy data). Normalise to 0–5.
        rating: float | None = None
        raw_r = r.get("Rating")
        if isinstance(raw_r, (int, float)) and raw_r > 0:
            rating = round(float(raw_r) / 2.0, 1)

        # Date: ".NET Date" /Date(1618081071433)/ — milliseconds since epoch
        date_iso: str | None = None
        wo = r.get("WrittenOn") or ""
        m = re.search(r"/Date\((\d+)", wo)
        if m:
            try:
                ts_ms = int(m.group(1))
                from datetime import datetime as _dt, timezone as _tz
                date_iso = _dt.fromtimestamp(ts_ms / 1000, tz=_tz.utc).date().isoformat()
            except Exception:
                date_iso = None
        if not date_iso and r.get("WrittenOnStr"):
            date_iso = str(r.get("WrittenOnStr"))

        helpful = 0
        h = r.get("Helpfulness") or {}
        if isinstance(h, dict):
            try:
                helpful = int(h.get("HelpfulYesTotal") or 0)
            except (ValueError, TypeError):
                helpful = 0

        title = (r.get("Title") or "").strip()
        # Prepend title if it adds info beyond the body
        if title and title.lower() not in text.lower():
            text = f"{title}. {text}"

        return {
            "text": text,
            "source": "dr",
            "rating": rating,
            "date": date_iso,
            "author": author,
            "verified_purchase": False,
            "helpful_count": helpful,
        }

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: Any) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 500:
            return False
        markers = ("dr.com.tr", "product-card", "js-prd-item", "prd-price")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(".product-card")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("dr.com.tr item parse error", error=str(exc))

        logger.info("dr.com.tr listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        # --- GTM data (primary data source) ---
        gtm_str = item.get("data-gtm", "")
        gtm: dict = {}
        if gtm_str:
            try:
                gtm = json.loads(gtm_str)
            except json.JSONDecodeError:
                pass

        # --- name ---
        name_raw = gtm.get("item_name", "").strip()
        if not name_raw:
            return None
        name = normalize_product_name(name_raw)

        # --- URL ---
        link_el = item.select_one("a.js-search-prd-item, a[href*='/urunno=']")
        href = link_el.get("href", "") if link_el else ""
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # --- price ---
        price: float | None = None
        price_el = item.select_one(".prd-price, [class*=price]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        # Reconstruct original price from GTM discount_rate
        original_price: float | None = None
        discount_rate = gtm.get("discount_rate", 0)
        if price is not None and discount_rate and discount_rate > 0:
            try:
                dr = float(discount_rate)
                original_price = round(price / (1 - dr / 100), 2)
            except (ZeroDivisionError, ValueError):
                pass

        # --- rating (GTM gives 0–10 scale, normalise to 0–5) ---
        rating: float | None = None
        item_rating = gtm.get("item_rating")
        if item_rating is not None:
            try:
                rating = round(float(item_rating) / 2.0, 1)
            except (ValueError, TypeError):
                pass

        review_count: int | None = None
        n_comments = gtm.get("number_of_comments")
        if n_comments is not None:
            try:
                review_count = int(n_comments)
            except (ValueError, TypeError):
                pass

        # --- stock ---
        in_stock_str = gtm.get("item_stock", "Yes")
        availability = "in_stock" if str(in_stock_str).lower() in ("yes", "true", "1") else "out_of_stock"

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img.lazyload, img[data-src]")
        if img_el:
            src = img_el.get("data-src") or img_el.get("src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        # --- specs from GTM ---
        specs: dict[str, Any] = {}
        if gtm.get("author"):
            specs["author"] = gtm["author"]
        if gtm.get("publisher"):
            specs["publisher"] = gtm["publisher"]
        if gtm.get("item_category"):
            specs["category"] = gtm["item_category"]
        if gtm.get("item_variant"):
            specs["variant"] = gtm["item_variant"]
        if gtm.get("item_id"):
            specs["item_id"] = gtm["item_id"]

        return Product(
            name=name,
            url=url,
            source="dr",
            original_price=original_price,
            discounted_price=price,
            discount_percentage=float(discount_rate) if discount_rate else None,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating,
            review_count=review_count,
            availability=availability,
            fetched_at=now_iso,
        )

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse a product detail page."""
        soup = BeautifulSoup(html, "lxml")
        now_iso = datetime.now(timezone.utc).isoformat()

        name_el = soup.select_one("h1")
        if not name_el:
            return None
        name = normalize_product_name(name_el.get_text(strip=True))

        price: float | None = None
        price_el = soup.select_one(".prd-price, [class*=sell-price], [class*=price]")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        image_url: str | None = None
        img_el = soup.select_one(".product-image img, [class*=product-img] img")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = src

        return Product(
            name=name,
            url=url,
            source="dr",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
