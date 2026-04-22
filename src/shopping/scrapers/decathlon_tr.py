"""decathlon.com.tr scraper — Decathlon Turkey's sports goods retailer.

Specialises in:
  - Bicycles (mountain, road, city, kids)
  - Camping & outdoor equipment
  - Fitness & gym gear
  - Swimming, running, football, basketball equipment
  - Hiking boots, sports apparel

Architecture notes:
  - Search: GET /search?Ntt=QUERY
  - Product items: div[data-supermodelid]
  - Name: span.vh (visually hidden full name inside the product link)
  - URL: a.dpb-product-model-link[href] (relative, prepend base)
  - Price: span.vtmn-price text (format: "₺20.990")
  - Brand: 3rd pipe-separated text segment in the item's full text
  - Review count: span.vtmn-rating_comment--secondary (format: "(1760)")
  - Image: img[src] or img[srcset] inside the item
  - No numeric rating score on listing page (only review count)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from .base import BaseScraper, register_scraper
from ..models import Product
from ..text_utils import normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.decathlon_tr")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 not installed -- decathlon.com.tr scraper disabled")

_BASE_URL = "https://www.decathlon.com.tr"

# Decathlon prices use ₺ symbol with Turkish number format (. as thousand sep)
_PRICE_RE = re.compile(r"[₺\s]*([\d.,]+)")


def _parse_decathlon_price(text: str) -> float | None:
    """Parse Decathlon price strings like '₺20.990' or '₺1.299,90'."""
    if not text:
        return None
    m = _PRICE_RE.search(text.strip())
    if not m:
        return None
    raw = m.group(1)
    # Decathlon uses period as thousands separator, comma as decimal
    # e.g. "20.990" -> 20990, "1.299,90" -> 1299.90
    if "," in raw:
        # Has decimal: remove thousands dots, replace comma decimal
        raw = raw.replace(".", "").replace(",", ".")
    else:
        # No comma: period is thousands separator
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


@register_scraper("decathlon")
class DecathlonTrScraper(BaseScraper):
    """decathlon.com.tr sports equipment scraper.

    Decathlon Turkey renders search results server-side with Svelte components.
    The HTML is fully present in the initial response without JS execution,
    making it scrapable via TLS tier.
    """

    _BASE_URL = _BASE_URL
    _SEARCH_URL = f"{_BASE_URL}/search"

    def __init__(self) -> None:
        super().__init__(domain="decathlon")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search decathlon.com.tr for products matching *query*."""
        if not _BS4:
            return []

        url = f"{self._SEARCH_URL}?Ntt={quote_plus(query)}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing(resp.text, max_results)
        except Exception as exc:
            logger.debug("decathlon search failed", query=query, error=str(exc))
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
            logger.debug("decathlon get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Fetch reviews from the dedicated Decathlon reviews page.

        Strategy:
          1. Cache check (``shopping.cache.get_cached_reviews``).
          2. Derive the review URL by rewriting ``/p/<slug>/_/R-p-<id>`` to
             ``/r/<slug>/_/R-p-<id>``. Decathlon TR mirrors the product slug
             on its review subpage and renders the first ~10 reviews
             server-side (Svelte SSR).
          3. Parse ``article.review-item`` blocks for body, rating, date,
             author. Reviews live in ``.review-item__body`` (primary) with
             fallbacks to ``.review-content`` / ``[itemprop=reviewBody]`` for
             selector renames.

        ``max_pages`` is currently advisory — Decathlon's review pagination is
        client-side ``?from=N`` appended to the same URL; the first SSR page
        already returns ~10 reviews which is enough for sentiment.
        """
        from ..cache import get_cached_reviews, cache_reviews

        if not _BS4:
            return []

        cached = await get_cached_reviews(url, "decathlon")
        if cached is not None:
            return cached

        try:
            review_url = self._derive_review_url(url)
            if not review_url:
                logger.debug("decathlon reviews: no review_url", url=url)
                return []

            html = await self._fetch_html_stealth(review_url)
            if not html:
                return []

            reviews = self._parse_reviews_html(html)
            await cache_reviews(url, reviews, "decathlon")
            logger.info(
                "decathlon reviews fetched",
                url=url,
                review_url=review_url,
                count=len(reviews),
            )
            return reviews
        except Exception as exc:
            logger.debug("decathlon get_reviews failed", url=url, error=str(exc))
            return []

    async def _fetch_html_stealth(self, url: str) -> str | None:
        """Fetch HTML with STEALTH tier — Decathlon TR returns 403 at TLS tier
        for review subpages, so we bypass BaseScraper._fetch (which caps at TLS).
        """
        try:
            from src.tools.scraper import scrape_url, ScrapeTier
            result = await scrape_url(url, max_tier=ScrapeTier.STEALTH, timeout=30.0)
            if result.status >= 400 or not result.html:
                return None
            return result.html
        except Exception as exc:
            logger.debug("decathlon stealth fetch failed", url=url, error=str(exc))
            return None

    def _derive_review_url(self, product_url: str) -> str | None:
        """Convert /p/<slug>/_/R-p-<id> → /r/<slug>/_/R-p-<id> (drop query)."""
        # Strip query string
        base = product_url.split("?", 1)[0]
        # Match /p/.../_/R-p-<id>
        m = re.match(r"(https?://[^/]+)/p/([^?]+?)/_/R-p-(\d+)/?$", base)
        if m:
            host, slug, pid = m.group(1), m.group(2), m.group(3)
            return f"{host}/r/{slug}/_/R-p-{pid}"
        # Already a review URL?
        if "/r/" in base and "/R-p-" in base:
            return base
        # Fallback: try just appending /reviews
        return None

    def _parse_reviews_html(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(
            "article.review-item, .review-item, [itemtype*='Review']"
        )
        out: list[dict] = []
        for el in items:
            try:
                review = self._parse_one_review(el)
                if review and review.get("text"):
                    out.append(review)
            except Exception as exc:
                logger.debug("decathlon review parse error", error=str(exc))
        return out

    def _parse_one_review(self, el: Any) -> dict | None:
        # --- text/body ---
        body_el = el.select_one(
            ".review-item__body, .review-content, [itemprop='reviewBody'], p.review-text"
        )
        text = body_el.get_text(" ", strip=True) if body_el else ""
        if not text:
            return None

        # --- title (prepend if distinct) ---
        title_el = el.select_one(".review-title, .review-item__title, h3")
        title = title_el.get_text(strip=True) if title_el else ""
        if title and title.lower() not in text.lower():
            text = f"{title}. {text}"

        # --- author ---
        # The author block is the first .vtmn-text-content-secondary > span
        author = None
        author_el = el.select_one(
            ".vtmn-text-content-secondary span, [itemprop='author'], .review-author, .reviewer-name"
        )
        if author_el:
            txt = author_el.get_text(strip=True)
            # Skip pipe-separated country tokens
            if txt and "|" not in txt and len(txt) < 60:
                author = txt

        # --- rating: parse "5/5" text or count star icons ---
        rating: float | None = None
        rcomment = el.select_one(".vtmn-rating_comment--primary, .rating-value")
        if rcomment:
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*5", rcomment.get_text())
            if m:
                try:
                    rating = float(m.group(1).replace(",", "."))
                except ValueError:
                    rating = None
        if rating is None:
            stars = el.select(".vtmx-star-fill")
            # The author block also has stars; cap at 5
            if stars:
                rating = float(min(5, len(stars)))

        # --- date ---
        date_iso: str | None = None
        time_el = el.select_one("time[datetime]")
        if time_el:
            date_iso = time_el.get("datetime") or time_el.get_text(strip=True)
        elif el.select_one("time"):
            date_iso = el.select_one("time").get_text(strip=True)

        # --- verified purchase ---
        verified = bool(
            el.select_one(".vtmx-checkbox-circle-line")
            or "Doğrulanmış" in el.get_text()
        )

        return {
            "text": text,
            "source": "decathlon",
            "rating": rating,
            "date": date_iso,
            "author": author,
            "verified_purchase": verified,
            "helpful_count": 0,
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
        markers = ("decathlon", "dpb-product-model-link", "vtmn-price", "data-supermodelid")
        return any(m in text for m in markers)

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_search_html(self, html: str, max_results: int = 20) -> list[Product]:
        """Public alias for testability."""
        return self._parse_listing(html, max_results)

    def _parse_listing(self, html: str, max_results: int) -> list[Product]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select("[data-supermodelid]")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items[:max_results]:
            try:
                product = self._parse_item(item, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("decathlon item parse error", error=str(exc))

        logger.info("decathlon listing parsed", count=len(products))
        return products

    def _parse_item(self, item: Any, now_iso: str) -> Product | None:
        # --- name & URL ---
        link_el = item.select_one("a.dpb-product-model-link")
        if not link_el:
            return None

        href = link_el.get("href", "")
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # Name is in the visually-hidden span inside the link
        name_el = link_el.select_one("span.vh")
        name = normalize_product_name(
            name_el.get_text(strip=True) if name_el else link_el.get_text(strip=True)
        )
        if len(name) < 3:
            return None

        # --- price ---
        price: float | None = None
        price_el = item.select_one("span.vtmn-price")
        if price_el:
            price = _parse_decathlon_price(price_el.get_text(strip=True))

        # --- brand (extracted from pipe-separated full text) ---
        # Pattern: "ProductName|Price|BrandName|ProductName again|(ReviewCount)|..."
        full_text = item.get_text(separator="|", strip=True)
        parts = [p.strip() for p in full_text.split("|") if p.strip()]
        brand: str | None = None
        # Brand is usually after the price and before the repeated product name
        price_idx = -1
        for i, part in enumerate(parts):
            if "₺" in part or "TL" in part:
                price_idx = i
                break
        if price_idx >= 0 and price_idx + 1 < len(parts):
            candidate = parts[price_idx + 1]
            # Brand is typically short (1–3 words) and not a product sentence
            if len(candidate.split()) <= 4 and len(candidate) < 40:
                brand = candidate

        # --- review count ---
        review_count: int | None = None
        rating_el = item.select_one(".vtmn-rating_comment--secondary")
        if rating_el:
            rating_text = rating_el.get_text(strip=True)
            m = re.search(r"(\d+)", rating_text)
            if m:
                review_count = int(m.group(1))

        # --- image ---
        image_url: str | None = None
        img_el = item.select_one("img[src]")
        if img_el:
            src = img_el.get("src", "")
            # Prefer higher-quality srcset variant
            srcset = img_el.get("srcset", "")
            if srcset:
                # Take the last (largest) srcset entry
                last_entry = srcset.strip().split(",")[-1].strip()
                src_candidate = last_entry.split(" ")[0]
                if src_candidate:
                    src = src_candidate
            if src and not src.startswith("data:"):
                image_url = src

        specs: dict[str, Any] = {}
        if brand:
            specs["brand"] = brand
        product_id = item.get("data-supermodelid")
        if product_id:
            specs["model_id"] = product_id

        # --- SKU: data-supermodelid is the canonical Decathlon model id ---
        sku: str | None = None
        if product_id:
            sku = f"dc-{product_id}"
        else:
            # Fall back to numeric id in URL: /p/<slug>/_/R-p-<id>
            m_sku = re.search(r"/R-p-(\d+)", url)
            if m_sku:
                sku = f"dc-{m_sku.group(1)}"

        return Product(
            name=name,
            url=url,
            source="decathlon",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            review_count=review_count,
            fetched_at=now_iso,
            sku=sku,
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
        price_el = soup.select_one("span.vtmn-price, [class*=price]")
        if price_el:
            price = _parse_decathlon_price(price_el.get_text(strip=True))

        image_url: str | None = None
        og_img = soup.select_one('meta[property="og:image"]')
        if og_img:
            image_url = og_img.get("content")

        return Product(
            name=name,
            url=url,
            source="decathlon",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            fetched_at=now_iso,
        )
