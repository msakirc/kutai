"""Epey.com scraper — Turkey's most detailed product comparison site.

85+ spec fields per product, price aggregation across stores,
filter by exact specs (RAM, GPU, screen size, etc.).

Architecture note:
  - Category listing pages render via AJAX from /kat/listele/ (POST)
  - Product rows are <ul class="metin row" id="PRODUCT_ID">
  - Price is in <li class="fiyat cell">
  - Specs are <li class="ozellik ozellikNNNN cell"> text values
  - Detail page has <li id="idNNNN"><strong class="ozellikNNNN">LABEL</strong><span class="cell">VALUE</span></li>
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import cache_reviews, get_cached_reviews
from ..models import Product
from ..text_utils import parse_turkish_price, normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.epey")

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    logger.warning("bs4 (BeautifulSoup) is not installed -- Epey scraper disabled")

try:
    import curl_cffi.requests as _curl_requests
    _CURL_CFFI = True
except ImportError:
    _curl_requests = None
    _CURL_CFFI = False

# Category IDs for the /kat/listele/ AJAX endpoint.
# Discovered by fetching each category page and extracting kategori_id from JS.
_CATEGORY_IDS: dict[str, int] = {
    "laptop": 15,
    "akilli-telefonlar": 1,
    "televizyon": 3,
    "kulaklik": 63,
    "tablet": 2,
    "monitor": 139,
    "camasir-makinesi": 9,
    "bulasik-makinesi": 7,
    "klima": 28,
    "robot-supurge": 257,
}

# Map search query keywords to category slugs
_QUERY_CATEGORY_MAP: dict[str, str] = {
    "laptop": "laptop",
    "notebook": "laptop",
    "dizüstü": "laptop",
    "dizustu": "laptop",
    "telefon": "akilli-telefonlar",
    "phone": "akilli-telefonlar",
    "smartphone": "akilli-telefonlar",
    "akıllı": "akilli-telefonlar",
    "tv": "televizyon",
    "televizyon": "televizyon",
    "television": "televizyon",
    "kulaklik": "kulaklik",
    "kulaklık": "kulaklik",
    "headphone": "kulaklik",
    "tablet": "tablet",
    "monitor": "monitor",
    "monitör": "monitor",
    "camasir": "camasir-makinesi",
    "çamaşır": "camasir-makinesi",
    "bulasik": "bulasik-makinesi",
    "bulaşık": "bulasik-makinesi",
    "klima": "klima",
    "robot": "robot-supurge",
    "süpürge": "robot-supurge",
}


@register_scraper("epey")
class EpeyScraper(BaseScraper):
    """Epey.com product comparison scraper.

    Epey is Turkey's most spec-rich product comparison site with 85+ spec
    fields per product and price aggregation from all major Turkish retailers.

    The listing data is loaded via POST to /kat/listele/ — this endpoint
    requires curl_cffi for the TLS fingerprint needed to avoid blocks.
    """

    _BASE_URL = "https://www.epey.com"
    _LISTELE_URL = "https://www.epey.com/kat/listele/"
    _LISTELE_LIMIT = 35  # Epey's default page size

    def __init__(self) -> None:
        super().__init__(domain="epey")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search epey.com for products.

        Detects a category from the query and POSTs to /kat/listele/ for
        structured product rows.  Falls back to the search endpoint if no
        category is matched.
        """
        if not _BS4:
            return []

        category_slug = self._detect_category(query)

        if category_slug:
            products = await self._search_by_category(category_slug, query, max_results)
            if products:
                return products

        return await self._search_generic(query, max_results)

    def _detect_category(self, query: str) -> str | None:
        """Detect which epey category slug matches the query."""
        q_lower = query.lower()
        for keyword, slug in _QUERY_CATEGORY_MAP.items():
            if keyword in q_lower:
                return slug
        return None

    async def _search_by_category(
        self, category_slug: str, query: str, max_results: int
    ) -> list[Product]:
        """POST to /kat/listele/ with the category ID.

        Optionally adds an 'ara:QUERY' filtrele string if the query is more
        specific than just the category keyword.
        """
        category_id = _CATEGORY_IDS.get(category_slug)
        if not category_id:
            return []

        # Build filtrele: include keyword search if query has extra terms
        filtrele = ""
        query_words = query.lower().strip().split()
        # If query has words beyond the category keyword, add search filter
        extra_words = [
            w for w in query_words
            if w not in _QUERY_CATEGORY_MAP and len(w) > 2
        ]
        if extra_words:
            search_term = " ".join(extra_words)
            filtrele = f"ara:{search_term}"

        return await self._fetch_listele(category_id, filtrele, max_results)

    async def _fetch_listele(
        self, category_id: int, filtrele: str, max_results: int
    ) -> list[Product]:
        """POST to /kat/listele/ and parse the product rows."""
        limit = min(max_results, self._LISTELE_LIMIT)

        try:
            if _CURL_CFFI:
                html = await self._post_with_curl(category_id, limit, filtrele)
            else:
                html = await self._post_with_httpx(category_id, limit, filtrele)

            if not html:
                return []

            return self._parse_listele_html(html, max_results)

        except Exception as exc:
            logger.debug(
                "epey listele fetch failed",
                category_id=category_id,
                filtrele=filtrele,
                error=str(exc),
            )
            return []

    async def _post_with_curl(self, category_id: int, limit: int, filtrele: str) -> str:
        """POST to /kat/listele/ using curl_cffi TLS impersonation."""
        import asyncio
        loop = asyncio.get_event_loop()

        def _do_post() -> str:
            session = _curl_requests.Session(impersonate="chrome120")
            resp = session.post(
                self._LISTELE_URL,
                data={
                    "kategori_id": category_id,
                    "limit": limit,
                    "filtrele": filtrele,
                },
                headers={
                    "Referer": f"{self._BASE_URL}/laptop/",
                    "X-Requested-With": "XMLHttpRequest",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.text
            logger.warning(
                "epey listele non-200",
                status=resp.status_code,
                category_id=category_id,
            )
            return ""

        return await loop.run_in_executor(None, _do_post)

    async def _post_with_httpx(self, category_id: int, limit: int, filtrele: str) -> str:
        """Fallback POST via httpx when curl_cffi is unavailable."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.post(
                self._LISTELE_URL,
                data={
                    "kategori_id": category_id,
                    "limit": limit,
                    "filtrele": filtrele,
                },
                headers={
                    "User-Agent": self._random_ua(),
                    "Referer": f"{self._BASE_URL}/laptop/",
                    "X-Requested-With": "XMLHttpRequest",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.5,en;q=0.3",
                },
            )
            if resp.status_code == 200:
                return resp.text
            return ""

    async def _search_generic(self, query: str, max_results: int) -> list[Product]:
        """Search via the /ara/ endpoint (generic site search)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"{self._BASE_URL}/ara/{encoded}"
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return []
            return self._parse_listing_page(resp.text, max_results)
        except Exception as exc:
            logger.debug("epey generic search failed", query=query, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # get_product
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """Fetch a product detail page and return a richly-populated Product."""
        if not _BS4:
            return None
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return None
            return self._parse_product_page(url, resp.text)
        except Exception as exc:
            logger.debug("epey get_product failed", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # get_reviews
    # ------------------------------------------------------------------

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract user comments embedded in an Epey product page.

        Reviews live inside ``<div id="yorumlar">`` on the product detail
        page itself (no separate ``/yorumlar/`` subpage exists).  Each
        top-level review is a ``<div class="yorum row ...">`` and replies
        are ``<div class="yanit row ...">``.  Pagination, when present,
        follows ``?p=N`` on the product URL.
        """
        if not _BS4:
            return []

        # Cache check
        try:
            cached = await get_cached_reviews(url, "epey")
            if cached is not None:
                logger.debug("epey reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        all_reviews: list[dict] = []
        seen_ids: set[str] = set()

        for page in range(1, max_pages + 1):
            page_url = url if page == 1 else (
                f"{url}?p={page}" if "?" not in url else f"{url}&p={page}"
            )
            try:
                resp = await self._fetch(page_url)
            except Exception as exc:
                logger.debug("epey reviews fetch failed", url=page_url, error=str(exc))
                break
            if resp.status_code != 200:
                break
            page_reviews = self._parse_reviews_html(resp.text)
            new_count = 0
            for r in page_reviews:
                rid = r.get("entry_id") or f"{r.get('author','')}|{r.get('text','')[:60]}"
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                all_reviews.append(r)
                new_count += 1
            if new_count == 0:
                break

        # Sort by helpful_count descending so top reviews come first
        all_reviews.sort(key=lambda r: r.get("helpful_count", 0), reverse=True)

        if all_reviews:
            try:
                await cache_reviews(url, all_reviews, "epey")
            except Exception:
                pass

        logger.info("epey reviews fetched", url=url, count=len(all_reviews))
        return all_reviews

    def _parse_reviews_html(self, html: str) -> list[dict]:
        """Parse top-level reviews and replies from a product page HTML."""
        reviews: list[dict] = []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return []

        yorumlar = soup.select_one("#yorumlar") or soup
        # Top-level reviews + replies. Multi-strategy fallback for resilience.
        review_els = (
            yorumlar.select("div.yorum.row, div.yanit.row")
            or yorumlar.select("div[class~='yorum'][class~='row']")
            or yorumlar.select("div[id^='4']")
        )

        for el in review_els:
            try:
                classes = el.get("class", []) or []
                # Skip the empty "yorumyaz" template card
                if "yorumyaz" in classes:
                    continue

                # Author — multi-strategy
                author_el = (
                    el.select_one("span.adi")
                    or el.select_one(".cell .adi")
                    or el.select_one("b")
                )
                author = author_el.get_text(strip=True) if author_el else None

                # Text
                text_el = (
                    el.select_one("span.metin")
                    or el.select_one(".cell.c86a45 span.metin")
                    or el.select_one("div.cell span:not(.adi)")
                )
                if text_el is None:
                    continue
                text = re.sub(r"\s+", " ", text_el.get_text(separator=" ", strip=True)).strip()
                if not text or len(text) < 3:
                    continue

                # Helpful count — number inside .dugme.faydali > span
                helpful = 0
                fav_el = (
                    el.select_one("span.dugme.faydali span")
                    or el.select_one(".faydali span")
                )
                if fav_el:
                    try:
                        helpful = int(re.sub(r"[^\d]", "", fav_el.get_text(strip=True)) or 0)
                    except (ValueError, TypeError):
                        pass

                # Unhelpful (dislikes)
                unhelpful = 0
                unfav_el = (
                    el.select_one("span.dugme.faydasiz span")
                    or el.select_one(".faydasiz span")
                )
                if unfav_el:
                    try:
                        unhelpful = int(re.sub(r"[^\d]", "", unfav_el.get_text(strip=True)) or 0)
                    except (ValueError, TypeError):
                        pass

                # Date — relative phrase like "5 gün önce" inside .secenek
                date_str: str | None = None
                for sp in el.select(
                    "div.secenek span, div.secyorum span, div.secyanit span"
                ):
                    txt = sp.get_text(strip=True)
                    if re.search(r"(önce|dakika|saat|gün|hafta|ay|yıl)", txt, re.I):
                        date_str = txt.lstrip("- ").strip()
                        break

                # Variant info (e.g., " - 1 TB") — bold span at end
                variant: str | None = None
                bold_spans = el.select("div.secenek span[style*='bold']") or []
                if bold_spans:
                    raw = bold_spans[-1].get_text(strip=True).lstrip("- ").strip()
                    if raw:
                        variant = raw

                entry_id = el.get("id")
                is_reply = "yanit" in classes

                review: dict[str, Any] = {
                    "text": text,
                    "source": "epey",
                    "author": author,
                    "date": date_str,
                    "rating": None,
                    "helpful_count": helpful,
                    "unhelpful_count": unhelpful,
                    "is_reply": is_reply,
                }
                if entry_id:
                    review["entry_id"] = entry_id
                if variant:
                    review["variant"] = variant

                reviews.append(review)
            except Exception as exc:
                logger.debug("epey review parse error", error=str(exc))
                continue

        return reviews

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: Any) -> bool:
        """Return True if the response looks like real Epey content."""
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 200:
            return False
        markers = ("epey.com", "metin row", "ozellik", "fiyat", "urunadi", "listele")
        return any(m in text.lower() for m in markers)

    # ------------------------------------------------------------------
    # get_product_details  (public convenience)
    # ------------------------------------------------------------------

    async def get_product_details(self, url: str) -> dict[str, str]:
        """Fetch detailed specs from a product page.

        Returns a flat dict of spec_name → spec_value with 85+ entries.
        """
        try:
            resp = await self._fetch(url)
            if not self.validate_response(resp):
                return {}
            return self._parse_product_specs(resp.text)
        except Exception as exc:
            logger.debug("epey product detail fetch failed", url=url, error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # HTML parsers
    # ------------------------------------------------------------------

    def _parse_listele_html(self, html: str, max_results: int) -> list[Product]:
        """Parse product rows from /kat/listele/ response.

        Each product is a <ul class="metin row" id="PRODUCT_ID"> element.
        """
        soup = BeautifulSoup(html, "lxml")
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        rows = soup.select("ul.metin.row")
        for row in rows[:max_results]:
            try:
                product = self._parse_product_row(row, now_iso)
                if product is not None:
                    products.append(product)
            except Exception as exc:
                logger.debug("epey row parse error", error=str(exc))

        logger.info("epey listing parsed", count=len(products))
        return products

    def _parse_product_row(self, row: Any, now_iso: str) -> Product | None:
        """Parse a single <ul class='metin row'> element into a Product."""
        # --- name & URL ---
        name_el = row.select_one("a.urunadi")
        if not name_el:
            return None

        name = (
            name_el.get("title", "").strip()
            or name_el.get_text(strip=True)
        )
        if len(name) < 3:
            return None
        name = normalize_product_name(name)

        href = name_el.get("href", "")
        url = href if href.startswith("http") else f"{self._BASE_URL}{href}"

        # --- price ---
        price: float | None = None
        store_count_str = ""
        fiyat_li = row.select_one("li.fiyat")
        if fiyat_li:
            fiyat_a = fiyat_li.select_one("a")
            if fiyat_a:
                # Price text is the direct text node, store count is in <span>
                store_span = fiyat_a.select_one("span")
                if store_span:
                    store_count_str = store_span.get_text(strip=True)
                    store_span.decompose()  # remove span to get clean price text
                price_text = fiyat_a.get_text(strip=True)
                parsed = parse_turkish_price(price_text)
                if parsed is not None:
                    price = parsed

        # --- inline specs (ozellikNNNN cells) ---
        specs: dict[str, Any] = {}
        for ozellik_li in row.select("li[class*='ozellik']"):
            classes = ozellik_li.get("class", [])
            # Find the ozellikNNNN class to use as spec key
            spec_key = None
            for cls in classes:
                if re.match(r"ozellik\d+", cls):
                    spec_key = cls
                    break
            if spec_key:
                val = ozellik_li.get_text(strip=True)
                if val and val != "-":
                    specs[spec_key] = val

        if store_count_str:
            m = re.search(r"(\d+)", store_count_str)
            if m:
                specs["store_count"] = int(m.group(1))

        # --- rating (circliful widget data-percent) ---
        rating: float | None = None
        puan_div = row.select_one("div[data-percent]")
        if puan_div:
            try:
                rating = float(puan_div.get("data-percent", 0))
            except (ValueError, TypeError):
                pass

        # --- image ---
        image_url: str | None = None
        img_el = row.select_one("img")
        if img_el:
            src = img_el.get("src") or img_el.get("data-src") or ""
            if src and not src.startswith("data:"):
                image_url = f"https:{src}" if src.startswith("//") else src

        return Product(
            name=name,
            url=url,
            source="epey",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            rating=rating,
            fetched_at=now_iso,
        )

    def _parse_listing_page(self, html: str, max_results: int) -> list[Product]:
        """Parse listing page HTML (fallback for generic search results).

        The generic search page may also contain metin rows inside the
        main #listele div.
        """
        return self._parse_listele_html(html, max_results)

    def _parse_product_page(self, url: str, html: str) -> Product | None:
        """Parse a product detail page into a Product with 85+ spec fields."""
        soup = BeautifulSoup(html, "lxml")
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- name ---
        name_el = soup.select_one("h1")
        if not name_el:
            return None
        name = normalize_product_name(name_el.get_text(strip=True))

        # Strip trailing " Laptop Fiyatları" / " Telefon Fiyatları" suffixes
        name = re.sub(r"\s+(Laptop|Telefon|Tablet|TV|Monitor)\s+Fiyatlar[ıi].*$", "", name, flags=re.I)

        # --- price ---
        price: float | None = None
        price_el = soup.select_one(".minFiyat, .enDusuk, [class*='minFiyat'], [class*='fiyat']")
        if price_el:
            price = parse_turkish_price(price_el.get_text(strip=True))

        # --- specs (85+ fields) ---
        specs = self._parse_product_specs(html)

        # --- image ---
        image_url: str | None = None
        img_el = soup.select_one(".urunResim img, .product-image img, img[itemprop='image']")
        if not img_el:
            # Try OG image tag
            og = soup.select_one('meta[property="og:image"]')
            if og:
                image_url = og.get("content")
        if img_el:
            src = img_el.get("data-src") or img_el.get("src") or ""
            if src and not src.startswith("data:"):
                image_url = f"https:{src}" if src.startswith("//") else src

        return Product(
            name=name,
            url=url,
            source="epey",
            discounted_price=price,
            currency="TRY",
            image_url=image_url,
            specs=specs,
            fetched_at=now_iso,
        )

    def _parse_product_specs(self, html: str) -> dict[str, str]:
        """Parse all spec fields from a product detail page.

        Epey uses <li id="idNNNN"><strong class="ozellikNNNN">LABEL</strong>
        <span class="cell csN"><span>VALUE</span></li> for each spec.

        Returns a flat dict of label → value with 85+ entries.
        """
        soup = BeautifulSoup(html, "lxml")
        specs: dict[str, str] = {}

        # Primary pattern: li[id^="id"] containing strong.ozellikNNNN + span.cell
        for li in soup.select('li[id^="id"]'):
            label_el = li.select_one("strong[class*='ozellik']")
            value_container = li.select_one("span.cell")
            if label_el and value_container:
                label = label_el.get_text(strip=True)
                value = value_container.get_text(strip=True, separator=" ").strip()
                if label and value and value != "-":
                    specs[label] = value

        return specs
