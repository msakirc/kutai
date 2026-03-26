"""Turkish tech forum scrapers -- Technopat and DonanımHaber.

These scrapers pull user discussions and reviews from Turkey's two
largest tech forums.  They use simple httpx + BeautifulSoup with light
anti-bot handling (User-Agent rotation, polite delays).

Both scrapers:
  - search: hit the forum's search URL, extract thread titles/URLs,
    score threads for relevance.
  - get_reviews: extract posts from up to 3 threads (max 50 posts each),
    with special handling for ``çözüm`` (solution) tagged posts.
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
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.forums")

# Graceful bs4 import
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("bs4 not installed -- forum scrapers disabled")

# ---------------------------------------------------------------------------
# Relevance scoring keywords
# ---------------------------------------------------------------------------

_POSITIVE_KEYWORDS = [
    "inceleme", "review", "deneyim", "tavsiye", "öneri",
    "karşılaştırma", "comparison", "memnun", "şikayet",
    "kullanıcı yorumu", "test", "benchmark", "performans",
    "fiyat", "kalite", "sorun", "problem", "çözüm",
]

_NEGATIVE_KEYWORDS = [
    "satılık", "takas", "acil", "hediye",
]


def _score_thread(title: str) -> float:
    """Score a thread title for relevance (0.0 - 1.0)."""
    title_lower = title.lower()
    score = 0.5

    for kw in _POSITIVE_KEYWORDS:
        if kw in title_lower:
            score += 0.1

    for kw in _NEGATIVE_KEYWORDS:
        if kw in title_lower:
            score -= 0.2

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Technopat
# ---------------------------------------------------------------------------


@register_scraper("technopat")
class TechnopatScraper(BaseScraper):
    """Scrape discussions from Technopat Social forum."""

    _BASE_URL = "https://www.technopat.net/sosyal"
    _SEARCH_URL = "https://www.technopat.net/sosyal/search/search"

    def __init__(self) -> None:
        super().__init__(domain="technopat")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search Technopat forum for discussion threads.

        Returns Product objects where URL points to the thread and
        ``specs`` contains thread metadata (reply_count, view_count, etc.).
        """
        if not _BS4_AVAILABLE:
            return []

        params = {
            "keywords": query,
            "type": "post",
            "order": "relevance",
        }

        try:
            response = await self._fetch(
                self._SEARCH_URL, params=params
            )
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        return self._parse_search(response.text, max_results)

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        """Parse Technopat search results."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        threads = (
            soup.select("li.block-row")
            or soup.select("div.contentRow")
            or soup.select("ol.listPlain li")
        )

        for thread in threads[:max_results]:
            try:
                # Title and URL
                title_el = (
                    thread.select_one("h3.contentRow-title a")
                    or thread.select_one("a.contentRow-title")
                    or thread.select_one("a[data-tp-title]")
                )
                if title_el is None:
                    continue

                title = title_el.get_text(strip=True)
                if not title:
                    continue

                href = title_el.get("href", "")
                thread_url = (
                    href
                    if href.startswith("http")
                    else f"{self._BASE_URL}/{href.lstrip('/')}"
                )

                # Relevance score
                score = _score_thread(title)
                if score < 0.3:
                    continue

                # Metadata
                specs: dict[str, Any] = {
                    "type": "forum_thread",
                    "forum": "technopat",
                    "relevance_score": round(score, 2),
                }

                # Reply / view counts
                meta_el = thread.select_one("ul.listInline--bullet")
                if meta_el:
                    meta_text = meta_el.get_text()
                    reply_m = re.search(r"(\d+)\s*(?:yanıt|cevap|reply)", meta_text, re.IGNORECASE)
                    view_m = re.search(r"(\d+)\s*(?:görüntü|view)", meta_text, re.IGNORECASE)
                    if reply_m:
                        specs["reply_count"] = int(reply_m.group(1))
                    if view_m:
                        specs["view_count"] = int(view_m.group(1))

                # Date
                time_el = thread.select_one("time")
                if time_el:
                    specs["date"] = time_el.get("datetime", time_el.get_text(strip=True))

                products.append(
                    Product(
                        name=title,
                        url=thread_url,
                        source="technopat",
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("thread parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    async def get_product(self, url: str) -> Product | None:
        """Forums do not have product pages. Returns None."""
        return None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract posts from Technopat thread(s).

        Fetches up to *max_pages* (treated as max threads) with up to
        50 posts each.  Posts tagged as ``çözüm`` (solution) are flagged.
        """
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_reviews(url, "technopat")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        all_posts = await self._extract_thread_posts(url, max_posts=50)

        if all_posts:
            try:
                await cache_reviews(url, all_posts, "technopat")
            except Exception:
                pass

        return all_posts

    async def _extract_thread_posts(
        self, url: str, max_posts: int = 50
    ) -> list[dict]:
        """Extract posts from a single Technopat thread."""
        posts: list[dict] = []

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("thread fetch failed", url=url, error=str(exc))
            return []

        if response.status_code != 200:
            return []

        try:
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            return []

        post_els = (
            soup.select("article.message")
            or soup.select("div.message--post")
            or soup.select("li.block-row--messages")
        )

        for post_el in post_els[:max_posts]:
            try:
                # Post body
                body_el = (
                    post_el.select_one("div.message-body")
                    or post_el.select_one("article.message-body")
                    or post_el.select_one("div.bbWrapper")
                )
                if body_el is None:
                    continue

                text = body_el.get_text(separator="\n", strip=True)
                if not text or len(text) < 20:
                    continue

                # Author
                author_el = (
                    post_el.select_one("a.message-name")
                    or post_el.select_one("span.message-name")
                )
                author = author_el.get_text(strip=True) if author_el else None

                # Date
                time_el = post_el.select_one("time")
                date_str = (
                    time_el.get("datetime", time_el.get_text(strip=True))
                    if time_el
                    else None
                )

                # Solution tag
                is_solution = bool(
                    post_el.select_one("span.label--solution")
                    or post_el.select_one("[data-solution]")
                    or ("çözüm" in post_el.get_text().lower()[:100])
                )

                # Likes
                like_count = 0
                like_el = post_el.select_one("span.reactionsBar-count")
                if like_el:
                    m = re.search(r"(\d+)", like_el.get_text())
                    if m:
                        like_count = int(m.group(1))

                posts.append({
                    "text": text,
                    "source": "technopat",
                    "author": author,
                    "date": date_str,
                    "is_solution": is_solution,
                    "helpful_count": like_count,
                    "url": url,
                })
            except Exception as exc:
                logger.debug("post parse error", error=str(exc))
                continue

        return posts

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("technopat", "xenforo", "message-body", "contentRow")
        return any(marker in text.lower() for marker in markers)


# ---------------------------------------------------------------------------
# DonanımHaber
# ---------------------------------------------------------------------------


@register_scraper("donanimhaber")
class DonanimHaberScraper(BaseScraper):
    """Scrape discussions from DonanımHaber forum."""

    _BASE_URL = "https://forum.donanimhaber.com"
    _SEARCH_URL = "https://forum.donanimhaber.com/arama"

    def __init__(self) -> None:
        super().__init__(domain="donanimhaber")

    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search DonanımHaber forum threads."""
        if not _BS4_AVAILABLE:
            return []

        params = {"kelime": query, "tip": "konu"}

        try:
            response = await self._fetch(self._SEARCH_URL, params=params)
        except Exception as exc:
            logger.error("search fetch failed", query=query, error=str(exc))
            return []

        if response.status_code != 200:
            logger.warning("search non-200", query=query, status=response.status_code)
            return []

        return self._parse_search(response.text, max_results)

    def _parse_search(self, html: str, max_results: int) -> list[Product]:
        """Parse DonanımHaber search results."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("search HTML parse failed", error=str(exc))
            return []

        threads = (
            soup.select("div.konu-list-item")
            or soup.select("li.konu-baslik")
            or soup.select("tr.topic-list-item")
        )

        for thread in threads[:max_results]:
            try:
                title_el = (
                    thread.select_one("a.konu-baslik")
                    or thread.select_one("a.topic-title")
                    or thread.select_one("h3 a")
                )
                if title_el is None:
                    continue

                title = title_el.get_text(strip=True)
                if not title:
                    continue

                href = title_el.get("href", "")
                thread_url = (
                    href
                    if href.startswith("http")
                    else f"{self._BASE_URL}{href}"
                )

                score = _score_thread(title)
                if score < 0.3:
                    continue

                specs: dict[str, Any] = {
                    "type": "forum_thread",
                    "forum": "donanimhaber",
                    "relevance_score": round(score, 2),
                }

                # Reply / view counts
                stat_els = thread.select("span.konu-istatistik") or thread.select("span.stat")
                for stat in stat_els:
                    stat_text = stat.get_text()
                    reply_m = re.search(r"(\d+)\s*(?:mesaj|yanıt|cevap)", stat_text, re.IGNORECASE)
                    view_m = re.search(r"(\d+)\s*(?:okunma|görüntü)", stat_text, re.IGNORECASE)
                    if reply_m:
                        specs["reply_count"] = int(reply_m.group(1))
                    if view_m:
                        specs["view_count"] = int(view_m.group(1))

                products.append(
                    Product(
                        name=title,
                        url=thread_url,
                        source="donanimhaber",
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("thread parse error", error=str(exc))
                continue

        logger.info("search parsed", count=len(products))
        return products

    async def get_product(self, url: str) -> Product | None:
        """Forums do not have product pages. Returns None."""
        return None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Extract posts from DonanımHaber thread(s)."""
        if not _BS4_AVAILABLE:
            return []

        # Cache
        try:
            cached = await get_cached_reviews(url, "donanimhaber")
            if cached is not None:
                logger.debug("reviews cache hit", url=url, count=len(cached))
                return cached
        except Exception:
            pass

        all_posts = await self._extract_thread_posts(url, max_posts=50)

        if all_posts:
            try:
                await cache_reviews(url, all_posts, "donanimhaber")
            except Exception:
                pass

        return all_posts

    async def _extract_thread_posts(
        self, url: str, max_posts: int = 50
    ) -> list[dict]:
        """Extract posts from a DonanımHaber thread."""
        posts: list[dict] = []

        try:
            response = await self._fetch(url)
        except Exception as exc:
            logger.error("thread fetch failed", url=url, error=str(exc))
            return []

        if response.status_code != 200:
            return []

        try:
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            return []

        post_els = (
            soup.select("div.mesaj-icerik")
            or soup.select("article.message-content")
            or soup.select("div.post-content")
        )

        for post_el in post_els[:max_posts]:
            try:
                text = post_el.get_text(separator="\n", strip=True)
                if not text or len(text) < 20:
                    continue

                # Author
                parent = post_el.find_parent("div", class_="mesaj-kutu") or post_el.parent
                author_el = None
                if parent:
                    author_el = (
                        parent.select_one("a.kullanici-adi")
                        or parent.select_one("span.username")
                    )
                author = author_el.get_text(strip=True) if author_el else None

                # Date
                time_el = None
                if parent:
                    time_el = parent.select_one("time") or parent.select_one("span.tarih")
                date_str = (
                    time_el.get("datetime", time_el.get_text(strip=True))
                    if time_el
                    else None
                )

                # Solution tag
                is_solution = bool(
                    post_el.find_parent(attrs={"data-cozum": True})
                    or ("çözüm" in text.lower()[:100])
                )

                posts.append({
                    "text": text,
                    "source": "donanimhaber",
                    "author": author,
                    "date": date_str,
                    "is_solution": is_solution,
                    "helpful_count": 0,
                    "url": url,
                })
            except Exception as exc:
                logger.debug("post parse error", error=str(exc))
                continue

        return posts

    def validate_response(self, response: httpx.Response) -> bool:
        if response.status_code >= 400:
            return False
        text = response.text
        if not text or len(text) < 300:
            return False
        markers = ("donanimhaber", "forum", "mesaj", "konu")
        return any(marker in text.lower() for marker in markers)
