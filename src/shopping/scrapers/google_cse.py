"""Google Custom Search Engine (CSE) scraper.

Uses the Google Custom Search JSON API to perform site-scoped searches
across Turkish shopping and review sites.

Features:
  - Quota tracking: 100 queries per day (free tier).
  - Result caching: 48-hour TTL to minimise API usage.
  - Smart query batching: groups related queries where possible.

Requires ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_CX`` environment
variables (or credential store entries).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import BaseScraper, register_scraper
from ..cache import cache_search, get_cached_search
from ..models import Product
from ..text_utils import normalize_product_name
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.google_cse")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSE_API_URL = "https://www.googleapis.com/customsearch/v1"
_DAILY_QUOTA = 100
_CACHE_TTL_HOURS = 48

# Site restrictions for Turkish shopping queries
_DEFAULT_SITE_RESTRICT = [
    "trendyol.com",
    "hepsiburada.com",
    "amazon.com.tr",
    "akakce.com",
    "n11.com",
    "gittigidiyor.com",
    "teknosa.com",
    "mediamarkt.com.tr",
    "vatanbilgisayar.com",
]


# ---------------------------------------------------------------------------
# Quota tracker (in-memory, per-process)
# ---------------------------------------------------------------------------


class _QuotaTracker:
    """Simple in-memory daily quota tracker for CSE API."""

    def __init__(self, daily_limit: int = _DAILY_QUOTA) -> None:
        self.daily_limit = daily_limit
        self._count = 0
        self._reset_date: str = ""

    def _maybe_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._count = 0
            self._reset_date = today

    def can_query(self) -> bool:
        self._maybe_reset()
        return self._count < self.daily_limit

    def record_query(self) -> None:
        self._maybe_reset()
        self._count += 1

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self.daily_limit - self._count)


_quota = _QuotaTracker()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


@register_scraper("google_cse")
class GoogleCSEScraper(BaseScraper):
    """Search via Google Custom Search JSON API.

    Performs site-scoped searches across Turkish shopping sites.
    Tracks daily quota (100/day free tier) and caches results for 48h.
    """

    def __init__(self) -> None:
        super().__init__(domain="google_cse")
        self._api_key = os.environ.get("GOOGLE_CSE_API_KEY", "")
        self._cx = os.environ.get("GOOGLE_CSE_CX", "")

    @property
    def is_configured(self) -> bool:
        """Return True if API key and CX are available."""
        return bool(self._api_key and self._cx)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, max_results: int = 10) -> list[Product]:
        """Execute a Google CSE search.

        Results are capped at 10 per query (API limit per page).
        """
        if not self.is_configured:
            logger.warning("Google CSE not configured (missing API key or CX)")
            return []

        if not _quota.can_query():
            logger.warning(
                "Google CSE daily quota exhausted",
                remaining=_quota.remaining,
            )
            return []

        # Cache check (48h TTL)
        try:
            cached = await get_cached_search(query, "google_cse")
            if cached is not None:
                logger.debug("CSE cache hit", query=query, count=len(cached))
                return [self._dict_to_product(p) for p in cached]
        except Exception:
            pass

        # Build query with site restriction
        products = await self._execute_search(query, max_results)

        # Cache results
        if products:
            try:
                await cache_search(
                    query,
                    "google_cse",
                    [self._product_to_dict(p) for p in products],
                )
            except Exception:
                pass

        return products

    async def _execute_search(
        self, query: str, max_results: int
    ) -> list[Product]:
        """Execute a single CSE API call."""
        products: list[Product] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        params: dict[str, str] = {
            "key": self._api_key,
            "cx": self._cx,
            "q": query,
            "num": str(min(max_results, 10)),  # API max is 10
            "lr": "lang_tr",
            "gl": "tr",
        }

        try:
            # Use httpx directly (not _fetch) to avoid domain-specific
            # rate limiting -- Google CSE has its own quota system
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
            ) as client:
                response = await client.get(_CSE_API_URL, params=params)

            _quota.record_query()

            if response.status_code != 200:
                logger.warning(
                    "CSE API non-200",
                    status=response.status_code,
                    query=query,
                )
                return []

            data = response.json()

        except Exception as exc:
            logger.error("CSE API request failed", query=query, error=str(exc))
            return []

        # Parse results
        items = data.get("items", [])

        for item in items:
            try:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")

                if not title or not link:
                    continue

                name = normalize_product_name(title)

                # Determine source from URL
                source = "google_cse"
                for site in _DEFAULT_SITE_RESTRICT:
                    if site in link:
                        source = site.split(".")[0]
                        break

                # Try to extract price from snippet
                price: float | None = None
                price_match = re.search(
                    r"([\d.,]+)\s*(?:TL|₺)", snippet
                ) if snippet else None
                if price_match:
                    try:
                        price_text = price_match.group(1).replace(".", "").replace(",", ".")
                        price = float(price_text)
                    except (ValueError, TypeError):
                        pass

                # Metadata from CSE
                pagemap = item.get("pagemap", {})
                specs: dict[str, Any] = {
                    "type": "cse_result",
                    "snippet": snippet,
                    "display_link": item.get("displayLink", ""),
                }

                # Extract structured data from pagemap
                if "product" in pagemap:
                    product_data = pagemap["product"]
                    if isinstance(product_data, list) and product_data:
                        pd = product_data[0]
                        if price is None:
                            price = _safe_float(pd.get("price"))
                        if pd.get("ratingvalue"):
                            specs["rating"] = pd["ratingvalue"]

                # Image from pagemap
                image_url: str | None = None
                if "cse_image" in pagemap:
                    images = pagemap["cse_image"]
                    if isinstance(images, list) and images:
                        image_url = images[0].get("src")

                products.append(
                    Product(
                        name=name,
                        url=link,
                        source=source,
                        discounted_price=price,
                        currency="TRY",
                        image_url=image_url,
                        specs=specs,
                        fetched_at=now_iso,
                    )
                )
            except Exception as exc:
                logger.debug("CSE item parse error", error=str(exc))
                continue

        logger.info(
            "CSE search complete",
            query=query,
            count=len(products),
            quota_remaining=_quota.remaining,
        )
        return products

    # ------------------------------------------------------------------
    # Site-scoped search helper
    # ------------------------------------------------------------------

    async def search_site(
        self,
        query: str,
        site: str,
        *,
        max_results: int = 10,
    ) -> list[Product]:
        """Execute a site-scoped search (e.g. ``site:trendyol.com query``).

        This is a convenience wrapper around ``search`` that prepends the
        site operator.
        """
        scoped_query = f"site:{site} {query}"
        return await self.search(scoped_query, max_results=max_results)

    # ------------------------------------------------------------------
    # Smart query batching
    # ------------------------------------------------------------------

    async def batch_search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
    ) -> dict[str, list[Product]]:
        """Execute multiple queries, deduplicating and batching where possible.

        Returns a dict mapping each original query to its results.
        Respects the daily quota -- stops early if quota is exhausted.
        """
        results: dict[str, list[Product]] = {}

        # Deduplicate queries
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            normalized = q.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(q)

        for q in unique_queries:
            if not _quota.can_query():
                logger.warning("quota exhausted during batch search, stopping")
                results[q] = []
                continue

            results[q] = await self.search(q, max_results=max_results_per_query)

        return results

    # ------------------------------------------------------------------
    # get_product / get_reviews (not applicable)
    # ------------------------------------------------------------------

    async def get_product(self, url: str) -> Product | None:
        """CSE is a search-only source. Returns None."""
        return None

    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """CSE is a search-only source. Returns empty."""
        return []

    # ------------------------------------------------------------------
    # validate_response
    # ------------------------------------------------------------------

    def validate_response(self, response: httpx.Response) -> bool:
        """Validate Google CSE API response."""
        if response.status_code >= 400:
            return False
        try:
            data = response.json()
            return isinstance(data, dict) and "items" in data
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Quota info
    # ------------------------------------------------------------------

    def get_quota_info(self) -> dict[str, Any]:
        """Return current quota status."""
        return {
            "daily_limit": _quota.daily_limit,
            "remaining": _quota.remaining,
            "configured": self.is_configured,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
