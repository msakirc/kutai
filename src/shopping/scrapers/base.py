"""Abstract base class for all shopping scrapers.

Provides rate limiting, retry logic, User-Agent rotation, request logging,
cache integration, and structured data extraction (JSON-LD, OpenGraph,
Schema.org).  Concrete scrapers inherit from ``BaseScraper`` and implement
the ``search``, ``get_product``, ``get_reviews``, and ``validate_response``
methods.
"""

from __future__ import annotations

import abc
import asyncio
import json
import random
import re
import time
from dataclasses import asdict
from typing import Any

import httpx

from ..config import get_rate_limit
from ..cache import (
    get_cached_product,
    cache_product,
    get_cached_search,
    cache_search,
)
from ..request_tracker import log_request, get_daily_request_count
from ..models import Product
from src.infra.logging_config import get_logger

logger = get_logger("shopping.scrapers.base")

# ---------------------------------------------------------------------------
# Tiered-scraper response shim
# ---------------------------------------------------------------------------

class _TieredResponse:
    """Minimal httpx.Response-compatible wrapper around a ScrapeResult.

    Allows the rest of BaseScraper (and all subclass ``validate_response``
    implementations) to use the same ``.status_code``, ``.text``,
    ``.headers``, and ``.json()`` interface regardless of which HTTP
    backend was used.
    """

    def __init__(self, html: str, status: int, headers: dict) -> None:
        self.status_code: int = status
        self.text: str = html
        self.headers: dict = headers
        self.content: bytes = html.encode("utf-8", errors="replace")

    def json(self, **kwargs: Any) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )

# ---------------------------------------------------------------------------
# User-Agent pool
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ---------------------------------------------------------------------------
# Scraper registry  (domain -> scraper class)
# ---------------------------------------------------------------------------

_registry: dict[str, type[BaseScraper]] = {}


def register_scraper(domain: str):
    """Class decorator that registers a scraper for *domain*."""

    def decorator(cls: type[BaseScraper]):
        _registry[domain] = cls
        return cls

    return decorator


def get_scraper(domain: str) -> type[BaseScraper] | None:
    """Return the scraper class for *domain*, or ``None``."""
    return _registry.get(domain)


def list_scrapers() -> dict[str, type[BaseScraper]]:
    """Return a copy of the full domain -> scraper mapping."""
    return dict(_registry)


# ---------------------------------------------------------------------------
# BaseScraper
# ---------------------------------------------------------------------------

class BaseScraper(abc.ABC):
    """Abstract base for site-specific scrapers.

    Concrete subclasses must implement :pymethod:`search`,
    :pymethod:`get_product`, :pymethod:`get_reviews`, and
    :pymethod:`validate_response`.
    """

    # Retry schedule (seconds between attempts)
    _BACKOFF_SCHEDULE: tuple[int, ...] = (5, 15, 45)

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self._rate_cfg = get_rate_limit(domain)
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def search(self, query: str, *, max_results: int = 20) -> list[Product]:
        """Search the site for *query* and return up to *max_results* products."""

    @abc.abstractmethod
    async def get_product(self, url: str) -> Product | None:
        """Fetch and parse a single product page."""

    @abc.abstractmethod
    async def get_reviews(self, url: str, *, max_pages: int = 3) -> list[dict]:
        """Fetch product reviews (up to *max_pages* pages)."""

    @abc.abstractmethod
    def validate_response(self, response: "_TieredResponse") -> bool:
        """Return ``True`` if the response contains real content (not a block page)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _fetch(self, url: str, **kwargs: Any) -> "_TieredResponse":
        """Fetch *url* with rate limiting, retries, UA rotation, and logging.

        Uses the tiered scraper (curl_cffi TLS fingerprinting) to bypass
        bot-detection on Turkish e-commerce sites.  Falls back to raw httpx
        if the tiered scraper module is unavailable.

        Keyword arguments are kept for API compatibility but most are ignored
        (the tiered scraper handles headers internally).  ``params`` is
        appended to the URL if present.

        Raises ``RuntimeError`` after exhausting retries.
        """
        # --- handle query params (forward-compat with callers that pass params=) ---
        params = kwargs.pop("params", None)
        if params:
            import urllib.parse as _up
            qs = _up.urlencode(params)
            url = f"{url}?{qs}" if "?" not in url else f"{url}&{qs}"

        # Pop headers kwarg (kept for compat; tiered scraper uses its own UA)
        kwargs.pop("headers", None)

        # --- daily budget guard ---
        daily = await get_daily_request_count(self.domain)
        budget = self._rate_cfg.get("daily_budget", 50)
        if daily >= budget:
            logger.warning(
                "daily request budget exhausted",
                domain=self.domain,
                used=daily,
                budget=budget,
            )
            raise RuntimeError(
                f"Daily request budget for {self.domain} exhausted "
                f"({daily}/{budget})"
            )

        # --- rate limiting (inter-request delay) ---
        delay = self._rate_cfg.get("delay_seconds", 10)
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < delay:
            wait = delay - elapsed + random.uniform(0.1, 1.0)
            logger.debug("rate-limit wait", domain=self.domain, wait_s=round(wait, 2))
            await asyncio.sleep(wait)

        # --- tiered scraper import (lazy, falls back to httpx if missing) ---
        try:
            from src.tools.scraper import scrape_url, ScrapeTier
            _USE_TIERED = True
        except ImportError:
            _USE_TIERED = False

        last_exc: BaseException | None = None
        start = time.monotonic()

        for attempt, backoff in enumerate(self._BACKOFF_SCHEDULE):
            try:
                self._last_request_time = time.monotonic()

                if _USE_TIERED:
                    result = await scrape_url(url, max_tier=ScrapeTier.TLS, timeout=30.0)
                    response = _TieredResponse(
                        html=result.html,
                        status=result.status,
                        headers=result.headers,
                    )
                else:
                    # Fallback: raw httpx (for environments without curl_cffi)
                    headers: dict = {}
                    headers.setdefault("User-Agent", self._random_ua())
                    headers.setdefault(
                        "Accept-Language", "tr-TR,tr;q=0.9,en-US;q=0.5,en;q=0.3"
                    )
                    async with httpx.AsyncClient(
                        follow_redirects=True,
                        timeout=httpx.Timeout(30.0, connect=10.0),
                    ) as client:
                        raw = await client.get(url, headers=headers)
                    response = _TieredResponse(
                        html=raw.text,
                        status=raw.status_code,
                        headers=dict(raw.headers),
                    )

                elapsed_ms = int((time.monotonic() - start) * 1000)

                # Log the request
                await log_request(
                    domain=self.domain,
                    url=url,
                    status_code=response.status_code,
                    response_time_ms=elapsed_ms,
                    cache_hit=False,
                    scraper_used=self.__class__.__name__,
                )

                # Validate the response (block-page detection)
                if not self.validate_response(response):
                    logger.warning(
                        "response failed validation (possible block)",
                        domain=self.domain,
                        url=url,
                        status=response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt < len(self._BACKOFF_SCHEDULE) - 1:
                        await asyncio.sleep(backoff + random.uniform(0, 2))
                        continue
                    # Last attempt -- return the response anyway and let
                    # callers decide.
                    return response

                # Retry server errors (5xx)
                if response.status_code >= 500:
                    logger.warning(
                        "server error, retrying",
                        domain=self.domain,
                        url=url,
                        status=response.status_code,
                        attempt=attempt + 1,
                        backoff=backoff,
                    )
                    if attempt < len(self._BACKOFF_SCHEDULE) - 1:
                        await asyncio.sleep(backoff + random.uniform(0, 2))
                        continue

                return response

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "request error, retrying",
                    domain=self.domain,
                    url=url,
                    error=str(exc),
                    attempt=attempt + 1,
                    backoff=backoff,
                )
                if attempt < len(self._BACKOFF_SCHEDULE) - 1:
                    await asyncio.sleep(backoff + random.uniform(0, 2))

        # All retries exhausted -- log failure and re-raise
        elapsed_ms = int((time.monotonic() - start) * 1000)
        await log_request(
            domain=self.domain,
            url=url,
            status_code=None,
            response_time_ms=elapsed_ms,
            cache_hit=False,
            scraper_used=self.__class__.__name__,
        )

        if last_exc is not None:
            raise last_exc
        # Should not happen, but satisfy the type checker.
        raise RuntimeError(f"All retries exhausted for {url}")

    # ------------------------------------------------------------------

    async def preflight_check(self) -> bool:
        """Quick health probe -- try to reach the domain root.

        Returns ``True`` if the domain responds with a 2xx/3xx status.
        """
        root_url = f"https://www.{self.domain}.com/"
        try:
            try:
                from src.tools.scraper import scrape_url, ScrapeTier
                result = await scrape_url(root_url, max_tier=ScrapeTier.TLS, timeout=15.0)
                ok = 200 <= result.status < 400
            except ImportError:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(15.0, connect=5.0),
                ) as client:
                    resp = await client.head(
                        root_url,
                        headers={"User-Agent": self._random_ua()},
                    )
                    ok = 200 <= resp.status_code < 400
            logger.info(
                "preflight check",
                domain=self.domain,
                ok=ok,
            )
            return ok
        except Exception as exc:
            logger.warning(
                "preflight check failed",
                domain=self.domain,
                error=str(exc),
            )
            return False

    # ------------------------------------------------------------------

    def extract_structured_data(self, html: str) -> dict:
        """Extract JSON-LD, OpenGraph, and Schema.org data from *html*.

        Returns a dict with optional keys ``json_ld``, ``opengraph``, and
        ``schema_org``.  Each value is either a dict or a list of dicts.
        Parsing errors are silently swallowed so callers always get a
        usable (possibly empty) dict.
        """
        result: dict[str, Any] = {}

        # --- JSON-LD ---
        try:
            ld_blocks: list[dict] = []
            for m in re.finditer(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                html,
                re.DOTALL | re.IGNORECASE,
            ):
                raw = m.group(1).strip()
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        ld_blocks.extend(parsed)
                    else:
                        ld_blocks.append(parsed)
            if ld_blocks:
                result["json_ld"] = ld_blocks if len(ld_blocks) > 1 else ld_blocks[0]
        except Exception:
            pass

        # --- OpenGraph ---
        try:
            og: dict[str, str] = {}
            for m in re.finditer(
                r'<meta\s+property=["\']og:(\w+)["\']\s+content=["\']([^"\']*)["\']',
                html,
                re.IGNORECASE,
            ):
                og[m.group(1)] = m.group(2)
            if og:
                result["opengraph"] = og
        except Exception:
            pass

        # --- Schema.org microdata (itemtype/itemprop) ---
        try:
            schema_items: list[dict[str, str]] = []
            for m in re.finditer(
                r'<[^>]+itemprop=["\'](\w+)["\'][^>]*>([^<]*)<',
                html,
                re.IGNORECASE,
            ):
                prop = m.group(1)
                value = m.group(2).strip()
                if value:
                    schema_items.append({"property": prop, "value": value})
            if schema_items:
                result["schema_org"] = schema_items
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _random_ua() -> str:
        """Return a random User-Agent string."""
        return random.choice(_USER_AGENTS)

    # ------------------------------------------------------------------
    # Convenience: serialise a Product to a cacheable dict
    # ------------------------------------------------------------------

    @staticmethod
    def _product_to_dict(product: Product) -> dict:
        return asdict(product)

    @staticmethod
    def _dict_to_product(data: dict) -> Product:
        return Product(**data)
