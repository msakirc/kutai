# src/tools/scraper.py
"""Tiered web scraper with auto-escalation.

Tiers:
- HTTP:    aiohttp (existing, zero extra deps)
- TLS:     curl_cffi with browser TLS fingerprints (~10-30MB)
- STEALTH: Scrapling StealthyFetcher with Camoufox (~300-500MB on-demand)
- BROWSER: Scrapling DynamicFetcher with Playwright (~500-800MB on-demand)

Auto-escalation: if a lower tier gets blocked (403, Cloudflare challenge),
the next tier is tried automatically up to max_tier.
"""

import asyncio
import enum
from dataclasses import dataclass, field

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("tools.scraper")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Phrases in HTML that indicate a Cloudflare/WAF challenge page
_CHALLENGE_MARKERS = [
    "just a moment",
    "checking your browser",
    "cdn-cgi/challenge-platform",
    "cf-browser-verification",
    "attention required",
    "ray id",
]


class ScrapeTier(enum.IntEnum):
    HTTP = 0
    TLS = 1
    STEALTH = 2
    BROWSER = 3


@dataclass
class ScrapeResult:
    html: str
    status: int
    tier: ScrapeTier
    url: str
    error: str | None = None
    headers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error and bool(self.html)


def _detect_block(status: int, html: str, headers: dict) -> bool:
    """Detect if a response is blocked by WAF/anti-bot."""
    if status in (403, 429, 402, 451):
        return True
    if status == 503 and "cloudflare" in str(headers.get("server", "")).lower():
        return True
    if status == 200 and html:
        html_lower = html[:2000].lower()
        if any(marker in html_lower for marker in _CHALLENGE_MARKERS):
            return True
    return False


async def _fetch_http(url: str, timeout: float = 10.0) -> ScrapeResult:
    """Tier 0: Plain aiohttp fetch."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": _USER_AGENT},
                allow_redirects=True,
                max_redirects=3,
            ) as resp:
                html = await resp.text(encoding=None)
                headers = dict(resp.headers)
                if _detect_block(resp.status, html, headers):
                    return ScrapeResult(
                        html=html, status=resp.status, tier=ScrapeTier.HTTP,
                        url=url, error="blocked", headers=headers,
                    )
                return ScrapeResult(
                    html=html, status=resp.status, tier=ScrapeTier.HTTP,
                    url=url, headers=headers,
                )
    except asyncio.TimeoutError:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.HTTP,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.HTTP,
                            url=url, error=str(e)[:200])


async def _fetch_tls(url: str, timeout: float = 12.0) -> ScrapeResult:
    """Tier 1: curl_cffi with browser TLS fingerprints."""
    try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(impersonate="chrome131") as session:
            resp = await asyncio.wait_for(
                session.get(url, allow_redirects=True, max_redirects=3),
                timeout=timeout,
            )
            html = resp.text
            headers = dict(resp.headers)
            if _detect_block(resp.status_code, html, headers):
                return ScrapeResult(
                    html=html, status=resp.status_code, tier=ScrapeTier.TLS,
                    url=url, error="blocked", headers=headers,
                )
            return ScrapeResult(
                html=html, status=resp.status_code, tier=ScrapeTier.TLS,
                url=url, headers=headers,
            )
    except ImportError:
        logger.warning("curl_cffi not installed, TLS tier unavailable")
        return ScrapeResult(html="", status=0, tier=ScrapeTier.TLS,
                            url=url, error="curl_cffi not installed")
    except asyncio.TimeoutError:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.TLS,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.TLS,
                            url=url, error=str(e)[:200])


async def _fetch_stealth(url: str, timeout: float = 25.0) -> ScrapeResult:
    """Tier 2: Scrapling StealthyFetcher (Camoufox)."""
    try:
        from scrapling import StealthyFetcher

        fetcher = StealthyFetcher()
        resp = await asyncio.wait_for(
            fetcher.async_fetch(url),
            timeout=timeout,
        )
        html = resp.html_content if hasattr(resp, "html_content") else str(resp)
        status = resp.status if hasattr(resp, "status") else 200
        return ScrapeResult(
            html=html, status=status, tier=ScrapeTier.STEALTH, url=url,
        )
    except ImportError:
        logger.warning("scrapling not installed, stealth tier unavailable")
        return ScrapeResult(html="", status=0, tier=ScrapeTier.STEALTH,
                            url=url, error="scrapling not installed")
    except asyncio.TimeoutError:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.STEALTH,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.STEALTH,
                            url=url, error=str(e)[:200])


async def _fetch_browser(url: str, timeout: float = 30.0) -> ScrapeResult:
    """Tier 3: Scrapling DynamicFetcher (Playwright Chromium)."""
    try:
        from scrapling import DynamicFetcher

        fetcher = DynamicFetcher()
        resp = await asyncio.wait_for(
            fetcher.async_fetch(url),
            timeout=timeout,
        )
        html = resp.html_content if hasattr(resp, "html_content") else str(resp)
        status = resp.status if hasattr(resp, "status") else 200
        return ScrapeResult(
            html=html, status=status, tier=ScrapeTier.BROWSER, url=url,
        )
    except ImportError:
        logger.warning("scrapling not installed, browser tier unavailable")
        return ScrapeResult(html="", status=0, tier=ScrapeTier.BROWSER,
                            url=url, error="scrapling not installed")
    except asyncio.TimeoutError:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.BROWSER,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.BROWSER,
                            url=url, error=str(e)[:200])


_TIER_FETCHER_NAMES = {
    ScrapeTier.HTTP: "_fetch_http",
    ScrapeTier.TLS: "_fetch_tls",
    ScrapeTier.STEALTH: "_fetch_stealth",
    ScrapeTier.BROWSER: "_fetch_browser",
}

# Module reference for dynamic lookup (allows patches to take effect)
import sys as _sys


async def scrape_url(
    url: str,
    max_tier: ScrapeTier = ScrapeTier.TLS,
    timeout: float | None = None,
) -> ScrapeResult:
    """Fetch a URL, auto-escalating through tiers on blocks.

    Starts at HTTP tier, escalates up to max_tier if blocked.
    Returns the first successful result, or the last failure.
    """
    last_result = None

    for tier in ScrapeTier:
        if tier > max_tier:
            break

        _this_mod = _sys.modules[__name__]
        fetcher = getattr(_this_mod, _TIER_FETCHER_NAMES[tier])
        result = await fetcher(url, timeout=timeout) if timeout else await fetcher(url)

        logger.debug(
            "scraper tier attempt",
            tier=tier.name, url=url[:80], status=result.status,
            ok=result.ok, error=result.error,
        )

        if result.ok:
            return result

        last_result = result

        # Only escalate on blocks, not on other errors
        if result.error and result.error != "blocked":
            return result

    return last_result or ScrapeResult(
        html="", status=0, tier=ScrapeTier.HTTP, url=url, error="all tiers failed",
    )


async def scrape_urls(
    urls: list[str],
    max_tier: ScrapeTier = ScrapeTier.TLS,
    max_concurrent: int = 5,
) -> dict[str, ScrapeResult]:
    """Scrape multiple URLs concurrently with auto-escalation.

    Returns {url: ScrapeResult} for all URLs attempted.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _scrape_one(url: str) -> tuple[str, ScrapeResult]:
        async with semaphore:
            result = await scrape_url(url, max_tier=max_tier)
            return url, result

    tasks = [_scrape_one(u) for u in urls if u.startswith(("http://", "https://"))]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = {}
    for r in results:
        if isinstance(r, tuple):
            out[r[0]] = r[1]
    return out
