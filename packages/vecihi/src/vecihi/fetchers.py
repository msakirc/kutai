"""Tiered fetchers with auto-escalation."""

import asyncio
import logging
import sys as _sys

import aiohttp

from .core import ScrapeTier, ScrapeResult, detect_block

logger = logging.getLogger("vecihi")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _suppress_browser_errors(loop, context):
    """Suppress orphaned patchright/playwright Future exceptions.

    When asyncio.wait_for cancels a browser fetch, patchright's internal
    navigation Future raises TargetClosedError after the browser context
    is cleaned up.  These are harmless — swallow them instead of letting
    asyncio log "Future exception was never retrieved".
    """
    exc = context.get("exception")
    if exc and "TargetClosedError" in type(exc).__name__:
        return  # swallow
    loop.default_exception_handler(context)


def install_browser_error_suppressor():
    """Install once from orchestrator startup or first scraper use."""
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_suppress_browser_errors)
    except RuntimeError:
        pass  # no running loop yet


async def fetch_http(url: str, timeout: float = 10.0) -> ScrapeResult:
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
                if detect_block(resp.status, html, headers):
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


async def fetch_tls(url: str, timeout: float = 12.0) -> ScrapeResult:
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
            if detect_block(resp.status_code, html, headers):
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


async def fetch_stealth(url: str, timeout: float = 25.0) -> ScrapeResult:
    """Tier 2: Scrapling StealthyFetcher (Camoufox)."""
    install_browser_error_suppressor()
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
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return ScrapeResult(html="", status=0, tier=ScrapeTier.STEALTH,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.STEALTH,
                            url=url, error=str(e)[:200])


async def fetch_browser(url: str, timeout: float = 30.0) -> ScrapeResult:
    """Tier 3: Scrapling DynamicFetcher (Playwright Chromium)."""
    install_browser_error_suppressor()
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
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return ScrapeResult(html="", status=0, tier=ScrapeTier.BROWSER,
                            url=url, error="timeout")
    except Exception as e:
        return ScrapeResult(html="", status=0, tier=ScrapeTier.BROWSER,
                            url=url, error=str(e)[:200])


_TIER_FETCHERS = {
    ScrapeTier.HTTP: "fetch_http",
    ScrapeTier.TLS: "fetch_tls",
    ScrapeTier.STEALTH: "fetch_stealth",
    ScrapeTier.BROWSER: "fetch_browser",
}


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
        fetcher = getattr(_this_mod, _TIER_FETCHERS[tier])
        result = await fetcher(url, timeout=timeout) if timeout else await fetcher(url)

        logger.debug(
            "scraper tier attempt: tier=%s url=%s status=%s ok=%s error=%s",
            tier.name, url[:80], result.status, result.ok, result.error,
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
