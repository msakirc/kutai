"""Vecihi — auto-escalating web scraper.

Tiers:
- HTTP:    aiohttp (zero extra deps)
- TLS:     curl_cffi with browser TLS fingerprints
- STEALTH: Scrapling StealthyFetcher with Camoufox
- BROWSER: Scrapling DynamicFetcher with Playwright

Auto-escalation: if a lower tier gets blocked (403, Cloudflare challenge),
the next tier is tried automatically up to max_tier.
"""

from .core import ScrapeTier, ScrapeResult, detect_block, CHALLENGE_MARKERS
from .fetchers import (
    scrape_url,
    scrape_urls,
    fetch_http,
    fetch_tls,
    fetch_stealth,
    fetch_browser,
    install_browser_error_suppressor,
)

__all__ = [
    "ScrapeTier",
    "ScrapeResult",
    "detect_block",
    "CHALLENGE_MARKERS",
    "scrape_url",
    "scrape_urls",
    "fetch_http",
    "fetch_tls",
    "fetch_stealth",
    "fetch_browser",
    "install_browser_error_suppressor",
]
