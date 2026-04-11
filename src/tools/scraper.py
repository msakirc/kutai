"""Shim — re-exports from the vecihi package.

All scraper functionality now lives in packages/vecihi/.
This module exists so existing ``from src.tools.scraper import ...``
imports throughout KutAI continue to work unchanged.
"""
from vecihi import (  # noqa: F401
    ScrapeTier,
    ScrapeResult,
    scrape_url,
    scrape_urls,
    detect_block,
    fetch_http,
    fetch_tls,
    fetch_stealth,
    fetch_browser,
    install_browser_error_suppressor,
)

# Backwards-compat aliases for old underscore-prefixed names
_detect_block = detect_block
_fetch_http = fetch_http
_fetch_tls = fetch_tls
_fetch_stealth = fetch_stealth
_fetch_browser = fetch_browser
