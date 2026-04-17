# Vecihi — Tiered Scraper Extraction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the tiered web scraper from `src/tools/scraper.py` into a standalone package called `vecihi` under `packages/vecihi/`, wire it as an editable install, and make KutAI's original module a thin re-export shim.

**Architecture:** The package lives at `packages/vecihi/` following the same `src/` layout as yazbunu. The sole KutAI dependency (`get_logger`) is replaced with stdlib `logging`. KutAI's `src/tools/scraper.py` becomes a shim that re-exports everything from `vecihi`, so no consumer code changes. Existing tests are updated to import from `vecihi` directly, while also verifying the shim path works.

**Tech Stack:** Python 3.10+, aiohttp, optional curl_cffi + scrapling, setuptools, pytest

---

## File Structure

```
packages/vecihi/
├── pyproject.toml                       # Package metadata, deps, optional extras
└── src/vecihi/
    ├── __init__.py                      # Public API re-exports
    ├── core.py                          # ScrapeTier, ScrapeResult, _detect_block
    └── fetchers.py                      # _fetch_http, _fetch_tls, _fetch_stealth, _fetch_browser, scrape_url, scrape_urls

Modified KutAI files:
  src/tools/scraper.py                   # Becomes thin re-export shim
  requirements.txt                       # Add -e ./packages/vecihi
  tests/test_scraper.py                  # Update patch targets to vecihi module paths
```

**Why split into core.py + fetchers.py instead of one file:**
- `core.py` has zero external deps (enum, dataclass, stdlib only) — importable without aiohttp
- `fetchers.py` needs aiohttp and conditionally imports curl_cffi/scrapling
- Users who only need `ScrapeTier`/`ScrapeResult` types (e.g. for type hints) don't pull in aiohttp

---

### Task 1: Create vecihi package skeleton

**Files:**
- Create: `packages/vecihi/pyproject.toml`
- Create: `packages/vecihi/src/vecihi/__init__.py`
- Create: `packages/vecihi/src/vecihi/core.py`
- Create: `packages/vecihi/src/vecihi/fetchers.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p packages/vecihi/src/vecihi
```

- [ ] **Step 2: Write pyproject.toml**

Write `packages/vecihi/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "vecihi"
version = "0.1.0"
description = "Auto-escalating web scraper: HTTP -> TLS fingerprint -> Stealth -> Browser"
requires-python = ">=3.10"
dependencies = ["aiohttp>=3.9.0"]

[project.optional-dependencies]
tls = ["curl_cffi>=0.7.0"]
stealth = ["scrapling>=0.2.0"]
all = ["curl_cffi>=0.7.0", "scrapling>=0.2.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write core.py — types and block detection**

Write `packages/vecihi/src/vecihi/core.py`:

```python
"""Core types and block detection for vecihi."""

import enum
from dataclasses import dataclass, field


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


# Phrases in HTML that indicate a Cloudflare/WAF challenge page
CHALLENGE_MARKERS = [
    "just a moment",
    "checking your browser",
    "cdn-cgi/challenge-platform",
    "cf-browser-verification",
    "attention required",
    "ray id",
]


def detect_block(status: int, html: str, headers: dict) -> bool:
    """Detect if a response is blocked by WAF/anti-bot."""
    if status in (403, 429, 402, 451):
        return True
    if status == 503 and "cloudflare" in str(headers.get("server", "")).lower():
        return True
    if status == 200 and html:
        html_lower = html[:2000].lower()
        if any(marker in html_lower for marker in CHALLENGE_MARKERS):
            return True
    return False
```

- [ ] **Step 4: Write fetchers.py — tier fetchers and orchestration**

Write `packages/vecihi/src/vecihi/fetchers.py`:

```python
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
```

**Note on naming:** The fetcher functions drop the leading underscore (`_fetch_http` → `fetch_http`) since they are now part of the package's public API. The `_detect_block` function becomes `detect_block` for the same reason.

- [ ] **Step 5: Write `__init__.py` — public API**

Write `packages/vecihi/src/vecihi/__init__.py`:

```python
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
```

- [ ] **Step 6: Verify the package imports cleanly**

```bash
pip install -e ./packages/vecihi
python -c "from vecihi import ScrapeTier, ScrapeResult, scrape_url, detect_block; print('vecihi OK')"
```

Expected: `vecihi OK`

- [ ] **Step 7: Commit**

```bash
git add packages/vecihi/
git commit -m "feat(vecihi): create standalone tiered scraper package"
```

---

### Task 2: Write vecihi's own test suite

**Files:**
- Create: `packages/vecihi/tests/test_core.py`
- Create: `packages/vecihi/tests/test_fetchers.py`

These are vecihi's own tests — they import from `vecihi`, not from `src.tools.scraper`. They prove the package works standalone.

- [ ] **Step 1: Write core tests**

Write `packages/vecihi/tests/test_core.py`:

```python
"""Tests for vecihi.core — types and block detection."""

from vecihi import ScrapeTier, ScrapeResult, detect_block


class TestScrapeTier:
    def test_tier_ordering(self):
        assert ScrapeTier.HTTP < ScrapeTier.TLS < ScrapeTier.STEALTH < ScrapeTier.BROWSER

    def test_tier_values(self):
        assert ScrapeTier.HTTP == 0
        assert ScrapeTier.TLS == 1
        assert ScrapeTier.STEALTH == 2
        assert ScrapeTier.BROWSER == 3


class TestScrapeResult:
    def test_ok_on_200_with_html(self):
        r = ScrapeResult(html="<html>test</html>", status=200,
                         tier=ScrapeTier.HTTP, url="https://example.com")
        assert r.ok is True

    def test_not_ok_on_non_200(self):
        r = ScrapeResult(html="Forbidden", status=403, tier=ScrapeTier.HTTP,
                         url="https://example.com")
        assert r.ok is False

    def test_not_ok_with_error(self):
        r = ScrapeResult(html="<html>test</html>", status=200,
                         tier=ScrapeTier.HTTP, url="https://example.com",
                         error="blocked")
        assert r.ok is False

    def test_not_ok_empty_html(self):
        r = ScrapeResult(html="", status=200, tier=ScrapeTier.HTTP,
                         url="https://example.com")
        assert r.ok is False


class TestDetectBlock:
    def test_403_is_blocked(self):
        assert detect_block(403, "", {}) is True

    def test_429_is_blocked(self):
        assert detect_block(429, "", {}) is True

    def test_402_is_blocked(self):
        assert detect_block(402, "", {}) is True

    def test_451_is_blocked(self):
        assert detect_block(451, "", {}) is True

    def test_503_with_cloudflare_header(self):
        assert detect_block(503, "", {"server": "cloudflare"}) is True

    def test_503_without_cloudflare_not_blocked(self):
        assert detect_block(503, "", {"server": "nginx"}) is False

    def test_200_with_cf_challenge(self):
        html = '<html><title>Just a moment...</title><body>Checking your browser</body></html>'
        assert detect_block(200, html, {}) is True

    def test_200_with_challenge_platform(self):
        html = '<html><body><form action="/cdn-cgi/challenge-platform">solve</form></body></html>'
        assert detect_block(200, html, {}) is True

    def test_200_with_attention_required(self):
        html = '<html><title>Attention Required</title></html>'
        assert detect_block(200, html, {}) is True

    def test_200_normal_page(self):
        html = '<html><title>Real Page</title><body>Content here</body></html>'
        assert detect_block(200, html, {}) is False

    def test_200_empty_html_not_blocked(self):
        assert detect_block(200, "", {}) is False
```

- [ ] **Step 2: Run core tests**

```bash
pytest packages/vecihi/tests/test_core.py -v
```

Expected: All 15 tests pass.

- [ ] **Step 3: Write fetcher tests**

Write `packages/vecihi/tests/test_fetchers.py`:

```python
"""Tests for vecihi.fetchers — tier fetchers and auto-escalation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from vecihi import ScrapeTier, ScrapeResult, scrape_url, scrape_urls


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestFetchHttp:
    def test_successful_fetch(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="<html><body>Hello</body></html>")
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            assert result.ok is True
            assert result.tier == ScrapeTier.HTTP
            assert "Hello" in result.html

    def test_403_returns_blocked(self):
        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")
        mock_resp.headers = {"server": "cloudflare"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            assert result.ok is False
            assert result.error == "blocked"


class TestAutoEscalation:
    def test_escalates_to_tls_on_block(self):
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")
        tls_ok = ScrapeResult(html="<html><body>TLS Success</body></html>",
                              status=200, tier=ScrapeTier.TLS,
                              url="https://example.com")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock, return_value=tls_ok):
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.TLS))
                assert result.ok is True
                assert result.tier == ScrapeTier.TLS

    def test_no_escalation_on_timeout(self):
        timeout_result = ScrapeResult(html="", status=0, tier=ScrapeTier.HTTP,
                                      url="https://example.com", error="timeout")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=timeout_result):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock) as mock_tls:
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.TLS))
                assert result.ok is False
                mock_tls.assert_not_called()

    def test_respects_max_tier(self):
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock) as mock_tls:
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
                assert result.ok is False
                mock_tls.assert_not_called()

    def test_full_escalation_chain(self):
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")
        stealth_ok = ScrapeResult(html="<html>Stealth</html>", status=200,
                                  tier=ScrapeTier.STEALTH, url="https://example.com")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock, return_value=blocked):
                with patch("vecihi.fetchers.fetch_stealth", new_callable=AsyncMock, return_value=stealth_ok):
                    result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.STEALTH))
                    assert result.ok is True
                    assert result.tier == ScrapeTier.STEALTH


class TestScrapeUrls:
    def test_scrape_multiple_urls(self):
        ok_result = ScrapeResult(html="<html>OK</html>", status=200,
                                 tier=ScrapeTier.HTTP, url="")

        with patch("vecihi.fetchers.scrape_url", new_callable=AsyncMock, return_value=ok_result):
            results = run_async(scrape_urls(
                ["https://a.com", "https://b.com"],
                max_tier=ScrapeTier.HTTP,
            ))
            assert len(results) == 2

    def test_filters_non_http_urls(self):
        ok_result = ScrapeResult(html="<html>OK</html>", status=200,
                                 tier=ScrapeTier.HTTP, url="")

        with patch("vecihi.fetchers.scrape_url", new_callable=AsyncMock, return_value=ok_result):
            results = run_async(scrape_urls(
                ["https://a.com", "ftp://b.com", "not-a-url"],
                max_tier=ScrapeTier.HTTP,
            ))
            assert len(results) == 1


class TestTimeout:
    def test_timeout_returns_error(self):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            assert result.ok is False
            assert "timeout" in (result.error or "").lower()
```

- [ ] **Step 4: Run fetcher tests**

```bash
pytest packages/vecihi/tests/test_fetchers.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 5: Run full vecihi test suite**

```bash
pytest packages/vecihi/tests/ -v
```

Expected: All 24 tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/vecihi/tests/
git commit -m "test(vecihi): add standalone test suite"
```

---

### Task 3: Wire vecihi into KutAI

**Files:**
- Modify: `src/tools/scraper.py` (replace with re-export shim)
- Modify: `requirements.txt:94` (add editable install)
- Modify: `tests/test_scraper.py` (update patch targets)

- [ ] **Step 1: Replace `src/tools/scraper.py` with a re-export shim**

Replace the entire contents of `src/tools/scraper.py` with:

```python
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
```

- [ ] **Step 2: Add editable install to requirements.txt**

Add this line after the yazbunu entry (currently line 94: `-e ./packages/yazbunu`):

```
-e ./packages/vecihi
```

- [ ] **Step 3: Install the editable package**

```bash
pip install -e ./packages/vecihi
```

- [ ] **Step 4: Verify KutAI shim imports work**

```bash
python -c "from src.tools.scraper import scrape_url, ScrapeTier, ScrapeResult, _detect_block, _fetch_http; print('shim OK')"
```

Expected: `shim OK`

- [ ] **Step 5: Update `tests/test_scraper.py` patch targets**

The existing tests patch `src.tools.scraper._fetch_http` etc. Since the shim now re-exports from vecihi, patches need to target the vecihi module where the actual functions live.

Replace the entire contents of `tests/test_scraper.py` with:

```python
# tests/test_scraper.py
"""Tests for the tiered scraper — verifies KutAI shim re-exports work."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.scraper import (
    scrape_url,
    scrape_urls,
    ScrapeTier,
    ScrapeResult,
    _detect_block,
    _fetch_http,
    _fetch_tls,
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestDetectBlock(unittest.TestCase):
    """Test Cloudflare/WAF block detection."""

    def test_403_is_blocked(self):
        self.assertTrue(_detect_block(403, "", {}))

    def test_503_with_cloudflare_header(self):
        self.assertTrue(_detect_block(503, "", {"server": "cloudflare"}))

    def test_200_with_cf_challenge(self):
        html = '<html><title>Just a moment...</title><body>Checking your browser</body></html>'
        self.assertTrue(_detect_block(200, html, {}))

    def test_200_normal_page(self):
        html = '<html><title>Real Page</title><body>Content here</body></html>'
        self.assertFalse(_detect_block(200, html, {}))

    def test_200_with_captcha_challenge(self):
        html = '<html><body><form action="/cdn-cgi/challenge-platform">solve</form></body></html>'
        self.assertTrue(_detect_block(200, html, {}))

    def test_429_rate_limited(self):
        self.assertTrue(_detect_block(429, "", {}))

    def test_402_paywall(self):
        self.assertTrue(_detect_block(402, "", {}))

    def test_451_unavailable_legal(self):
        self.assertTrue(_detect_block(451, "", {}))

    def test_200_empty_html_not_blocked(self):
        self.assertFalse(_detect_block(200, "", {}))

    def test_503_without_cloudflare_not_blocked(self):
        self.assertFalse(_detect_block(503, "", {"server": "nginx"}))

    def test_attention_required_marker(self):
        html = '<html><title>Attention Required</title></html>'
        self.assertTrue(_detect_block(200, html, {}))


class TestScrapeResult(unittest.TestCase):

    def test_result_ok_on_200(self):
        r = ScrapeResult(html="<html>test</html>", status=200,
                         tier=ScrapeTier.HTTP, url="https://example.com")
        self.assertTrue(r.ok)

    def test_result_not_ok_on_403(self):
        r = ScrapeResult(html="Forbidden", status=403, tier=ScrapeTier.HTTP,
                         url="https://example.com")
        self.assertFalse(r.ok)

    def test_result_not_ok_with_error(self):
        r = ScrapeResult(html="<html>test</html>", status=200,
                         tier=ScrapeTier.HTTP, url="https://example.com",
                         error="blocked")
        self.assertFalse(r.ok)

    def test_result_not_ok_empty_html(self):
        r = ScrapeResult(html="", status=200, tier=ScrapeTier.HTTP,
                         url="https://example.com")
        self.assertFalse(r.ok)

    def test_tier_values(self):
        self.assertEqual(ScrapeTier.HTTP, 0)
        self.assertEqual(ScrapeTier.TLS, 1)
        self.assertEqual(ScrapeTier.STEALTH, 2)
        self.assertEqual(ScrapeTier.BROWSER, 3)


class TestScrapeUrlHttpTier(unittest.TestCase):
    """Test HTTP tier (aiohttp)."""

    def test_successful_fetch(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="<html><body>Hello</body></html>")
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            self.assertTrue(result.ok)
            self.assertEqual(result.tier, ScrapeTier.HTTP)
            self.assertIn("Hello", result.html)

    def test_403_returns_blocked(self):
        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")
        mock_resp.headers = {"server": "cloudflare"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            self.assertFalse(result.ok)
            self.assertEqual(result.error, "blocked")


class TestScrapeUrlAutoEscalation(unittest.TestCase):
    """Test that scrape_url escalates from HTTP to TLS on block."""

    def test_escalates_to_tls_on_403(self):
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")
        tls_ok = ScrapeResult(html="<html><body>TLS Success</body></html>",
                              status=200, tier=ScrapeTier.TLS,
                              url="https://example.com")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock, return_value=tls_ok):
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.TLS))
                self.assertTrue(result.ok)
                self.assertEqual(result.tier, ScrapeTier.TLS)
                self.assertIn("TLS Success", result.html)

    def test_no_escalation_on_timeout(self):
        """Non-block errors (timeout) should NOT trigger escalation."""
        timeout_result = ScrapeResult(html="", status=0, tier=ScrapeTier.HTTP,
                                      url="https://example.com", error="timeout")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=timeout_result):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock) as mock_tls:
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.TLS))
                self.assertFalse(result.ok)
                mock_tls.assert_not_called()

    def test_respects_max_tier(self):
        """Should not escalate beyond max_tier."""
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock) as mock_tls:
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
                self.assertFalse(result.ok)
                mock_tls.assert_not_called()

    def test_full_escalation_chain(self):
        """HTTP blocked -> TLS blocked -> stealth succeeds."""
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")
        stealth_ok = ScrapeResult(html="<html>Stealth</html>", status=200,
                                  tier=ScrapeTier.STEALTH, url="https://example.com")

        with patch("vecihi.fetchers.fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("vecihi.fetchers.fetch_tls", new_callable=AsyncMock, return_value=blocked):
                with patch("vecihi.fetchers.fetch_stealth", new_callable=AsyncMock, return_value=stealth_ok):
                    result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.STEALTH))
                    self.assertTrue(result.ok)
                    self.assertEqual(result.tier, ScrapeTier.STEALTH)


class TestScrapeUrlTimeout(unittest.TestCase):

    def test_timeout_returns_error(self):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.HTTP))
            self.assertFalse(result.ok)
            self.assertIn("timeout", (result.error or "").lower())


class TestScrapeUrls(unittest.TestCase):

    def test_scrape_multiple_urls(self):
        ok_result = ScrapeResult(html="<html>OK</html>", status=200,
                                 tier=ScrapeTier.HTTP, url="")

        with patch("vecihi.fetchers.scrape_url", new_callable=AsyncMock, return_value=ok_result):
            results = run_async(scrape_urls(
                ["https://a.com", "https://b.com"],
                max_tier=ScrapeTier.HTTP,
            ))
            self.assertEqual(len(results), 2)

    def test_filters_non_http_urls(self):
        ok_result = ScrapeResult(html="<html>OK</html>", status=200,
                                 tier=ScrapeTier.HTTP, url="")

        with patch("vecihi.fetchers.scrape_url", new_callable=AsyncMock, return_value=ok_result):
            results = run_async(scrape_urls(
                ["https://a.com", "ftp://b.com", "not-a-url"],
                max_tier=ScrapeTier.HTTP,
            ))
            self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
```

**Key change:** All patch targets for escalation tests changed from `src.tools.scraper._fetch_*` to `vecihi.fetchers.fetch_*` since that's where the actual functions live now.

- [ ] **Step 6: Run KutAI's scraper tests**

```bash
pytest tests/test_scraper.py -v
```

Expected: All 18 tests pass.

- [ ] **Step 7: Run both test suites together**

```bash
pytest packages/vecihi/tests/ tests/test_scraper.py -v
```

Expected: All 42 tests pass (24 vecihi + 18 KutAI shim).

- [ ] **Step 8: Verify other consumers still work**

These files import from `src.tools.scraper` and should still work via the shim:

```bash
python -c "from src.tools.page_fetch import fetch_pages; print('page_fetch OK')"
python -c "from src.shopping.scrapers.base import BaseScraper; print('shopping base OK')"
```

Expected: Both print OK.

- [ ] **Step 9: Commit**

```bash
git add src/tools/scraper.py tests/test_scraper.py requirements.txt
git commit -m "refactor: wire vecihi into KutAI, replace scraper.py with shim"
```

---

### Task 4: Verify vecihi works independently

**Files:** None — verification only.

- [ ] **Step 1: Verify vecihi imports without KutAI on sys.path**

```bash
python -c "
import sys
sys.path = [p for p in sys.path if 'kutay' not in p.lower() or 'packages' in p.lower()]
from vecihi import ScrapeTier, ScrapeResult, scrape_url, scrape_urls, detect_block
print('vecihi standalone OK')
"
```

Expected: `vecihi standalone OK`

- [ ] **Step 2: Live smoke test (HTTP tier only)**

```bash
python -c "
import asyncio
from vecihi import scrape_url, ScrapeTier
result = asyncio.run(scrape_url('https://httpbin.org/get', max_tier=ScrapeTier.HTTP))
print(f'ok={result.ok}, status={result.status}, tier={result.tier.name}, len={len(result.html)}')
"
```

Expected: `ok=True, status=200, tier=HTTP, len=...`

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest packages/vecihi/tests/ tests/test_scraper.py -v --tb=short
```

Expected: All 42 tests pass.

- [ ] **Step 4: Delete the old plan file**

Remove the superseded Phase 1 plan:

```bash
git rm docs/superpowers/plans/2026-04-11-phase1-package-extraction.md
git commit -m "chore: remove superseded phase1 plan (replaced by vecihi extraction)"
```
