# Search Guard + Scraper Phases 2-3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix LLM hallucination-before-search issue with a search-required guard, fix the task_hints extraction bug, then implement tiered scraping (curl_cffi TLS bypass + Scrapling stealth/browser) with auto-escalation from the existing aiohttp fetcher.

**Architecture:** The search-required guard uses the classifier's `search_depth` field (already computed before agent runs) to reject `final_answer` when web_search hasn't been called. The scraper module (`src/tools/scraper.py`) provides a unified `scrape_url()` function with 4 tiers (http → tls → stealth → browser) that auto-escalates on 403/Cloudflare responses. It replaces direct `aiohttp` calls in `page_fetch.py` and `web_search.py`.

**Tech Stack:** Python 3.10, curl_cffi (TLS fingerprinting), scrapling (stealth browser), pytest, asyncio

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/agents/base.py` | Modified (done) | Search-required guard + `_get_search_depth()` helper |
| `src/tools/scraper.py` | Create | Tiered scraper: http → tls → stealth → browser |
| `src/tools/page_fetch.py` | Modify | Use scraper instead of raw aiohttp for fetches |
| `src/tools/web_search.py` | Modify | Use scraper-backed page_fetch |
| `tests/test_search_guard.py` | Create | Tests for the search-required guard |
| `tests/test_scraper.py` | Create | Tests for tiered scraper |
| `requirements.txt` | Modify | Add curl_cffi, scrapling[all] |

---

### Task 1: Search-Required Guard Tests

**Files:**
- Test: `tests/test_search_guard.py`
- Reference: `src/agents/base.py` (already modified)

The search-required guard is already implemented in `base.py`. This task writes tests for it and the `_get_search_depth()` helper.

- [ ] **Step 1: Write test for `_get_search_depth` helper**

```python
# tests/test_search_guard.py
"""Tests for the search-required guard in BaseAgent."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.base import BaseAgent


class TestGetSearchDepth(unittest.TestCase):

    def test_extracts_from_classification_context(self):
        task = {"context": json.dumps({"classification": {"search_depth": "deep"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "deep")

    def test_extracts_quick(self):
        task = {"context": json.dumps({"classification": {"search_depth": "quick"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "quick")

    def test_extracts_standard(self):
        task = {"context": json.dumps({"classification": {"search_depth": "standard"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "standard")

    def test_returns_none_for_no_search(self):
        task = {"context": json.dumps({"classification": {"search_depth": "none"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_no_context(self):
        task = {}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_context_is_empty_string(self):
        task = {"context": ""}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_context_is_invalid_json(self):
        task = {"context": "not json"}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_classification_missing(self):
        task = {"context": json.dumps({"other": "data"})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_search_depth_missing(self):
        task = {"context": json.dumps({"classification": {"agent_type": "coder"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_handles_dict_context(self):
        """Context might already be parsed as dict in some code paths."""
        task = {"context": {"classification": {"search_depth": "deep"}}}
        self.assertEqual(BaseAgent._get_search_depth(task), "deep")

    def test_handles_none_search_depth_value(self):
        task = {"context": json.dumps({"classification": {"search_depth": None}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_search_guard.py -v`
Expected: All 11 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_search_guard.py src/agents/base.py
git commit -m "feat(agents): search-required guard — reject final_answer when search_depth demands web_search"
```

---

### Task 2: Fix task_hints search_depth Extraction Bug

**Files:**
- Modify: `src/agents/base.py:1510-1514`

Line 1512 tries `task.get("context", {})` but context is a JSON string, not a dict. So `isinstance(task.get("context"), dict)` is always False and `search_depth` in hints is always None. Fix: reuse `_get_search_depth()`.

- [ ] **Step 1: Write a test that exposes the bug**

Add to `tests/test_search_guard.py`:

```python
class TestTaskHintsExtraction(unittest.TestCase):
    """Verify that task_hints correctly extracts search_depth from JSON context."""

    def test_hints_extraction_from_json_context(self):
        """The bug: context is a JSON string but code treats it as dict."""
        task = {"context": json.dumps({"classification": {"search_depth": "deep"}})}
        # _get_search_depth handles both str and dict context
        depth = BaseAgent._get_search_depth(task)
        self.assertEqual(depth, "deep")
```

- [ ] **Step 2: Fix the hints extraction in base.py**

Replace the inline extraction at line 1510-1514 with `_get_search_depth()`:

```python
                            # Build task hints for context-aware tools
                            _hints = {
                                "agent_type": self.name,
                                "search_depth": self._get_search_depth(task),
                                "shopping_sub_intent": task.get("shopping_sub_intent"),
                            }
```

- [ ] **Step 3: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_search_guard.py tests/test_deep_search_integration.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/agents/base.py tests/test_search_guard.py
git commit -m "fix(agents): task_hints search_depth extraction from JSON context string"
```

---

### Task 3: Phase 2 — curl_cffi TLS Fingerprint Scraper

**Files:**
- Create: `src/tools/scraper.py`
- Test: `tests/test_scraper.py`
- Modify: `requirements.txt`

The scraper provides a unified `scrape_url()` async function. Tier 0 (http) uses existing aiohttp. Tier 1 (tls) uses curl_cffi with browser TLS fingerprints. Auto-escalates on 403/Cloudflare detection.

- [ ] **Step 1: Add curl_cffi to requirements.txt**

Add after `ddgs>=9.0`:
```
curl_cffi>=0.7.0
```

- [ ] **Step 2: Install curl_cffi**

Run: `.venv/Scripts/pip install curl_cffi>=0.7.0`

- [ ] **Step 3: Write failing tests for scraper**

```python
# tests/test_scraper.py
"""Tests for the tiered scraper module."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.scraper import (
    scrape_url,
    ScrapeTier,
    ScrapeResult,
    _detect_block,
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


class TestScrapeResult(unittest.TestCase):

    def test_result_fields(self):
        r = ScrapeResult(html="<html>test</html>", status=200,
                         tier=ScrapeTier.HTTP, url="https://example.com")
        self.assertEqual(r.html, "<html>test</html>")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.tier, ScrapeTier.HTTP)
        self.assertTrue(r.ok)

    def test_failed_result(self):
        r = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                         url="https://example.com", error="Blocked")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "Blocked")


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


class TestScrapeUrlAutoEscalation(unittest.TestCase):
    """Test that scrape_url escalates from HTTP to TLS on block."""

    def test_escalates_to_tls_on_403(self):
        # HTTP tier returns 403
        mock_resp_403 = AsyncMock()
        mock_resp_403.status = 403
        mock_resp_403.text = AsyncMock(return_value="Forbidden")
        mock_resp_403.headers = {"server": "cloudflare"}
        mock_resp_403.__aenter__ = AsyncMock(return_value=mock_resp_403)
        mock_resp_403.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp_403)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # TLS tier succeeds
        mock_tls_resp = MagicMock()
        mock_tls_resp.status_code = 200
        mock_tls_resp.text = "<html><body>TLS Success</body></html>"
        mock_tls_resp.headers = {"content-type": "text/html"}

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("src.tools.scraper._fetch_tls", new_callable=AsyncMock,
                       return_value=mock_tls_resp):
                result = run_async(scrape_url("https://example.com", max_tier=ScrapeTier.TLS))
                self.assertTrue(result.ok)
                self.assertEqual(result.tier, ScrapeTier.TLS)
                self.assertIn("TLS Success", result.html)


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_scraper.py -v`
Expected: ImportError — `src.tools.scraper` doesn't exist yet

- [ ] **Step 5: Implement scraper.py**

```python
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


_TIER_FETCHERS = {
    ScrapeTier.HTTP: _fetch_http,
    ScrapeTier.TLS: _fetch_tls,
    ScrapeTier.STEALTH: _fetch_stealth,
    ScrapeTier.BROWSER: _fetch_browser,
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

        fetcher = _TIER_FETCHERS[tier]
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
            # Non-block error (timeout, connection error) — don't escalate
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

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_scraper.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/tools/scraper.py tests/test_scraper.py requirements.txt
git commit -m "feat(tools): tiered scraper with curl_cffi TLS fingerprint bypass (Phase 2)"
```

---

### Task 4: Wire Scraper into page_fetch and web_search

**Files:**
- Modify: `src/tools/page_fetch.py`
- Modify: `src/tools/web_search.py`
- Modify: `tests/test_page_fetch.py` (if any tests break)

Replace direct aiohttp calls in `page_fetch.py` with the scraper's `scrape_url()`. The deep pipeline in `web_search.py` already calls `page_fetch.fetch_pages()`, so it will automatically benefit.

- [ ] **Step 1: Modify `fetch_page_content` in page_fetch.py to use scraper**

The key change: `fetch_page_content` tries `scrape_url()` first, falls back to the existing aiohttp path if scraper module fails to import. `fetch_pages` gets a `max_tier` parameter.

```python
# Replace the existing fetch_page_content function.
# Keep extract_main_text unchanged (it's used as fallback by content_extract.py too).

async def fetch_page_content(
    session: aiohttp.ClientSession | None,
    url: str,
    max_chars: int = 1500,
    timeout: float = 8.0,
    max_tier: int = 1,  # ScrapeTier.TLS by default
) -> str | None:
    """Fetch a single page and extract main text.

    Uses tiered scraper (HTTP → TLS) with auto-escalation.
    Falls back to plain aiohttp if scraper not available.
    """
    try:
        from src.tools.scraper import scrape_url, ScrapeTier
        tier = ScrapeTier(min(max_tier, ScrapeTier.BROWSER))
        result = await scrape_url(url, max_tier=tier, timeout=timeout)
        if not result.ok:
            logger.debug("page_fetch: scraper failed", url=url[:80],
                         tier=result.tier.name, error=result.error)
            return None
        text = extract_main_text(result.html, max_chars=max_chars)
        if not text or len(text) < 50:
            return None
        return text
    except ImportError:
        pass  # Fall through to legacy aiohttp path

    # Legacy path (no scraper available)
    if session is None:
        return None
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
            max_redirects=3,
        ) as resp:
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/xhtml" not in content_type:
                return None
            if resp.status != 200:
                return None
            html = await resp.text(encoding=None)
            text = extract_main_text(html, max_chars=max_chars)
            if not text or len(text) < 50:
                return None
            return text
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.debug("page_fetch: error", url=url[:80], error=str(e)[:100])
        return None
```

Update `fetch_pages` to pass `session=None` when using scraper (scraper manages its own sessions):

```python
async def fetch_pages(
    urls: list[str],
    max_pages: int = 3,
    max_chars: int = 1500,
    total_timeout: float = 12.0,
    max_tier: int = 1,
) -> dict[str, str]:
    valid_urls = [u for u in urls if u.startswith(("http://", "https://"))][:max_pages]
    if not valid_urls:
        return {}

    results = {}
    try:
        # Try scraper-based path first (no aiohttp session needed)
        try:
            from src.tools.scraper import ScrapeTier  # noqa: F401
            tasks = [
                fetch_page_content(None, url, max_chars=max_chars,
                                   timeout=min(total_timeout, 10.0), max_tier=max_tier)
                for url in valid_urls
            ]
        except ImportError:
            # Legacy aiohttp path
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_page_content(session, url, max_chars=max_chars)
                    for url in valid_urls
                ]
                fetched = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=total_timeout,
                )
                for url, content in zip(valid_urls, fetched):
                    if isinstance(content, str) and content:
                        results[url] = content
                return results

        fetched = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=total_timeout,
        )
        for url, content in zip(valid_urls, fetched):
            if isinstance(content, str) and content:
                results[url] = content

    except asyncio.TimeoutError:
        logger.debug("page_fetch: total timeout reached", fetched=len(results))
    except Exception as e:
        logger.debug("page_fetch: batch error", error=str(e)[:100])

    return results
```

- [ ] **Step 2: Run existing page_fetch tests**

Run: `.venv/Scripts/python -m pytest tests/test_page_fetch.py -v`
Expected: Tests should still pass (scraper import may be mocked or available)

- [ ] **Step 3: Run web_search integration tests**

Run: `.venv/Scripts/python -m pytest tests/test_web_search_integration.py tests/test_deep_search_integration.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/tools/page_fetch.py
git commit -m "feat(tools): wire scraper into page_fetch for auto-escalation on blocked sites"
```

---

### Task 5: Phase 3 — Scrapling Stealth/Browser Tiers

**Files:**
- Modify: `requirements.txt`
- Modify: `tests/test_scraper.py` (add stealth/browser tests)

Scrapling tiers are already coded in `scraper.py` (Task 3). This task adds the dependency and tests. The tiers use lazy imports — if scrapling isn't installed, they gracefully return an error result without crashing.

- [ ] **Step 1: Add scrapling to requirements.txt**

Add after `curl_cffi>=0.7.0`:
```
scrapling[all]>=0.4.0
```

- [ ] **Step 2: Install scrapling**

Run: `.venv/Scripts/pip install "scrapling[all]>=0.4.0"`

Note: This may take a while as it installs Camoufox and Playwright. If it fails on Windows, the stealth/browser tiers will gracefully degrade (ImportError caught in scraper.py). The HTTP and TLS tiers still work.

- [ ] **Step 3: Add stealth tier tests**

Add to `tests/test_scraper.py`:

```python
class TestStealthTier(unittest.TestCase):
    """Test stealth tier (Scrapling StealthyFetcher)."""

    def test_stealth_unavailable_graceful(self):
        """When scrapling not installed, stealth tier returns error gracefully."""
        with patch.dict("sys.modules", {"scrapling": None}):
            # Force reimport to trigger ImportError
            result = run_async(
                _fetch_stealth_direct("https://example.com")
            )
            # Should not crash, just return error result
            self.assertFalse(result.ok)

    def test_escalation_http_to_tls_to_stealth(self):
        """Full escalation chain: HTTP blocked → TLS blocked → stealth succeeds."""
        blocked = ScrapeResult(html="", status=403, tier=ScrapeTier.HTTP,
                               url="https://example.com", error="blocked")

        mock_stealth_result = ScrapeResult(
            html="<html><body>Stealth content</body></html>",
            status=200, tier=ScrapeTier.STEALTH, url="https://example.com",
        )

        with patch("src.tools.scraper._fetch_http", new_callable=AsyncMock, return_value=blocked):
            with patch("src.tools.scraper._fetch_tls", new_callable=AsyncMock, return_value=blocked):
                with patch("src.tools.scraper._fetch_stealth", new_callable=AsyncMock,
                           return_value=mock_stealth_result):
                    result = run_async(scrape_url("https://example.com",
                                                  max_tier=ScrapeTier.STEALTH))
                    self.assertTrue(result.ok)
                    self.assertEqual(result.tier, ScrapeTier.STEALTH)
```

- [ ] **Step 4: Run all scraper tests**

Run: `.venv/Scripts/python -m pytest tests/test_scraper.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_scraper.py
git commit -m "feat(tools): add Scrapling stealth/browser tiers (Phase 3)"
```

---

### Task 6: Deep Pipeline Uses Higher Tiers for Blocked Pages

**Files:**
- Modify: `src/tools/web_search.py`

The deep search pipeline should use TLS tier (default), but for product/market/research intents, allow escalation to stealth tier if available.

- [ ] **Step 1: Add max_tier parameter to deep pipeline**

In `web_search.py`, modify `_deep_search_pipeline` to pass tier info:

```python
async def _deep_search_pipeline(
    query: str, ddgs_results: list, urls: list, intent: str, params: _SearchParams
) -> str:
    """Deep path: fetch pages -> Trafilatura -> BM25 -> budget allocation."""
    from src.tools.page_fetch import fetch_pages
    from src.tools.content_extract import extract_content
    from src.tools.relevance import score_and_budget

    # Higher intents get higher scraper tiers
    _INTENT_TIER = {
        "factual": 0,   # HTTP only
        "product": 1,   # Up to TLS
        "reviews": 1,   # Up to TLS
        "market": 2,    # Up to stealth (if available)
        "research": 2,  # Up to stealth (if available)
    }
    max_tier = _INTENT_TIER.get(intent, 1)

    page_htmls = await fetch_pages(
        urls, max_pages=params.max_results, max_chars=50000,
        max_tier=max_tier,
    )
    # ... rest unchanged
```

- [ ] **Step 2: Run all search tests**

Run: `.venv/Scripts/python -m pytest tests/test_web_search_integration.py tests/test_deep_search_integration.py tests/test_page_fetch.py tests/test_scraper.py tests/test_content_extract.py tests/test_relevance.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/tools/web_search.py
git commit -m "feat(search): deep pipeline escalates scraper tier by intent (market/research get stealth)"
```

---

### Task 7: Full Test Suite + Final Commit

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/Scripts/python -m pytest tests/ -q --ignore=tests/test_phase2.py --ignore=tests/test_phase2_structured.py -k "not idea_to_product"`
Expected: All pass (existing + new tests)

- [ ] **Step 2: Verify imports**

Run: `.venv/Scripts/python -c "from src.tools.scraper import scrape_url, ScrapeTier; print('scraper OK')"`
Run: `.venv/Scripts/python -c "from src.agents.base import BaseAgent; print('base OK')"`

- [ ] **Step 3: Update web-search-xray.md**

Add scraper documentation to the existing architecture doc.

- [ ] **Step 4: Final commit if any docs changed**

```bash
git add docs/web-search-xray.md
git commit -m "docs: update web search x-ray with scraper tiers"
```
