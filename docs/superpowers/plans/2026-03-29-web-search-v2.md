# Web Search v2: ddgs + Page Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Perplexica/SearXNG as the primary web search path with ddgs (snippets) + async page fetching (full content), eliminating container dependencies, GPU contention, and timeout issues.

**Architecture:** `web_search()` becomes: ChromaDB cache check → ddgs for URLs+snippets → async fetch top 3 pages → extract main text with BeautifulSoup → return formatted snippets+content to agent. Perplexica moves to a gated fallback (disabled by default, future work). No separate synthesis LLM call — the agent reasons over raw results in its normal ReAct iteration.

**Tech Stack:** `ddgs` (already installed), `aiohttp` (already installed), `beautifulsoup4` (installed but not in requirements.txt), `lxml` (installed, fast HTML parser).

---

### Task 1: Add beautifulsoup4 to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add beautifulsoup4 and lxml to requirements.txt**

Add these two lines to `requirements.txt` (alphabetical placement):

```
beautifulsoup4>=4.12.0
lxml>=5.0.0
```

- [ ] **Step 2: Verify installation**

Run: `pip install -r requirements.txt`
Expected: "Requirement already satisfied" for both (they're installed, just not tracked).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add beautifulsoup4 and lxml to requirements.txt"
```

---

### Task 2: Write the page fetcher module

**Files:**
- Create: `src/tools/page_fetch.py`
- Test: `tests/test_page_fetch.py`

This module fetches URLs and extracts main text content. It is a standalone utility used by `web_search.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_page_fetch.py`:

```python
# tests/test_page_fetch.py
"""Tests for page_fetch module — HTML content extraction."""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.page_fetch import extract_main_text, fetch_page_content, fetch_pages


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestExtractMainText(unittest.TestCase):
    """Test HTML text extraction logic."""

    def test_extracts_article_content(self):
        html = """
        <html><body>
        <nav>Menu items</nav>
        <article><p>This is the main article content about coffee machines.</p></article>
        <footer>Copyright 2026</footer>
        </body></html>
        """
        result = extract_main_text(html, max_chars=1500)
        self.assertIn("main article content", result)
        self.assertNotIn("Menu items", result)
        self.assertNotIn("Copyright", result)

    def test_extracts_main_tag(self):
        html = """
        <html><body>
        <header>Site header</header>
        <main><p>Main content here.</p><p>More details.</p></main>
        <aside>Sidebar</aside>
        </body></html>
        """
        result = extract_main_text(html, max_chars=1500)
        self.assertIn("Main content here", result)
        self.assertNotIn("Site header", result)
        self.assertNotIn("Sidebar", result)

    def test_falls_back_to_body(self):
        html = """
        <html><body>
        <div><p>Just body content with no article or main tag.</p></div>
        </body></html>
        """
        result = extract_main_text(html, max_chars=1500)
        self.assertIn("body content", result)

    def test_strips_script_and_style(self):
        html = """
        <html><body>
        <script>var x = 1;</script>
        <style>.foo { color: red; }</style>
        <p>Visible content only.</p>
        </body></html>
        """
        result = extract_main_text(html, max_chars=1500)
        self.assertIn("Visible content", result)
        self.assertNotIn("var x", result)
        self.assertNotIn("color: red", result)

    def test_truncates_to_max_chars(self):
        html = "<html><body><p>" + "word " * 500 + "</p></body></html>"
        result = extract_main_text(html, max_chars=100)
        self.assertLessEqual(len(result), 110)  # small margin for word boundary

    def test_collapses_whitespace(self):
        html = "<html><body><p>Hello   \n\n\n   world</p></body></html>"
        result = extract_main_text(html, max_chars=1500)
        self.assertNotIn("\n\n\n", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_empty_html_returns_empty(self):
        result = extract_main_text("", max_chars=1500)
        self.assertEqual(result, "")

    def test_non_html_returns_empty(self):
        result = extract_main_text("just plain text without tags", max_chars=1500)
        # Should still extract text content even without proper HTML
        self.assertIn("just plain text", result)


class TestFetchPageContent(unittest.TestCase):
    """Test single page fetching with mocked HTTP."""

    def test_successful_fetch(self):
        html = "<html><body><article><p>Great article about Python.</p></article></body></html>"
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.text = AsyncMock(return_value=html)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        result = run_async(fetch_page_content(mock_session, "https://example.com/article", max_chars=1500))
        self.assertIn("Great article about Python", result)

    def test_non_html_content_type_returns_none(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.text = AsyncMock(return_value="")

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        result = run_async(fetch_page_content(mock_session, "https://example.com/doc.pdf", max_chars=1500))
        self.assertIsNone(result)

    def test_http_error_returns_none(self):
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = AsyncMock(return_value="Not found")

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        result = run_async(fetch_page_content(mock_session, "https://example.com/missing", max_chars=1500))
        self.assertIsNone(result)

    def test_timeout_returns_none(self):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=asyncio.TimeoutError),
            __aexit__=AsyncMock(return_value=False),
        ))

        result = run_async(fetch_page_content(mock_session, "https://example.com/slow", max_chars=1500))
        self.assertIsNone(result)


class TestFetchPages(unittest.TestCase):
    """Test parallel page fetching."""

    def test_fetches_multiple_pages(self):
        pages = {
            "https://a.com": "<html><body><p>Page A content</p></body></html>",
            "https://b.com": "<html><body><p>Page B content</p></body></html>",
        }

        async def mock_get(url, **kwargs):
            resp = AsyncMock()
            resp.status = 200
            resp.headers = {"content-type": "text/html"}
            resp.text = AsyncMock(return_value=pages.get(url, ""))
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        with patch("aiohttp.ClientSession") as mock_cls:
            session = AsyncMock()
            session.get = mock_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = run_async(fetch_pages(
                ["https://a.com", "https://b.com"],
                max_pages=3,
                max_chars=1500,
            ))

        self.assertEqual(len(results), 2)
        self.assertIn("Page A content", results["https://a.com"])
        self.assertIn("Page B content", results["https://b.com"])

    def test_limits_to_max_pages(self):
        """Only fetches up to max_pages URLs."""
        urls = [f"https://example.com/{i}" for i in range(10)]

        with patch("aiohttp.ClientSession") as mock_cls:
            session = AsyncMock()
            call_count = {"n": 0}

            async def mock_get(url, **kwargs):
                call_count["n"] += 1
                resp = AsyncMock()
                resp.status = 200
                resp.headers = {"content-type": "text/html"}
                resp.text = AsyncMock(return_value=f"<html><body><p>Content {url}</p></body></html>")
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(return_value=resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx

            session.get = mock_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = run_async(fetch_pages(urls, max_pages=3, max_chars=1500))

        self.assertLessEqual(call_count["n"], 3)

    def test_skips_non_http_urls(self):
        """Skip URLs that aren't http/https."""
        urls = ["ftp://example.com/file", "https://real.com"]

        with patch("aiohttp.ClientSession") as mock_cls:
            session = AsyncMock()

            async def mock_get(url, **kwargs):
                resp = AsyncMock()
                resp.status = 200
                resp.headers = {"content-type": "text/html"}
                resp.text = AsyncMock(return_value="<html><body><p>Real content</p></body></html>")
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(return_value=resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx

            session.get = mock_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = run_async(fetch_pages(urls, max_pages=3, max_chars=1500))

        self.assertNotIn("ftp://example.com/file", results)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_page_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.page_fetch'`

- [ ] **Step 3: Write the page_fetch module**

Create `src/tools/page_fetch.py`:

```python
# src/tools/page_fetch.py
"""Async page fetcher with HTML content extraction using BeautifulSoup."""

import asyncio
import re

import aiohttp
from bs4 import BeautifulSoup

from src.infra.logging_config import get_logger

logger = get_logger("tools.page_fetch")

# Standard browser User-Agent to avoid bot blocks
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Tags that contain non-content elements
_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"]


def extract_main_text(html: str, max_chars: int = 1500) -> str:
    """Extract main text content from HTML, stripping boilerplate.

    Priority: <article> → <main> → <body>.
    Strips script, style, nav, header, footer, aside.
    Collapses whitespace. Truncates to max_chars on word boundary.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove non-content tags
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # Find main content container
    content = soup.find("article") or soup.find("main") or soup.find("body")
    if not content:
        return ""

    # Get text, collapse whitespace
    text = content.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = text.strip()

    # Truncate on word boundary
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return text


async def fetch_page_content(
    session: aiohttp.ClientSession,
    url: str,
    max_chars: int = 1500,
    timeout: float = 8.0,
) -> str | None:
    """Fetch a single page and extract main text.

    Returns extracted text, or None on any error (timeout, non-HTML, HTTP error).
    """
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
            max_redirects=3,
        ) as resp:
            # Skip non-HTML responses
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/xhtml" not in content_type:
                logger.debug("page_fetch: skipping non-HTML", url=url[:80], ct=content_type[:50])
                return None

            if resp.status != 200:
                logger.debug("page_fetch: HTTP error", url=url[:80], status=resp.status)
                return None

            html = await resp.text(encoding=None)  # let aiohttp detect encoding
            text = extract_main_text(html, max_chars=max_chars)
            if not text or len(text) < 50:
                logger.debug("page_fetch: too little content", url=url[:80], text_len=len(text))
                return None

            logger.debug("page_fetch: ok", url=url[:80], text_len=len(text))
            return text

    except asyncio.TimeoutError:
        logger.debug("page_fetch: timeout", url=url[:80])
        return None
    except Exception as e:
        logger.debug("page_fetch: error", url=url[:80], error=str(e)[:100])
        return None


async def fetch_pages(
    urls: list[str],
    max_pages: int = 3,
    max_chars: int = 1500,
    total_timeout: float = 12.0,
) -> dict[str, str]:
    """Fetch multiple pages in parallel, returning {url: extracted_text}.

    Only fetches http/https URLs. Limits to max_pages.
    Returns dict of successfully fetched pages (may be empty).
    """
    # Filter to http(s) only and limit count
    valid_urls = [u for u in urls if u.startswith(("http://", "https://"))][:max_pages]

    if not valid_urls:
        return {}

    results = {}
    try:
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

    except asyncio.TimeoutError:
        logger.debug("page_fetch: total timeout reached", fetched=len(results))
    except Exception as e:
        logger.debug("page_fetch: batch error", error=str(e)[:100])

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_page_fetch.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/page_fetch.py tests/test_page_fetch.py
git commit -m "feat(search): add page_fetch module for async HTML content extraction"
```

---

### Task 3: Rewrite web_search to use ddgs + page fetch as primary

**Files:**
- Modify: `src/tools/web_search.py`
- Test: `tests/test_web_search_integration.py` (modify existing)

- [ ] **Step 1: Write new tests for the ddgs + page fetch flow**

Add to `tests/test_web_search_integration.py`, a new test class after the existing ones:

```python
# ===========================================================================
# 5. TestDDGSWithPageFetch — new primary search path
# ===========================================================================

class TestDDGSWithPageFetch(unittest.TestCase):
    """Tests for the ddgs + page fetch primary search path."""

    def setUp(self):
        _reset_ws_state()

    def tearDown(self):
        _reset_ws_state()

    @patch.object(_ws_mod, "_search_perplexica", new_callable=AsyncMock, return_value=None)
    @patch.object(_ws_mod, "_search_searxng_direct", new_callable=AsyncMock, return_value=None)
    def test_ddgs_with_page_fetch_returns_content(self, mock_searxng, mock_perp):
        """ddgs results + page content should be included in output."""
        mock_ddgs_results = [
            {"title": "Coffee Guide", "body": "Best coffee machines reviewed", "href": "https://example.com/coffee"},
            {"title": "Machine Reviews", "body": "Top picks for 2026", "href": "https://example.com/reviews"},
        ]

        mock_page_content = {
            "https://example.com/coffee": "Detailed review of coffee machines including DeLonghi and Philips models with prices.",
        }

        with patch.object(_ws_mod._DDGS, "__call__") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_ddgs_results
            mock_ddgs_cls.return_value = mock_instance

            with patch("src.tools.page_fetch.fetch_pages", new_callable=AsyncMock, return_value=mock_page_content):
                result = run_async(_ws_mod.web_search("coffee machine reviews", max_results=5))

        self.assertIn("Coffee Guide", result)
        self.assertIn("DeLonghi", result)  # from page content

    @patch.object(_ws_mod, "_search_perplexica", new_callable=AsyncMock, return_value=None)
    @patch.object(_ws_mod, "_search_searxng_direct", new_callable=AsyncMock, return_value=None)
    def test_ddgs_without_page_fetch_still_works(self, mock_searxng, mock_perp):
        """If page fetch fails entirely, snippets-only results should still be returned."""
        mock_ddgs_results = [
            {"title": "Python Docs", "body": "Official Python documentation", "href": "https://docs.python.org"},
        ]

        with patch.object(_ws_mod._DDGS, "__call__") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_ddgs_results
            mock_ddgs_cls.return_value = mock_instance

            with patch("src.tools.page_fetch.fetch_pages", new_callable=AsyncMock, return_value={}):
                result = run_async(_ws_mod.web_search("Python tutorial", max_results=3))

        self.assertIn("Python Docs", result)
        self.assertIn("Official Python documentation", result)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_web_search_integration.py::TestDDGSWithPageFetch -v`
Expected: FAIL — page fetch not yet integrated into `web_search()`.

- [ ] **Step 3: Rewrite the ddgs path in web_search() to include page fetching**

In `src/tools/web_search.py`, replace the ddgs section (the `# Method 3` block, lines 466-487) with:

```python
    # Method 3: duckduckgo-search package (ddgs 9.x) + page content fetch
    if _DDGS is not None:
        try:
            # ddgs 9.x: DDGS().text() returns a list directly
            results = _DDGS().text(query, max_results=max_results)
            if results:
                logger.debug("ddgs search ok", count=len(results))

                # Fetch full page content for top results
                urls = [r.get("href", "") for r in results if r.get("href")]
                page_contents = {}
                if urls:
                    try:
                        from src.tools.page_fetch import fetch_pages
                        page_contents = await fetch_pages(urls, max_pages=3, max_chars=1500)
                        logger.debug("page_fetch: fetched pages", count=len(page_contents))
                    except Exception as e:
                        logger.debug("page_fetch: skipped", error=str(e)[:100])

                # Format: snippets + page content where available
                lines = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    body = r.get("body", "")[:200]
                    href = r.get("href", "")
                    parts = [f"{i}. **{title}**\n   {body}\n   {href}"]
                    if href in page_contents:
                        parts.append(f"   ---\n   {page_contents[href]}")
                    lines.append("\n".join(parts))

                result_text = f"Search results for '{query}':\n\n" + "\n\n".join(lines)

                # Phase D: Embed results
                await _embed_web_results(query, result_text)

                return result_text
            else:
                return f"No results found for '{query}'"
        except Exception as e:
            logger.warning("duckduckgo search failed, using curl fallback", error=str(e))
```

- [ ] **Step 4: Run all web search tests**

Run: `pytest tests/test_web_search_integration.py -v`
Expected: All tests PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/tools/web_search.py tests/test_web_search_integration.py
git commit -m "feat(search): integrate page fetch into ddgs search path"
```

---

### Task 4: Promote ddgs + page fetch to primary, gate Perplexica behind flag

**Files:**
- Modify: `src/tools/web_search.py`

- [ ] **Step 1: Write test for new fallback order**

Add to `tests/test_web_search_integration.py`:

```python
class TestFallbackOrder(unittest.TestCase):
    """Verify ddgs is tried BEFORE Perplexica."""

    def setUp(self):
        _reset_ws_state()

    def tearDown(self):
        _reset_ws_state()

    def test_ddgs_is_primary_perplexica_is_fallback(self):
        """ddgs should be called first; Perplexica only if ddgs fails."""
        call_order = []

        async def mock_perp(*args, **kwargs):
            call_order.append("perplexica")
            return {"answer": "Perplexica answer", "sources": [{"title": "t", "url": "u", "snippet": "s"}]}

        original_ddgs = _ws_mod._DDGS

        class MockDDGS:
            def text(self, *a, **kw):
                call_order.append("ddgs")
                return [{"title": "DDG Result", "body": "snippet", "href": "https://example.com"}]

        _ws_mod._DDGS = MockDDGS
        try:
            with patch.object(_ws_mod, "_search_perplexica", side_effect=mock_perp):
                with patch("src.tools.page_fetch.fetch_pages", new_callable=AsyncMock, return_value={}):
                    result = run_async(_ws_mod.web_search("test query"))

            # ddgs should be first, perplexica should NOT be called
            self.assertEqual(call_order[0], "ddgs")
            self.assertNotIn("perplexica", call_order)
        finally:
            _ws_mod._DDGS = original_ddgs

    def test_perplexica_called_when_ddgs_fails(self):
        """If ddgs returns nothing, Perplexica should be tried as fallback."""
        call_order = []

        async def mock_perp(*args, **kwargs):
            call_order.append("perplexica")
            return {"answer": "Perplexica answer", "sources": [{"title": "t", "url": "u", "snippet": "s"}]}

        original_ddgs = _ws_mod._DDGS

        class MockDDGS:
            def text(self, *a, **kw):
                call_order.append("ddgs")
                return []  # empty results

        _ws_mod._DDGS = MockDDGS
        try:
            with patch.object(_ws_mod, "_search_perplexica", side_effect=mock_perp):
                result = run_async(_ws_mod.web_search("test query"))

            self.assertIn("ddgs", call_order)
            self.assertIn("perplexica", call_order)
            self.assertIn("Perplexica", result)
        finally:
            _ws_mod._DDGS = original_ddgs
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_web_search_integration.py::TestFallbackOrder -v`
Expected: FAIL — current code tries Perplexica first.

- [ ] **Step 3: Reorder the fallback chain in web_search()**

Rewrite `web_search()` in `src/tools/web_search.py` (the function body starting after the ChromaDB cache check). The new order is:

1. ChromaDB cache (unchanged)
2. ddgs + page fetch (primary)
3. Perplexica (fallback, only if ddgs returned nothing)
4. SearXNG direct (fallback)
5. curl (last resort)

Replace the function body from `# Check for degraded capability` (line 421) onwards with:

```python
    # Check for degraded capability
    try:
        from src.infra.runtime_state import runtime_state
        is_degraded = "web_search" in runtime_state.get("degraded_capabilities", [])
    except Exception:
        is_degraded = False

    # Method 1 (primary): DuckDuckGo + page content fetch
    if _DDGS is not None:
        try:
            results = _DDGS().text(query, max_results=max_results)
            if results:
                logger.debug("ddgs search ok", count=len(results))

                # Fetch full page content for top results
                urls = [r.get("href", "") for r in results if r.get("href")]
                page_contents = {}
                if urls:
                    try:
                        from src.tools.page_fetch import fetch_pages
                        page_contents = await fetch_pages(urls, max_pages=3, max_chars=1500)
                        logger.debug("page_fetch: fetched pages", count=len(page_contents))
                    except Exception as e:
                        logger.debug("page_fetch: skipped", error=str(e)[:100])

                # Format: snippets + page content where available
                lines = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    body = r.get("body", "")[:200]
                    href = r.get("href", "")
                    parts = [f"{i}. **{title}**\n   {body}\n   {href}"]
                    if href in page_contents:
                        parts.append(f"   ---\n   {page_contents[href]}")
                    lines.append("\n".join(parts))

                result_text = f"Search results for '{query}':\n\n" + "\n\n".join(lines)
                await _embed_web_results(query, result_text)
                return result_text
        except Exception as e:
            logger.warning("ddgs primary search failed", error=str(e))

    # Method 2 (fallback): Perplexica/Vane AI synthesis
    if not is_degraded:
        perplexica_result = await _search_perplexica(query, max_results, search_type)
        if perplexica_result:
            logger.debug("using perplexica fallback for web search")
            lines = [
                "## AI-Synthesized Answer (from Perplexica)\n",
                perplexica_result["answer"],
            ]
            if perplexica_result["sources"]:
                lines.append("\n### Sources")
                for i, src in enumerate(perplexica_result["sources"], 1):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    lines.append(f"- [{title}]({url})")
            lines.append(
                "\n**Note: This answer is already synthesized from multiple "
                "sources. Use it as your final answer unless something "
                "specific is missing.**"
            )
            result_text = "\n".join(lines)
            await _embed_web_results(query, result_text)
            return result_text

    # Method 3 (fallback): SearXNG direct (raw results, no LLM)
    searxng_result = await _search_searxng_direct(query, max_results)
    if searxng_result and searxng_result.count("**") >= 6:
        asyncio.ensure_future(_embed_web_results(query, searxng_result))
        return searxng_result

    # Method 4 (last resort): curl DuckDuckGo
    try:
        safe_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&no_redirect=1"

        result = await run_shell(
            f'curl -s --max-time 10 "{url}"',
            timeout=15,
        )

        if result.startswith("\u2705"):
            result = result[1:].strip()

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return f"Search returned non-JSON response for '{query}':\n{result[:1000]}"

        lines = []
        if data.get("Abstract"):
            lines.append(f"**Summary:** {data['Abstract']}")
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                lines.append(f"- {text}\n  {url}")

        if lines:
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)

        scrape_url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        scrape_result = await run_shell(
            f'curl -s --max-time 10 "{scrape_url}" | grep -oP \'<a rel="nofollow" class="result__a" href="[^"]*">[^<]*</a>\' | head -5',
            timeout=15,
        )

        if scrape_result.startswith("\u2705"):
            scrape_result = scrape_result[1:].strip()

        if scrape_result and "\u274c" not in scrape_result:
            return f"Search results for '{query}':\n\n{scrape_result}"

        return f"No results found for '{query}'. All search backends failed."

    except Exception as e:
        logger.exception("web search all backends failed", error=str(e))
        return f"Search error: {e}"
```

- [ ] **Step 4: Update the module docstring**

Replace line 3 of `src/tools/web_search.py`:

Old:
```python
"""
Web search using Perplexica/Vane (primary) — with DuckDuckGo and curl fallback.
"""
```

New:
```python
"""
Web search using DuckDuckGo + page content fetch (primary).
Fallback chain: ddgs+pages → Perplexica/Vane → SearXNG direct → curl.
"""
```

Also update the `web_search()` docstring to reflect the new order:

Old:
```python
    """
    Search the web using Perplexica/Vane (primary), with SearXNG direct and DuckDuckGo fallbacks.

    Fallback chain: Perplexica (45s, AI-synthesized) → SearXNG direct (12s, raw results)
    → DuckDuckGo package → curl/DuckDuckGo API.
```

New:
```python
    """
    Search the web using DuckDuckGo + page content fetch (primary).

    Fallback chain: ddgs + page fetch (fast, no GPU) → Perplexica/Vane (AI-synthesized)
    → SearXNG direct (raw results) → curl/DuckDuckGo API.
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_web_search_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tools/web_search.py tests/test_web_search_integration.py
git commit -m "feat(search): promote ddgs+pagefetch to primary, Perplexica becomes fallback"
```

---

### Task 5: End-to-end smoke test

**Files:**
- No code changes — just verification

- [ ] **Step 1: Run the full unit test suite**

Run: `pytest tests/test_web_search_integration.py tests/test_page_fetch.py -v`
Expected: All PASS.

- [ ] **Step 2: Run a live search test**

Run:
```bash
python -c "
import asyncio
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(web_search('best coffee machine 2026', max_results=5))
print(result[:2000])
print('---')
print(f'Total length: {len(result)} chars')
"
```

Expected: Output should contain numbered results with titles, snippets, URLs, and `---` separated page content for at least 1-2 results.

- [ ] **Step 3: Test with a factual query (snippets sufficient)**

Run:
```bash
python -c "
import asyncio
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(web_search('capital of Australia'))
print(result[:1000])
"
```

Expected: Results mentioning "Canberra".

- [ ] **Step 4: Test with a deep research query**

Run:
```bash
python -c "
import asyncio
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(web_search('how does transformer attention mechanism work'))
print(result[:2000])
"
```

Expected: Results with page content containing technical details beyond snippets.

- [ ] **Step 5: Verify import works from agent context**

Run:
```bash
python -c "from src.tools import web_search; print('web_search tool importable:', callable(web_search))"
```

Expected: `web_search tool importable: True`

- [ ] **Step 6: Commit (if any fixes were needed)**

```bash
git add -u
git commit -m "fix(search): address issues found in e2e smoke testing"
```

Only commit if changes were needed. Skip if everything passed clean.
