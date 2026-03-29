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
