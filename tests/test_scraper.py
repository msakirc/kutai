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
