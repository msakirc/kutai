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
