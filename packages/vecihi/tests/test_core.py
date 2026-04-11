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
