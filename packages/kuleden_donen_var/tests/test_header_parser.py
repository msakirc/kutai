"""Tests for provider-specific rate limit header parsing."""
import time
from kuleden_donen_var.header_parser import parse_rate_limit_headers, RateLimitSnapshot


def test_snapshot_has_any_data_empty():
    snap = RateLimitSnapshot()
    assert snap.has_any_data() is False


def test_snapshot_has_any_data_with_rpm():
    snap = RateLimitSnapshot(rpm_limit=30)
    assert snap.has_any_data() is True


def test_openai_style_headers():
    headers = {
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-requests": "25",
        "x-ratelimit-reset-requests": "6s",
        "x-ratelimit-limit-tokens": "131072",
        "x-ratelimit-remaining-tokens": "100000",
        "x-ratelimit-reset-tokens": "12ms",
    }
    snap = parse_rate_limit_headers("groq", headers)
    assert snap is not None
    assert snap.rpm_limit == 30
    assert snap.rpm_remaining == 25
    assert snap.tpm_limit == 131072
    assert snap.tpm_remaining == 100000
    assert snap.rpm_reset_at is not None
    assert snap.rpm_reset_at > time.time()


def test_anthropic_headers():
    headers = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "48",
        "anthropic-ratelimit-requests-reset": "2026-01-27T12:00:30Z",
        "anthropic-ratelimit-tokens-limit": "80000",
        "anthropic-ratelimit-tokens-remaining": "75000",
        "anthropic-ratelimit-tokens-reset": "2026-01-27T12:00:30Z",
    }
    snap = parse_rate_limit_headers("anthropic", headers)
    assert snap is not None
    assert snap.rpm_limit == 50
    assert snap.rpm_remaining == 48
    assert snap.tpm_limit == 80000


def test_cerebras_daily_limits():
    headers = {
        "x-ratelimit-limit-tokens-minute": "131072",
        "x-ratelimit-remaining-tokens-minute": "100000",
        "x-ratelimit-reset-tokens-minute": "30.0",
        "x-ratelimit-limit-requests-day": "1000",
        "x-ratelimit-remaining-requests-day": "950",
        "x-ratelimit-reset-requests-day": "33011.382867",
    }
    snap = parse_rate_limit_headers("cerebras", headers)
    assert snap is not None
    assert snap.tpm_limit == 131072
    assert snap.rpd_limit == 1000
    assert snap.rpd_remaining == 950


def test_llm_provider_prefix_stripped():
    headers = {
        "llm_provider-x-ratelimit-limit-requests": "15",
        "llm_provider-x-ratelimit-remaining-requests": "10",
    }
    snap = parse_rate_limit_headers("gemini", headers)
    assert snap is not None
    assert snap.rpm_limit == 15
    assert snap.rpm_remaining == 10


def test_empty_headers_returns_none():
    assert parse_rate_limit_headers("openai", {}) is None
    assert parse_rate_limit_headers("openai", None) is None


def test_unknown_provider_uses_openai_style():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    snap = parse_rate_limit_headers("some_new_provider", headers)
    assert snap is not None
    assert snap.rpm_limit == 100
