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
    # Daily axes absent → snapshot leaves them None
    assert snap.rpd_limit is None
    assert snap.tpd_limit is None


def test_groq_with_daily_axes():
    """Groq paid-tier responses with both minute + daily axes populate
    all four cells. Validates RPM/TPM/RPD/TPD plumbing through to snapshot."""
    headers = {
        # Minute axis
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "950",
        "x-ratelimit-reset-requests": "30s",
        "x-ratelimit-limit-tokens": "200000",
        "x-ratelimit-remaining-tokens": "180000",
        "x-ratelimit-reset-tokens": "45s",
        # Daily axis
        "x-ratelimit-limit-requests-day": "100000",
        "x-ratelimit-remaining-requests-day": "95000",
        "x-ratelimit-reset-requests-day": "3600s",
        "x-ratelimit-limit-tokens-day": "10000000",
        "x-ratelimit-remaining-tokens-day": "9500000",
        "x-ratelimit-reset-tokens-day": "3600s",
    }
    snap = parse_rate_limit_headers("groq", headers)
    assert snap is not None
    assert snap.rpm_limit == 1000
    assert snap.rpm_remaining == 950
    assert snap.tpm_limit == 200000
    assert snap.tpm_remaining == 180000
    assert snap.rpd_limit == 100000
    assert snap.rpd_remaining == 95000
    assert snap.rpd_reset_at is not None
    assert snap.rpd_reset_at > time.time()
    assert snap.tpd_limit == 10000000
    assert snap.tpd_remaining == 9500000
    assert snap.tpd_reset_at is not None


def test_groq_daily_only():
    """Daily axes alone (no minute) — useful for tier checks."""
    headers = {
        "x-ratelimit-limit-tokens-day": "5000000",
        "x-ratelimit-remaining-tokens-day": "4800000",
        "x-ratelimit-reset-tokens-day": "7200s",
    }
    snap = parse_rate_limit_headers("groq", headers)
    assert snap is not None
    assert snap.tpd_limit == 5000000
    assert snap.tpd_remaining == 4800000
    assert snap.rpm_limit is None
    assert snap.tpm_limit is None


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
    # Combined-only response → input/output splits stay None
    assert snap.itpm_limit is None
    assert snap.otpm_limit is None


def test_anthropic_input_output_token_splits():
    """Anthropic paid plans expose separate input/output token sub-budgets.
    These map to itpm / otpm matrix cells (and itpd / otpd on tiers
    that include the day suffix)."""
    headers = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "48",
        "anthropic-ratelimit-tokens-limit": "200000",
        "anthropic-ratelimit-tokens-remaining": "180000",
        "anthropic-ratelimit-input-tokens-limit": "150000",
        "anthropic-ratelimit-input-tokens-remaining": "140000",
        "anthropic-ratelimit-input-tokens-reset": "2026-01-27T12:00:30Z",
        "anthropic-ratelimit-output-tokens-limit": "50000",
        "anthropic-ratelimit-output-tokens-remaining": "40000",
        "anthropic-ratelimit-output-tokens-reset": "2026-01-27T12:00:30Z",
    }
    snap = parse_rate_limit_headers("anthropic", headers)
    assert snap is not None
    assert snap.itpm_limit == 150000
    assert snap.itpm_remaining == 140000
    assert snap.otpm_limit == 50000
    assert snap.otpm_remaining == 40000
    # Combined still parsed alongside the splits
    assert snap.tpm_limit == 200000


def test_anthropic_input_output_token_day_axes():
    """Day-axis variants (some Anthropic tiers) populate itpd / otpd."""
    headers = {
        "anthropic-ratelimit-input-tokens-day-limit": "50000000",
        "anthropic-ratelimit-input-tokens-day-remaining": "47000000",
        "anthropic-ratelimit-input-tokens-day-reset": "2026-01-28T00:00:00Z",
        "anthropic-ratelimit-output-tokens-day-limit": "10000000",
        "anthropic-ratelimit-output-tokens-day-remaining": "9000000",
        "anthropic-ratelimit-output-tokens-day-reset": "2026-01-28T00:00:00Z",
    }
    snap = parse_rate_limit_headers("anthropic", headers)
    assert snap is not None
    assert snap.itpd_limit == 50_000_000
    assert snap.itpd_remaining == 47_000_000
    assert snap.otpd_limit == 10_000_000
    assert snap.otpd_remaining == 9_000_000
    assert snap.itpd_reset_at is not None


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


def test_cerebras_full_axis_set():
    """Cerebras success responses expose minute + hour + day buckets on both
    request and token axes. Hour bucket has no tracker field; minute + day
    must populate rpm/tpm/rpd/tpd. Live header sample (no reset values)."""
    headers = {
        "x-ratelimit-remaining-requests-minute": "29",
        "x-ratelimit-remaining-requests-hour":   "899",
        "x-ratelimit-remaining-requests-day":    "14399",
        "x-ratelimit-remaining-tokens-minute":   "59995",
        "x-ratelimit-remaining-tokens-hour":     "999995",
        "x-ratelimit-remaining-tokens-day":      "999995",
        "x-ratelimit-limit-requests-minute":     "30",
        "x-ratelimit-limit-requests-hour":       "900",
        "x-ratelimit-limit-requests-day":        "14400",
        "x-ratelimit-limit-tokens-minute":       "60000",
        "x-ratelimit-limit-tokens-hour":         "1000000",
        "x-ratelimit-limit-tokens-day":          "1000000",
    }
    snap = parse_rate_limit_headers("cerebras", headers)
    assert snap is not None
    assert snap.rpm_limit == 30
    assert snap.rpm_remaining == 29
    assert snap.tpm_limit == 60_000
    assert snap.tpm_remaining == 59_995
    assert snap.rpd_limit == 14_400
    assert snap.rpd_remaining == 14_399
    assert snap.tpd_limit == 1_000_000
    assert snap.tpd_remaining == 999_995


def test_retry_after_numeric_seconds():
    """RFC 7231 delta-seconds form — most LLM APIs use this."""
    snap = parse_rate_limit_headers("cerebras", {"retry-after": "12"})
    assert snap is not None
    assert snap.retry_after_seconds == 12.0


def test_retry_after_float_seconds():
    snap = parse_rate_limit_headers("cerebras", {"retry-after": "12.5"})
    assert snap is not None
    assert snap.retry_after_seconds == 12.5


def test_retry_after_negative_clamped_to_zero():
    snap = parse_rate_limit_headers("cerebras", {"retry-after": "-3"})
    assert snap is not None
    assert snap.retry_after_seconds == 0.0


def test_retry_after_case_insensitive():
    snap = parse_rate_limit_headers("cerebras", {"Retry-After": "5"})
    assert snap is not None
    assert snap.retry_after_seconds == 5.0


def test_retry_after_litellm_prefix_stripped():
    """LiteLLM wraps provider headers under `llm_provider-` — central
    parser strips this prefix before dispatch, so retry-after works
    via the prefixed key too."""
    snap = parse_rate_limit_headers("cerebras", {"llm_provider-retry-after": "8"})
    assert snap is not None
    assert snap.retry_after_seconds == 8.0


def test_retry_after_only_still_returns_snapshot():
    """When Retry-After is the ONLY rate-limit signal, the snapshot must
    survive — has_any_data() includes retry_after_seconds. This covers
    degraded 429s where x-ratelimit-* headers are absent."""
    snap = parse_rate_limit_headers("groq", {"retry-after": "30"})
    assert snap is not None
    assert snap.retry_after_seconds == 30.0


def test_retry_after_alongside_bucket_headers():
    """Both signals coexist — retry-after fills its dedicated field
    without disturbing bucket parsing."""
    snap = parse_rate_limit_headers("cerebras", {
        "retry-after": "7",
        "x-ratelimit-limit-requests-minute": "30",
        "x-ratelimit-remaining-requests-minute": "0",
    })
    assert snap is not None
    assert snap.retry_after_seconds == 7.0
    assert snap.rpm_limit == 30
    assert snap.rpm_remaining == 0


def test_retry_after_imf_fixdate():
    """Defensive: RFC 7231 also permits HTTP-date format. Compute
    seconds-from-now relative to a future date."""
    import time as _t
    future = _t.gmtime(_t.time() + 60)
    fixdate = _t.strftime("%a, %d %b %Y %H:%M:%S GMT", future)
    snap = parse_rate_limit_headers("cerebras", {"retry-after": fixdate})
    assert snap is not None
    # Allow a little wall-clock drift between header gen and parse.
    assert 50.0 <= snap.retry_after_seconds <= 70.0


def test_retry_after_malformed_returns_none():
    """Garbage value drops the field but bucket parsing still wins."""
    snap = parse_rate_limit_headers("cerebras", {
        "retry-after": "soon",
        "x-ratelimit-limit-requests-minute": "30",
    })
    assert snap is not None
    assert snap.retry_after_seconds is None
    assert snap.rpm_limit == 30


def test_cerebras_429_with_reset_seconds():
    """On 429, Cerebras emits the same headers plus reset values as float
    seconds. Parser must coerce them to absolute reset timestamps."""
    headers = {
        "x-ratelimit-limit-requests-minute":     "30",
        "x-ratelimit-remaining-requests-minute": "0",
        "x-ratelimit-reset-requests-minute":     "12.5",
        "x-ratelimit-limit-tokens-minute":       "60000",
        "x-ratelimit-remaining-tokens-minute":   "0",
        "x-ratelimit-reset-tokens-minute":       "12.5",
    }
    snap = parse_rate_limit_headers("cerebras", headers)
    assert snap is not None
    assert snap.rpm_remaining == 0
    assert snap.rpm_reset_at is not None
    assert snap.tpm_reset_at is not None


def test_llm_provider_prefix_stripped():
    headers = {
        "llm_provider-x-ratelimit-limit-requests": "15",
        "llm_provider-x-ratelimit-remaining-requests": "10",
    }
    snap = parse_rate_limit_headers("gemini", headers)
    assert snap is not None
    assert snap.rpm_limit == 15
    assert snap.rpm_remaining == 10


def test_gemini_full_axis_set():
    """Gemini paid tier exposes RPM + TPM + RPD + TPD; all four axes
    must populate so admission gate can see daily exhaustion before 429."""
    headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "950",
        "x-ratelimit-limit-tokens": "1000000",
        "x-ratelimit-remaining-tokens": "900000",
        "x-ratelimit-limit-requests-day": "50000",
        "x-ratelimit-remaining-requests-day": "47000",
        "x-ratelimit-reset-requests-day": "7200s",
        "x-ratelimit-limit-tokens-day": "50000000",
        "x-ratelimit-remaining-tokens-day": "48000000",
        "x-ratelimit-reset-tokens-day": "7200s",
    }
    snap = parse_rate_limit_headers("gemini", headers)
    assert snap is not None
    assert snap.rpm_limit == 1000
    assert snap.tpm_limit == 1_000_000
    assert snap.rpd_limit == 50_000
    assert snap.rpd_remaining == 47_000
    assert snap.tpd_limit == 50_000_000
    assert snap.tpd_remaining == 48_000_000
    assert snap.tpd_reset_at is not None


def test_empty_headers_returns_none():
    assert parse_rate_limit_headers("openai", {}) is None


# ── 429 body parser (gemini RESOURCE_EXHAUSTED) ──────────────────────────


def test_parse_429_body_gemini_limit_zero():
    """Gemini RESOURCE_EXHAUSTED with limit:0 → snapshot writes
    rpd_remaining=0, rpd_reset_at=now+retry_secs so S1 depletion fires
    negative pressure on next selection."""
    from kuleden_donen_var.header_parser import parse_429_body
    msg = (
        "GeminiException BadRequestError - 429 RESOURCE_EXHAUSTED. "
        "Quota exceeded for metric: generativelanguage.googleapis.com/"
        "generate_content_free_tier_requests, limit: 0, model: "
        "gemini-2.0-flash. Please retry in 40.121s."
    )
    snap = parse_429_body("gemini", msg)
    assert snap is not None
    assert snap.rpd_remaining == 0
    assert snap.rpd_reset_at is not None
    # retry_secs of 40s clamps to >=60s floor
    delta = snap.rpd_reset_at - time.time()
    assert 50 <= delta <= 80


def test_parse_429_body_gemini_picks_smallest_limit():
    """Multi-line body lists multiple quotaMetric lines; smallest limit
    is the binding one."""
    from kuleden_donen_var.header_parser import parse_429_body
    msg = (
        "RESOURCE_EXHAUSTED Quota exceeded for metric: ...rpm, limit: 5, "
        "Quota exceeded for metric: ...rpd, limit: 0, "
        "Quota exceeded for metric: ...input_tokens, limit: 250000. "
        "Please retry in 86400s."
    )
    snap = parse_429_body("gemini", msg)
    assert snap is not None
    # 0 limit becomes 1 marker (so cell isn't dropped)
    assert snap.rpd_limit == 1


def test_parse_429_body_no_match_returns_none():
    """Non-429 messages, non-gemini providers, and empty messages all
    short-circuit without producing a synthetic snapshot."""
    from kuleden_donen_var.header_parser import parse_429_body
    assert parse_429_body("gemini", "") is None
    assert parse_429_body("gemini", "Connection error") is None
    assert parse_429_body("groq", "RESOURCE_EXHAUSTED limit: 0") is None  # groq path not impl
    assert parse_429_body("openai", "rate_limit_exceeded") is None


def test_parse_429_body_default_24h_when_no_retry_hint():
    """Body without 'Please retry in Xs' falls back to 86400s window."""
    from kuleden_donen_var.header_parser import parse_429_body
    msg = "Quota exceeded for metric: ...free_tier_requests, limit: 0"
    snap = parse_429_body("gemini", msg)
    assert snap is not None
    delta = snap.rpd_reset_at - time.time()
    # 86400s default (24h)
    assert 86_300 <= delta <= 86_500
    assert parse_rate_limit_headers("openai", None) is None


def test_unknown_provider_uses_openai_style():
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    snap = parse_rate_limit_headers("some_new_provider", headers)
    assert snap is not None
    assert snap.rpm_limit == 100
