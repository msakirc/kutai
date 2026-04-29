"""Adapter that converts KDV internal state to nerd_herd.CloudProviderState."""
import pytest

from kuleden_donen_var import KuledenConfig, KuledenDonenVar
from kuleden_donen_var.nerd_herd_adapter import (
    build_cloud_provider_state,
    make_state_getter,
)


@pytest.fixture
def kdv():
    return KuledenDonenVar(KuledenConfig())


def _seed(kdv, provider, model_id, **rpd):
    kdv.register(model_id, provider, rpm=60, tpm=100_000)
    state = kdv._rate_limiter.model_limits.get(model_id)
    assert state is not None, "register() must populate rate_limiter.model_limits"
    if "rpd_limit" in rpd:
        state.rpd_limit = rpd["rpd_limit"]
    if "rpd_remaining" in rpd:
        state.rpd_remaining = rpd["rpd_remaining"]
    if "rpd_reset_at" in rpd:
        state.rpd_reset_at = rpd["rpd_reset_at"]


def test_build_returns_none_for_unknown_provider(kdv):
    assert build_cloud_provider_state(kdv, "nobody") is None


def test_build_forwards_rpd_limits(kdv):
    _seed(kdv, "anthropic", "claude-sonnet-4-6",
          rpd_limit=30, rpd_remaining=22, rpd_reset_at=1_800_000_000.0)
    out = build_cloud_provider_state(kdv, "anthropic")
    assert out is not None
    assert out.provider == "anthropic"
    m = out.models["claude-sonnet-4-6"]
    assert m.limits.rpd.limit == 30
    assert m.limits.rpd.remaining == 22
    assert m.limits.rpd.reset_at == 1_800_000_000


def test_make_state_getter_adapts(kdv):
    _seed(kdv, "groq", "llama-3.1-70b",
          rpd_limit=500, rpd_remaining=100, rpd_reset_at=None)
    getter = make_state_getter(kdv)
    out = getter("groq")
    assert out.models["llama-3.1-70b"].limits.rpd.remaining == 100
    assert getter("unknown") is None


def test_build_handles_missing_rpd_fields(kdv):
    _seed(kdv, "x", "m")
    out = build_cloud_provider_state(kdv, "x")
    assert out is not None
    rpd = out.models["m"].limits.rpd
    assert rpd.limit is None
    assert rpd.remaining is None


def test_adapter_copies_tpm_cell_when_present():
    from kuleden_donen_var.kdv import KuledenDonenVar
    from kuleden_donen_var import KuledenConfig
    from kuleden_donen_var.nerd_herd_adapter import build_cloud_provider_state
    kdv = KuledenDonenVar(KuledenConfig())
    kdv.register("groq/llama", "groq", rpm=30, tpm=6000)
    # Seed rpd so we can also assert rpd cell (matches register API)
    state = kdv._rate_limiter.model_limits["groq/llama"]
    state.rpd_limit = 14_400
    state.rpd_remaining = 14_400
    cloud_state = build_cloud_provider_state(kdv, "groq")
    m = cloud_state.models["groq/llama"]
    assert m.limits.tpm.limit == 6000
    assert m.limits.rpm.limit == 30
    assert m.limits.rpd.limit == 14_400


def test_adapter_forwards_anthropic_token_splits_end_to_end(kdv):
    """Header → snapshot → state → adapter → matrix.itpm/otpm/itpd/otpd.

    Anthropic exposes input/output token sub-budgets which map to the
    matrix's split-axis cells. Without end-to-end plumbing, S2 burden
    only sees the combined tpm cell and misses an output-burst exhaustion
    on long-completion calls.
    """
    from kuleden_donen_var.header_parser import parse_rate_limit_headers
    kdv.register("anthropic/claude-sonnet-4-6", "anthropic", rpm=50, tpm=200_000)
    state = kdv._rate_limiter.model_limits["anthropic/claude-sonnet-4-6"]
    headers = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "48",
        "anthropic-ratelimit-tokens-limit": "200000",
        "anthropic-ratelimit-tokens-remaining": "180000",
        "anthropic-ratelimit-input-tokens-limit": "150000",
        "anthropic-ratelimit-input-tokens-remaining": "140000",
        "anthropic-ratelimit-output-tokens-limit": "50000",
        "anthropic-ratelimit-output-tokens-remaining": "40000",
        "anthropic-ratelimit-input-tokens-day-limit": "50000000",
        "anthropic-ratelimit-input-tokens-day-remaining": "47000000",
        "anthropic-ratelimit-output-tokens-day-limit": "10000000",
        "anthropic-ratelimit-output-tokens-day-remaining": "9000000",
    }
    snap = parse_rate_limit_headers("anthropic", headers)
    state.update_from_snapshot(snap)
    cloud_state = build_cloud_provider_state(kdv, "anthropic")
    m = cloud_state.models["anthropic/claude-sonnet-4-6"]
    # Combined cells
    assert m.limits.rpm.limit == 50
    assert m.limits.tpm.limit == 200_000
    # Split cells (the new piece)
    assert m.limits.itpm.limit == 150_000
    assert m.limits.itpm.remaining == 140_000
    assert m.limits.otpm.limit == 50_000
    assert m.limits.otpm.remaining == 40_000
    assert m.limits.itpd.limit == 50_000_000
    assert m.limits.itpd.remaining == 47_000_000
    assert m.limits.otpd.limit == 10_000_000
    assert m.limits.otpd.remaining == 9_000_000


def test_adapter_forwards_gemini_full_axes_end_to_end(kdv):
    """Gemini RPM + TPM + RPD + TPD all reach matrix cells."""
    from kuleden_donen_var.header_parser import parse_rate_limit_headers
    kdv.register("gemini/gemini-2.0-flash", "gemini", rpm=1000, tpm=1_000_000)
    state = kdv._rate_limiter.model_limits["gemini/gemini-2.0-flash"]
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
    state.update_from_snapshot(snap)
    cloud_state = build_cloud_provider_state(kdv, "gemini")
    m = cloud_state.models["gemini/gemini-2.0-flash"]
    assert m.limits.rpm.limit == 1000
    assert m.limits.tpm.limit == 1_000_000
    assert m.limits.rpd.limit == 50_000
    assert m.limits.rpd.remaining == 47_000
    assert m.limits.tpd.limit == 50_000_000
    assert m.limits.tpd.remaining == 48_000_000


def test_adapter_forwards_groq_daily_headers_end_to_end(kdv):
    """Header → snapshot → state → adapter → matrix.tpd / matrix.rpd.

    Validates the full pipeline works for Groq daily axes: parser
    populates snapshot, state.update_from_snapshot writes through,
    adapter forwards to matrix cells consumed by S1/S2/S9 signals.
    """
    from kuleden_donen_var.header_parser import parse_rate_limit_headers
    kdv.register("groq/llama-3.1-70b", "groq", rpm=1000, tpm=200_000)
    state = kdv._rate_limiter.model_limits["groq/llama-3.1-70b"]
    headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "950",
        "x-ratelimit-limit-tokens": "200000",
        "x-ratelimit-remaining-tokens": "180000",
        "x-ratelimit-limit-requests-day": "100000",
        "x-ratelimit-remaining-requests-day": "95000",
        "x-ratelimit-reset-requests-day": "3600s",
        "x-ratelimit-limit-tokens-day": "10000000",
        "x-ratelimit-remaining-tokens-day": "9500000",
        "x-ratelimit-reset-tokens-day": "3600s",
    }
    snap = parse_rate_limit_headers("groq", headers)
    state.update_from_snapshot(snap)
    cloud_state = build_cloud_provider_state(kdv, "groq")
    m = cloud_state.models["groq/llama-3.1-70b"]
    # Minute axis
    assert m.limits.rpm.limit == 1000
    assert m.limits.tpm.limit == 200_000
    # Daily axis (the new piece)
    assert m.limits.rpd.limit == 100_000
    assert m.limits.rpd.remaining == 95_000
    assert m.limits.tpd.limit == 10_000_000
    assert m.limits.tpd.remaining == 9_500_000
