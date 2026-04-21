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
