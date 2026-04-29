# packages/kuleden_donen_var/tests/test_rate_limiter.py
"""Tests for two-tier rate limiting."""
import time
import pytest
from kuleden_donen_var.rate_limiter import RateLimitState, RateLimitManager
from kuleden_donen_var.header_parser import RateLimitSnapshot


# -- RateLimitState --

def test_state_initially_has_capacity():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.has_capacity() is True
    assert state.current_rpm == 0
    assert state.current_tpm == 0


def test_state_headroom():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.rpm_headroom == 30
    assert state.tpm_headroom == 100000


def test_state_utilization_initially_zero():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    assert state.utilization_pct() == 0.0


def test_state_record_tokens():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.record_tokens(50000)
    assert state.current_tpm == 50000
    assert state.utilization_pct() == 50.0


def test_state_429_reduces_limits():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.record_429()
    assert state.rpm_limit < 30
    assert state.tpm_limit < 100000


def test_state_429_floors_at_half_original():
    state = RateLimitState(rpm_limit=10, tpm_limit=10000)
    for _ in range(20):
        state.record_429()
    assert state.rpm_limit >= 5  # 50% of original
    assert state.tpm_limit >= 5000


def test_state_update_from_snapshot():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    snap = RateLimitSnapshot(rpm_limit=60, rpm_remaining=55)
    state.update_from_snapshot(snap)
    assert state.rpm_limit == 60
    assert state._header_rpm_remaining == 55


def test_state_update_from_snapshot_tpd():
    """tpd_* fields on the snapshot land on RateLimitState; nerd_herd
    adapter reads them via getattr to populate the matrix's tpd cell."""
    state = RateLimitState(rpm_limit=1000, tpm_limit=200000)
    reset_at = time.time() + 3600
    snap = RateLimitSnapshot(
        tpd_limit=10_000_000, tpd_remaining=9_500_000, tpd_reset_at=reset_at,
    )
    state.update_from_snapshot(snap)
    assert state.tpd_limit == 10_000_000
    assert state.tpd_remaining == 9_500_000
    assert state.tpd_reset_at == reset_at


def test_state_daily_limit_exhaustion():
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    assert state.has_capacity() is False


# -- RateLimitManager --

def test_manager_register_and_has_capacity():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    assert mgr.has_capacity("groq/llama-8b", "groq") is True


def test_manager_record_tokens():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_tokens("groq/llama-8b", "groq", 50000)
    util = mgr.get_utilization("groq/llama-8b")
    assert util > 0


def test_manager_record_429():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_429("groq/llama-8b", "groq")
    state = mgr.model_limits["groq/llama-8b"]
    assert state.rpm_limit < 30


def test_manager_update_from_headers():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    snap = RateLimitSnapshot(rpm_limit=60, rpm_remaining=55)
    mgr.update_from_headers("groq/llama-8b", "groq", snap)
    state = mgr.model_limits["groq/llama-8b"]
    assert state.rpm_limit == 60


def test_manager_daily_exhausted():
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=131072)
    state = mgr.model_limits["cerebras/llama-8b"]
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    assert mgr.is_daily_exhausted("cerebras/llama-8b") is True


def test_manager_provider_aggregate():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072,
                        provider_aggregate_rpm=100, provider_aggregate_tpm=500000)
    mgr.register_model("groq/mixtral", "groq", rpm=30, tpm=131072)
    assert "groq" in mgr._provider_limits
    prov_util = mgr.get_provider_utilization("groq")
    assert prov_util == 0.0


def test_manager_restore_limits():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    mgr.record_429("groq/llama-8b", "groq")
    state = mgr.model_limits["groq/llama-8b"]
    reduced_rpm = state.rpm_limit
    state._last_429_at = time.time() - 700  # 11+ minutes ago
    mgr.restore_limits()
    assert state.rpm_limit > reduced_rpm


def test_manager_unregistered_model_has_capacity():
    mgr = RateLimitManager()
    assert mgr.has_capacity("unknown/model", "unknown") is True


@pytest.mark.asyncio
async def test_manager_wait_and_acquire_no_wait():
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    waited = await mgr.wait_and_acquire("groq/llama-8b", "groq")
    assert waited == 0.0


def test_manager_get_status():
    mgr = RateLimitManager()
    mgr.register_model(
        "groq/llama-8b", "groq",
        rpm=30, tpm=131072,
        provider_aggregate_rpm=300, provider_aggregate_tpm=500_000,
    )
    status = mgr.get_status()
    assert "groq/llama-8b" in status["models"]
    assert "groq" in status["providers"]


def test_register_skips_provider_state_when_aggregate_unset():
    """Provider state is created only when an aggregate is supplied.

    Without aggregates the per-model buckets gate alone, and
    has_capacity short-circuits via the `if provider_state else True` guard.
    """
    mgr = RateLimitManager()
    mgr.register_model("groq/llama-8b", "groq", rpm=30, tpm=131072)
    assert "groq" not in mgr._provider_limits
    assert mgr.has_capacity("groq/llama-8b", "groq") is True
    # Subsequent register on same provider WITH aggregate creates the entry.
    mgr.register_model(
        "groq/llama-70b", "groq",
        rpm=30, tpm=12000,
        provider_aggregate_rpm=300, provider_aggregate_tpm=500_000,
    )
    assert "groq" in mgr._provider_limits
