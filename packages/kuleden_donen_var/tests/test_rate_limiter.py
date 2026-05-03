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


def test_state_retry_after_raises_rpm_reset_floor():
    """Provider Retry-After header sets a hard 'do not call before T'
    floor on _header_rpm_reset_at, also forcing remaining=0. Stricter
    signal wins — a 5s bucket reset cannot override a 30s retry-after."""
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state._header_rpm_reset_at = time.time() + 5.0
    state._header_rpm_remaining = 25
    snap = RateLimitSnapshot(retry_after_seconds=30.0)
    state.update_from_snapshot(snap)
    assert state._header_rpm_remaining == 0
    # Must have moved to ~now+30s, not stuck at now+5s
    assert state._header_rpm_reset_at >= time.time() + 25.0


def test_state_retry_after_does_not_lower_existing_floor():
    """When existing reset_at is later than now+retry_after, keep it.
    The bucket may legitimately recover later than the provider hint."""
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    far_future = time.time() + 120.0
    state._header_rpm_reset_at = far_future
    snap = RateLimitSnapshot(retry_after_seconds=10.0)
    state.update_from_snapshot(snap)
    assert state._header_rpm_reset_at == far_future
    assert state._header_rpm_remaining == 0  # still forced to 0


def test_state_retry_after_zero_is_noop():
    """retry_after_seconds=0 is a degenerate hint (provider says 'now ok'
    after a 429 — uncommon but possible). Treat as no signal: don't
    install a floor at exactly `now`, don't stomp remaining."""
    state = RateLimitState(rpm_limit=30, tpm_limit=100000)
    state._header_rpm_remaining = 25
    snap = RateLimitSnapshot(retry_after_seconds=0.0)
    state.update_from_snapshot(snap)
    assert state._header_rpm_remaining == 25  # untouched


def test_cold_start_backfill_appends_ghost_timestamps():
    """First authoritative snapshot post-restart: provider says 25/30
    consumed, local has 1 (this call's record_request). Backfill 24
    ghost entries so subsequent admissions see current_rpm=25 and the
    gate denies until the bucket actually clears."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    state._request_timestamps.append(time.time())  # the first call
    snap = RateLimitSnapshot(rpm_limit=30, rpm_remaining=5, rpm_reset_at=time.time() + 10.0)
    state.update_from_snapshot(snap)
    # 30 - 5 = 25 consumed, local had 1, gap = 24
    assert len(state._request_timestamps) == 25
    assert state._backfilled is True


def test_cold_start_backfill_no_op_when_local_already_matches():
    """Steady-state: provider remaining matches local count. No ghosts."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    now = time.time()
    state._request_timestamps.extend([now] * 5)
    snap = RateLimitSnapshot(rpm_limit=30, rpm_remaining=25)  # 5 consumed
    state.update_from_snapshot(snap)
    assert len(state._request_timestamps) == 5
    assert state._backfilled is True


def test_cold_start_backfill_no_op_when_local_exceeds_provider():
    """Provider remaining > local-derived headroom (clock skew or our own
    state ahead). Don't subtract — leave local alone, mark backfilled."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    now = time.time()
    state._request_timestamps.extend([now] * 10)
    snap = RateLimitSnapshot(rpm_limit=30, rpm_remaining=25)  # only 5 consumed
    state.update_from_snapshot(snap)
    assert len(state._request_timestamps) == 10
    assert state._backfilled is True


def test_cold_start_backfill_only_fires_once():
    """Mid-runtime re-saturation must NOT trigger another backfill —
    record_attempt is bumping _request_timestamps in real time, and a
    second backfill would double-count its own slots."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    state._request_timestamps.append(time.time())
    snap1 = RateLimitSnapshot(rpm_limit=30, rpm_remaining=20)  # 10 consumed
    state.update_from_snapshot(snap1)
    assert len(state._request_timestamps) == 10
    # Second snapshot showing further drain — must not backfill again.
    snap2 = RateLimitSnapshot(rpm_limit=30, rpm_remaining=5)  # 25 consumed
    state.update_from_snapshot(snap2)
    assert len(state._request_timestamps) == 10  # unchanged


def test_cold_start_backfill_ghost_age_uses_reset_at():
    """Ghost timestamps should age out exactly when provider's bucket
    clears. With reset_at = now+10s, ghost age = 60-10 = 50s — they'll
    expire 10s from now via _cleanup."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    now = time.time()
    snap = RateLimitSnapshot(rpm_limit=30, rpm_remaining=25, rpm_reset_at=now + 10.0)
    state.update_from_snapshot(snap)
    # 5 ghosts (consumed=5, local=0)
    assert len(state._request_timestamps) == 5
    for ts in state._request_timestamps:
        # age = now - ts; expected ~50s (60 - 10)
        age = now - ts
        assert 49.0 < age < 51.0


def test_cold_start_backfill_default_reset_when_no_reset_at():
    """Cerebras success responses don't return reset values. Fall back
    to 30s (mid-window): ghost age = 30s."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    now = time.time()
    snap = RateLimitSnapshot(rpm_limit=30, rpm_remaining=25)  # no reset_at
    state.update_from_snapshot(snap)
    for ts in state._request_timestamps:
        age = now - ts
        assert 29.0 < age < 31.0


def test_cold_start_backfill_tpm_mirror():
    """Token axis backfilled the same way: single _token_log entry
    carrying the gap."""
    state = RateLimitState(rpm_limit=30, tpm_limit=60_000)
    snap = RateLimitSnapshot(
        tpm_limit=60_000, tpm_remaining=10_000,  # 50_000 consumed
    )
    state.update_from_snapshot(snap)
    # local started at 0 tokens, gap = 50_000
    total = sum(c for _, c in state._token_log)
    assert total == 50_000


def test_is_rpm_cooldown_returns_true_when_floor_active():
    """RateLimitManager.is_rpm_cooldown reads raw _header_* fields directly,
    bypassing the freshness-windowed rpm_remaining property. Required for
    retry-after horizons longer than 5s (most cases)."""
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=60_000)
    state = mgr.model_limits["cerebras/llama-8b"]
    state._header_rpm_remaining = 0
    state._header_rpm_reset_at = time.time() + 30.0
    assert mgr.is_rpm_cooldown("cerebras/llama-8b") is True


def test_is_rpm_cooldown_false_when_floor_passed():
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=60_000)
    state = mgr.model_limits["cerebras/llama-8b"]
    state._header_rpm_remaining = 0
    state._header_rpm_reset_at = time.time() - 1.0  # already past
    assert mgr.is_rpm_cooldown("cerebras/llama-8b") is False


def test_is_rpm_cooldown_false_when_remaining_positive():
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=60_000)
    state = mgr.model_limits["cerebras/llama-8b"]
    state._header_rpm_remaining = 5
    state._header_rpm_reset_at = time.time() + 30.0
    assert mgr.is_rpm_cooldown("cerebras/llama-8b") is False


def test_is_rpm_cooldown_false_for_unknown_model():
    mgr = RateLimitManager()
    assert mgr.is_rpm_cooldown("nonexistent/model") is False


def test_is_rpm_cooldown_survives_freshness_window_expiry():
    """The whole point of bypassing rpm_remaining property: cooldown must
    persist past the 5s freshness window. Set _last_header_update far
    enough in the past that the property would fall back to sliding
    window — is_rpm_cooldown still returns True via raw field read."""
    mgr = RateLimitManager()
    mgr.register_model("cerebras/llama-8b", "cerebras", rpm=30, tpm=60_000)
    state = mgr.model_limits["cerebras/llama-8b"]
    state._header_rpm_remaining = 0
    state._header_rpm_reset_at = time.time() + 30.0
    state._last_header_update = time.time() - 60.0  # well past 5s freshness
    assert mgr.is_rpm_cooldown("cerebras/llama-8b") is True


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


def test_rpm_remaining_falls_back_to_sliding_window():
    """Pool pressure depends on matrix.rpm.remaining being populated.
    Gemini-class providers emit nothing in response headers — without
    a sliding-window fallback, S1 sees None, treats as 0 via the
    exhausted-neutral path on time_bucketed pools, gives no signal,
    selector keeps picking the saturated model, hits 429 by surprise.

    Property must return rpm_limit when no calls + no headers (full
    bucket), shrink as calls land, bottom at 0."""
    state = RateLimitState(rpm_limit=5, tpm_limit=250_000)
    # Cold: full bucket
    assert state.rpm_remaining == 5
    # Three calls in this minute → 2 left
    state.record_attempt = lambda: state._request_timestamps.append(time.time())
    state._request_timestamps.append(time.time())
    state._request_timestamps.append(time.time())
    state._request_timestamps.append(time.time())
    assert state.rpm_remaining == 2
    # Saturate
    for _ in range(10):
        state._request_timestamps.append(time.time())
    assert state.rpm_remaining == 0


def test_rpm_remaining_prefers_fresh_header():
    """When provider headers ARE present (Groq/Anthropic), authoritative
    header value wins over the local sliding-window estimate."""
    state = RateLimitState(rpm_limit=100, tpm_limit=200_000)
    # Local count says 90 left
    state._request_timestamps.append(time.time())
    assert state.rpm_remaining == 99
    # But provider header says 50 left and is fresh
    state._header_rpm_remaining = 50
    state._last_header_update = time.time()
    assert state.rpm_remaining == 50
    # Stale header → fall back to local count
    state._last_header_update = time.time() - 60
    assert state.rpm_remaining == 99


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
