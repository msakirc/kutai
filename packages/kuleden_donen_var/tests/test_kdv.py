"""Tests for KuledenDonenVar main class."""
import time
import pytest
from kuleden_donen_var import KuledenDonenVar, KuledenConfig
from kuleden_donen_var.config import CapacityEvent, PreCallResult


@pytest.fixture
def events():
    return []


@pytest.fixture
def kdv(events):
    cfg = KuledenConfig(
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown_seconds=0.5,
        on_capacity_change=lambda evt: events.append(evt),
    )
    return KuledenDonenVar(cfg)


@pytest.fixture
def kdv_with_model(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    return kdv


# -- register --

def test_register_model(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    status = kdv.status
    assert "groq" in status
    assert "groq/llama-8b" in status["groq"].models


def test_register_multiple_models_same_provider(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    kdv.register("groq/mixtral", "groq", rpm=30, tpm=131072)
    assert len(kdv.status["groq"].models) == 2


def test_register_with_provider_aggregate(kdv):
    kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072,
                 provider_aggregate_rpm=100, provider_aggregate_tpm=500000)
    assert "groq" in kdv.status


# -- pre_call --

def test_pre_call_allowed(kdv_with_model):
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is True
    assert result.wait_seconds == 0.0
    assert result.daily_exhausted is False


def test_pre_call_circuit_breaker_blocks(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False


def test_pre_call_daily_exhausted(kdv_with_model):
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() + 3600
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False
    assert result.daily_exhausted is True


# -- post_call --

def test_record_attempt_bumps_rpm(kdv_with_model):
    """Per-minute request counter is bumped at admission time by
    record_attempt — NOT in post_call. post_call only adds tokens.

    This contract was inverted in the original implementation
    (post_call bumped RPM) and changed when failed-call quota visibility
    was added. Now record_attempt is the single bookkeeping site for
    request counting, regardless of call outcome.
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.current_rpm == 0
    kdv_with_model.record_attempt("groq/llama-8b", "groq")
    assert state.current_rpm == 1
    kdv_with_model.record_attempt("groq/llama-8b", "groq")
    assert state.current_rpm == 2


def test_post_call_does_not_double_bump_rpm(kdv_with_model):
    """post_call must NOT bump RPM — record_attempt already counted at
    admission. Without this guarantee, every successful call would be
    counted twice.
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    kdv_with_model.record_attempt("groq/llama-8b", "groq")
    assert state.current_rpm == 1
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=1000)
    assert state.current_rpm == 1  # unchanged


def test_record_attempt_reserves_tpm(kdv_with_model):
    """Concurrent admissions on a tight tpm budget must see each other's
    reservations. Without provisional reservation, both admissions pass
    has_capacity simultaneously and the provider returns 429.
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.tpm_limit = 6000  # Tighten to free-tier groq qwen3-32b shape.
    state._original_tpm = 6000
    # First admission reserves 5000.
    kdv_with_model.record_attempt(
        "groq/llama-8b", "groq", estimated_tokens=5000,
    )
    assert state.current_tpm == 5000
    # Second admission with est=2000 should now see headroom = 1000 < 2000
    # and be refused.
    assert state.has_capacity(estimated_tokens=2000) is False
    # Same model with est=1000 still fits.
    assert state.has_capacity(estimated_tokens=1000) is True


def test_post_call_corrects_reservation_to_actual(kdv_with_model):
    """post_call records (actual - reserved) so the running TPM converges
    to real usage. Reserved 5000 then actual 4200 → token_log nets to 4200.
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    kdv_with_model.record_attempt(
        "groq/llama-8b", "groq", estimated_tokens=5000,
    )
    assert state.current_tpm == 5000
    kdv_with_model.post_call(
        "groq/llama-8b", "groq",
        headers={}, token_count=4200, reserved_tokens=5000,
    )
    # token_log entries: +5000, +(4200-5000)=-800. sum = 4200.
    assert state.current_tpm == 4200


def test_release_reservation_rolls_back(kdv_with_model):
    """Failed call: full reservation is rolled back so subsequent calls
    don't see a phantom 60s reservation against the bucket.
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    kdv_with_model.record_attempt(
        "groq/llama-8b", "groq", estimated_tokens=5000,
    )
    assert state.current_tpm == 5000
    kdv_with_model.release_reservation("groq/llama-8b", "groq", reserved_tokens=5000)
    assert state.current_tpm == 0


def test_release_reservation_keeps_rpm_consumed(kdv_with_model):
    """Release rolls back TPM only — RPM stays consumed because the request
    slot WAS used (provider counted it regardless of outcome).
    """
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    kdv_with_model.record_attempt(
        "groq/llama-8b", "groq", estimated_tokens=5000,
    )
    assert state.current_rpm == 1
    kdv_with_model.release_reservation("groq/llama-8b", "groq", reserved_tokens=5000)
    assert state.current_rpm == 1  # unchanged
    assert state.current_tpm == 0  # rolled back


def test_post_call_records_tokens(kdv_with_model):
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=5000)
    util = kdv_with_model._rate_limiter.get_utilization("groq/llama-8b")
    assert util > 0


def test_post_call_parses_headers(kdv_with_model):
    headers = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "55",
    }
    kdv_with_model.post_call("groq/llama-8b", "groq", headers=headers, token_count=1000)
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpm_limit == 60


def test_post_call_records_circuit_breaker_success(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert kdv_with_model.pre_call("groq/llama-8b", "groq").allowed is False
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)
    assert kdv_with_model.pre_call("groq/llama-8b", "groq").allowed is True


# -- record_failure --

def test_record_failure_rate_limit(kdv_with_model, events):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpm_limit < 30
    assert any(e.event_type == "limit_hit" for e in events)


def test_record_failure_server_error_trips_breaker(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert any(e.event_type == "circuit_breaker_tripped" for e in events)


def test_record_failure_timeout_trips_breaker(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "timeout")
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is False


# -- reliability window (S10) must NOT see capacity rejections --
# A 429 / daily-exhaustion is a CAPACITY condition (recoverable, already gated
# by record_429 + mark_daily_exhausted). Counting it as a model-QUALITY failure
# craters recent_success_rate → S10 = -1.0, and provider_prior_rate spreads it
# to healthy full-quota siblings → phantom uniform -1.0 pressure (live outage
# 2026-06-17). Only genuine quality failures may enter the outcome window.

def test_rate_limited_not_recorded_as_reliability_failure(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limited")
    assert kdv_with_model.recent_samples_n("groq/llama-8b") == 0


def test_rate_limit_legacy_not_recorded_as_reliability_failure(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    assert kdv_with_model.recent_samples_n("groq/llama-8b") == 0


def test_daily_exhausted_not_recorded_as_reliability_failure(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "daily_exhausted")
    assert kdv_with_model.recent_samples_n("groq/llama-8b") == 0


def test_server_error_recorded_as_reliability_failure(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert kdv_with_model.recent_samples_n("groq/llama-8b") == 1


def test_rate_limit_still_tracks_capacity(kdv_with_model):
    """Excluding 429 from the reliability window must NOT disable the
    dedicated capacity machinery (adaptive rpm reduction via record_429)."""
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpm_limit < 30


def test_restore_limits(kdv_with_model):
    """Watchdog calls restore_limits to undo adaptive 429 reductions."""
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    reduced_rpm = state.rpm_limit
    state._last_429_at = time.time() - 700  # 11+ minutes ago
    kdv_with_model.restore_limits()
    assert state.rpm_limit > reduced_rpm


def test_record_failure_auth_ignored(kdv_with_model):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "auth")
    result = kdv_with_model.pre_call("groq/llama-8b", "groq")
    assert result.allowed is True


# -- status --

def test_status_empty(kdv):
    assert kdv.status == {}


def test_status_reflects_state(kdv_with_model):
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=5000)
    status = kdv_with_model.status
    assert status["groq"].provider == "groq"
    assert status["groq"].circuit_breaker_open is False
    model_status = status["groq"].models["groq/llama-8b"]
    assert model_status.model_id == "groq/llama-8b"


def test_status_circuit_breaker_reflected(kdv_with_model):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    assert kdv_with_model.status["groq"].circuit_breaker_open is True


# -- on_capacity_change --

def test_capacity_change_on_limit_hit(kdv_with_model, events):
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limit")
    assert len(events) >= 1
    assert events[-1].event_type == "limit_hit"
    assert events[-1].provider == "groq"
    assert events[-1].snapshot.provider == "groq"


def test_capacity_change_on_circuit_breaker_trip(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    cb_events = [e for e in events if e.event_type == "circuit_breaker_tripped"]
    assert len(cb_events) == 1


def test_capacity_change_on_circuit_breaker_reset(kdv_with_model, events):
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    events.clear()
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)
    cb_events = [e for e in events if e.event_type == "circuit_breaker_reset"]
    assert len(cb_events) == 1


# -- recent_success_rate (reliability tracking) --


def test_recent_success_rate_default_when_no_data(kdv_with_model):
    """Until MIN_SAMPLES outcomes accumulate, return 1.0 (no penalty)."""
    assert kdv_with_model.recent_success_rate("groq/llama-8b") == 1.0


def test_recent_success_rate_tracks_outcomes(kdv_with_model):
    """Mix of successes + failures lands in (0, 1)."""
    for _ in range(8):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(2):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    rate = kdv_with_model.recent_success_rate("groq/llama-8b")
    assert 0.7 < rate < 0.85, f"got {rate}"


def test_recent_success_rate_excludes_auth_failures(kdv_with_model):
    """auth_failure is a credentials problem, not a model-quality
    signal — outcome window must NOT include it."""
    for _ in range(6):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "auth_failure")
    assert kdv_with_model.recent_success_rate("groq/llama-8b") == 1.0


def test_recent_success_rate_excludes_quota_failures(kdv_with_model):
    """rate_limited / daily_exhausted are CAPACITY, not model quality. They
    are gated separately (record_429 → rpm cooldown; mark_daily_exhausted →
    daily veto) and must NOT enter the reliability window. Counting them here
    craters S10 and provider_prior_rate spreads the penalty to healthy
    full-quota siblings → phantom uniform -1.0 pressure (live outage
    2026-06-17). Supersedes the old "includes_quota_failures" contract."""
    for _ in range(6):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(7):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limited")
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "daily_exhausted")
    assert kdv_with_model.recent_success_rate("groq/llama-8b") == 1.0


def test_recent_success_rate_unknown_model(kdv):
    """Unregistered ids return 1.0 (no data → no penalty)."""
    assert kdv.recent_success_rate("nonexistent/model") == 1.0


# -- Daily-cap local decrement (preemptive daily-exhaustion) --


def test_record_attempt_decrements_rpd_when_limit_set(kdv_with_model):
    """Production 2026-05-02 root: gemini's free-tier 20-req/day cap
    was invisible to selector because the provider doesn't return
    x-ratelimit-remaining-requests-day headers — rpd_remaining stayed
    at registration value (20) until a 429 body parse flipped it.
    Decrement locally per call so daily exhaustion fires preemptively."""
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.rpd_limit = 20
    state.rpd_remaining = 20
    state.rpd_reset_at = None

    for i in range(15):
        kdv_with_model.record_attempt("groq/llama-8b", "groq")

    # 15 attempts → 5 remaining
    assert state.rpd_remaining == 5
    # is_daily_exhausted should NOT yet fire
    assert not kdv_with_model._rate_limiter.is_daily_exhausted("groq/llama-8b")

    for i in range(5):
        kdv_with_model.record_attempt("groq/llama-8b", "groq")

    # 20 attempts total → 0 remaining → daily exhausted
    assert state.rpd_remaining == 0
    assert kdv_with_model._rate_limiter.is_daily_exhausted("groq/llama-8b")


def test_record_attempt_no_op_when_rpd_limit_unset(kdv_with_model):
    """Models without a daily cap (most paid tiers) shouldn't get a
    fabricated rpd_remaining."""
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    assert state.rpd_limit is None
    for i in range(50):
        kdv_with_model.record_attempt("groq/llama-8b", "groq")
    assert state.rpd_limit is None
    assert state.rpd_remaining is None


def test_record_attempt_resets_rpd_window_after_24h(kdv_with_model):
    """When rpd_reset_at is in the past, refresh both remaining and
    the reset clock — covers the daily window rollover."""
    import time
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.rpd_limit = 20
    state.rpd_remaining = 0
    state.rpd_reset_at = time.time() - 1.0  # past

    kdv_with_model.record_attempt("groq/llama-8b", "groq")
    assert state.rpd_remaining == 19  # reset to 20, then decremented by 1
    assert state.rpd_reset_at > time.time()


def test_calendar_reset_aligns_with_utc_midnight(kdv_with_model):
    """Daily reset clock should land on next UTC midnight, not
    rolling-24h-from-first-call. Provider quota windows are typically
    calendar-aligned so this stays in sync."""
    import time, datetime as _dt
    state = kdv_with_model._rate_limiter.model_limits["groq/llama-8b"]
    state.rpd_limit = 20

    kdv_with_model.record_attempt("groq/llama-8b", "groq")
    reset_at = state.rpd_reset_at
    reset_dt = _dt.datetime.utcfromtimestamp(reset_at)
    assert reset_dt.hour == 0 and reset_dt.minute == 0


# -- Canary gate (uncertainty-period throttling) --


def test_canary_blocks_concurrent_admissions_until_first_returns(kdv_with_model):
    """Cold start: first call admits as canary, second is refused
    with reason=canary_in_flight until canary returns."""
    r1 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r1.allowed is True

    r2 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r2.allowed is False
    assert r2.reason == "canary_in_flight"


def test_canary_success_unlocks_subsequent_admissions(kdv_with_model):
    """Canary returns 200 → provider verified → bursts admitted."""
    kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)

    # Now multiple admissions allowed.
    for _ in range(3):
        r = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
        assert r.allowed is True


def test_canary_failure_re_locks_provider(kdv_with_model):
    """Canary fails → provider goes back to unverified → next admission
    becomes the new canary, others held."""
    kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")

    # First call: admitted as new canary.
    r1 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r1.allowed is True
    # Second call: blocked by canary gate.
    r2 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r2.allowed is False
    assert r2.reason == "canary_in_flight"


def test_canary_failure_during_burst_re_locks_provider(kdv_with_model):
    """Steady-state defense: provider was verified, then a call fails.
    Subsequent admissions go back through canary gate to validate
    quota didn't run out mid-burst."""
    # Verify provider.
    kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=100)
    # Verify burst flows.
    r = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r.allowed is True

    # A failure lands.
    kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limited")

    # Next admission: canary mode. First admits, second blocks.
    r1 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r1.allowed is True
    r2 = kdv_with_model.pre_call("groq/llama-8b", "groq", estimated_tokens=10)
    assert r2.allowed is False
    assert r2.reason == "canary_in_flight"


# -- provider_prior_rate (Step 6, 2026-05-04) --


def test_provider_prior_returns_none_when_no_data(kdv_with_model):
    """Cold start: no outcomes anywhere → prior is None (caller falls
    back to neutral)."""
    assert kdv_with_model.provider_prior_rate("groq") is None


def test_provider_prior_returns_none_when_below_min_samples(kdv_with_model):
    """Below min_samples, prior is None — under-sampled aggregate is
    not yet trustworthy."""
    kdv_with_model.post_call("groq/llama-8b", "groq", headers={},
                             token_count=10)
    assert kdv_with_model.provider_prior_rate("groq", min_samples=3) is None


def test_provider_prior_aggregates_across_siblings(kdv):
    """Aggregate counts every sibling's outcomes inside the window."""
    kdv.register("groq/m1", "groq", rpm=30, tpm=131072)
    kdv.register("groq/m2", "groq", rpm=30, tpm=131072)
    # 4 successes on m1, 4 failures on m2 → aggregate 0.5.
    for _ in range(4):
        kdv.post_call("groq/m1", "groq", headers={}, token_count=10)
    for _ in range(4):
        kdv.record_failure("groq/m2", "groq", "server_error")
    rate = kdv.provider_prior_rate("groq", min_samples=3)
    assert rate == pytest.approx(0.5, abs=0.01)


def test_provider_prior_window_filters_old_entries(kdv_with_model):
    """Outcomes older than window_secs do NOT contribute — recent
    behavior is what matters for the prior."""
    from collections import deque
    now = time.time()
    dq = deque(maxlen=kdv_with_model._OUTCOME_MAX_LEN)
    for _ in range(10):
        dq.append((now - 1000.0, False))  # outside 300s window
    for _ in range(4):
        dq.append((now - 10.0, True))     # inside window
    kdv_with_model._outcomes["groq/llama-8b"] = dq

    rate = kdv_with_model.provider_prior_rate(
        "groq", window_secs=300.0, min_samples=3,
    )
    # Only the 4 fresh successes count → 1.0.
    assert rate == pytest.approx(1.0, abs=0.01)


def test_provider_prior_member_ids_overrides_provider_lookup(kdv):
    """Caller can pass an explicit member set — used by the adapter to
    aggregate openrouter by sub-vendor instead of by provider."""
    kdv.register("openrouter/anthropic/claude:free", "openrouter",
                 rpm=30, tpm=131072)
    kdv.register("openrouter/tencent/hy3:free", "openrouter",
                 rpm=30, tpm=131072)
    # anthropic side: all good. tencent side: all bad.
    for _ in range(4):
        kdv.post_call("openrouter/anthropic/claude:free", "openrouter",
                      headers={}, token_count=10)
    for _ in range(4):
        kdv.record_failure("openrouter/tencent/hy3:free", "openrouter",
                           "server_error")

    anthropic_rate = kdv.provider_prior_rate(
        "openrouter",
        member_ids={"openrouter/anthropic/claude:free"},
        min_samples=3,
    )
    tencent_rate = kdv.provider_prior_rate(
        "openrouter",
        member_ids={"openrouter/tencent/hy3:free"},
        min_samples=3,
    )
    assert anthropic_rate == pytest.approx(1.0, abs=0.01)
    assert tencent_rate == pytest.approx(0.0, abs=0.01)


# -- snapshot_state / restore_state: outcomes window (Step 5c) --


def test_snapshot_includes_outcomes(kdv_with_model):
    """snapshot_state captures the per-model _outcomes deque so the
    24h reliability window survives process restart."""
    for _ in range(4):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={},
                                 token_count=10)
    kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")

    snap = kdv_with_model.snapshot_state()
    assert "outcomes" in snap
    assert "groq/llama-8b" in snap["outcomes"]
    entries = snap["outcomes"]["groq/llama-8b"]
    assert len(entries) == 5
    # JSON-safe shape: list of [ts:float, success:bool] pairs.
    for ts, success in entries:
        assert isinstance(ts, float)
        assert isinstance(success, bool)
    assert sum(1 for _, s in entries if s) == 4


def test_restore_outcomes_round_trip(kdv_with_model):
    """restore_state(snapshot_state()) preserves recent_success_rate
    on a fresh KDV instance — the cold-start gap is the whole point of
    this persistence path."""
    # Build a >MIN_SAMPLES window so the rate is meaningful (not the
    # 1.0 no-data sentinel).
    for _ in range(7):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={},
                                 token_count=10)
    for _ in range(3):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "server_error")
    pre = kdv_with_model.recent_success_rate("groq/llama-8b")
    pre_n = kdv_with_model.recent_samples_n("groq/llama-8b")
    assert pre_n == 10
    assert 0.65 < pre < 0.75

    snap = kdv_with_model.snapshot_state()

    # Fresh instance, register the same model so restore can match.
    fresh = KuledenDonenVar(KuledenConfig())
    fresh.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    fresh.restore_state(snap)

    assert fresh.recent_samples_n("groq/llama-8b") == pre_n
    assert fresh.recent_success_rate("groq/llama-8b") == pytest.approx(pre)


def test_restore_outcomes_drops_aged_entries(kdv_with_model):
    """Entries older than _OUTCOME_MAX_AGE_SECONDS are filtered on
    restore — a snapshot from yesterday shouldn't resurrect verdicts
    that have aged past the rolling window."""
    now = time.time()
    aged = now - kdv_with_model._OUTCOME_MAX_AGE_SECONDS - 60.0  # 1m past cutoff
    fresh_ts = now - 30.0  # well within window

    # Hand-crafted snap with mixed-age entries.
    snap = {
        "outcomes": {
            "groq/llama-8b": [
                [aged, True], [aged, False], [aged, True],  # all stale
                [fresh_ts, True], [fresh_ts, False],         # both kept
            ],
        },
    }

    fresh = KuledenDonenVar(KuledenConfig())
    fresh.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    fresh.restore_state(snap)

    # Only the 2 fresh entries should land. recent_samples_n returns 2,
    # which is below MIN_SAMPLES — recent_success_rate falls back to 1.0.
    assert fresh.recent_samples_n("groq/llama-8b") == 2


def test_restore_outcomes_handles_malformed_entries(kdv_with_model):
    """Garbage in the snapshot (wrong shape, non-numeric ts) is silently
    skipped — restore must never raise, persistence is best-effort."""
    now = time.time()
    snap = {
        "outcomes": {
            "groq/llama-8b": [
                [now - 10, True],          # good
                ["not-a-float", False],    # bad ts
                [now - 5],                 # missing success
                None,                      # not even a list
                [now - 1, True],           # good
            ],
        },
    }

    fresh = KuledenDonenVar(KuledenConfig())
    fresh.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    fresh.restore_state(snap)  # must not raise

    assert fresh.recent_samples_n("groq/llama-8b") == 2
