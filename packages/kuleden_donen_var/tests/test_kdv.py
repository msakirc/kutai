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


def test_recent_success_rate_includes_quota_failures(kdv_with_model):
    """A frequently-rate-limited model IS less reliable from the
    dispatcher's POV — track quota/rate_limited as a failure outcome."""
    for _ in range(3):
        kdv_with_model.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(7):
        kdv_with_model.record_failure("groq/llama-8b", "groq", "rate_limited")
    rate = kdv_with_model.recent_success_rate("groq/llama-8b")
    assert rate == 0.3, f"got {rate}"


def test_recent_success_rate_unknown_model(kdv):
    """Unregistered ids return 1.0 (no data → no penalty)."""
    assert kdv.recent_success_rate("nonexistent/model") == 1.0
