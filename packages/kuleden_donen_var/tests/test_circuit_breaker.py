"""Tests for per-provider circuit breaker."""
import time
from kuleden_donen_var.circuit_breaker import CircuitBreaker


def test_initially_not_degraded():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    assert cb.is_degraded is False


def test_single_failure_not_degraded():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    assert cb.is_degraded is False


def test_threshold_failures_trips_breaker():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_degraded is True


def test_success_resets_failures():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=300, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.is_degraded is False


def test_cooldown_expires():
    cb = CircuitBreaker(failure_threshold=2, window_seconds=300, cooldown_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_degraded is True
    time.sleep(0.15)
    assert cb.is_degraded is False


def test_old_failures_outside_window_ignored():
    cb = CircuitBreaker(failure_threshold=3, window_seconds=0.1, cooldown_seconds=600)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    cb.record_failure()
    assert cb.is_degraded is False
