import pytest
from nerd_herd.signals.s10_failure import s10_failure


# ── Legacy consecutive_failures path ─────────────────────────────────


def test_s10_zero_failures():
    assert s10_failure(consecutive_failures=0) == 0.0


def test_s10_one_failure():
    assert s10_failure(consecutive_failures=1) == pytest.approx(-0.2, abs=0.01)


def test_s10_three_failures():
    assert s10_failure(consecutive_failures=3) == pytest.approx(-0.5, abs=0.01)


def test_s10_clamps_at_minus_half():
    assert s10_failure(consecutive_failures=10) == pytest.approx(-0.5, abs=0.01)


# ── New per-model rate path (2026-05-03) ─────────────────────────────


def test_s10_rate_no_data_neutral():
    """samples_n below MIN_SAMPLES → signal is 0 regardless of rate.
    Critical: prevents freshly-revived models from ranking as "perfectly
    reliable" on an empty outcome window."""
    assert s10_failure(success_rate=0.0, samples_n=0) == 0.0
    assert s10_failure(success_rate=0.5, samples_n=4) == 0.0


def test_s10_rate_healthy_neutral():
    """Above HEALTHY_THRESHOLD (0.95), no penalty."""
    assert s10_failure(success_rate=1.0, samples_n=10) == 0.0
    assert s10_failure(success_rate=0.96, samples_n=10) == 0.0


def test_s10_rate_broken_minus_one():
    """Below BROKEN_THRESHOLD (0.20), full penalty."""
    assert s10_failure(success_rate=0.20, samples_n=10) == pytest.approx(-1.0, abs=0.01)
    assert s10_failure(success_rate=0.0, samples_n=10) == pytest.approx(-1.0, abs=0.01)


def test_s10_rate_linear_interp():
    """Between thresholds, linear from (0.95, 0) to (0.20, -1)."""
    # midpoint: (0.95+0.20)/2 = 0.575 → -0.5
    assert s10_failure(success_rate=0.575, samples_n=10) == pytest.approx(-0.5, abs=0.01)
    # 0.80 → -((0.95-0.80)/0.75) = -0.2
    assert s10_failure(success_rate=0.80, samples_n=10) == pytest.approx(-0.2, abs=0.01)


def test_s10_combines_rate_and_streak_worst_wins():
    """When both rate and streak signals fire, take the worst (most
    negative). Prevents either source from masking the other."""
    # streak=-0.2, rate=-0.5 → -0.5
    val = s10_failure(success_rate=0.575, samples_n=10, consecutive_failures=1)
    assert val == pytest.approx(-0.5, abs=0.01)
    # streak=-0.5, rate=0 (healthy) → -0.5
    val = s10_failure(success_rate=1.0, samples_n=10, consecutive_failures=5)
    assert val == pytest.approx(-0.5, abs=0.01)


def test_s10_rate_window_boundary():
    """At MIN_SAMPLES exactly, signal activates."""
    # samples_n=5 (== MIN_SAMPLES), rate=0.0 → -1.0
    assert s10_failure(success_rate=0.0, samples_n=5) == pytest.approx(-1.0, abs=0.01)
