"""S9 free-cloud perishability — continuity (no 'until too late' cliff).

Locks the 2026-06-04 reversion of the 2026-05-03 hard 1h window back to a
continuous exp(-reset_in/τ) decay. The window made perishability flat-zero
all day then cliff in the final hour; these tests assert the signal is now
strictly positive and monotonic across the whole cycle.
"""
import math
import time
from types import SimpleNamespace

from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.signals.s9_perishability import (
    s9_perishability,
    FREE_CLOUD_DECAY_TAU_SECS,
)


def _free_model():
    return SimpleNamespace(is_local=False, is_free=True, name="free/m", provider="free")


def _matrix(reset_in_secs: float):
    now = time.time()
    return RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=900, reset_at=int(now + reset_in_secs))
    )


def _s9(reset_in_secs: float) -> float:
    return s9_perishability(
        _free_model(), local=None, vram_avail_mb=0,
        matrix=_matrix(reset_in_secs), task_difficulty=3,
    )


def test_no_cliff_positive_far_from_reset():
    """The old 1h window returned exactly 0 past 3600s. Continuous decay
    must stay strictly > 0 even a full cycle out."""
    for hours in (2, 6, 12, 23):
        assert _s9(hours * 3600) > 0.0, f"S9 must be > 0 at {hours}h to reset"


def test_monotonic_decreasing_with_distance():
    """Closer to reset ⇒ stronger perishability, smoothly."""
    distances = [600, 1800, 3600, 6 * 3600, 12 * 3600, 24 * 3600]
    vals = [_s9(d) for d in distances]
    for a, b in zip(vals, vals[1:]):
        assert a > b, f"S9 must strictly decrease with reset distance: {vals}"


def test_matches_exp_decay_form():
    """Value is exp(-reset_in/τ) at the configured τ."""
    d = 3 * 3600.0
    expected = math.exp(-d / FREE_CLOUD_DECAY_TAU_SECS)
    # rel_tol loose: reset_at is int()-rounded and time.time() drifts a hair
    # between calls. The point is exp form (≈0.607 at 3h/τ=6h), not the window
    # (which would give a linear 0.0 here).
    assert math.isclose(_s9(d), expected, rel_tol=1e-3)


def test_continuity_across_old_window_boundary():
    """No jump at the old 3600s boundary — values just inside and just
    outside 1h differ only infinitesimally (continuity, not a step)."""
    just_in = _s9(3590)
    just_out = _s9(3610)
    assert abs(just_in - just_out) < 0.01, "must be continuous across the old 1h gate"


def test_near_reset_strong():
    """Imminent reset still yields a strong flush signal (PP4 contract: >0.7
    at 600s out, preserved under τ=6h)."""
    assert _s9(600) > 0.7


def test_zero_when_nothing_remaining():
    """Nothing left to flush ⇒ no perishability signal."""
    now = time.time()
    empty = RateLimitMatrix(rpd=RateLimit(limit=1000, remaining=0, reset_at=int(now + 600)))
    v = s9_perishability(_free_model(), local=None, vram_avail_mb=0,
                         matrix=empty, task_difficulty=3)
    assert v == 0.0
