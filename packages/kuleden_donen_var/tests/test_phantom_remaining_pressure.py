"""A rate-limit axis with a known LIMIT but unknown REMAINING (provider sends
no header for that axis) must NOT read as depleted.

Live 2026-06-17: groq/cerebras free models that PASSED eligibility (KDV's
daily_exhausted flag = False → KDV considers them healthy) were still scored
pressure scalar = -1.00 and hard-vetoed by the pool-pressure gate, so every
low-urgency overhead task (graders/critics) got `select=None` and sat ready,
undispatched, for ~hours while local was down.

Root: the adapter emitted RateLimit(limit=X, remaining=None) for an axis with
no provider header, and every budget signal does `(remaining or 0)` → frac 0 →
S1 depletion_max = -1.0. Genuine exhaustion is surfaced separately via the
daily_exhausted / rpm_cooldown flags at eligibility — a missing header is
"unknown", not "empty", and must be treated as full for pressure.
"""
from types import SimpleNamespace

from kuleden_donen_var.nerd_herd_adapter import _rl
from nerd_herd.signals.s1_remaining import s1_remaining
from nerd_herd.types import RateLimitMatrix


def test_rl_unknown_remaining_assumes_full():
    # Provider exposes a daily LIMIT but never a daily REMAINING header.
    st = SimpleNamespace(rpd_limit=14400, rpd_remaining=None, rpd_reset_at=0)
    rl = _rl(st, "rpd")
    assert rl.limit == 14400
    assert rl.remaining == 14400, "unknown remaining must be treated as full, not 0"


def test_phantom_daily_axis_does_not_force_s1_depletion():
    # Healthy per-minute axis + a daily axis that has a limit but no remaining
    # header. Pre-fix S1 read the daily axis as frac 0 → -1.0 worst-wins.
    st = SimpleNamespace(
        rpm_limit=30, rpm_remaining=30, rpm_reset_at=0,
        rpd_limit=14400, rpd_remaining=None, rpd_reset_at=0,
    )
    m = RateLimitMatrix()
    m.rpm = _rl(st, "rpm")
    m.rpd = _rl(st, "rpd")
    s1 = s1_remaining(m, profile="time_bucketed")
    assert s1 >= 0.0, f"phantom unknown-remaining axis forced depletion: s1={s1}"


def test_genuine_zero_remaining_still_depletes():
    # A real header that says remaining=0 MUST still fire depletion (the
    # daily_exhausted flag also catches it at eligibility, but the signal must
    # not be neutered for genuinely-empty axes).
    st = SimpleNamespace(rpd_limit=14400, rpd_remaining=0, rpd_reset_at=0)
    rl = _rl(st, "rpd")
    assert rl.remaining == 0
    m = RateLimitMatrix()
    m.rpd = rl
    assert s1_remaining(m, profile="time_bucketed") == -1.0
