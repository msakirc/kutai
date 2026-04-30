import pytest

from nerd_herd.modifiers import (
    M1_capacity_amplifier,
    M2_perishability_dampener,
    M3_difficulty_weights,
)


# M1
def test_m1_small_pool_amplifies():
    assert M1_capacity_amplifier(limit=10) > 1.3
    assert M1_capacity_amplifier(limit=10) <= 2.0


def test_m1_medium_pool_neutral():
    assert M1_capacity_amplifier(limit=100) == pytest.approx(1.0, abs=0.05)


def test_m1_large_pool_dampens():
    assert M1_capacity_amplifier(limit=1000) < 0.7


def test_m1_clamps():
    assert M1_capacity_amplifier(limit=1) == 2.0
    assert M1_capacity_amplifier(limit=1_000_000) == 0.5


# M2
def test_m2_strong_perishability_no_damp():
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.7) == 1.0


def test_m2_no_perishability_full_damp():
    # fit_excess=0.5 → 1 - 0.5*0.5 = 0.75
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.0) == pytest.approx(0.75, abs=0.01)


def test_m2_mild_perishability_partial_damp():
    # 0.2 < S9=0.3 < 0.5 → partial: 1 - 0.5*0.25 = 0.875
    assert M2_perishability_dampener(fit_excess=0.5, s9_value=0.3) == pytest.approx(0.875, abs=0.01)


# M3
def test_m3_easy_downweights_burden():
    w = M3_difficulty_weights(difficulty=2)
    assert w["S2"] == 0.5
    assert w["S3"] == 0.5
    assert w["S4"] == 1.5


def test_m3_hard_upweights_burden():
    w = M3_difficulty_weights(difficulty=9)
    assert w["S2"] == 1.5
    assert w["S4"] == 0.7


def test_m3_mid_neutral():
    w = M3_difficulty_weights(difficulty=5)
    assert all(v == 1.0 for k, v in w.items() if k in ("S2", "S3", "S4"))


# ── M1 monotonicity + boundaries ────────────────────────────────────────


def test_m1_monotonic_decreasing_in_limit():
    """log10-driven amplifier must monotonically decrease as pool size
    grows — bigger pool, smaller amplification per failure."""
    prev = float("inf")
    for limit in [1, 5, 10, 50, 100, 500, 1000, 10_000, 1_000_000]:
        v = M1_capacity_amplifier(limit=limit)
        assert v <= prev, f"M1 must be monotonically non-increasing: {limit}→{v} after {prev}"
        prev = v


def test_m1_zero_or_negative_limit_neutral():
    """Defensive: bad input shouldn't blow up — return neutral 1.0."""
    assert M1_capacity_amplifier(limit=0) == 1.0
    assert M1_capacity_amplifier(limit=-5) == 1.0


# ── M2 boundary conditions ──────────────────────────────────────────────


def test_m2_zero_fit_excess_neutral_at_all_perishability_levels():
    """When the model is perfectly fitted (no over-qualification),
    M2 must always be 1.0 regardless of S9 — no damp without excess."""
    for s9 in [0.0, 0.1, 0.3, 0.6, 1.0]:
        m2 = M2_perishability_dampener(fit_excess=0.0, s9_value=s9)
        assert m2 == pytest.approx(1.0, abs=0.001), \
            f"S9={s9}: M2 must be 1.0 with zero excess, got {m2}"


def test_m2_high_excess_clamps_to_zero_floor():
    """Catastrophic over-qualification with no perishability → full
    floor (0.0). Locks the lower clamp."""
    m2 = M2_perishability_dampener(fit_excess=10.0, s9_value=0.0)
    assert m2 == pytest.approx(0.0, abs=0.001)


def test_m2_partial_damp_floor_is_0_5():
    """0.2 < S9 ≤ 0.5 branch clamps damp at 0.5 floor — paid cloud
    with mild perishability never gets fully suppressed even on
    extreme over-qualification."""
    m2 = M2_perishability_dampener(fit_excess=10.0, s9_value=0.3)
    assert m2 == pytest.approx(0.5, abs=0.001)


def test_m2_negative_fit_excess_treated_as_zero():
    """Under-qualified model (fit_excess < 0) must NOT inflate M2 above
    1.0 via negative subtraction. The max(0, ...) guard."""
    m2 = M2_perishability_dampener(fit_excess=-1.0, s9_value=0.0)
    assert m2 == pytest.approx(1.0, abs=0.001)


# ── M3 paid-vs-free S9 inversion ────────────────────────────────────────


def test_m3_easy_paid_cloud_downweights_s9():
    """Easy task + paid cloud → don't waste paid quota on cheap work.
    S9 weight 0.7. Easy + free cloud → free is right tool, S9 weight 1.5."""
    paid = M3_difficulty_weights(difficulty=2, model_is_paid=True)
    free = M3_difficulty_weights(difficulty=2, model_is_paid=False)
    assert paid["S9"] == 0.7
    assert free["S9"] == 1.5
    assert paid["S9"] < free["S9"]


def test_m3_hard_paid_cloud_upweights_s9():
    """Hard task + paid cloud → right tool, fire S9. Hard + free → free
    isn't the right tool for d≥7, S9 down-weighted."""
    paid = M3_difficulty_weights(difficulty=8, model_is_paid=True)
    free = M3_difficulty_weights(difficulty=8, model_is_paid=False)
    assert paid["S9"] == 1.5
    assert free["S9"] == 0.7
    assert paid["S9"] > free["S9"]


def test_m3_weights_cover_all_signals():
    """Every signal name (S1-S11 except S8 dropped) must have a weight
    or combine_signals will treat it as 0."""
    expected = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11"}
    for d in (2, 5, 9):
        w = M3_difficulty_weights(difficulty=d)
        assert set(w.keys()) == expected, \
            f"d={d}: missing weights for {expected - set(w.keys())}"


def test_m3_boundary_difficulty_3_treated_as_easy():
    """d=3 is the boundary — the if condition is `<= 3`, so d=3 should
    take the easy branch."""
    w = M3_difficulty_weights(difficulty=3)
    assert w["S2"] == 0.5  # easy branch


def test_m3_boundary_difficulty_4_treated_as_mid():
    """d=4 must NOT trigger the easy branch (which is <= 3)."""
    w = M3_difficulty_weights(difficulty=4)
    assert w["S2"] == 1.0  # mid branch


def test_m3_boundary_difficulty_7_treated_as_hard():
    """d=7 is the boundary — the if condition is `>= 7`, so d=7 should
    take the hard branch."""
    w = M3_difficulty_weights(difficulty=7)
    assert w["S2"] == 1.5  # hard branch


def test_m3_boundary_difficulty_6_treated_as_mid():
    """d=6 must NOT trigger the hard branch (which is >= 7)."""
    w = M3_difficulty_weights(difficulty=6)
    assert w["S2"] == 1.0  # mid branch
