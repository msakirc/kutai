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
