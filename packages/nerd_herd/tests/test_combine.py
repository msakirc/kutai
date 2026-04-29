import pytest

from nerd_herd.combine import combine_signals


def test_all_neutral_returns_zero():
    sigs = {k: 0.0 for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11")}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    assert breakdown.scalar == 0.0


def test_burden_bucket_takes_min():
    sigs = {"S1": 0, "S2": -0.3, "S3": -0.5, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # burden_neg = min(-0.3, -0.5) = -0.5; W_burden=0.5 → -0.25
    assert breakdown.bucket_totals["burden"] == pytest.approx(-0.25, abs=0.01)


def test_abundance_gated_off_by_significant_negative():
    sigs = {"S1": 0.6, "S2": 0, "S3": 0, "S4": -0.5, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0.4, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # queue_neg=-0.5; weighted=-0.35 (<-0.2 gate). abundance gated off.
    assert breakdown.positive_total == 0.0
    assert breakdown.negative_total < -0.2


def test_abundance_fires_with_no_significant_negative():
    sigs = {"S1": 0.6, "S2": 0, "S3": 0, "S4": -0.1, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0.4, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    # queue_neg=-0.1; weighted=-0.07 (>-0.2). Abundance fires: max(S1, S9) = 0.6
    assert breakdown.positive_total == pytest.approx(0.6, abs=0.05)
    assert breakdown.scalar > 0.4


def test_scalar_clipped():
    sigs = {"S1": 1.0, "S2": -1.0, "S3": -1.0, "S4": -1.0, "S5": -1.0, "S6": -1.0,
            "S7": -1.0, "S9": -1.0, "S10": -1.0, "S11": -1.0}
    weights = {k: 1.0 for k in sigs}
    breakdown = combine_signals(signals=sigs, weights=weights)
    assert breakdown.scalar == pytest.approx(-1.0, abs=0.01)
