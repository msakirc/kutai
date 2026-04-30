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


# ── Bucket independence: each bucket's worst signal stands alone ────────


def test_burden_bucket_ignores_positive_signals():
    """Worst-of-negatives semantic — a positive S2 must not flip the
    bucket sign. Lock the `min(... if < 0)` filter."""
    sigs = {"S1": 0, "S2": +0.5, "S3": -0.4, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # min(-0.4) = -0.4; W=0.5 → -0.2
    assert br.bucket_totals["burden"] == pytest.approx(-0.2, abs=0.01)


def test_queue_bucket_takes_min():
    sigs = {"S1": 0, "S2": 0, "S3": 0, "S4": -0.6, "S5": -0.3, "S6": -0.4,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # min(-0.6, -0.3, -0.4) = -0.6; W=0.7 → -0.42
    assert br.bucket_totals["queue"] == pytest.approx(-0.42, abs=0.01)


def test_other_bucket_takes_min_across_s1_s7_s9_s10_s11():
    """S1 (negative-arm in this case), S7, S9, S10, S11 all live in
    OTHER bucket. Worst wins."""
    sigs = {"S1": -0.2, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0,
            "S7": -0.5, "S9": -0.3, "S10": -0.8, "S11": -0.4}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # min = -0.8 from S10; W=1.0 → -0.8
    assert br.bucket_totals["other"] == pytest.approx(-0.8, abs=0.01)


# ── Abundance gate ──────────────────────────────────────────────────────


def test_abundance_gate_below_threshold_suppresses():
    """When negative_total drops below the -0.2 gate, abundance is
    suppressed entirely. Locks the gate semantic (significant negative
    pressure must NOT be hidden by an unrelated positive signal)."""
    sigs = {"S1": 0.7, "S2": 0, "S3": 0, "S4": -0.4, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # queue_neg = -0.4; weighted = 0.7 * -0.4 = -0.28; gate: -0.28 > -0.2 → False
    assert br.negative_total < -0.2
    assert br.positive_total == 0.0


def test_abundance_gate_strict_inequality():
    """Just above -0.2 → gate passes → abundance fires."""
    sigs = {"S1": 0.7, "S2": 0, "S3": 0, "S4": -0.2, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # queue_neg = -0.2; weighted = -0.14. gate: -0.14 > -0.2 → True
    assert br.positive_total == pytest.approx(0.7, abs=0.001)


def test_abundance_only_s1_and_s9_contribute_positive():
    """Positive S2, S4, S10 must NOT contribute to positive_total —
    only S1 and S9 are in the abundance arm. Locks POSITIVE_ARM_SIGNALS."""
    sigs = {"S1": 0.0, "S2": 0.5, "S3": 0.0, "S4": 0.5, "S5": 0.0, "S6": 0.0,
            "S7": 0.5, "S9": 0.0, "S10": 0.5, "S11": 0.5}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # No negatives → gate passes. But max(S1, S9) where both = 0 → 0
    assert br.positive_total == 0.0
    assert br.scalar == 0.0


def test_abundance_picks_max_of_s1_s9():
    """When both S1 and S9 are positive, the larger one wins (not sum,
    not avg). Prevents double-counting two manifestations of the
    same 'this model has spare capacity' signal."""
    sigs = {"S1": 0.3, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0.8, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    assert br.positive_total == pytest.approx(0.8, abs=0.001)


# ── Weights propagate ───────────────────────────────────────────────────


def test_weights_apply_per_signal_before_bucketing():
    """M3-style weights must scale each signal independently before
    the bucket-min picks the worst. Without this, all signals would
    have equal voice regardless of difficulty."""
    sigs = {"S1": 0, "S2": -0.4, "S3": -0.4, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights_easy = {"S1": 1, "S2": 0.5, "S3": 0.5, "S4": 1, "S5": 1, "S6": 1,
                    "S7": 1, "S9": 1, "S10": 1, "S11": 1}
    weights_hard = {"S1": 1, "S2": 1.5, "S3": 1.5, "S4": 1, "S5": 1, "S6": 1,
                    "S7": 1, "S9": 1, "S10": 1, "S11": 1}
    easy = combine_signals(signals=sigs, weights=weights_easy)
    hard = combine_signals(signals=sigs, weights=weights_hard)
    # Easy: weighted -0.4*0.5 = -0.2; bucket=-0.2, W_burden=0.5 → -0.1
    # Hard: weighted -0.4*1.5 = -0.6; bucket=-0.6, W_burden=0.5 → -0.3
    assert easy.bucket_totals["burden"] == pytest.approx(-0.1, abs=0.01)
    assert hard.bucket_totals["burden"] == pytest.approx(-0.3, abs=0.01)


def test_zero_weight_eliminates_signal():
    """A signal with weight=0 must contribute nothing, even if extreme."""
    sigs = {"S1": 0, "S2": -1.0, "S3": 0, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    weights["S2"] = 0.0
    br = combine_signals(signals=sigs, weights=weights)
    assert br.bucket_totals["burden"] == 0.0
    assert br.scalar == 0.0


def test_missing_signal_treated_as_zero():
    """combine_signals must tolerate a missing key — defaults to 0,
    no KeyError. Guards against signal-rename refactors."""
    sigs = {"S1": -0.5}  # only S1 set; S2..S11 missing
    weights = {k: 1.0 for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7",
                                 "S9", "S10", "S11")}
    br = combine_signals(signals=sigs, weights=weights)
    # Only S1 contributed; bucket=other; W=1.0 → -0.5
    assert br.bucket_totals["other"] == pytest.approx(-0.5, abs=0.01)
    assert br.scalar == pytest.approx(-0.5, abs=0.01)


# ── Scalar clamp boundary ───────────────────────────────────────────────


def test_scalar_clamps_to_plus_one():
    """Even if positive_total + negative_total > 1, output clamps to 1."""
    sigs = {"S1": 5.0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0,
            "S7": 0, "S9": 5.0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    assert br.scalar == pytest.approx(1.0, abs=0.001)


def test_negative_total_sums_buckets():
    """Three buckets all negative — final negative_total is the sum,
    not min. Locks the `sum(bucket_totals.values())` semantic."""
    sigs = {"S1": -0.4, "S2": -0.4, "S3": 0, "S4": -0.4, "S5": 0, "S6": 0,
            "S7": 0, "S9": 0, "S10": 0, "S11": 0}
    weights = {k: 1.0 for k in sigs}
    br = combine_signals(signals=sigs, weights=weights)
    # burden: 0.5 * -0.4 = -0.2
    # queue: 0.7 * -0.4 = -0.28
    # other (S1): 1.0 * -0.4 = -0.4
    # sum = -0.88
    assert br.negative_total == pytest.approx(-0.88, abs=0.01)
