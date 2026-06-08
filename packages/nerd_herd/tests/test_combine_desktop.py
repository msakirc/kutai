from nerd_herd.combine import combine_signals, OTHER_BUCKET


def test_s13_s14_in_other_bucket():
    assert "S13" in OTHER_BUCKET
    assert "S14" in OTHER_BUCKET


def test_s13_graded_negative_flows_through():
    sig = {k: 0.0 for k in ("S1","S2","S3","S4","S5","S6","S7","S9","S10","S11","S12","S13","S14")}
    sig["S13"] = -0.6
    out = combine_signals(signals=sig, weights={})
    assert out.scalar < 0.0


def test_s14_sentinel_pegs_minus_one():
    sig = {k: 0.0 for k in ("S1","S2","S3","S4","S5","S6","S7","S9","S10","S11","S12","S13","S14")}
    sig["S14"] = -10.0
    out = combine_signals(signals=sig, weights={})
    assert out.scalar == -1.0
