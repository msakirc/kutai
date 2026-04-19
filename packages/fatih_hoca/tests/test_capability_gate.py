"""Tests for the urgency multiplier + capability gate (Phase 2c).

Uses the helper-function approach: calls _apply_urgency_layer directly with
fake ScoredModel objects so we don't need to build full ModelInfo+snapshot fakes.

CAP_GATE_RATIO = 0.85 (default):
  - top_cap=90 → cap_threshold=76.5
  - candidate with cap=70 is gated (70 < 76.5)
  - candidate with cap=80 passes (80 >= 76.5)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fatih_hoca.ranking import ScoredModel, _apply_urgency_layer, CAP_GATE_RATIO
from fatih_hoca.pools import URGENCY_MAX_BONUS


def _make_sm(
    cap_score_raw: float,   # 0–10 (stored as capability_score)
    composite: float,
    pool: str = "local",
    urgency_return: float = 0.0,
) -> ScoredModel:
    """Build a minimal ScoredModel with a mocked model for urgency computation.

    cap_score_raw is stored as sm.capability_score (0–10). Inside
    _apply_urgency_layer it is multiplied by 10 to get the 0–100 scale.

    urgency_return is what compute_urgency will return for this model.
    We patch this by storing the expected urgency on the mock — the real
    compute_urgency is NOT called (we supply a pre-computed urgency mock below).
    """
    mock_model = MagicMock()
    mock_model._urgency_return = urgency_return
    mock_model._pool = pool
    sm = ScoredModel(
        model=mock_model,
        score=composite,
        capability_score=cap_score_raw,
        composite_score=composite,
    )
    return sm


def _run_urgency(items: list[ScoredModel], snapshot=None, monkeypatch_module=None) -> None:
    """Apply urgency layer using mocked compute_urgency and classify_pool.

    Each sm.model._urgency_return is used as the urgency value.
    Each sm.model._pool is used as the pool string.
    """
    import fatih_hoca.ranking as ranking_mod
    from fatih_hoca.pools import Pool

    orig_compute = ranking_mod.compute_urgency
    orig_classify = ranking_mod.classify_pool

    def mock_compute(model, snap):
        return getattr(model, "_urgency_return", 0.0)

    def mock_classify(model):
        pool_str = getattr(model, "_pool", "per_call")
        return Pool(pool_str)

    ranking_mod.compute_urgency = mock_compute
    ranking_mod.classify_pool = mock_classify
    try:
        _apply_urgency_layer(items, snapshot or MagicMock())
    finally:
        ranking_mod.compute_urgency = orig_compute
        ranking_mod.classify_pool = orig_classify


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_below_gate_no_boost():
    """Candidate with cap_score well below gate threshold is NOT boosted."""
    # top_cap = 90 (raw=9.0), cap_threshold = 0.85*90 = 76.5
    # other: cap_score_raw=7.0 → cap_score_100=70 → 70 < 76.5 → gated
    sm_top = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=1.0)
    sm_other = _make_sm(cap_score_raw=7.0, composite=100.0, pool="local", urgency_return=1.0)

    _run_urgency([sm_top, sm_other])

    # top was boosted (cap=90 >= 76.5), other was gated (cap=70 < 76.5)
    assert sm_top.score > sm_other.score, (
        f"top.score={sm_top.score:.3f} should be > other.score={sm_other.score:.3f}"
    )
    # top should have urgency= reason, other should have urgency_gated= reason
    assert any("urgency=local" in r for r in sm_top.reasons), sm_top.reasons
    assert any("urgency_gated=local" in r for r in sm_other.reasons), sm_other.reasons


def test_near_peer_both_boosted():
    """If other_cap >= 0.85 * top_cap, both candidates receive urgency boost."""
    # top_cap=90 (raw=9.0), cap_threshold=76.5
    # other: cap_score_raw=8.0 → cap_score_100=80 → 80 >= 76.5 → both boosted
    sm_top = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=1.0)
    sm_other = _make_sm(cap_score_raw=8.0, composite=100.0, pool="local", urgency_return=1.0)

    _run_urgency([sm_top, sm_other])

    # Both should have urgency= (not urgency_gated=) reason
    assert any("urgency=local" in r for r in sm_top.reasons), sm_top.reasons
    assert any("urgency=local" in r for r in sm_other.reasons), sm_other.reasons

    # Since both started at same composite=100, same urgency=1.0, same mult → same score
    expected_mult = 1.0 + URGENCY_MAX_BONUS * 1.0
    assert abs(sm_top.score - 100.0 * expected_mult) < 0.01
    assert abs(sm_other.score - 100.0 * expected_mult) < 0.01


def test_near_peer_ratio_preserved():
    """Both candidates boosted → their score ratio changes by <2%."""
    # Baseline: urgency=0 → no boost
    sm_top_b = _make_sm(cap_score_raw=9.0, composite=120.0, pool="local", urgency_return=0.0)
    sm_other_b = _make_sm(cap_score_raw=8.0, composite=100.0, pool="local", urgency_return=0.0)
    _run_urgency([sm_top_b, sm_other_b])
    ratio_before = sm_top_b.score / sm_other_b.score

    # With urgency=1.0 → both boosted by same multiplier
    sm_top = _make_sm(cap_score_raw=9.0, composite=120.0, pool="local", urgency_return=1.0)
    sm_other = _make_sm(cap_score_raw=8.0, composite=100.0, pool="local", urgency_return=1.0)
    _run_urgency([sm_top, sm_other])
    ratio_after = sm_top.score / sm_other.score

    assert abs(ratio_after - ratio_before) < 0.02, (
        f"ratio_before={ratio_before:.4f} ratio_after={ratio_after:.4f} — "
        "near-peer urgency should not change the relative ordering significantly"
    )


def test_zero_urgency_no_boost():
    """urgency=0 → no composite change, no urgency reason appended."""
    sm = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=0.0)
    _run_urgency([sm])
    assert sm.score == 100.0
    assert sm.composite_score == 100.0
    assert not any("urgency" in r for r in sm.reasons), sm.reasons


def test_zero_urgency_does_not_change_relative_order():
    """With urgency=0, both candidates are unchanged."""
    sm_top = _make_sm(cap_score_raw=9.0, composite=150.0, pool="local", urgency_return=0.0)
    sm_other = _make_sm(cap_score_raw=7.0, composite=100.0, pool="local", urgency_return=0.0)
    _run_urgency([sm_top, sm_other])
    assert sm_top.score == 150.0
    assert sm_other.score == 100.0


def test_top_cap_candidate_gets_full_bonus_at_urgency_1():
    """Top-cap candidate with urgency=1.0 gets exactly URGENCY_MAX_BONUS applied."""
    sm = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=1.0)
    _run_urgency([sm])
    expected = 100.0 * (1.0 + URGENCY_MAX_BONUS * 1.0)
    assert abs(sm.score - expected) < 0.01, f"expected {expected:.3f}, got {sm.score:.3f}"


def test_urgency_partial_boost():
    """urgency=0.5 applies half the max bonus."""
    sm = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=0.5)
    _run_urgency([sm])
    expected = 100.0 * (1.0 + URGENCY_MAX_BONUS * 0.5)
    assert abs(sm.score - expected) < 0.01, f"expected {expected:.3f}, got {sm.score:.3f}"


def test_per_call_pool_no_urgency():
    """PER_CALL pool always returns urgency=0 → no boost."""
    sm = _make_sm(cap_score_raw=9.0, composite=100.0, pool="per_call", urgency_return=0.0)
    _run_urgency([sm])
    assert sm.score == 100.0
    assert sm.pool == "per_call"


def test_pool_and_urgency_fields_populated():
    """After _apply_urgency_layer, sm.pool and sm.urgency are set."""
    sm = _make_sm(cap_score_raw=9.0, composite=100.0, pool="local", urgency_return=0.7)
    _run_urgency([sm])
    assert sm.pool == "local"
    assert abs(sm.urgency - 0.7) < 0.001


def test_empty_list_no_error():
    """Empty scored list should not raise."""
    _run_urgency([])  # no exception expected


def test_single_candidate_always_qualifies():
    """Single candidate is always the top_cap — always passes the gate."""
    sm = _make_sm(cap_score_raw=5.0, composite=80.0, pool="local", urgency_return=0.8)
    _run_urgency([sm])
    # top_cap = 50.0, cap_threshold = 42.5, sm.cap_score_100 = 50.0 >= 42.5 → boost
    expected = 80.0 * (1.0 + URGENCY_MAX_BONUS * 0.8)
    assert abs(sm.score - expected) < 0.01, f"expected {expected:.3f}, got {sm.score:.3f}"
