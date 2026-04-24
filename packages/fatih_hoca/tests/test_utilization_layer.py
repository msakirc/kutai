"""Tests for the Phase 2d unified utilization layer."""
from types import SimpleNamespace

from fatih_hoca.ranking import ScoredModel, _apply_utilization_layer


def _sm(name, cap_score_1_to_10, score, is_local=False, is_free=False, is_loaded=False,
        provider=""):
    model = SimpleNamespace(
        name=name,
        litellm_name=name,
        is_local=is_local,
        is_free=is_free,
        is_loaded=is_loaded,
        provider=provider,
    )
    return ScoredModel(
        model=model,
        score=score,
        capability_score=cap_score_1_to_10,
        composite_score=score,
    )


def _blank_snapshot():
    local = SimpleNamespace(
        model_name="",
        idle_seconds=0.0,
        measured_tps=0.0,
        thinking_enabled=False,
        requests_processing=0,
    )
    return SimpleNamespace(local=local, cloud={})


def test_zero_scarcity_leaves_score_unchanged():
    sm = _sm("x", cap_score_1_to_10=6.0, score=100.0, is_local=True)
    # local with no idle and not-loaded → scarcity 0
    snap = _blank_snapshot()
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert sm.score == 100.0


def test_positive_scarcity_under_qualified_model_gets_full_boost():
    # local, loaded, saturated idle → +0.5 scarcity
    # cap_score_100=50, d=5 → cap_needed=45, fit_excess=0.05; (1-0.05)=0.95
    # composite *= 1 + 1.0 * 0.5 * 0.95 = 1.475
    sm = _sm("loaded-local", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="loaded-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert 146 < sm.score < 149


def test_over_qualified_model_ignores_positive_scarcity():
    # cap_score_100=95, d=3 → cap_needed=30, fit_excess=0.65
    # (1 - 0.65) = 0.35 → adjustment only 35% of K*scarcity
    # with scarcity +0.5: composite *= 1 + 1.0 * 0.5 * 0.35 = 1.175
    sm = _sm("overq-local", cap_score_1_to_10=9.5, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="overq-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=3)
    assert 117 < sm.score < 118


def test_under_qualified_model_positive_scarcity_dampened_symmetrically():
    # Symmetric fit dampener on positive scarcity (2026-04-20): under-qualified
    # candidates don't get the full "burn me" boost — burning a wrong tool
    # is itself wasteful.
    # cap_score_100=25, d=5 → cap_needed=45, fit_excess=-0.2
    # For scarcity > 0: dampener = 1 - abs(-0.2) = 0.8
    # composite *= 1 + 1.0 * 0.5 * 0.8 = 1.4
    sm = _sm("weak-local", cap_score_1_to_10=2.5, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="weak-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert 139 < sm.score < 141


def test_pool_and_urgency_fields_populated():
    sm = _sm("x", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="x", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert sm.pool == "local"
    # urgency field is repurposed to store the scalar scarcity for telemetry continuity
    assert 0.4 <= sm.urgency <= 0.5


def test_empty_list_is_no_op():
    _apply_utilization_layer([], _blank_snapshot(), task_difficulty=5)
