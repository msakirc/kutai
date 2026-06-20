"""Tests for the Phase 2d unified utilization layer (post-Task-23).

After Task 23 the utilization layer calls snapshot.pressure_for() directly;
scarcity.py is deleted.  Tests use real SystemSnapshot objects so pressure_for
is available.
"""
from types import SimpleNamespace

from fatih_hoca.ranking import ScoredModel, _apply_utilization_layer
from nerd_herd.types import SystemSnapshot, LocalModelState


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


def _blank_snapshot() -> SystemSnapshot:
    return SystemSnapshot(
        local=LocalModelState(model_name="", idle_seconds=0.0),
    )


def _loaded_snapshot(model_name: str, idle_seconds: float = 600.0) -> SystemSnapshot:
    return SystemSnapshot(
        local=LocalModelState(
            model_name=model_name,
            idle_seconds=idle_seconds,
            measured_tps=20.0,
        ),
    )


def test_cold_local_gets_small_positive_pressure():
    # Cold (not-loaded) local: S9 = COLD_LOCAL_VRAM_OK = 0.4 (size_mb=0)
    # For d=5 (mid), S9 weight=1.0; positive_total=0.4; scalar=0.4
    # adjustment = 1 + 1.0 * 0.4 = 1.4 → score ≈ 140
    sm = _sm("x", cap_score_1_to_10=6.0, score=100.0, is_local=True, is_loaded=False)
    snap = _blank_snapshot()
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert 138 < sm.score < 142, f"expected ~140, got {sm.score}"


def test_loaded_idle_local_gets_boosted():
    # Loaded + saturated idle (600s) → S9 = min(1, 600/60) * 0.5 = 0.5
    # d=5 (mid): S9 weight=1.0; positive_total=0.5; scalar=0.5
    # adjustment = 1 + 1.0 * 0.5 = 1.5 → score ≈ 150
    sm = _sm("loaded-local", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = _loaded_snapshot("loaded-local", idle_seconds=600.0)
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert 148 < sm.score < 152, f"expected ~150, got {sm.score}"


def test_loaded_idle_local_easy_task_higher_boost():
    # d=3 (easy, local not paid): S9 weight=1.5; S9=0.5; weighted_S9=0.75
    # scalar=0.75; adjustment=1.75 → score ≈ 175
    sm = _sm("overq-local", cap_score_1_to_10=9.5, score=100.0, is_local=True, is_loaded=True)
    snap = _loaded_snapshot("overq-local", idle_seconds=600.0)
    _apply_utilization_layer([sm], snap, task_difficulty=3)
    assert 173 < sm.score < 177, f"expected ~175, got {sm.score}"


def test_pool_and_urgency_fields_populated():
    sm = _sm("x", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = _loaded_snapshot("x", idle_seconds=600.0)
    _apply_utilization_layer([sm], snap, task_difficulty=5)
    assert sm.pool == "local"
    # urgency stores the pressure scalar for telemetry continuity
    assert 0.4 <= sm.urgency <= 0.6, f"expected scalar ~0.5, got {sm.urgency}"


def test_empty_list_is_no_op():
    _apply_utilization_layer([], _blank_snapshot(), task_difficulty=5)


def test_utilization_layer_passes_learned_btable_to_estimate_for(monkeypatch):
    """_apply_utilization_layer must use get_btable(), not hardcode {}.

    Spy variant: monkeypatch general_beckman.btable_cache.get_btable to return
    a sentinel dict, then monkeypatch fatih_hoca.ranking.estimate_for to capture
    the btable kwarg passed. Assert every call received the sentinel, not {}.
    """
    import fatih_hoca.ranking as _ranking_mod
    from fatih_hoca.estimates import Estimates

    sentinel = {("researcher", "1.0a", "research"): {"samples_n": 10,
                "in_p90": 1000, "out_p90": 500, "iters_p90": 2}}
    captured_btables: list[dict] = []

    # Monkeypatch get_btable in general_beckman.btable_cache (source)
    import general_beckman.btable_cache as _btcache
    monkeypatch.setattr(_btcache, "_BTABLE", sentinel)
    monkeypatch.setattr(_btcache, "get_btable", lambda: sentinel)

    # Spy on estimate_for as used by ranking
    _real_estimate_for = _ranking_mod.estimate_for

    def _spy_estimate_for(task, *, btable, model_is_thinking=False):
        captured_btables.append(btable)
        return _real_estimate_for(task, btable=btable, model_is_thinking=model_is_thinking)

    monkeypatch.setattr(_ranking_mod, "estimate_for", _spy_estimate_for)

    sm1 = _sm("a", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    sm2 = _sm("b", cap_score_1_to_10=5.0, score=90.0, is_local=False, is_free=True,
              provider="groq")
    snap = _loaded_snapshot("a")
    _apply_utilization_layer([sm1, sm2], snap, task_difficulty=5)

    assert len(captured_btables) >= 1, "estimate_for was never called"
    for bt in captured_btables:
        assert bt is sentinel, (
            f"estimate_for received btable={bt!r} instead of sentinel — "
            "btable still hardcoded to {{}}"
        )
