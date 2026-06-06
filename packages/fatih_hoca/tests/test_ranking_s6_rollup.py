"""_apply_utilization_layer must build a capable-supply rollup and thread
now/burn_log into pressure_for. We capture the kwargs pressure_for receives."""
from types import SimpleNamespace

from fatih_hoca import ranking
from fatih_hoca.ranking import ScoredModel


def test_eligible_models_and_now_threaded(monkeypatch):
    captured = {}

    def fake_pressure_for(model, **kwargs):
        captured.setdefault("calls", []).append((getattr(model, "name", "?"), kwargs))
        return SimpleNamespace(scalar=0.0, signals={}, modifiers={},
                               bucket_totals={}, positive_total=0.0, negative_total=0.0)

    snap = SimpleNamespace(
        cloud={}, queue_profile=None, local=SimpleNamespace(model_name=None),
        pressure_for=fake_pressure_for,
    )
    m1 = SimpleNamespace(name="a/m", provider="a", is_local=False, is_free=True,
                         capabilities={"vision"}, is_loaded=False)
    m2 = SimpleNamespace(name="b/m", provider="b", is_local=False, is_free=True,
                         capabilities=set(), is_loaded=False)
    scored = [ScoredModel(model=m1, score=10.0), ScoredModel(model=m2, score=9.0)]

    ranking._apply_utilization_layer(scored, snap, task_difficulty=5, reqs=None,
                                     now=12345.0, burn_log="BL")

    first = captured["calls"][0][1]
    assert first["now"] == 12345.0
    assert first["burn_log"] == "BL"
    names = {getattr(m, "name", "?") for m in first["eligible_models"]}
    assert names == {"a/m", "b/m"}
