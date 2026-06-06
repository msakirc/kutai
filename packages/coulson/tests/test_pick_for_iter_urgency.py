"""Mid-task re-selection must use the task's admission urgency + finish-bias,
not a flat 0.5 — a started task is never judged stricter than a fresh one.
Drives the REAL pick_for_iter; only the fatih_hoca selector boundary is stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import fatih_hoca
import coulson.dispatch_helpers as dh
from fatih_hoca.types import Pick, Failure
from fatih_hoca.requirements import ModelRequirements


def _pick(name: str) -> Pick:
    model = SimpleNamespace(name=name, litellm_name=name, is_local=True, is_loaded=True)
    return Pick(model=model, min_time_seconds=1.0)


def _reqs() -> ModelRequirements:
    return ModelRequirements(task="coder", agent_type="coder", difficulty=5)


def _tracking_select(monkeypatch, returns: Pick):
    calls: list[dict] = []

    def _sel(**kwargs):
        calls.append(kwargs)
        return returns

    monkeypatch.setattr(fatih_hoca, "select", _sel)
    return calls


def test_reselect_uses_admission_urgency_plus_finish_bias(monkeypatch):
    # held no longer servable → re-select; admission urgency 0.8 → 0.8 + 0.1
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: False)
    fresh = _pick("gemini/gemma-4-26b")
    calls = _tracking_select(monkeypatch, fresh)

    task = {"id": 1, "preselected_pick": _pick("held"), "_admission_urgency": 0.8}
    result = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0,
    )
    assert result is fresh
    assert abs(calls[0]["urgency"] - 0.9) < 1e-9


def test_reselect_default_when_unstamped(monkeypatch):
    # no _admission_urgency → base 0.5 + 0.1 finish-bias = 0.6
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: False)
    calls = _tracking_select(monkeypatch, _pick("x"))
    task = {"id": 1, "preselected_pick": _pick("held")}
    dh.pick_for_iter(reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0)
    assert abs(calls[0]["urgency"] - 0.6) < 1e-9


def test_failures_stack_failure_bump(monkeypatch):
    # admission 0.5 + finish 0.1 + failure 0.1 = 0.7
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: True)
    calls = _tracking_select(monkeypatch, _pick("fallback"))
    task = {"id": 1, "preselected_pick": _pick("flaky"), "_admission_urgency": 0.5}
    dh.pick_for_iter(
        reqs=_reqs(), task=task,
        failures=[Failure(model="flaky", reason="timeout")],
        iteration=1, remaining_budget=5.0,
    )
    assert abs(calls[0]["urgency"] - 0.7) < 1e-9
