# test_pick_for_iter_reuse.py
"""RC-A (mission 74): the worker must reuse the model Beckman reserved for
a task across *every* no-failure iteration — not just iteration 0 — as long
as that model is still servable right now. Re-selecting fresh each turn
re-races the live pool and is the `no_candidates` mechanism.

Drives the REAL ``pick_for_iter``; only the fatih_hoca selector boundary
(``select`` / ``is_servable``) is stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import fatih_hoca
import coulson.dispatch_helpers as dh
from fatih_hoca.types import Pick, Failure
from fatih_hoca.requirements import ModelRequirements


def _pick(name: str) -> Pick:
    model = SimpleNamespace(name=name, litellm_name=name,
                            is_local=True, is_loaded=True)
    return Pick(model=model, min_time_seconds=1.0)


def _reqs() -> ModelRequirements:
    return ModelRequirements(task="coder", agent_type="coder", difficulty=5)


def _no_select(monkeypatch):
    """Install a select() that fails the test if the worker re-races the pool."""
    def _boom(**kwargs):
        raise AssertionError("pick_for_iter re-selected when it should have reused")
    monkeypatch.setattr(fatih_hoca, "select", _boom)


def _tracking_select(monkeypatch, returns: Pick):
    calls: list[dict] = []

    def _sel(**kwargs):
        calls.append(kwargs)
        return returns

    monkeypatch.setattr(fatih_hoca, "select", _sel)
    return calls


# ─── Reuse path ───────────────────────────────────────────────────────────────

def test_reuses_preselected_across_later_iters_when_servable(monkeypatch):
    """iter ≥ 1, no failures, held model still servable → reuse it, no select."""
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: True)
    _no_select(monkeypatch)

    held = _pick("Qwen3.5-9B-thinking")
    task = {"id": 164601, "preselected_pick": held}

    result = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=3, remaining_budget=5.0,
    )
    assert result is held


def test_reuses_on_iter_zero_when_servable(monkeypatch):
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: True)
    _no_select(monkeypatch)

    held = _pick("alpha")
    task = {"id": 1, "preselected_pick": held}
    result = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=0, remaining_budget=5.0,
    )
    assert result is held


# ─── Re-select path ─────────────────────────────────────────────────────────

def test_reselects_when_held_no_longer_servable(monkeypatch):
    """Held model slipped (swapped-out / rate-limited) → re-select fresh."""
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: False)
    fresh = _pick("gemini/gemma-4-26b")
    calls = _tracking_select(monkeypatch, fresh)

    held = _pick("Qwen3.5-9B-thinking")
    task = {"id": 1, "preselected_pick": held}
    result = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0,
    )
    assert result is fresh
    assert len(calls) == 1


def test_reselects_when_failures_present(monkeypatch):
    """Failures present → failure-adaptation re-selects even if held is servable."""
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: True)
    fresh = _pick("fallback")
    calls = _tracking_select(monkeypatch, fresh)

    held = _pick("flaky")
    task = {"id": 1, "preselected_pick": held}
    result = dh.pick_for_iter(
        reqs=_reqs(), task=task,
        failures=[Failure(model="flaky", reason="timeout")],
        iteration=1, remaining_budget=5.0,
    )
    assert result is fresh
    assert len(calls) == 1
    # mid-task urgency = base 0.5 + finish-bias 0.1 + failure-bump 0.1
    assert abs(calls[0]["urgency"] - 0.7) < 1e-9


def test_held_pick_updated_after_reselect_then_reused(monkeypatch):
    """After a failure-driven re-select to Q, a subsequent no-failure iter
    reuses Q (the model now actually running) — NOT the stale preselected P."""
    state = {"servable": True}
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: state["servable"])

    P = _pick("preselected-P")
    Q = _pick("reselected-Q")
    task = {"id": 1, "preselected_pick": P}

    # Iter with a failure → re-select to Q, stamps the held pick.
    monkeypatch.setattr(fatih_hoca, "select", lambda **kw: Q)
    r1 = dh.pick_for_iter(
        reqs=_reqs(), task=task,
        failures=[Failure(model="preselected-P", reason="timeout")],
        iteration=1, remaining_budget=5.0,
    )
    assert r1 is Q

    # Next iter, no failures, Q servable → reuse Q, never touch select().
    _no_select(monkeypatch)
    r2 = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0,
    )
    assert r2 is Q
