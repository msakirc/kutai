"""Regression guard — inject_chain_context's prior-steps cap must TERMINATE.

Production crash loop (2026-05-26): the truncation while-loop in
`src/core/context_injection.py` only ever truncated the longest sibling
result to a ~524-char floor and never dropped a step. When a task's parent
had enough completed siblings that `N * 524 > MAX_CONTEXT_CHAIN_LENGTH`
(observed: task #178354 under parent #166114 with 23 reviewer siblings →
23*524 = 12052 > 12000), re-truncating already-floored results never reduced
the total below the cap → an INFINITE LOOP that spun the orchestrator's
event-loop thread at 100% CPU, starved the Yaşar Usta heartbeat (pure 15s
liveness writer), and triggered the watchdog to kill+restart the orchestrator
every ~3 min — forever. py-spy on the live hung process caught MainThread
`active+gil` inside this loop (lines 51→52→54→56) across consecutive samples.

These tests drive the real helper. The 23-sibling case is the exact prod
shape; it MUST finish (the old code would hang here) and stay bounded.
"""
from __future__ import annotations

import json
import threading

import pytest

import src.core.context_injection as ci
from src.core.context_injection import _cap_prior_steps, _TRUNC_FLOOR, _is_raw_dispatch
from src.core.task_context import parse_context


def _run_with_timeout(fn, seconds: float = 5.0) -> bool:
    """Return True if ``fn`` finished within ``seconds`` (else it hung)."""
    done = threading.Event()

    def _target() -> None:
        fn()
        done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return done.wait(timeout=seconds)


def test_cap_terminates_with_many_siblings_prod_repro():
    # Exact production shape: 23 siblings, each well above the floor, with a
    # cap (12000) that N*floor (12052) exceeds — the infinite-loop trigger.
    steps = [{"result": "x" * 1000} for _ in range(23)]
    finished = _run_with_timeout(lambda: _cap_prior_steps(steps, 12000))
    assert finished, "cap loop hung — infinite-loop regression reintroduced"
    total = sum(len(s["result"]) for s in steps)
    assert total <= 12000, f"total {total} still over cap"
    # Truncation alone cannot get under cap here (23*floor > 12000), so the
    # helper must have dropped at least one step.
    assert len(steps) < 23


def test_cap_truncates_without_dropping_when_few_large_steps():
    # 2 large results over a 6000 cap: truncating to the floor is enough,
    # so no step should be dropped.
    steps = [{"result": "y" * 5000} for _ in range(2)]
    finished = _run_with_timeout(lambda: _cap_prior_steps(steps, 6000))
    assert finished
    assert sum(len(s["result"]) for s in steps) <= 6000
    # No step dropped — truncating the longest is enough to clear the cap.
    assert len(steps) == 2
    # The loop stops as soon as total <= cap, so at least one (not necessarily
    # every) result is truncated down to the floor.
    assert any(len(s["result"]) <= _TRUNC_FLOOR for s in steps)


def test_cap_noop_when_already_under_cap():
    steps = [{"result": "z" * 100} for _ in range(3)]
    before = [s["result"] for s in steps]
    _cap_prior_steps(steps, 12000)
    assert [s["result"] for s in steps] == before


def test_cap_handles_empty():
    steps: list[dict] = []
    _cap_prior_steps(steps, 12000)
    assert steps == []


# ── P2c: raw_dispatch grade children must not enter chain context ───────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows

    async def close(self):
        pass


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.queried = False

    async def execute(self, sql, params):
        self.queried = True
        return _FakeCursor(self._rows)


@pytest.mark.asyncio
async def test_raw_dispatch_task_skips_chain_injection(monkeypatch):
    # A raw_dispatch task (inline grade) must return early WITHOUT querying the
    # DB for siblings — it is self-contained and pulling siblings is the path
    # that fed the crash loop.
    async def _boom():
        raise AssertionError("get_db must not be called for raw_dispatch tasks")

    monkeypatch.setattr(ci, "get_db", _boom)
    task = {
        "id": 178354,
        "parent_task_id": 166114,
        "agent_type": "reviewer",
        "context": json.dumps({"llm_call": {"raw_dispatch": True, "messages": []}}),
    }
    out = await ci.inject_chain_context(task)
    assert "prior_steps" not in parse_context(out)


@pytest.mark.asyncio
async def test_grade_children_excluded_from_prior_steps(monkeypatch):
    # A normal task (e.g. the grader AGENT) sharing a parent with raw_dispatch
    # grade children must NOT pull those children into prior_steps.
    rows = [
        {"id": 2, "title": "real step", "result": "R" * 100,
         "agent_type": "coder", "status": "completed", "context": "{}"},
        {"id": 3, "title": "grader:task#99:000000-abc", "result": "VERDICT: pass",
         "agent_type": "reviewer", "status": "completed",
         "context": json.dumps({"llm_call": {"raw_dispatch": True}})},
    ]
    fake = _FakeDB(rows)

    async def _get_db():
        return fake

    monkeypatch.setattr(ci, "get_db", _get_db)
    task = {"id": 1, "parent_task_id": 99, "agent_type": "grader", "context": "{}"}
    out = await ci.inject_chain_context(task)
    ps = parse_context(out).get("prior_steps", [])
    titles = [s["title"] for s in ps]
    assert titles == ["real step"]  # grade child #3 excluded


def test_is_raw_dispatch_detects_both_dict_and_json():
    assert _is_raw_dispatch({"llm_call": {"raw_dispatch": True}}) is True
    assert _is_raw_dispatch(json.dumps({"llm_call": {"raw_dispatch": True}})) is True
    assert _is_raw_dispatch({"llm_call": {"raw_dispatch": False}}) is False
    assert _is_raw_dispatch({}) is False
    assert _is_raw_dispatch("not json") is False
    assert _is_raw_dispatch(None) is False
