"""T7 (Phase 3) — identical re-attempt → escalate via _dlq_write.

The proven symptom: a quality-rejected task emitted byte-identical output
across many re-dispatches (observed 48×) and never converged. `_retry_or_dlq`
must detect when THIS attempt's output hash equals the immediately-prior
`_rejection_ledger` entry's `out_hash` (both non-None) and escalate to the DLQ
instead of re-dispatching a 49th identical attempt.

Pure tests: dabidabi get_task/update_task and `_dlq_write` are monkeypatched.
No live DB is touched.
"""
from __future__ import annotations

import json

import pytest

from src.workflows.engine.hooks import _output_hash


def _install(monkeypatch, store: dict, calls: dict):
    """Patch the DB + DLQ surface that ``_retry_or_dlq`` reaches."""
    import general_beckman.apply as apply_mod

    async def fake_get_task(tid):
        return store.get(int(tid))

    async def fake_update_task(tid, **kwargs):
        calls.setdefault("update_task", []).append((int(tid), kwargs))
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)
        return None

    async def fake_dlq_write(task, *, error, category, attempts):
        calls.setdefault("dlq", []).append(
            {"id": task.get("id"), "error": error,
             "category": category, "attempts": attempts}
        )

    # _retry_or_dlq does `from dabidabi import get_task as _get_task, update_task`
    monkeypatch.setattr("dabidabi.get_task", fake_get_task, raising=False)
    monkeypatch.setattr("dabidabi.update_task", fake_update_task, raising=False)
    # update_exclusions_on_failure pulls task_state.used_model — keep it inert.
    monkeypatch.setattr(apply_mod, "_dlq_write", fake_dlq_write)


def _task(result: str, ledger: list | None, *, worker_attempts: int = 1):
    ctx: dict = {}
    if ledger is not None:
        ctx["_rejection_ledger"] = ledger
    return {
        "id": 4242,
        "status": "ungraded",
        "result": result,
        "context": json.dumps(ctx),
        "task_state": "{}",
        "worker_attempts": worker_attempts,   # 1 → < cap → normal retry branch
        "max_worker_attempts": 15,
        "title": "competitive_positioning",
        "mission_id": None,                    # avoid workflow-advance side effects
        "model": "weak-9b",
    }


def _did_repend(calls: dict) -> bool:
    return any(
        kwargs.get("status") == "pending"
        for _tid, kwargs in calls.get("update_task", [])
    )


@pytest.mark.asyncio
async def test_identical_output_escalates_to_dlq(monkeypatch):
    """Prior ledger out_hash == hash(current result) → DLQ, no re-pend."""
    import general_beckman.apply as apply_mod

    output = "The same exact draft that never converges.\n" * 20
    prior_hash = _output_hash(output)
    assert prior_hash is not None
    ledger = [{"attempt": 1, "category": "quality",
               "reason": "grade: shallow", "out_hash": prior_hash}]
    task = _task(output, ledger)
    store = {4242: task}
    calls: dict = {}
    _install(monkeypatch, store, calls)

    await apply_mod._retry_or_dlq(task, category="quality", error="grade: shallow")

    assert calls.get("dlq"), "expected _dlq_write to be called on identical repeat"
    dlq = calls["dlq"][0]
    assert dlq["category"] == "quality"
    assert "not converging" in dlq["error"]
    assert not _did_repend(calls), "must NOT re-pend a repeat"


@pytest.mark.asyncio
async def test_different_output_normal_retry(monkeypatch):
    """Current output differs from prior ledger out_hash → normal retry."""
    import general_beckman.apply as apply_mod

    prior_hash = _output_hash("an entirely different earlier draft")
    ledger = [{"attempt": 1, "category": "quality",
               "reason": "grade: shallow", "out_hash": prior_hash}]
    task = _task("a brand new, different draft this attempt", ledger)
    store = {4242: task}
    calls: dict = {}
    _install(monkeypatch, store, calls)

    await apply_mod._retry_or_dlq(task, category="quality", error="grade: shallow")

    assert not calls.get("dlq"), "must NOT DLQ when output changed"
    assert _did_repend(calls), "expected a normal re-pend"


@pytest.mark.asyncio
async def test_empty_ledger_normal_retry(monkeypatch):
    """No ledger → no repeat-detection → normal retry."""
    import general_beckman.apply as apply_mod

    task = _task("first attempt output", ledger=None)
    store = {4242: task}
    calls: dict = {}
    _install(monkeypatch, store, calls)

    await apply_mod._retry_or_dlq(task, category="quality", error="grade: shallow")

    assert not calls.get("dlq")
    assert _did_repend(calls)


@pytest.mark.asyncio
async def test_prior_out_hash_none_normal_retry(monkeypatch):
    """Prior entry carried a null out_hash (degenerate/empty) → normal retry,
    even if the current output also hashes to None."""
    import general_beckman.apply as apply_mod

    ledger = [{"attempt": 1, "category": "quality",
               "reason": "empty result", "out_hash": None}]
    # current output empty → _output_hash → None, must NOT match prior None.
    task = _task("", ledger)
    store = {4242: task}
    calls: dict = {}
    _install(monkeypatch, store, calls)

    await apply_mod._retry_or_dlq(task, category="quality", error="empty result")

    assert not calls.get("dlq"), "None == None must not trigger a false repeat"
    assert _did_repend(calls)
