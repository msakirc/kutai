"""SP3b Task 4 — post-hook result-REWRITE verdict path.

The existing post-hook framework only ever *gates* (pass/fail → retry/DLQ) or
*surfaces* (founder_action). SP3b's reflection + constrained_emit post-hooks
need a third verdict shape: REWRITE the source task's ``result`` in place
(reflection's corrected_result; emit's schema-conforming JSON), then let the
ordered post-hook chain (Task 6) advance to the next hook.

These tests exercise ``_apply_posthook_verdict`` directly. The DB layer is
monkeypatched with in-memory fakes — the same pattern used by
``test_posthook_kind.py`` (no real SQLite DB needed). ``_apply_posthook_verdict``
imports ``get_task`` / ``update_task`` from ``src.infra.db`` *inside the
function body*, so patching those names is honored at call time.
"""
from __future__ import annotations

import pytest


def _install_fake_db(monkeypatch, store: dict):
    """Wire src.infra.db.get_task / update_task to an in-memory ``store``.

    ``store`` maps task id -> task dict. update_task mutates in place so the
    test can read the post-rewrite result back via get_task.
    """

    async def fake_get_task(tid):
        return store.get(int(tid))

    async def fake_update_task(tid, **kwargs):
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)
        return None

    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)


@pytest.mark.asyncio
async def test_rewrite_verdict_replaces_result(monkeypatch):
    """A verdict with action="rewrite" overwrites the source task's result."""
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    store = {
        7: {
            "id": 7,
            "status": "ungraded",
            "result": "DRAFT",
            "context": "{}",
        }
    }
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=7,
        kind="constrained_emit",
        passed=True,
        raw={},
        action="rewrite",
        new_result='{"ok": true}',
    )

    # First arg is the (child) post-hook task; the apply uses a.source_task_id
    # to resolve the source. Pass the source row to mirror the plan's intent.
    await apply_mod._apply_posthook_verdict(store[7], verdict)

    refreshed = await __import__(
        "src.infra.db", fromlist=["get_task"]
    ).get_task(7)
    assert refreshed["result"] == '{"ok": true}'


@pytest.mark.asyncio
async def test_rewrite_is_idempotent(monkeypatch):
    """Applying the same terminal rewrite twice is safe (no error, no garbling).

    The SP1/SP3 claim-then-fire CAS guarantees a single fire per child terminal
    event; this test only proves the apply itself is naturally idempotent —
    update_task to the same value yields the same result, with no exception."""
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    store = {
        9: {
            "id": 9,
            "status": "ungraded",
            "result": "DRAFT",
            "context": "{}",
        }
    }
    _install_fake_db(monkeypatch, store)

    verdict = PostHookVerdict(
        source_task_id=9,
        kind="constrained_emit",
        passed=True,
        raw={},
        action="rewrite",
        new_result='{"ok": true}',
    )

    await apply_mod._apply_posthook_verdict(store[9], verdict)
    # Second call must not error or double-apply garbage.
    await apply_mod._apply_posthook_verdict(store[9], verdict)

    refreshed = await __import__(
        "src.infra.db", fromlist=["get_task"]
    ).get_task(9)
    assert refreshed["result"] == '{"ok": true}'


@pytest.mark.asyncio
async def test_empty_string_rewrite_does_not_clear_result(monkeypatch):
    """SP3b Task 5 hardening: an empty-string rewrite must NOT silently clear
    the source result. The rewrite branch guards on ``bool(new_result)``
    (truthy), not ``is not None`` — so action="rewrite" with new_result="" must
    fall through and leave the draft intact.

    Because the empty rewrite falls through to the gate path, this verdict is
    shaped as kind="self_reflect" (a warning-severity, no-pending kind) so the
    gate path is a clean no-op rather than a grade FAIL retry.
    """
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    DRAFT = "the original good draft"
    store = {
        11: {
            "id": 11,
            "status": "ungraded",
            "result": DRAFT,
            "context": "{}",
        }
    }

    update_calls: list[dict] = []

    async def fake_get_task(tid):
        return store.get(int(tid))

    async def recording_update_task(tid, **kwargs):
        update_calls.append({"tid": int(tid), "kwargs": kwargs})
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)

    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", recording_update_task)

    verdict = PostHookVerdict(
        source_task_id=11,
        kind="self_reflect",
        passed=True,
        raw={},
        action="rewrite",
        new_result="",  # empty → must NOT clear the draft
    )

    await apply_mod._apply_posthook_verdict(store[11], verdict)

    # The rewrite branch must not have fired with result="" .
    rewrite_calls = [
        c for c in update_calls if c["kwargs"].get("result") == ""
    ]
    assert rewrite_calls == [], (
        f"Empty rewrite cleared the source result: {rewrite_calls}"
    )
    assert store[11]["result"] == DRAFT, (
        f"Source draft was clobbered: {store[11]['result']!r}"
    )


@pytest.mark.asyncio
async def test_gate_default_action_unchanged(monkeypatch):
    """Regression: a verdict with the default action ("gate") must NOT touch
    result via the rewrite path. A gate-shaped verdict (action="gate", the
    default) for kind="grade"/passed=True falls through to the existing
    per-kind dispatch — the rewrite branch must be inert.

    Behavioral assertion: after calling _apply_posthook_verdict with a gate
    verdict, update_task is never called with result=<anything> (the rewrite
    branch's signature call). The source task's result stays "ORIGINAL".
    """
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    ORIGINAL_RESULT = "ORIGINAL"

    store = {
        3: {
            "id": 3,
            "status": "ungraded",
            "result": ORIGINAL_RESULT,
            "context": "{}",
            "worker_attempts": 0,
            "max_worker_attempts": 15,
            "title": "test-task",
        }
    }

    # Track every update_task call so we can inspect kwargs.
    update_calls: list[dict] = []

    async def fake_get_task(tid):
        return store.get(int(tid))

    async def recording_update_task(tid, **kwargs):
        update_calls.append({"tid": int(tid), "kwargs": kwargs})
        row = store.get(int(tid))
        if row is not None:
            row.update(kwargs)

    monkeypatch.setattr("src.infra.db.get_task", fake_get_task)
    monkeypatch.setattr("src.infra.db.update_task", recording_update_task)

    # Stub out downstream helpers that the grade+pass path calls — we only
    # care about the rewrite branch, not the full grade dispatch.
    async def _noop_summary_kinds(source, ctx):
        return []

    async def _noop_spawn(*args, **kwargs):
        pass

    async def _noop_confidence(*args, **kwargs):
        pass

    monkeypatch.setattr(apply_mod, "_summary_kinds_for_source", _noop_summary_kinds)
    monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop_spawn)
    monkeypatch.setattr(apply_mod, "_record_and_resolve_confidence", _noop_confidence)

    # Default action is "gate"; new_result defaults to None.
    verdict = PostHookVerdict(
        source_task_id=3,
        kind="grade",
        passed=True,
        raw={},
    )
    assert verdict.action == "gate"
    assert verdict.new_result is None

    # --- BEHAVIORAL ASSERTION ---
    await apply_mod._apply_posthook_verdict(store[3], verdict)

    # The rewrite branch calls update_task(id, result=new_result) then returns.
    # Verify no call carried result= — the rewrite branch was not taken.
    rewrite_calls = [c for c in update_calls if "result" in c["kwargs"]]
    assert rewrite_calls == [], (
        f"Rewrite branch fired unexpectedly: {rewrite_calls}"
    )

    # Double-check via the store: the source result is still ORIGINAL.
    assert store[3]["result"] == ORIGINAL_RESULT, (
        f"Source result was mutated: {store[3]['result']!r}"
    )
