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
async def test_gate_default_action_unchanged(monkeypatch):
    """Regression: a verdict with the default action ("gate") must NOT touch
    result via the rewrite path. constrained_emit's gate-shaped verdict (no
    new_result) falls through to the existing per-kind dispatch."""
    import general_beckman.apply as apply_mod
    from general_beckman.result_router import PostHookVerdict

    # Default action is "gate"; new_result defaults to None.
    verdict = PostHookVerdict(
        source_task_id=3,
        kind="grade",
        passed=True,
        raw={},
    )
    assert verdict.action == "gate"
    assert verdict.new_result is None
