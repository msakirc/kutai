"""A mechanical post-hook task that returns needs_review must go TERMINAL.

Bug (2026-05-26): find_similar_missions correctly returns
status='needs_review' (3 prior missions matched → founder Continue/Branch/
Abort). mr_roboto enqueues the founder notice and rewrite Rule 0c'
synthesises the SOURCE's PostHookVerdict — but _apply_review only spawned a
reviewer task and never marked the originating mechanical post-hook task
terminal. Left non-terminal it was swept back, re-run, and DLQ'd at the
worker_attempts cap (task #166396 DLQ'd 6× with its own needs_review string
as the "error"). For a mechanical gate the review surfacing is already done;
the post-hook task itself must complete, and no reviewer agent should spawn.
"""
from __future__ import annotations

import json

import pytest

import general_beckman.apply as apply_mod
from general_beckman.result_router import RequestReview


@pytest.mark.asyncio
async def test_mechanical_posthook_needs_review_completes_no_reviewer(monkeypatch):
    updated: dict = {}
    added: list = []

    async def fake_update_task(tid, **kw):
        updated[tid] = kw

    async def fake_add_task(**kw):
        added.append(kw)
        return 1

    async def fake_get_db():
        raise AssertionError("mechanical post-hook must skip the reviewer dedup path")

    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)
    monkeypatch.setattr("src.infra.db.add_task", fake_add_task)
    monkeypatch.setattr("src.infra.db.get_db", fake_get_db)

    task = {
        "id": 166396,
        "agent_type": "mechanical",
        "title": "Find similar missions for #166099",
        "context": json.dumps({
            "posthook_kind": "find_similar_missions",
            "source_task_id": 166099,
        }),
    }
    a = RequestReview(task_id=166396, summary="matches=3", raw={})

    await apply_mod._apply_review(task, a)

    assert updated.get(166396, {}).get("status") == "completed", \
        "mechanical post-hook needs_review must mark its task completed, not retry→DLQ"
    assert added == [], "must NOT spawn a reviewer agent for a mechanical gate"


@pytest.mark.asyncio
async def test_agent_task_needs_review_still_spawns_reviewer(monkeypatch):
    """Regression: a non-mechanical (agent) task still gets a reviewer task."""
    added: list = []

    class _Cur:
        async def fetchone(self):
            return None

    class _Conn:
        async def execute(self, *a):
            return _Cur()

    async def fake_add_task(**kw):
        added.append(kw)
        return 1

    async def fake_get_db():
        return _Conn()

    async def fake_update_task(*a, **k):
        return None

    monkeypatch.setattr("src.infra.db.add_task", fake_add_task)
    monkeypatch.setattr("src.infra.db.get_db", fake_get_db)
    monkeypatch.setattr("src.infra.db.update_task", fake_update_task)

    task = {"id": 500, "agent_type": "coder", "title": "build X", "context": "{}"}
    a = RequestReview(task_id=500, summary="please review", raw={})

    await apply_mod._apply_review(task, a)

    assert len(added) == 1 and added[0]["agent_type"] == "reviewer"
