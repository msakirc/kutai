"""Regression: _refresh_workflow_step_config must PERSIST the refreshed
description to the DB, not just update it in memory.

Bug: worker rebuilds its prompt from the live workflow JSON (via the
in-memory refresh), but the grader and self_reflect children read
``tasks.description`` straight from the DB. The refresh persisted only
``context`` — never ``description`` — so after a workflow-instruction edit
the grader kept judging the artifact against the STALE spec and DLQ'd a
correct artifact (task #259351: instruction dropped founding_year/
funding_total, worker emitted the light shape, grader kept demanding the
heavy fields → eternal COMPLETE: NO).
"""
import asyncio
import json
from unittest.mock import patch


def _run_refresh(task, task_ctx, live_instruction):
    from coulson import _refresh_workflow_step_config

    captured = {}

    class FakeCursor:
        async def fetchone(self):
            return (json.dumps({"workflow_name": "i2p_v3"}),)

        async def close(self):
            pass

    class FakeDB:
        async def execute(self, *a, **k):
            return FakeCursor()

    async def fake_get_db():
        return FakeDB()

    class FakeWF:
        def get_step(self, _sid):
            return {"instruction": live_instruction}

    def fake_load_workflow(_name):
        return FakeWF()

    async def fake_update_task(task_id, **kwargs):
        captured["task_id"] = task_id
        captured.update(kwargs)

    with patch("src.infra.db.get_db", fake_get_db), \
         patch("src.infra.db.update_task", fake_update_task), \
         patch("src.workflows.engine.loader.load_workflow", fake_load_workflow):
        asyncio.run(_refresh_workflow_step_config(task, task_ctx))

    return captured


def test_refreshed_description_is_persisted_to_db():
    new_instr = "Identify direct competitors: return ONLY name, website_url, status."
    task = {
        "id": 1,
        "mission_id": 80,
        "description": "OLD: collect founding_year, funding_total, team_size_estimate.",
    }
    task_ctx = {"workflow_step_id": "1.3"}

    captured = _run_refresh(task, task_ctx, new_instr)

    # In-memory refresh already worked before the fix.
    assert task["description"] == new_instr
    # The fix: the persisted DB write must carry the new description so the
    # grader (which re-reads the DB row) sees the same spec the worker ran.
    assert captured.get("description") == new_instr


def test_description_not_persisted_when_unchanged():
    same = "Identical instruction, nothing to re-sync."
    task = {"id": 2, "mission_id": 80, "description": same}
    task_ctx = {"workflow_step_id": "1.3"}

    captured = _run_refresh(task, task_ctx, same)

    # No divergence -> no description key forced into the write (and ideally
    # no write at all when nothing changed).
    assert captured.get("description") is None
