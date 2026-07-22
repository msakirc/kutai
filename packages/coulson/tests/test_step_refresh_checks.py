"""_refresh_workflow_step_config must live-reload the `checks` field.

Bug: `checks` (parameterized mechanical verifiers) freeze in task.context at
expansion time. When a workflow JSON edit ADDS a check to a step (e.g. the
requirement-conservation gate on 3.9a/3.10a/3.10b), already-expanded tasks
keep running WITHOUT it — the instruction live-reloads but the gate stays
inert (m90 3.10b dropped FR-012..015 unchecked). determine_posthooks reads
task_ctx["checks"] at COMPLETION, so refreshing that field at dispatch is
enough for the gate to fire.
"""
import asyncio
import json
from unittest.mock import patch


def _run_refresh(task, task_ctx, live_step):
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
            return live_step

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


_CONS = [{"kind": "verify_requirement_conservation",
          "payload": {"action": "verify_requirement_conservation"}}]


def test_newly_declared_check_is_synced_onto_frozen_task():
    # Task expanded before the check existed → context has no checks.
    task = {"id": 3, "mission_id": 90, "description": "assemble spec"}
    task_ctx = {"workflow_step_id": "3.10b"}
    captured = _run_refresh(
        task, task_ctx,
        {"instruction": "assemble spec", "checks": _CONS},
    )
    # In-memory: the gate is now on the task so determine_posthooks enqueues it.
    assert task_ctx.get("checks") == _CONS
    # Persisted: the completion path re-reads the DB, so it must be written.
    assert json.loads(captured["context"])["checks"] == _CONS


def test_checks_unchanged_no_spurious_write():
    task = {"id": 4, "mission_id": 90, "description": "assemble spec"}
    task_ctx = {"workflow_step_id": "3.10b", "checks": _CONS}
    captured = _run_refresh(
        task, task_ctx,
        {"instruction": "assemble spec", "checks": _CONS},
    )
    # Nothing diverged → no context write forced.
    assert captured.get("context") is None
