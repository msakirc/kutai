"""Verify the orchestrator pump calls intersect.flash before dispatch."""
import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_run_loop_calls_flash_before_dispatch(monkeypatch):
    from src.core.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)  # skip __init__ (Telegram)
    orch.running = True
    orch._shutting_down = False
    orch.shutdown_event = asyncio.Event()
    orch._running_futures = []
    orch.requested_exit_code = None

    seen = {"flashed": None, "dispatched": None}
    sample = {"id": 4242, "title": "t", "context": json.dumps({})}

    import general_beckman

    async def _next_task():
        orch.running = False  # one iteration only
        return dict(sample)

    monkeypatch.setattr(general_beckman, "next_task", _next_task)

    import intersect

    async def _flash(task):
        seen["flashed"] = task["id"]
        task["skills"] = [{"artifact_id": 1, "exposure_class": "inject"}]
        return task

    monkeypatch.setattr(intersect, "flash", _flash)

    async def _dispatch(task):
        seen["dispatched"] = task.get("skills")

    monkeypatch.setattr(orch, "_dispatch", _dispatch)

    # Avoid the workspace/git init branch + founder sweep import.
    import src.tools.git_ops as _git

    async def _noop_git(*a, **k):
        return None

    monkeypatch.setattr(_git, "ensure_git_repo", _noop_git)

    await orch.run_loop()
    # Give the fire-and-forget _dispatch task a tick.
    await asyncio.sleep(0)
    assert seen["flashed"] == 4242
    assert seen["dispatched"] == [{"artifact_id": 1,
                                   "exposure_class": "inject"}]
