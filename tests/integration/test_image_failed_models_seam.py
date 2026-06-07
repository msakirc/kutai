"""FIX 8c — REAL orchestrator-failure → on_task_finished → failed_models seam.

The exact live chain is:

    husam._run_image raises ModelCallFailed(call_id=<model.name>, "availability")
      → orchestrator's `except ModelCallFailed` builds
            {"status":"failed", ..., "model": mcf.call_id}
      → REAL general_beckman.on_task_finished appends result["model"] to
        context.failed_models (so the next retry's selector excludes it).

The two pre-existing half-tests cover the orchestrator half (mocked husam +
mocked on_task_finished) and the husam half separately. This drives the WHOLE
chain end-to-end against an isolated tmp DB: real next_task admission (gets a
real preselected_pick), real husam._run_image (paintress.generate mocked to a
quality_failure), real orchestrator failure-dict construction, and the REAL
on_task_finished writing failed_models into the persisted task context.

Isolated via the _fresh_db tmp-DB helper + the _close_db_conn autouse fixture
(copied from tests/integration/test_image_e2e.py) so it NEVER touches the live
kutai.db and never hangs teardown on the aiosqlite background thread.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

import src.infra.db as _db_mod
from src.infra.db import init_db, get_task, get_db
import general_beckman  # noqa: F401  (import side-effects)
from general_beckman import enqueue, next_task


@pytest.fixture(autouse=True)
async def _close_db_conn():
    """Close the aiosqlite connection after each test. Its non-daemon background
    thread otherwise blocks interpreter exit and hangs pytest teardown for hours."""
    yield
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _fresh_db(tmp_path, monkeypatch):
    """Reset DB to a fresh temp file for isolation (copy of test_image_e2e.py)."""
    db_file = tmp_path / "t.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None
    import general_beckman.cron_seed as cs_mod
    monkeypatch.setattr(cs_mod, "_seeded", False)
    from general_beckman import paused_patterns as _pp
    _pp._patterns.clear()
    await init_db()
    conn = await get_db()
    await conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '+1 hour')"
    )
    await conn.commit()


def _make_orch():
    from src.core.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.telegram = None
    orch.shutdown_event = None
    orch._shutting_down = False
    orch._running_futures = []
    orch.running = False
    return orch


@pytest.mark.asyncio
async def test_image_failed_models_seam_full_dispatch(monkeypatch, tmp_path):
    await _fresh_db(tmp_path, monkeypatch)

    # paintress.generate returns a quality_failure → the REAL husam._run_image
    # raises ModelCallFailed(call_id=model.name, error_category="availability").
    import paintress

    async def _quality_fail(p, spec):
        from paintress import ImageResult
        return ImageResult(provider="pollinations", model="pollinations/flux",
                           error="quality_failure:blank")
    monkeypatch.setattr("paintress.generate", _quality_fail)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    spec = {
        "title": "seam", "description": "seam image",
        "agent_type": "image", "kind": "image", "priority": 5,
        "context": {"image_call": {
            "raw_dispatch": True, "prompt": "a mountain lake",
            "out_dir": str(tmp_path), "width": 64, "height": 64,
        }},
    }
    tid = await enqueue(spec)
    assert isinstance(tid, int)

    task = await next_task()
    assert task is not None and task["id"] == tid
    pick = task["preselected_pick"]
    assert pick.model.name == "pollinations/flux"

    orch = _make_orch()

    # Drive the REAL _dispatch path. Mock only the SAME helpers the existing
    # orchestrator image-route test mocks — NOT on_task_finished (real one runs
    # against the tmp DB) and NOT husam.run (real husam._run_image runs).
    with patch(
        "src.core.orchestrator.inject_chain_context",
        new_callable=AsyncMock,
        return_value=task,
    ), patch(
        "src.core.orchestrator.release_task_locks", new_callable=AsyncMock
    ), patch(
        "src.workflows.engine.hooks.should_skip_workflow_step",
        new_callable=AsyncMock,
        return_value=(False, ""),
    ), patch(
        "src.core.orchestrator.get_agent",
        return_value=type("P", (), {"enable_self_reflection": False})(),
    ), patch(
        "src.core.metrics_push.push_metrics", new_callable=AsyncMock
    ):
        await orch._dispatch(task)

    # The whole live chain ran: read the persisted task row and assert the
    # picked model name landed in context.failed_models via the REAL
    # on_task_finished (driven by the orchestrator's mcf.call_id → result["model"]).
    row = await get_task(tid)
    ctx = json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
    failed = ctx.get("failed_models") or []
    assert "pollinations/flux" in failed, (
        f"failed_models did not capture the picked model; got {failed!r}. "
        "The orchestrator → on_task_finished failed_models seam is broken."
    )
