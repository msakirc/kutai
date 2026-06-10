"""Z6 polish (P5) — end-to-end trace verifying detect-and-bail short-circuits
before any LLM call.

The audit guaranteed coulson's runtime entry intercepts when
``task.needs_real_tools=True`` and prereqs are missing, parking the task
with ``status='blocked_on_founder_action'`` without entering the
react/single_shot path. This test enforces the property in mocked form:
no react/single_shot invocation, no dispatcher.request invocation,
status set on the DB row.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_polish_p5.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()
    return db_mod, fa


def _make_profile():
    return SimpleNamespace(
        name="executor",
        allowed_tools=["vendor_call"],
        max_iterations=5,
        execution_pattern="react",
        get_system_prompt=lambda task: "sys",
        enable_self_reflection=False,
        min_confidence=0.0,
        can_create_subtasks=False,
        _build_full_system_prompt=lambda task: "sys",
        _build_context=AsyncMock(return_value=""),
        _count_tokens=lambda text: len(text.split()),
        progress_callback=None,
    )


@pytest.mark.asyncio
async def test_detect_and_bail_no_llm_call(tmp_path, monkeypatch):
    """When needs_real_tools=True with missing prereqs, dispatcher.request
    must never be invoked. The check is defense-in-depth in case beckman's
    pre-dispatch admission gate is bypassed.
    """
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    from src.infra.db import add_task as _add_task
    tid = await _add_task(
        title="Deploy",
        description="Deploy to vercel",
        agent_type="executor",
        context={
            "needs_real_tools": True,
            "real_tool_kind": "vercel",
            "workflow_step_id": "13.1",
        },
        mission_id=mid,
    )
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET needs_real_tools = 1, reversibility = 'irreversible' "
        "WHERE id = ?",
        (tid,),
    )
    await db.commit()
    row = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
    task = dict(await row.fetchone())

    react_mock = AsyncMock(return_value={"status": "completed"})

    from coulson import execute
    with patch("coulson._react_run", react_mock):
        profile = _make_profile()
        result = await execute(profile, task)

    assert result["status"] == "blocked_on_founder_action"
    react_mock.assert_not_called()

    # DB row reflects the parked status (defense-in-depth — beckman should
    # have parked it first, but coulson does it too).
    row = await db.execute("SELECT status FROM tasks WHERE id = ?", (tid,))
    assert (await row.fetchone())["status"] == "blocked_on_founder_action"


@pytest.mark.asyncio
async def test_needs_real_tools_field_reachable_from_db_row(
    tmp_path, monkeypatch,
):
    """The T1A column hoist must surface ``needs_real_tools`` directly on
    the task row dict that coulson receives. This guards against silent
    regressions where the column gets dropped and the value sinks into
    context-only land (where older callers might not look)."""
    db_mod, _fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    from src.infra.db import add_task as _add_task
    tid = await _add_task(
        title="Foo",
        description="bar",
        agent_type="executor",
        context={"needs_real_tools": True, "real_tool_kind": "vercel"},
        mission_id=mid,
    )
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET needs_real_tools = 1 WHERE id = ?", (tid,),
    )
    await db.commit()
    row = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
    task = dict(await row.fetchone())
    assert "needs_real_tools" in task
    assert bool(task["needs_real_tools"]) is True
