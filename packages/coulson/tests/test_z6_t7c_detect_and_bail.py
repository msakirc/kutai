"""Z6 T7C — coulson detect needs_real_tools and short-circuit or inject."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t7c.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    return db_mod, fa


def _make_profile(execution_pattern: str = "react"):
    """Cheap duck-typed profile stand-in."""
    return SimpleNamespace(
        name="executor",
        allowed_tools=["vendor_call"],
        max_iterations=5,
        execution_pattern=execution_pattern,
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
async def test_needs_real_tools_missing_creds_short_circuits(
    tmp_path, monkeypatch,
):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    from src.infra.db import add_task as _add_task
    tid = await _add_task(
        title="Deploy", description="Deploy to vercel",
        agent_type="executor",
        context={
            "needs_real_tools": True,
            "real_tool_kind": "vercel",
            "workflow_step_id": "13.1",
        },
        mission_id=mid,
    )
    # Manually mirror the column hoist so admission's column-pref path wins.
    db = await db_mod.get_db()
    await db.execute(
        "UPDATE tasks SET needs_real_tools = 1, reversibility = 'irreversible' "
        "WHERE id = ?",
        (tid,),
    )
    await db.commit()
    row = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
    task = dict(await row.fetchone())

    # Stub registry so it has no adapter — forces vendor_enroll path
    from coulson import execute
    react_mock = AsyncMock(return_value={"status": "completed"})
    with patch("coulson._react_run", react_mock):
        profile = _make_profile()
        result = await execute(profile, task)

    assert result["status"] == "blocked_on_founder_action"
    react_mock.assert_not_called()

    # A founder_action must have been emitted.
    actions = await fa.list_pending()
    assert len(actions) >= 1


@pytest.mark.asyncio
async def test_needs_real_tools_with_creds_proceeds_with_injection(
    tmp_path, monkeypatch,
):
    """When adapter + credentials are present, runtime proceeds but injects
    the real-world warning block into the task description."""
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    from src.infra.db import add_task as _add_task
    tid = await _add_task(
        title="Deploy", description="Original task body",
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
        "UPDATE tasks SET needs_real_tools = 1, reversibility = 'partial' "
        "WHERE id = ?",
        (tid,),
    )
    await db.commit()
    row = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
    task = dict(await row.fetchone())

    # Stub admission to return admit=True (mock adapter+creds satisfied).
    from general_beckman.z6_admission import AdmissionResult
    admit_ok = AsyncMock(
        return_value=AdmissionResult(admit=True, reason="prereqs ok"),
    )
    react_mock = AsyncMock(return_value={"status": "completed", "result": "ok"})
    from coulson import execute
    with patch(
        "general_beckman.z6_admission.check_z6_admission", admit_ok,
    ), patch("coulson._react_run", react_mock):
        profile = _make_profile()
        result = await execute(profile, task)

    assert result["status"] == "completed"
    react_mock.assert_awaited_once()
    # The task passed into react.run must carry the injected warning block.
    called_task = react_mock.await_args.args[1]
    from coulson.system_prompt_blocks import REAL_WORLD_BLOCK_MARKER
    assert REAL_WORLD_BLOCK_MARKER in called_task["description"]
    assert "Original task body" in called_task["description"]


@pytest.mark.asyncio
async def test_task_without_flag_proceeds_normally(tmp_path, monkeypatch):
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    task = {
        "id": 999,
        "mission_id": mid,
        "title": "Normal",
        "description": "Plain task body",
        "context": json.dumps({}),
        "needs_real_tools": 0,
    }
    react_mock = AsyncMock(return_value={"status": "completed"})
    from coulson import execute
    with patch("coulson._react_run", react_mock):
        profile = _make_profile()
        await execute(profile, task)
    react_mock.assert_awaited_once()
    called_task = react_mock.await_args.args[1]
    from coulson.system_prompt_blocks import REAL_WORLD_BLOCK_MARKER
    # No injection.
    assert REAL_WORLD_BLOCK_MARKER not in called_task["description"]
    # No founder_action either.
    import src.founder_actions as fa2
    assert await fa2.list_pending() == []


@pytest.mark.asyncio
async def test_injection_is_idempotent(tmp_path, monkeypatch):
    """Re-running execute on the same task does not double-inject."""
    db_mod, fa = await _setup_db(tmp_path, monkeypatch)
    mid = await db_mod.add_mission("M", "")
    from src.infra.db import add_task as _add_task
    tid = await _add_task(
        title="Deploy", description="Body",
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
        "UPDATE tasks SET needs_real_tools = 1 WHERE id = ?", (tid,),
    )
    await db.commit()
    row = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
    task = dict(await row.fetchone())

    from general_beckman.z6_admission import AdmissionResult
    admit_ok = AsyncMock(
        return_value=AdmissionResult(admit=True, reason="prereqs ok"),
    )
    react_mock = AsyncMock(return_value={"status": "completed"})
    from coulson import execute
    with patch(
        "general_beckman.z6_admission.check_z6_admission", admit_ok,
    ), patch("coulson._react_run", react_mock):
        profile = _make_profile()
        await execute(profile, task)
        # Second invocation on the (mutated) task.
        await execute(profile, task)
    from coulson.system_prompt_blocks import REAL_WORLD_BLOCK_MARKER
    # Only one marker — second run saw it already present and skipped.
    assert task["description"].count(REAL_WORLD_BLOCK_MARKER) == 1
