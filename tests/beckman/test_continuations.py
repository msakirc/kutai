"""TDD tests for continuation registry + terminal hook (Task 3)."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
import general_beckman as _gb
from general_beckman import enqueue, on_task_finished
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp
from general_beckman.continuations import register, dispatch_on_complete, _HANDLERS


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "continuations.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_register_and_dispatch_handler(tmp_path, monkeypatch):
    """register() + dispatch_on_complete() must invoke the handler with correct args."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        calls = []

        async def my_handler(task_id: int, result: dict) -> None:
            calls.append((task_id, result))

        register("test.handler", my_handler)
        await dispatch_on_complete("test.handler", 42, {"x": 1})
        assert calls == [(42, {"x": 1})], f"unexpected calls: {calls}"
    finally:
        _HANDLERS.pop("test.handler", None)
        await _close_db()


@pytest.mark.asyncio
async def test_handler_crash_does_not_propagate(tmp_path, monkeypatch):
    """A crashing handler must not raise — dispatch swallows it."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        async def bad_handler(task_id: int, result: dict) -> None:
            raise RuntimeError("boom")

        register("test.crash", bad_handler)
        # Must NOT raise
        await dispatch_on_complete("test.crash", 1, {})
    finally:
        _HANDLERS.pop("test.crash", None)
        await _close_db()


@pytest.mark.asyncio
async def test_terminal_router_fires_on_complete(tmp_path, monkeypatch):
    """on_task_finished with on_complete in context must invoke the registered handler."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        invoked = []

        async def my_resume(task_id: int, result: dict) -> None:
            invoked.append(task_id)

        register("my.handler", my_resume)

        task_id = await enqueue(
            {"title": "t", "description": "d", "agent_type": "coder"},
            on_complete="my.handler",
        )
        # Simulate terminal — on_task_finished handles completion routing
        await on_task_finished(task_id, {"status": "completed", "result": "ok"})

        # The handler is dispatched in a detached create_task; give it a tick
        await asyncio.sleep(0.05)
        assert task_id in invoked, f"handler not invoked; invoked={invoked}"
    finally:
        _HANDLERS.pop("my.handler", None)
        await _close_db()


@pytest.mark.asyncio
async def test_terminal_router_chains_next_task_spec(tmp_path, monkeypatch):
    """on_task_finished with next_task_spec must create a child task in the DB."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        parent_id = await enqueue(
            {"title": "parent", "description": "d", "agent_type": "coder"},
            next_task_spec={"title": "child", "description": "cd", "agent_type": "coder"},
        )
        await on_task_finished(parent_id, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)

        db = await _db_mod.get_db()
        cursor = await db.execute(
            "SELECT id, parent_task_id FROM tasks WHERE parent_task_id = ?",
            (parent_id,),
        )
        rows = list(await cursor.fetchall())
        assert len(rows) == 1, f"Expected 1 child task, got {len(rows)}: {rows}"
        assert rows[0]["parent_task_id"] == parent_id
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_terminal_router_resolves_inline_waiter(tmp_path, monkeypatch):
    """on_task_finished must resolve any pending await_inline waiter for the task."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import _inline_waiters

        spec = {"title": "inline_t", "description": "d", "agent_type": "coder"}
        bg = asyncio.ensure_future(enqueue(spec, await_inline=True))
        await asyncio.sleep(0.05)

        # Find the waiter
        assert len(_inline_waiters) >= 1, "No inline waiter registered"
        waiter_id = next(iter(_inline_waiters))

        # Simulate terminal via on_task_finished
        await on_task_finished(waiter_id, {"status": "completed", "result": "done"})
        await asyncio.sleep(0.05)

        final = await asyncio.wait_for(bg, timeout=5)
        assert final.status == "completed"
    finally:
        await _close_db()
