"""TDD tests for widened general_beckman.enqueue() signature (Task 2)."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
import general_beckman as _gb
from general_beckman import enqueue
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "enqueue_contract.db"
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
async def test_enqueue_default_returns_task_id(tmp_path, monkeypatch):
    """Old behaviour: plain enqueue(spec) returns an int task_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        spec = {"title": "hello", "description": "world", "agent_type": "coder"}
        result = await enqueue(spec)
        assert isinstance(result, int), f"Expected int task_id, got {type(result)}"
        assert result > 0
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_propagates_kind_to_db_row(tmp_path, monkeypatch):
    """spec['kind'] must be persisted to the tasks.kind column."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        spec = {
            "title": "overhead task",
            "description": "d",
            "agent_type": "grader",
            "kind": "overhead",
        }
        task_id = await enqueue(spec)
        assert task_id is not None
        row = await _db_mod.get_task(task_id)
        assert row["kind"] == "overhead", f"Expected 'overhead', got {row['kind']!r}"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_propagates_parent_id_to_db_row(tmp_path, monkeypatch):
    """parent_id kwarg must be stored in tasks.parent_task_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        parent_id = await enqueue({"title": "parent", "description": "p", "agent_type": "coder"})
        child_id = await enqueue(
            {"title": "child", "description": "c", "agent_type": "coder"},
            parent_id=parent_id,
        )
        assert child_id is not None
        row = await _db_mod.get_task(child_id)
        assert row["parent_task_id"] == parent_id, (
            f"Expected parent_task_id={parent_id}, got {row['parent_task_id']}"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_stores_continuation_in_context_envelope(tmp_path, monkeypatch):
    """on_complete and next_task_spec must land in context['beckman']."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        next_spec = {"title": "followup", "description": "f"}
        task_id = await enqueue(
            {"title": "main", "description": "m", "agent_type": "coder"},
            on_complete="agent.resume",
            next_task_spec=next_spec,
        )
        assert task_id is not None
        row = await _db_mod.get_task(task_id)
        ctx_raw = row.get("context") or "{}"
        ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
        beckman = ctx.get("beckman", {})
        assert beckman.get("on_complete") == "agent.resume", (
            f"on_complete missing/wrong: {beckman}"
        )
        assert beckman.get("next_task_spec") == next_spec, (
            f"next_task_spec missing/wrong: {beckman}"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_await_inline_blocks_until_resolved(tmp_path, monkeypatch):
    """await_inline=True should block until resolve_inline() is called."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import resolve_inline, TaskResult

        captured_id: list[int] = []

        async def _background():
            spec = {"title": "inline", "description": "d", "agent_type": "coder"}
            # We need the task_id before blocking; patch timeout high
            task_id_holder: list[int] = []

            # To get the task_id we do a dry enqueue first
            dry_id = await enqueue({"title": "inline_dry", "description": "d", "agent_type": "executor"})
            captured_id.append(dry_id)

            result = await enqueue(spec, await_inline=True)
            return result

        # Start background enqueue
        # We need a way to get the task_id that will be created — we know it
        # will be > the last inserted id. So we sniff the DB after a tiny sleep.
        bg = asyncio.ensure_future(_background())
        await asyncio.sleep(0.05)  # let the enqueue start

        # The inline waiter should have registered by now; find the task_id
        from general_beckman import _inline_waiters
        assert len(_inline_waiters) >= 1, "No inline waiter registered"
        waiter_id = next(iter(_inline_waiters))

        tr = TaskResult(status="completed", result={"x": 1}, error=None)
        resolve_inline(waiter_id, tr)

        final = await asyncio.wait_for(bg, timeout=5)
        assert final.result == {"x": 1}
        assert final.status == "completed"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_await_inline_raises_on_timeout(tmp_path, monkeypatch):
    """await_inline=True should raise asyncio.TimeoutError when not resolved."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        # Patch INLINE_TIMEOUT to a tiny value
        monkeypatch.setattr(_gb, "INLINE_TIMEOUT", 0.05)
        spec = {"title": "timeout_test", "description": "d", "agent_type": "coder"}
        with pytest.raises(asyncio.TimeoutError):
            await enqueue(spec, await_inline=True)
    finally:
        await _close_db()
