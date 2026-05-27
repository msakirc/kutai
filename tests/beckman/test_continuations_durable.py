"""SP1 durable continuation substrate — host-path, DB-isolated tests."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated temp DB per test (mirrors tests/beckman/test_continuations.py)."""
    db_file = tmp_path / "cps.db"
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
async def test_continuations_table_and_index_created(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='continuations'"
        )
        assert await cur.fetchone() is not None, "continuations table missing"
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_continuations_pending'"
        )
        assert await cur.fetchone() is not None, "status index missing"
        cur = await db.execute("PRAGMA table_info(continuations)")
        cols = {row[1] for row in await cur.fetchall()}
        assert cols == {
            "child_task_id", "resume_name", "on_error_name",
            "state_json", "status", "created_at",
        }, f"unexpected columns: {cols}"
    finally:
        await _close_db()


async def _add(**kw):
    from src.infra.db import add_task
    base = dict(title="t", description="d", agent_type="coder")
    base.update(kw)
    return await add_task(**base)


@pytest.mark.asyncio
async def test_add_task_writes_continuation_row_atomically(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        cid = await _add(on_complete="x.resume", cont_state={"k": 1})
        assert isinstance(cid, int)
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, on_error_name, state_json, status "
            "FROM continuations WHERE child_task_id = ?", (cid,)
        )
        row = await cur.fetchone()
        assert row is not None, "continuation row not written"
        assert row[0] == "x.resume"
        assert row[1] is None
        assert json.loads(row[2]) == {"k": 1}
        assert row[3] == "pending"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_add_task_on_error_only_writes_empty_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        cid = await _add(on_error="x.fail")
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, on_error_name FROM continuations WHERE child_task_id = ?",
            (cid,),
        )
        row = await cur.fetchone()
        assert row[0] == ""        # empty resume sentinel
        assert row[1] == "x.fail"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_continuation_children_are_never_deduped(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        a = await _add(on_complete="x.resume")
        b = await _add(on_complete="x.resume")
        assert a != b, "continuation children were deduped (PK collision risk)"
        db = await _db_mod.get_db()
        cur = await db.execute("SELECT COUNT(*) FROM continuations")
        assert (await cur.fetchone())[0] == 2
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_plain_tasks_still_dedup(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        a = await _add()
        b = await _add()        # identical → deduped to None
        assert isinstance(a, int)
        assert b is None, f"expected dedup (None), got {b}"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_claim_for_fire_is_single_winner(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import claim_for_fire
        cid = await _add(on_complete="x.resume", cont_state={"v": 9})
        first = await claim_for_fire(cid)
        second = await claim_for_fire(cid)
        assert first is not None and first["resume_name"] == "x.resume"
        assert first["state"] == {"v": 9}
        assert second is None, "second claim must lose"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_for_task_success_dispatches_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, fire_for_task, _HANDLERS,
        )
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, result, state))

        register_resume("t.resume", resume)
        cid = await _add(on_complete="t.resume", cont_state={"s": 1})
        fired = await fire_for_task(cid, {"status": "completed", "result": "ok"},
                                    "completed")
        await asyncio.sleep(0.05)
        assert fired is True
        assert seen == [(cid, {"status": "completed", "result": "ok"}, {"s": 1})]
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_fire_failed_with_on_error_dispatches_on_error(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, fire_for_task, _HANDLERS,
        )
        errs = []

        async def on_err(task_id, result, state):
            errs.append((task_id, result.get("status"), state))

        register_resume("t.err", on_err)
        cid = await _add(on_error="t.err", cont_state={"p": 2})
        fired = await fire_for_task(cid, {"status": "failed", "error": "boom"},
                                    "failed")
        await asyncio.sleep(0.05)
        assert fired is True
        assert errs == [(cid, "failed", {"p": 2})]
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_fire_failed_without_on_error_is_noop(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        cid = await _add(on_complete="t.resume")     # resume only, no on_error
        fired = await fire_for_task(cid, {"status": "failed"}, "failed")
        assert fired is True
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id = ?", (cid,)
        )
        assert (await cur.fetchone())[0] == "fired"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_needs_clarification_leaves_pending(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        cid = await _add(on_complete="t.resume")
        fired = await fire_for_task(cid, {"status": "needs_clarification"},
                                    "needs_clarification")
        assert fired is False, "needs_clarification must not fire"
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id = ?", (cid,)
        )
        assert (await cur.fetchone())[0] == "pending", "row must stay pending"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_no_row_returns_false(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        plain = await _add()    # no continuation
        assert await fire_for_task(plain, {"status": "completed"}, "completed") is False
    finally:
        await _close_db()
