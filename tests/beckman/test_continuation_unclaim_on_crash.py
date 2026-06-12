"""FIX 2.2 — un-claim the continuation row when its handler crashes.

The claim (pending→fired) happens BEFORE handler dispatch; a handler that
raised used to leave the row consumed — the continuation was silently lost
(swap-chain ledgers stalled forever). Now ``dispatch_on_complete`` flips the
row back to 'pending' on a handler exception so the periodic reconcile
retries it — but ONLY while the row is inside CONTINUATION_TTL_SECONDS.
Past the TTL the row stays consumed, so a forever-crashing handler (incl.
the TTL-expiry on_error dispatch itself) can never produce an infinite
crash→un-claim→re-fire loop.
"""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
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


async def _add_child(resume="t.crash.resume"):
    from src.infra.db import add_task
    return await add_task(
        title="child", description="d", agent_type="coder",
        on_complete=resume, cont_state={"k": 1},
    )


async def _row_status(child_task_id):
    db = await _db_mod.get_db()
    cur = await db.execute(
        "SELECT status FROM continuations WHERE child_task_id=?",
        (child_task_id,),
    )
    row = await cur.fetchone()
    return row[0] if row else None


async def _backdate_row(child_task_id, seconds):
    """Shift continuations.created_at into the past by `seconds`."""
    db = await _db_mod.get_db()
    await db.execute(
        "UPDATE continuations SET created_at = "
        "datetime('now', '-' || ? || ' seconds') WHERE child_task_id=?",
        (seconds, child_task_id),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_handler_crash_unclaims_row_within_ttl(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            fire_for_task, _HANDLERS,
        )

        calls = []

        async def _boom(task_id, result, state):
            calls.append(task_id)
            raise RuntimeError("handler crashed")

        monkeypatch.setitem(_HANDLERS, "t.crash.resume", _boom)
        cid = await _add_child()

        fired = await fire_for_task(cid, {"status": "completed"}, "completed")
        assert fired is True
        await asyncio.sleep(0.05)  # detached dispatch + un-claim

        assert calls == [cid]
        assert await _row_status(cid) == "pending", (
            "a crashed handler must UN-CLAIM the row (fired→pending) so "
            "reconcile can retry it"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_handler_crash_past_ttl_stays_consumed(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            fire_for_task, _HANDLERS, CONTINUATION_TTL_SECONDS,
        )

        async def _boom(task_id, result, state):
            raise RuntimeError("handler crashed")

        monkeypatch.setitem(_HANDLERS, "t.crash.resume", _boom)
        cid = await _add_child()
        await _backdate_row(cid, CONTINUATION_TTL_SECONDS + 60)

        fired = await fire_for_task(cid, {"status": "completed"}, "completed")
        assert fired is True
        await asyncio.sleep(0.05)

        assert await _row_status(cid) == "fired", (
            "past the TTL a crashed handler must leave the row consumed — "
            "no infinite crash→un-claim→re-fire loop"
        )
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_handler_success_keeps_row_fired(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task, _HANDLERS

        async def _ok(task_id, result, state):
            return None

        monkeypatch.setitem(_HANDLERS, "t.crash.resume", _ok)
        cid = await _add_child()

        fired = await fire_for_task(cid, {"status": "completed"}, "completed")
        assert fired is True
        await asyncio.sleep(0.05)

        assert await _row_status(cid) == "fired"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_unclaimed_row_is_retried_by_reconcile(tmp_path, monkeypatch):
    """End-to-end recovery: crash → un-claim → reconcile re-fires → a now-
    healthy handler consumes the row."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            fire_for_task, reconcile_continuations, _HANDLERS,
        )

        calls = []
        crash = {"on": True}

        async def _flaky(task_id, result, state):
            calls.append(task_id)
            if crash["on"]:
                raise RuntimeError("transient handler crash")

        monkeypatch.setitem(_HANDLERS, "t.crash.resume", _flaky)
        cid = await _add_child()

        # Child terminal in the DB so reconcile reconstructs + re-fires.
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE tasks SET status='completed', result=? WHERE id=?",
            (json.dumps({"content": "x"}), cid),
        )
        await db.commit()

        await fire_for_task(cid, {"status": "completed"}, "completed")
        await asyncio.sleep(0.05)
        assert await _row_status(cid) == "pending"

        crash["on"] = False
        await reconcile_continuations()
        await asyncio.sleep(0.05)

        assert calls == [cid, cid], f"reconcile must retry the handler: {calls}"
        assert await _row_status(cid) == "fired"
    finally:
        await _close_db()
