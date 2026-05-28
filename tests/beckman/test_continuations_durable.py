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


@pytest.mark.asyncio
async def test_enqueue_with_continuation_returns_fresh_id_and_row(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue
        cid = await enqueue(
            {"title": "child", "description": "d", "agent_type": "coder"},
            on_complete="t.resume", cont_state={"parent_id": 7},
        )
        assert isinstance(cid, int)
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, state_json FROM continuations WHERE child_task_id=?",
            (cid,),
        )
        row = await cur.fetchone()
        assert row[0] == "t.resume"
        assert json.loads(row[1]) == {"parent_id": 7}
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_rejects_await_inline_plus_on_complete(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue
        with pytest.raises(ValueError):
            await enqueue(
                {"title": "x", "description": "d", "agent_type": "coder"},
                await_inline=True, on_complete="t.resume",
            )
    finally:
        await _close_db()


async def _set_task(task_id, status, result_json=None):
    db = await _db_mod.get_db()
    if result_json is None:
        await db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    else:
        await db.execute(
            "UPDATE tasks SET status=?, result=? WHERE id=?",
            (status, result_json, task_id),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_on_task_finished_fires_resume_with_state(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, state))

        register_resume("t.resume", resume)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_complete="t.resume", cont_state={"parent_id": 99},
        )
        # SP1.1: fire trigger reads DB status; must be terminal before calling.
        await _set_task(cid, "completed", json.dumps({"result": "ok"}))
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)
        assert seen == [(cid, {"parent_id": 99})]
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_double_on_task_finished_fires_once(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        calls = []

        async def resume(task_id, result, state):
            calls.append(task_id)

        register_resume("t.resume", resume)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_complete="t.resume",
        )
        # SP1.1: fire trigger reads DB status; must be terminal before calling.
        await _set_task(cid, "completed", json.dumps({"result": "ok"}))
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)
        assert calls == [cid], f"expected single fire, got {calls}"
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


async def _set_task_status(task_id, status, result_json=None):
    db = await _db_mod.get_db()
    if result_json is None:
        await db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    else:
        await db.execute(
            "UPDATE tasks SET status=?, result=? WHERE id=?",
            (status, result_json, task_id),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_reconcile_fires_terminal_child_with_reconstructed_result(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, reconcile_continuations, _HANDLERS,
        )
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, result.get("verdict"), state))

        register_resume("t.resume", resume)
        cid = await _add(on_complete="t.resume", cont_state={"p": 5})
        await _set_task_status(cid, "completed", json.dumps({"verdict": "pass"}))
        await reconcile_continuations()
        await asyncio.sleep(0.05)
        assert seen == [(cid, "pass", {"p": 5})], (
            f"reconcile must fire with reconstructed result: {seen}"
        )
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_ttl_expires_dead_stale_child(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, reconcile_continuations, _HANDLERS,
        )
        errs = []

        async def on_err(task_id, result, state):
            errs.append(task_id)

        register_resume("t.err", on_err)
        cid = await _add(on_error="t.err")
        await _set_task_status(cid, "processing")
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE continuations SET created_at='2000-01-01 00:00:00' "
            "WHERE child_task_id=?", (cid,))
        await db.commit()
        await reconcile_continuations(ttl_seconds=3600)
        await asyncio.sleep(0.05)
        assert errs == [cid], "dead stale child must expire via on_error"
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_ttl_leaves_alive_child_pending(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import src.core.in_flight as _if
        from general_beckman.continuations import reconcile_continuations
        cid = await _add(on_complete="t.resume")
        await _set_task_status(cid, "processing")
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE continuations SET created_at='2000-01-01 00:00:00' "
            "WHERE child_task_id=?", (cid,))
        await db.commit()

        class _E:
            task_id = cid
        monkeypatch.setattr(_if, "in_flight_snapshot", lambda: [_E()])

        await reconcile_continuations(ttl_seconds=3600)
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id=?", (cid,))
        assert (await cur.fetchone())[0] == "pending", "alive child must stay pending"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_existing_handlers_fire_with_empty_state(tmp_path, monkeypatch):
    """analytics_digest + classify_signals handlers run under the durable path
    with state={} (they don't use state). Proves the 3-arg migration."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import register_startup_handlers, _HANDLERS

        register_startup_handlers()
        assert "growth.store_weekly_digest" in _HANDLERS
        assert "growth.classify_signals_complete" in _HANDLERS

        # Calling each with 3 args must not raise (they early-return on empty input).
        await _HANDLERS["growth.store_weekly_digest"](1, {"result": ""}, {})
        await _HANDLERS["growth.classify_signals_complete"](1, {"result": {}}, {})
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_failed_then_retried_completed_fires_success_resume(tmp_path, monkeypatch):
    """C1 regression: a transient failed → retry → success sequence MUST fire
    the success resume on the eventual completion. Pre-fix the row claimed on
    the first failed attempt and silently suppressed the retry's resume."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, result.get("status"), state))

        register_resume("t.resume", resume)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_complete="t.resume", cont_state={"p": 1},
        )
        # Simulate: agent reports failed → apply_failed re-pends.
        await _set_task(cid, "pending")
        await on_task_finished(cid, {"status": "failed", "error": "transient"})
        await asyncio.sleep(0.05)
        assert seen == [], (
            f"resume must NOT fire on a re-pended transient failure: {seen}"
        )

        # Retry succeeds.
        await _set_task(cid, "completed", json.dumps({"result": "ok"}))
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)
        assert seen == [(cid, "completed", {"p": 1})], (
            f"resume MUST fire on the retry's completion: {seen}"
        )
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_failed_exhausted_fires_on_error_on_true_terminal(tmp_path, monkeypatch):
    """A truly terminal failed task (retries exhausted, DB status='failed')
    fires on_error with the agent's last result."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        errs = []

        async def on_err(task_id, result, state):
            errs.append((task_id, result.get("error"), state))

        register_resume("t.err", on_err)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_error="t.err", cont_state={"p": 2},
        )
        # Simulate: retries exhausted — set worker_attempts to max so that
        # _retry_or_dlq fires DLQ (writes status='failed') rather than re-pend.
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE tasks SET worker_attempts=15, max_worker_attempts=15 WHERE id=?",
            (cid,),
        )
        await db.commit()
        await on_task_finished(cid, {"status": "failed", "error": "final"})
        await asyncio.sleep(0.05)
        assert errs == [(cid, "final", {"p": 2})], errs
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_handlers_see_agent_result_not_posthook_flip(tmp_path, monkeypatch):
    """C2 regression: if post_execute_workflow_step flips result['status'] or
    rewrites result['result'], the handler must still see the AGENT's snapshot
    captured before the post-hook ran.

    Scenario: agent reports completed with AGENT_TRUE_OUTPUT; the post-hook
    flips result['status'] to 'failed' and overwrites result['result']. Retries
    are exhausted so _retry_or_dlq fires DLQ → DB status stays 'failed' → the
    on_error handler fires with the AGENT snapshot (not the post-hook mutation).
    """
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import on_task_finished, enqueue
        from general_beckman.continuations import register_resume, _HANDLERS
        import src.workflows.engine.hooks as _hooks_mod

        seen_result = []
        seen_status = []

        async def on_err(task_id, result, state):
            seen_result.append(result.get("result"))
            seen_status.append(result.get("status"))

        register_resume("t.err", on_err)

        # Patch is_workflow_step to True and post_execute_workflow_step to
        # MUTATE the in-memory result IN PLACE (simulating a degenerate-output flip).
        async def _evil_posthook(task, result):
            if isinstance(result, dict):
                result["status"] = "failed"
                result["result"] = "POST_HOOK_OVERWROTE_THIS"

        monkeypatch.setattr(_hooks_mod, "is_workflow_step", lambda ctx: True)
        monkeypatch.setattr(_hooks_mod, "post_execute_workflow_step", _evil_posthook)

        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder",
             "context": {"workflow_step": True}},
            on_error="t.err",
        )
        # Exhaust retries so _retry_or_dlq fires DLQ (DB status stays 'failed')
        # rather than re-pending. This simulates the last retry attempt.
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE tasks SET worker_attempts=15, max_worker_attempts=15 WHERE id=?",
            (cid,),
        )
        await db.commit()

        await on_task_finished(cid, {"status": "completed", "result": "AGENT_TRUE_OUTPUT"})
        await asyncio.sleep(0.05)

        # The post-hook DID overwrite the in-memory result (to failed +
        # POST_HOOK_OVERWROTE_THIS), but the on_error handler must have received
        # the snapshot taken BEFORE the post-hook ran: status=completed,
        # result=AGENT_TRUE_OUTPUT.
        assert seen_result == ["AGENT_TRUE_OUTPUT"], (
            f"handler must see agent's pre-posthook result, got: {seen_result}"
        )
        assert seen_status == ["completed"], (
            f"handler must see agent's pre-posthook status, got: {seen_status}"
        )
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_reconcile_one_bad_row_does_not_poison_pass(tmp_path, monkeypatch):
    """Important #2 regression: per-row try/except in reconcile so one bad
    row doesn't abort the rest of the pass."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, reconcile_continuations, _HANDLERS,
        )
        fired = []

        async def resume(task_id, result, state):
            fired.append(task_id)

        register_resume("t.resume", resume)
        good = await _add(on_complete="t.resume", cont_state={"k": 1})
        bad = await _add(on_complete="t.resume", cont_state={"k": 2})
        # Both terminal in DB so reconcile would normally fire both.
        await _set_task(good, "completed", json.dumps({"result": "ok"}))
        await _set_task(bad, "completed", json.dumps({"result": "ok"}))

        # Poison: monkeypatch get_db to fail on the SECOND tasks SELECT.
        import src.infra.db as _db_mod
        orig_get_db = _db_mod.get_db
        calls = {"n": 0}

        class _PoisonConn:
            def __init__(self, real):
                self._real = real
            async def execute(self, sql, params=()):
                if "FROM tasks" in sql:
                    calls["n"] += 1
                    if calls["n"] == 2:
                        raise RuntimeError("simulated transient row failure")
                return await self._real.execute(sql, params)
            async def commit(self):
                return await self._real.commit()

        async def _poisoned_get_db():
            return _PoisonConn(await orig_get_db())

        monkeypatch.setattr(_db_mod, "get_db", _poisoned_get_db)
        await reconcile_continuations()
        await asyncio.sleep(0.05)
        # At least one row must still have fired (the pass kept going).
        assert len(fired) >= 1, f"reconcile aborted after one bad row: {fired}"
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()
