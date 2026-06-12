"""Sweep section 8 (pending-past-cap → force DLQ) must honor the SAME effective
cap as decide_retry and the admission cap-guard.

mission_79 #225600 (2026-05-31): a reviewer task correctly classified
error_category="availability" sat pending at worker_attempts=6 / raw
max_worker_attempts=6, waiting out a daily-quota reset (decide_retry gives
transient categories an effective cap of the full 15-step ladder, and the
admission cap-guard already applies effective_max_attempts). But sweep
section 8 compares ``worker_attempts >= COALESCE(max_worker_attempts, 15)``
against the RAW stored cap (6) and force-DLQ'd it with
"Worker attempts exceeded: 6/6" — defeating the transient ladder. The
"3-site unification" missed this SQL: the pure function was added and unit
tested, but the sweep never actually called it.

A genuine quality task at 6/6 MUST still be swept (regression guard).
"""
from __future__ import annotations

import pytest
import aiosqlite


def _reset_db(db_module, db_path: str):
    db_module._db_connection = None
    db_module._db_connection_path = None
    db_module.DB_PATH = db_path


async def _insert_pending_overcap(db_path, *, category, attempts=6, max_att=6):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, status, agent_type, runner, "
            "worker_attempts, max_worker_attempts, error_category) "
            "VALUES ('t', 'pending', 'reviewer', 'react', ?, ?, ?)",
            (attempts, max_att, category),
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        tid = (await cur.fetchone())[0]
        await db.commit()
    return tid


async def _status(db_path, tid):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT status FROM tasks WHERE id=?", (tid,))
        row = await cur.fetchone()
    return row[0] if row else None


async def _run_sweep(monkeypatch, db_module, db_path):
    _reset_db(db_module, db_path)
    # Silence the Telegram notifier — DLQ writes call it.
    import general_beckman.sweep as sweep_mod

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(sweep_mod, "_notify", _noop)
    await sweep_mod.sweep_queue()


@pytest.mark.asyncio
async def test_sweep_does_not_dlq_availability_below_effective_cap(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    tid = await _insert_pending_overcap(db_path, category="availability")
    await _run_sweep(monkeypatch, db_module, db_path)

    # availability effective cap is the 15-step ladder, so 6/6 is NOT exhausted —
    # the task must keep waiting, not be force-DLQ'd.
    assert await _status(db_path, tid) == "pending"


@pytest.mark.asyncio
async def test_sweep_still_dlqs_quality_at_cap(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    tid = await _insert_pending_overcap(db_path, category="quality")
    await _run_sweep(monkeypatch, db_module, db_path)

    # quality keeps the raw cap — 6/6 is genuinely exhausted → swept to DLQ.
    assert await _status(db_path, tid) == "failed"


@pytest.mark.asyncio
async def test_sweep_section1_writes_ladder_to_context_not_infra_resets(tmp_path, monkeypatch):
    """A task stuck >5min in 'processing' (and not in_flight) is re-pended
    with availability backoff. The ladder's seconds-of-last-delay must land
    in context['last_avail_delay'], NOT the infra_resets column (which stays
    a pure reset count). First step of the ladder is 60s."""
    import json
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    import src.infra.db as db_module
    _reset_db(db_module, db_path)
    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tasks (title, status, agent_type, runner, "
            "infra_resets, started_at) "
            "VALUES ('stuck', 'processing', 'coder', 'react', 0, "
            "datetime('now', '-10 minutes'))"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        tid = (await cur.fetchone())[0]
        await db.commit()

    await _run_sweep(monkeypatch, db_module, db_path)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT status, infra_resets, retry_reason, context "
            "FROM tasks WHERE id=?", (tid,)
        )
        row = await cur.fetchone()

    assert row["status"] == "pending"
    assert row["retry_reason"] == "availability"
    assert row["infra_resets"] == 0  # NOT overloaded with ladder seconds
    assert json.loads(row["context"] or "{}").get("last_avail_delay") == 60
