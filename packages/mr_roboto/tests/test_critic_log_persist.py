"""SP6 T2 — critic_gate._persist must actually write a critic_log row.

Regression guard: ``_persist`` previously used ``async with get_db() as db:``,
but ``src.infra.db.get_db`` is an ``async def`` that RETURNS a bare aiosqlite
connection (it is NOT an async context manager). ``async with get_db()`` raised
TypeError, which the function's best-effort ``except Exception`` swallowed →
zero critic_log rows were ever written (and the un-awaited coroutine produced a
``RuntimeWarning: coroutine 'get_db' was never awaited``).
"""
from __future__ import annotations

import json

import src.infra.db as _db_mod
import mr_roboto.critic_gate as _critic


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "critic_persist.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def test_persist_writes_one_critic_log_row(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _critic._persist(
            mission_id=7,
            action_name="git_commit",
            verdict="veto",
            reasons=["leaks"],
            payload_hash="hash123",
        )

        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT COUNT(*) FROM critic_log WHERE action_name = ?",
            ("git_commit",),
        )
        (count,) = await cur.fetchone()
        assert count == 1, f"expected exactly 1 critic_log row, got {count}"

        cur = await db.execute(
            "SELECT mission_id, verdict, reasons_json, redacted_payload_hash "
            "FROM critic_log WHERE action_name = ?",
            ("git_commit",),
        )
        row = await cur.fetchone()
        assert row[0] == 7
        assert row[1] == "veto"
        assert json.loads(row[2]) == ["leaks"]
        assert row[3] == "hash123"
    finally:
        await _close_db()
