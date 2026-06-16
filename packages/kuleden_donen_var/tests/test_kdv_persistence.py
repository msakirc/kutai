"""KDV-state persistence + schema ownership, relocated into kuleden_donen_var.

Phase B §4: the kdv_state table's DDL + read/write helpers now live in this
package (owner), not in the dabidabi engine. These tests lock in:
  1. importing the package registers the kdv schema with dabidabi,
  2. create_kdv_schema actually creates the table,
  3. the save -> sqlite -> load_sync round-trip still works end to end.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

import aiosqlite
import pytest

import dabidabi
import kuleden_donen_var
from kuleden_donen_var import persistence as kdv_persistence
from kuleden_donen_var.schema import create_kdv_schema
from kuleden_donen_var import KuledenConfig, KuledenDonenVar


def test_schema_registered_on_import():
    """Importing the package must register the kdv schema with the engine,
    so init_db() (which runs registered schemas) creates kdv_state on a
    fresh DB without the engine owning the DDL."""
    assert "kuleden_donen_var_kdv" in dabidabi._registered_schemas


def test_create_kdv_schema_makes_table(tmp_path):
    """create_kdv_schema runs the owned DDL against a connection."""
    db_path = str(tmp_path / "kdv.db")

    async def _run():
        async with aiosqlite.connect(db_path) as db:
            await create_kdv_schema(db)
            await db.commit()

    asyncio.run(_run())

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kdv_state'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == "kdv_state"


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "kdv_test.db")

    async def _run():
        async with aiosqlite.connect(p) as db:
            await create_kdv_schema(db)
            await db.commit()

    asyncio.run(_run())
    return p


def test_round_trip_save_load_sync(db_path):
    """save() writes per-scope rows; load_sync() restores them onto a fresh
    KDV. Exercises the relocated read/write helpers via the engine's
    connect_aux/connect_aux_sync (not src.infra.db)."""
    src = KuledenDonenVar(KuledenConfig())
    src.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    for _ in range(7):
        src.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    for _ in range(3):
        src.record_failure("groq/llama-8b", "groq", "server_error")
    expected_rate = src.recent_success_rate("groq/llama-8b")
    expected_n = src.recent_samples_n("groq/llama-8b")

    asyncio.run(kdv_persistence.save(src, db_path))

    dst = KuledenDonenVar(KuledenConfig())
    dst.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    report = kdv_persistence.load_sync(dst, db_path)

    assert report["outcomes"] == 1
    assert dst.recent_samples_n("groq/llama-8b") == expected_n
    assert dst.recent_success_rate("groq/llama-8b") == pytest.approx(expected_rate)


def test_load_sync_drops_stale_rows(db_path):
    src = KuledenDonenVar(KuledenConfig())
    src.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    for _ in range(6):
        src.post_call("groq/llama-8b", "groq", headers={}, token_count=10)
    asyncio.run(kdv_persistence.save(src, db_path))

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE kdv_state SET last_persisted = ? WHERE scope='outcomes'",
            (time.time() - 48 * 3600,),
        )
        conn.commit()
    finally:
        conn.close()

    dst = KuledenDonenVar(KuledenConfig())
    dst.register("groq/llama-8b", "groq", rpm=30, tpm=131072)
    report = kdv_persistence.load_sync(dst, db_path)
    assert report["outcomes"] == 0
    assert report["skipped_stale"] >= 1
