"""Z8 T5A — backup_verify executor tests (sqlite + postgres skip)."""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile

import pytest

from mr_roboto.executors.backup_verify import run as bv_run


@pytest.mark.asyncio
async def test_sqlite_backup_verify_happy_path(tmp_path):
    backup = tmp_path / "src.db"
    conn = sqlite3.connect(str(backup))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    sandbox = tmp_path / "sandbox"
    task = {"payload": {
        "action": "backup_verify",
        "backend": "sqlite",
        "backup_path": str(backup),
        "sandbox_dir": str(sandbox),
        "expect_tables": ["users", "orders"],
    }}
    res = await bv_run(task)
    assert res["ok"] is True
    assert res["backend"] == "sqlite"
    assert res["tables_seen"] == 2
    assert res["skipped"] is False


@pytest.mark.asyncio
async def test_sqlite_backup_verify_missing_table_fails(tmp_path):
    backup = tmp_path / "src.db"
    conn = sqlite3.connect(str(backup))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    task = {"payload": {
        "backend": "sqlite",
        "backup_path": str(backup),
        "sandbox_dir": str(tmp_path / "sandbox"),
        "expect_tables": ["users", "missing_table"],
    }}
    res = await bv_run(task)
    assert res["ok"] is False
    assert "missing expected tables" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_sqlite_backup_verify_no_backup_file(tmp_path):
    task = {"payload": {
        "backend": "sqlite",
        "backup_path": str(tmp_path / "nope.db"),
        "sandbox_dir": str(tmp_path / "sandbox"),
    }}
    res = await bv_run(task)
    assert res["ok"] is False
    assert "missing or not a file" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_postgres_skipped_when_pg_restore_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    task = {"payload": {
        "backend": "postgres",
        "dump_path": "/tmp/nope.dump",
    }}
    res = await bv_run(task)
    assert res["skipped"] is True
    assert "pg_restore not on PATH" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_unsupported_backend():
    res = await bv_run({"payload": {"backend": "redis"}})
    assert res["ok"] is False
    assert "unsupported backend" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_default_backend_is_sqlite(tmp_path):
    backup = tmp_path / "src.db"
    conn = sqlite3.connect(str(backup))
    conn.execute("CREATE TABLE t (x INT)")
    conn.commit()
    conn.close()
    res = await bv_run({"payload": {
        "backup_path": str(backup),
        "sandbox_dir": str(tmp_path / "sb"),
    }})
    assert res["backend"] == "sqlite"
    assert res["ok"] is True
