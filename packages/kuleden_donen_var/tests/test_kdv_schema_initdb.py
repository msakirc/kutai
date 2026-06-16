"""Fresh-DB init_db creates kdv_state via the registered schema.

This is the live-restart guard for Phase B §4: the engine (dabidabi) no longer
owns the kdv_state DDL inline, so init_db() must create the table through the
registered-schema loop. Importing kuleden_donen_var registers it; this proves
init_db on a brand-new DB then materialises the table.

Mutates the dabidabi singleton DB path, so it snapshots + restores it.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3

import dabidabi
import kuleden_donen_var  # noqa: F401  registers kdv_state schema


def test_init_db_creates_kdv_state_on_fresh_db(tmp_path):
    assert "kuleden_donen_var_kdv" in dabidabi._registered_schemas

    fresh = str(tmp_path / "fresh.db")
    saved_path = dabidabi.DB_PATH
    try:
        dabidabi.configure(fresh)  # absolute path required
        asyncio.run(dabidabi.init_db())
        conn = sqlite3.connect(fresh)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='kdv_state'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None and row[0] == "kdv_state"
    finally:
        # Restore the original singleton path so sibling tests are unaffected.
        if saved_path and os.path.isabs(saved_path):
            dabidabi.configure(saved_path)
        else:
            dabidabi.DB_PATH = saved_path
            dabidabi._db_connection = None
            dabidabi._db_connection_path = None
