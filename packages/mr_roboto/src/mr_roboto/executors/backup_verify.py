"""Z8 T5A — backup_verify mechanical executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "backup_verify"``
(or the cron verb ``"cron_backup_verify"``).

Two flavours:

- **sqlite** (``backend="sqlite"``) — copies ``backup_path`` into
  ``sandbox_dir`` and runs the recipe's smoke SELECT against the copy.
  Pure local, no external deps.
- **postgres** (``backend="postgres"``) — pg_restore into an ephemeral
  database via subprocess. Skipped when ``pg_restore`` is not on PATH.

Returns ``{"ok": bool, "backend": str, "tables_seen": int, "skipped": bool,
"reason": str|None}``.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.backup_verify")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    backend = (payload.get("backend") or "sqlite").lower()
    if backend == "sqlite":
        return _verify_sqlite(payload)
    if backend == "postgres":
        return _verify_postgres(payload)
    return {
        "ok": False,
        "backend": backend,
        "tables_seen": 0,
        "skipped": False,
        "reason": f"unsupported backend: {backend!r}",
    }


def _verify_sqlite(payload: dict) -> dict:
    backup_path = payload.get("backup_path") or ""
    sandbox_dir = payload.get("sandbox_dir") or "/tmp/backup_verify"
    expect_tables = payload.get("expect_tables") or []
    if isinstance(expect_tables, str):
        expect_tables = [t.strip() for t in expect_tables.split(",") if t.strip()]

    if not backup_path or not os.path.isfile(backup_path):
        return {
            "ok": False,
            "backend": "sqlite",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"backup_path missing or not a file: {backup_path!r}",
        }

    try:
        Path(sandbox_dir).mkdir(parents=True, exist_ok=True)
        restore_path = Path(sandbox_dir) / f"restored_{Path(backup_path).name}"
        shutil.copyfile(backup_path, restore_path)
    except Exception as e:
        return {
            "ok": False,
            "backend": "sqlite",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"copy failed: {e}",
        }

    try:
        conn = sqlite3.connect(str(restore_path))
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
    except Exception as e:
        return {
            "ok": False,
            "backend": "sqlite",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"smoke SELECT failed: {e}",
        }

    missing = [t for t in expect_tables if t not in tables]
    ok = not missing
    return {
        "ok": ok,
        "backend": "sqlite",
        "tables_seen": len(tables),
        "skipped": False,
        "reason": (
            None if ok else f"missing expected tables: {missing!r}"
        ),
        "restored_path": str(restore_path),
    }


def _verify_postgres(payload: dict) -> dict:
    dump_path = payload.get("dump_path") or ""
    db_name = payload.get("ephemeral_db_name") or "backup_verify_sandbox"
    expect_tables = payload.get("expect_tables") or []

    if shutil.which("pg_restore") is None:
        return {
            "ok": False,
            "backend": "postgres",
            "tables_seen": 0,
            "skipped": True,
            "reason": "pg_restore not on PATH",
        }

    if not dump_path or not os.path.isfile(dump_path):
        return {
            "ok": False,
            "backend": "postgres",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"dump_path missing or not a file: {dump_path!r}",
        }

    # v1: trust libpq env (PGHOST/PGUSER/PGPASSWORD). Caller must set them.
    try:
        # Recreate ephemeral DB.
        subprocess.run(
            ["dropdb", "--if-exists", db_name],
            check=False,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["createdb", db_name],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["pg_restore", "--dbname", db_name, "--no-owner", dump_path],
            check=True,
            capture_output=True,
            timeout=180,
        )
        # Smoke: count tables in public schema.
        out = subprocess.run(
            [
                "psql", "-d", db_name, "-At", "-c",
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public'",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        tables_seen = int((out.stdout or b"0").decode("utf-8").strip())
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "backend": "postgres",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"pg subprocess failed: {e.stderr!r}",
        }
    except Exception as e:
        return {
            "ok": False,
            "backend": "postgres",
            "tables_seen": 0,
            "skipped": False,
            "reason": f"pg verify failed: {e}",
        }

    ok = tables_seen > 0 and (
        not expect_tables or tables_seen >= len(expect_tables)
    )
    return {
        "ok": ok,
        "backend": "postgres",
        "tables_seen": tables_seen,
        "skipped": False,
        "reason": None if ok else "no tables found after restore",
    }
