"""Apply a database migration to an ephemeral sandbox — Z2 T3A.

Mechanical executor. No LLM. Stack-aware dispatch:

- ``stack_hint`` contains "sqlite" → create a temp SQLite DB, apply each
  target `.sql` file via the sqlite3 stdlib (or alembic if alembic.ini
  found), then discard.
- ``stack_hint`` contains "postgres" → testcontainers path. Only runs when
  ``enable_testcontainers=True`` in the payload.  If testcontainers is not
  installed, returns a soft-skip verdict.
- Unknown stack → alembic offline mode (``alembic upgrade head --sql``).
  Catches syntax errors, misses FK ordering and extension deps.

Return shape
------------
``{ok, error, exit, stdout_tail, stderr_tail, duration_s, applied_files,
skipped, stack_used}``

- ``ok=False`` on apply error; ``error`` carries a detailed string.
- ``ok=True, skipped=True`` when testcontainers absent (soft-skip).
- ``ok=True, warning="slow_migration"`` when apply takes >30 s.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.apply_migration")

_SLOW_THRESHOLD_S = 30.0


def _tail(text: str, max_bytes: int = 4096) -> str:
    if len(text) > max_bytes:
        return text[-max_bytes:]
    return text


async def _apply_sqlite(
    target_files: list[str],
    workspace_path: str,
    timeout_s: float,
) -> dict[str, Any]:
    """SQLite path: ephemeral in-memory DB; apply each .sql file in order.

    If alembic.ini is present in workspace_path, delegate to
    ``alembic upgrade head`` against a temp SQLite file URI (allows
    alembic migration scripts to find the DB).  Otherwise, run each
    target .sql file directly via the sqlite3 stdlib.
    """
    from mr_roboto.run_cmd import run_cmd

    start = time.monotonic()

    # Resolve target files under workspace_path when they're relative.
    resolved: list[str] = []
    for f in target_files:
        if os.path.isabs(f):
            resolved.append(f)
        else:
            resolved.append(os.path.join(workspace_path, f))

    alembic_ini = os.path.join(workspace_path, "alembic.ini")
    has_alembic = os.path.isfile(alembic_ini)

    if has_alembic:
        # Use alembic against a temp SQLite file.
        fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix="kutai_mig_")
        os.close(fd)
        try:
            db_url = f"sqlite:///{tmp_db}"
            env = dict(os.environ, DATABASE_URL=db_url)
            raw = await run_cmd(
                mission_id=None,
                cmd=["alembic", "-c", alembic_ini, "upgrade", "head"],
                cwd=None,
                timeout_s=timeout_s,
                env=env,
                require_exit_zero=False,
                workspace_path=workspace_path,
            )
        finally:
            try:
                os.unlink(tmp_db)
            except OSError:
                pass

        duration = time.monotonic() - start
        ok = raw.get("ok") and raw.get("exit") == 0
        result: dict[str, Any] = {
            "ok": ok,
            "error": "" if ok else (
                f"alembic upgrade head failed (exit={raw.get('exit')}): "
                f"{_tail(raw.get('stderr_tail') or raw.get('stdout_tail') or '', 400)}"
            ),
            "exit": raw.get("exit"),
            "stdout_tail": _tail(raw.get("stdout_tail") or ""),
            "stderr_tail": _tail(raw.get("stderr_tail") or ""),
            "duration_s": duration,
            "applied_files": target_files,
            "skipped": False,
            "stack_used": "sqlite_alembic",
        }
        if ok and duration > _SLOW_THRESHOLD_S:
            result["warning"] = "slow_migration"
        return result

    # No alembic.ini — apply each .sql file directly via sqlite3 stdlib.
    applied: list[str] = []
    error_str = ""
    stdout_lines: list[str] = []
    conn: sqlite3.Connection | None = None
    try:
        # Use a named temp file so we can discard after.
        fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix="kutai_sql_")
        os.close(fd)
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute("PRAGMA journal_mode=WAL")
            for path in resolved:
                if not os.path.isfile(path):
                    error_str = f"migration file not found: {path}"
                    break
                try:
                    sql_text = open(path, "r", encoding="utf-8").read()
                except OSError as exc:
                    error_str = f"cannot read {path}: {exc}"
                    break
                try:
                    conn.executescript(sql_text)
                    conn.commit()
                    applied.append(os.path.basename(path))
                    stdout_lines.append(f"applied: {os.path.basename(path)}")
                except sqlite3.Error as exc:
                    error_str = f"sqlite3 error in {os.path.basename(path)}: {exc}"
                    break
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                os.unlink(tmp_db)
            except OSError:
                pass
    except Exception as exc:
        error_str = f"unexpected error in sqlite path: {exc}"

    duration = time.monotonic() - start
    ok = not error_str
    result = {
        "ok": ok,
        "error": error_str,
        "exit": 0 if ok else 1,
        "stdout_tail": "\n".join(stdout_lines),
        "stderr_tail": error_str,
        "duration_s": duration,
        "applied_files": applied if ok else [],
        "skipped": False,
        "stack_used": "sqlite_direct",
    }
    if ok and duration > _SLOW_THRESHOLD_S:
        result["warning"] = "slow_migration"
    return result


async def _apply_postgres(
    target_files: list[str],
    workspace_path: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Postgres path via testcontainers (opt-in).

    Returns a soft-skip verdict if testcontainers is not installed.
    """
    try:
        from testcontainers.postgres import PostgresContainer  # type: ignore
    except ImportError:
        return {
            "ok": True,
            "error": "",
            "exit": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "applied_files": [],
            "skipped": True,
            "stack_used": "postgres_skipped",
            "reason": "testcontainers not installed",
        }

    from mr_roboto.run_cmd import run_cmd

    start = time.monotonic()
    alembic_ini = os.path.join(workspace_path, "alembic.ini")
    has_alembic = os.path.isfile(alembic_ini)

    try:
        with PostgresContainer("postgres:15") as pg:
            db_url = pg.get_connection_url()
            env = dict(os.environ, DATABASE_URL=db_url)

            if has_alembic:
                raw = await run_cmd(
                    mission_id=None,
                    cmd=["alembic", "-c", alembic_ini, "upgrade", "head"],
                    cwd=None,
                    timeout_s=timeout_s,
                    env=env,
                    require_exit_zero=False,
                    workspace_path=workspace_path,
                )
            else:
                # Apply each .sql via psql.
                raw = {"ok": True, "exit": 0, "stdout_tail": "", "stderr_tail": ""}
                for f in target_files:
                    path = f if os.path.isabs(f) else os.path.join(workspace_path, f)
                    r = await run_cmd(
                        mission_id=None,
                        cmd=["psql", db_url, "-f", path],
                        cwd=None,
                        timeout_s=timeout_s,
                        env=env,
                        require_exit_zero=False,
                        workspace_path=workspace_path,
                    )
                    if not (r.get("ok") and r.get("exit") == 0):
                        raw = r
                        break
    except Exception as exc:
        duration = time.monotonic() - start
        return {
            "ok": False,
            "error": f"testcontainers postgres error: {exc}",
            "exit": 1,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "duration_s": duration,
            "applied_files": [],
            "skipped": False,
            "stack_used": "postgres_testcontainers",
        }

    duration = time.monotonic() - start
    ok = bool(raw.get("ok")) and raw.get("exit") == 0
    result: dict[str, Any] = {
        "ok": ok,
        "error": "" if ok else (
            f"postgres migration failed (exit={raw.get('exit')}): "
            f"{_tail(raw.get('stderr_tail') or raw.get('stdout_tail') or '', 400)}"
        ),
        "exit": raw.get("exit"),
        "stdout_tail": _tail(raw.get("stdout_tail") or ""),
        "stderr_tail": _tail(raw.get("stderr_tail") or ""),
        "duration_s": duration,
        "applied_files": target_files if ok else [],
        "skipped": False,
        "stack_used": "postgres_testcontainers",
    }
    if ok and duration > _SLOW_THRESHOLD_S:
        result["warning"] = "slow_migration"
    return result


async def _apply_offline(
    target_files: list[str],
    workspace_path: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Unknown-stack path: alembic offline mode (``alembic upgrade head --sql``).

    Catches syntax errors; misses FK ordering and extension dependencies.
    Falls back to offline parse error when alembic is not present.
    """
    from mr_roboto.run_cmd import run_cmd

    start = time.monotonic()
    alembic_ini = os.path.join(workspace_path, "alembic.ini")
    has_alembic = os.path.isfile(alembic_ini)

    if has_alembic:
        raw = await run_cmd(
            mission_id=None,
            cmd=["alembic", "-c", alembic_ini, "upgrade", "head", "--sql"],
            cwd=None,
            timeout_s=timeout_s,
            require_exit_zero=False,
            workspace_path=workspace_path,
        )
    else:
        # Best-effort: check each .sql file exists and is non-empty UTF-8.
        errors: list[str] = []
        for f in target_files:
            path = f if os.path.isabs(f) else os.path.join(workspace_path, f)
            if not os.path.isfile(path):
                errors.append(f"not found: {f}")
                continue
            try:
                content = open(path, "r", encoding="utf-8").read().strip()
                if not content:
                    errors.append(f"empty: {f}")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"unreadable {f}: {exc}")
        duration = time.monotonic() - start
        ok = not errors
        return {
            "ok": ok,
            "error": "; ".join(errors) if errors else "",
            "exit": 0 if ok else 1,
            "stdout_tail": "",
            "stderr_tail": "; ".join(errors) if errors else "",
            "duration_s": duration,
            "applied_files": target_files if ok else [],
            "skipped": False,
            "stack_used": "offline_parse",
        }

    duration = time.monotonic() - start
    ok = bool(raw.get("ok")) and raw.get("exit") == 0
    result: dict[str, Any] = {
        "ok": ok,
        "error": "" if ok else (
            f"alembic offline failed (exit={raw.get('exit')}): "
            f"{_tail(raw.get('stderr_tail') or raw.get('stdout_tail') or '', 400)}"
        ),
        "exit": raw.get("exit"),
        "stdout_tail": _tail(raw.get("stdout_tail") or ""),
        "stderr_tail": _tail(raw.get("stderr_tail") or ""),
        "duration_s": duration,
        "applied_files": target_files if ok else [],
        "skipped": False,
        "stack_used": "alembic_offline",
    }
    if ok and duration > _SLOW_THRESHOLD_S:
        result["warning"] = "slow_migration"
    return result


async def apply_migration(
    mission_id: int | None,
    target_files: list[str],
    workspace_path: str,
    stack_hint: str,
    timeout_s: float = 120.0,
    enable_testcontainers: bool = False,
) -> dict[str, Any]:
    """Apply migration files to an ephemeral database sandbox.

    Parameters
    ----------
    mission_id:
        Owning mission (informational; not used for workspace resolution
        here since workspace_path is always provided by the posthook wiring).
    target_files:
        List of migration file paths to apply. Relative paths are resolved
        under ``workspace_path``.
    workspace_path:
        Absolute path to the mission workspace root.
    stack_hint:
        Free-form stack descriptor from the step context (e.g. "sqlite",
        "postgres+fastapi", "nextjs+sqlite"). Used to select the
        appropriate apply path.
    timeout_s:
        Hard timeout for the apply subprocess. Default 120 s.
    enable_testcontainers:
        When False (default), the postgres path returns a soft-skip verdict
        even when testcontainers is installed. Set True to opt in.

    Returns
    -------
    dict with keys: ``ok``, ``error``, ``exit``, ``stdout_tail``,
    ``stderr_tail``, ``duration_s``, ``applied_files``, ``skipped``,
    ``stack_used``.  Optionally ``warning`` when apply is slow (>30 s).
    """
    hint = (stack_hint or "").lower()

    if not target_files:
        return {
            "ok": True,
            "error": "",
            "exit": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "applied_files": [],
            "skipped": True,
            "stack_used": "none",
            "reason": "no target_files provided",
        }

    if not workspace_path:
        return {
            "ok": False,
            "error": "apply_migration: workspace_path is required",
            "exit": 1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "applied_files": [],
            "skipped": False,
            "stack_used": "none",
        }

    logger.info(
        "apply_migration: dispatching",
        stack_hint=stack_hint,
        target_files=target_files,
        enable_testcontainers=enable_testcontainers,
    )

    if "postgres" in hint:
        if not enable_testcontainers:
            return {
                "ok": True,
                "error": "",
                "exit": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "duration_s": 0.0,
                "applied_files": [],
                "skipped": True,
                "stack_used": "postgres_skipped",
                "reason": "enable_testcontainers not set",
            }
        return await _apply_postgres(target_files, workspace_path, timeout_s)

    if "sqlite" in hint:
        return await _apply_sqlite(target_files, workspace_path, timeout_s)

    # Unknown stack → alembic offline / static parse.
    return await _apply_offline(target_files, workspace_path, timeout_s)
