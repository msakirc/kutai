"""Z1 Tier 4 (C10) — companion to emit_preview_url.

Reads ``mission_{mission_id}/.tunnel.pid``, terminates the subprocess
(Windows-aware), removes the pidfile and the ``preview_url.txt`` surface.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.kill_preview_url")


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _terminate(pid: int) -> int:
    """Send SIGTERM-equivalent to ``pid``. Returns the subprocess exit code."""
    if sys.platform == "win32":
        # taskkill /T cleans up the new process group spawned by emit.
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
        return int(result.returncode)
    try:
        os.kill(pid, signal.SIGTERM)
        return 0
    except ProcessLookupError:
        return 0
    except Exception as e:  # pragma: no cover
        logger.warning(f"kill pid={pid} failed: {e}")
        return 1


async def kill_preview_url(
    mission_id: int,
    workspace_path: str | None = None,
    *,
    _silent: bool = False,
) -> dict[str, Any]:
    """Stop any preview tunnel for ``mission_id`` and clean up surface files."""
    workspace_path = _resolve_workspace(mission_id, workspace_path)
    pid_file = os.path.join(workspace_path, ".tunnel.pid")
    url_file = os.path.join(workspace_path, "preview_url.txt")

    killed_pid: int | None = None
    exit_code: int | None = None

    if os.path.exists(pid_file):
        try:
            raw = open(pid_file, encoding="utf-8").read().strip()
            killed_pid = int(raw) if raw else None
        except Exception:
            killed_pid = None
        if killed_pid:
            exit_code = _terminate(killed_pid)
        try:
            os.remove(pid_file)
        except OSError:
            pass

    if os.path.exists(url_file):
        try:
            os.remove(url_file)
        except OSError:
            pass

    if not _silent:
        try:
            from mr_roboto.emit_preview_url import _persist_to_db
            await _persist_to_db(
                mission_id, action="kill", url=None, exit_code=exit_code,
            )
        except Exception as e:  # pragma: no cover — best-effort
            logger.debug(f"preview_log DB persist skipped: {e}")

    logger.info(
        f"preview_url killed for mission {mission_id} (pid={killed_pid})"
    )
    return {
        "ok": True,
        "killed_pid": killed_pid,
        "exit_code": exit_code,
    }
