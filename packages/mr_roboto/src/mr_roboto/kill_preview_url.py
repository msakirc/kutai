"""Z1 Tier 4 (C10) — companion to emit_preview_url.

Reads ``mission_{mission_id}/.tunnel.pid`` and ``.httpserver.pid``, terminates
both subprocesses (Windows-aware), removes their pidfiles and the
``preview_url.txt`` surface. Both kills are best-effort and idempotent —
a missing pidfile is a no-op.
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


def _kill_pidfile(pid_file: str) -> tuple[int | None, int | None]:
    """Read pid from *pid_file*, terminate it, remove the file.

    Returns ``(killed_pid, exit_code)`` — both None if the file did not exist.
    """
    if not os.path.exists(pid_file):
        return None, None

    killed_pid: int | None = None
    exit_code: int | None = None

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

    return killed_pid, exit_code


async def kill_preview_url(
    mission_id: int,
    workspace_path: str | None = None,
    *,
    _silent: bool = False,
) -> dict[str, Any]:
    """Stop any preview tunnel + static server for ``mission_id``.

    Terminates both ``.tunnel.pid`` (cloudflared) and ``.httpserver.pid``
    (python -m http.server). Cleans up both pidfiles and ``preview_url.txt``.
    Both kills are best-effort and idempotent.
    """
    workspace_path = _resolve_workspace(mission_id, workspace_path)

    tunnel_pid_file = os.path.join(workspace_path, ".tunnel.pid")
    server_pid_file = os.path.join(workspace_path, ".httpserver.pid")
    url_file = os.path.join(workspace_path, "preview_url.txt")

    # Kill cloudflared tunnel.
    killed_pid, exit_code = _kill_pidfile(tunnel_pid_file)

    # Kill local static HTTP server (best-effort, idempotent).
    _kill_pidfile(server_pid_file)

    # Remove the URL surface file.
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
