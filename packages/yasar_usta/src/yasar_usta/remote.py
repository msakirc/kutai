"""Claude Code remote-control session management.

Starts Claude as a **detached process** so it survives guard restarts.
Tracks sessions in a directory — each session gets ``{pid}.url`` and
``{pid}.log`` files.  Multiple concurrent sessions are supported.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess as _sp
import sys
import time
from pathlib import Path

from .lock import is_pid_alive

logger = logging.getLogger("yasar_usta.remote")


# ── Discovery ────────────────────────────────────────────────────────

def find_claude_cmd(custom_path: str | None = None) -> str | None:
    """Find the claude CLI command. Returns None if not installed."""
    if custom_path and Path(custom_path).exists():
        return custom_path
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        candidate = Path(appdata) / "npm" / "claude.cmd"
        if candidate.exists():
            return str(candidate)
    import shutil
    which = shutil.which("claude")
    if which:
        return which
    return None


# ── Session tracking ─────────────────────────────────────────────────

def list_sessions(session_dir: str | Path) -> list[tuple[int, str | None]]:
    """Return ``[(pid, url), ...]`` for all alive sessions.

    Cleans up stale entries (dead PIDs) as a side-effect.
    """
    sdir = Path(session_dir)
    if not sdir.is_dir():
        return []

    alive: list[tuple[int, str | None]] = []
    for url_file in sdir.glob("*.url"):
        try:
            pid = int(url_file.stem)
        except ValueError:
            continue
        if is_pid_alive(pid):
            url = url_file.read_text(encoding="utf-8").strip() or None
            alive.append((pid, url))
        else:
            # Dead session — clean up
            _remove_session_files(sdir, pid)

    return alive


def stop_session(session_dir: str | Path, pid: int) -> None:
    """Kill a specific Claude remote session by PID."""
    logger.info("Stopping Claude remote session (PID %d)", pid)
    try:
        os.kill(pid, 9)
    except OSError as e:
        logger.error("Claude session stop error: %s", e)
    _remove_session_files(Path(session_dir), pid)


def stop_all_sessions(session_dir: str | Path) -> None:
    """Kill every tracked Claude remote session."""
    for pid, _ in list_sessions(session_dir):
        stop_session(session_dir, pid)


def _remove_session_files(session_dir: Path, pid: int) -> None:
    for suffix in (".url", ".log"):
        try:
            (session_dir / f"{pid}{suffix}").unlink(missing_ok=True)
        except Exception:
            pass


# ── Start ─────────────────────────────────────────────────────────────

async def start_claude_remote(
    claude_cmd: str,
    name: str = "App",
    cwd: str | None = None,
    session_dir: str | Path | None = None,
) -> tuple[int | None, str | None]:
    """Start a Claude Code remote-control session as a detached process.

    Returns:
        ``(pid, session_url)`` — pid is None on failure,
        session_url may be None if URL couldn't be extracted.
    """
    logger.info("Starting Claude Code remote-control (detached)")

    sdir = Path(session_dir) if session_dir else None
    log_path: Path | None = None
    out_fh = None

    try:
        kwargs: dict = {}

        if sdir:
            sdir.mkdir(parents=True, exist_ok=True)
            # Use PID-unique temp name to avoid collision between sessions
            log_path = sdir / f"_starting_{os.getpid()}.log"
            out_fh = open(log_path, "w", encoding="utf-8")
            kwargs["stdout"] = out_fh
            kwargs["stderr"] = out_fh

        if sys.platform == "win32":
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
            )
            kwargs["close_fds"] = True

        proc = _sp.Popen(
            [claude_cmd, "remote-control",
             "--name", name,
             "--permission-mode", "bypassPermissions"],
            cwd=cwd,
            **kwargs,
        )
    except FileNotFoundError:
        logger.warning("claude command not found at %s", claude_cmd)
        if out_fh:
            out_fh.close()
        return None, None
    except Exception as e:
        logger.error("Failed to start Claude remote-control: %s", e)
        if out_fh:
            out_fh.close()
        return None, None

    pid = proc.pid
    logger.info("Claude remote-control started (PID %d)", pid)

    # Close parent's file handle — child has its own via Popen.
    # This releases the Windows file lock so rename works.
    if out_fh:
        out_fh.close()

    # Rename log to {pid}.log
    if sdir and log_path and log_path.exists():
        final_log = sdir / f"{pid}.log"
        try:
            log_path.rename(final_log)
            log_path = final_log
        except Exception:
            pass  # keep temp name — URL polling still works

    # Poll log file for the session URL (or an error)
    session_url = None
    error = None
    if log_path:
        session_url, error = await _poll_log_for_url_or_error(log_path, timeout=30)

    if error:
        logger.warning("Claude session PID %d initial error: %s", pid, error)
        # Don't kill the process — it may recover (reconnection attempts).
        # Still track it so list_sessions can report it.
        if sdir:
            (sdir / f"{pid}.url").write_text("", encoding="utf-8")
        return None, error

    # Save session entry
    if sdir and session_url:
        (sdir / f"{pid}.url").write_text(session_url, encoding="utf-8")

    return pid, session_url


async def _poll_log_for_url_or_error(
    log_path: Path, timeout: float = 60,
) -> tuple[str | None, str | None]:
    """Poll a log file for URL or error.

    Returns:
        ``(url, None)`` on success, ``(None, error_msg)`` on failure,
        ``(None, None)`` on timeout.
    """
    deadline = time.monotonic() + timeout
    seen_bytes = 0
    last_error: str | None = None

    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        try:
            size = log_path.stat().st_size
            if size <= seen_bytes:
                continue
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(seen_bytes)
                new_text = f.read()
            seen_bytes = size
            for line in new_text.splitlines():
                line = line.strip("\x00 \t\r\n")
                if not line:
                    continue
                # URL found — success (takes priority over earlier errors)
                if "claude.ai" in line or "http" in line.lower():
                    url_match = re.search(r"https?://\S+", line)
                    if url_match:
                        url = url_match.group(0)
                        logger.info("Claude session URL: %s", url)
                        return url, None
                # Record errors but keep looking — the process may recover
                if line.lower().startswith("error:"):
                    last_error = line
                    logger.warning("Claude session error (continuing): %s", line)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning("Error polling Claude log: %s", e)

    if last_error:
        logger.warning("Timed out with last error: %s", last_error)
        return None, last_error
    logger.warning("Timed out waiting for Claude session URL")
    return None, None
