"""Claude Code remote-control trigger."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("yasar_usta.remote")


def find_claude_cmd(custom_path: str | None = None) -> str | None:
    """Find the claude CLI command. Returns None if not installed."""
    if custom_path and Path(custom_path).exists():
        return custom_path
    # Check common locations
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        candidate = Path(appdata) / "npm" / "claude.cmd"
        if candidate.exists():
            return str(candidate)
    # Check PATH
    import shutil
    which = shutil.which("claude")
    if which:
        return which
    return None


async def start_claude_remote(
    claude_cmd: str,
    name: str = "App",
    cwd: str | None = None,
) -> tuple[asyncio.subprocess.Process | None, str | None]:
    """Start a Claude Code remote-control session.

    Returns:
        (process, session_url) — process may be None on failure,
        session_url may be None if URL couldn't be extracted.
    """
    logger.info("Starting Claude Code remote-control server")
    try:
        proc = await asyncio.create_subprocess_exec(
            claude_cmd, "remote-control",
            "--name", name,
            "--permission-mode", "bypassPermissions",
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("claude command not found at %s", claude_cmd)
        return None, None
    except Exception as e:
        logger.error("Failed to start Claude remote-control: %s", e)
        return None, None

    # Read stdout to capture session URL
    session_url = None
    try:
        for _ in range(20):
            line_bytes = await asyncio.wait_for(
                proc.stdout.readline(), timeout=10,
            )
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            logger.info("[claude-rc] %s", line)
            if "claude.ai" in line or "http" in line.lower():
                url_match = re.search(r"https?://\S+", line)
                if url_match:
                    session_url = url_match.group(0)
                    break
    except asyncio.TimeoutError:
        pass

    return proc, session_url
