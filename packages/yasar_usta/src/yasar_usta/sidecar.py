"""Sidecar subprocess management (log viewer, etc.)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .lock import is_pid_alive

logger = logging.getLogger("yasar_usta.sidecar")


class SidecarManager:
    """Manages a detached sidecar subprocess with PID file tracking.

    Args:
        name: Display name.
        command: Command to run.
        pid_file: Path to PID file.
        health_url: HTTP URL to check for liveness (optional).
        health_timeout: Timeout for health check (seconds).
        log_file: Path to redirect sidecar stdout/stderr.
        cwd: Working directory.
        detached: Run as detached process (survives parent death).
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        pid_file: str | None = None,
        health_url: str | None = None,
        health_timeout: float = 3.0,
        log_file: str | None = None,
        cwd: str | None = None,
        detached: bool = True,
    ):
        self.name = name
        self.command = command
        self.pid_file = Path(pid_file) if pid_file else None
        self.health_url = health_url
        self.health_timeout = health_timeout
        self.log_file = Path(log_file) if log_file else None
        self.cwd = cwd
        self.detached = detached

    def pid_alive(self) -> int | None:
        """Return PID if the sidecar process is alive, else None."""
        if not self.pid_file:
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        if is_pid_alive(pid):
            return pid
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    async def http_alive(self) -> bool:
        """Check if sidecar is responding on its health URL."""
        if not self.health_url:
            return False
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    self.health_url,
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    async def is_alive(self) -> bool:
        """Check if sidecar is alive (PID or HTTP)."""
        if self.pid_alive():
            return True
        return await self.http_alive()

    async def start(self) -> None:
        """Start the sidecar subprocess."""
        if await self.is_alive():
            logger.info("%s already running", self.name)
            return

        if not self.command:
            logger.warning("No command configured for sidecar %s", self.name)
            return

        logger.info("Starting %s", self.name)
        try:
            import subprocess as _sp
            kwargs = {}
            out_fh = None
            if self.detached and sys.platform == "win32":
                kwargs["creationflags"] = (
                    _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
                )
                kwargs["close_fds"] = True
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                out_fh = open(self.log_file, "a", encoding="utf-8")
                kwargs["stdout"] = out_fh
                kwargs["stderr"] = out_fh

            proc = _sp.Popen(
                self.command,
                cwd=self.cwd,
                **kwargs,
            )
            if self.pid_file:
                self.pid_file.parent.mkdir(parents=True, exist_ok=True)
                self.pid_file.write_text(str(proc.pid))
            logger.info("%s started (PID %d)", self.name, proc.pid)
        except Exception as e:
            logger.error("Failed to start %s: %s", self.name, e)

    async def stop(self) -> None:
        """Kill the sidecar by PID."""
        pid = self.pid_alive()
        if not pid:
            return
        logger.info("Stopping %s (PID %d)", self.name, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            logger.error("%s stop error: %s", self.name, e)
        if self.pid_file:
            try:
                self.pid_file.unlink(missing_ok=True)
            except Exception:
                pass

    async def ensure(self) -> None:
        """Restart sidecar if it's dead."""
        if await self.is_alive():
            return
        logger.warning("%s not running, restarting", self.name)
        await self.start()
