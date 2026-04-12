"""Subprocess lifecycle management with heartbeat watchdog."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger("yasar_usta.subprocess")


class SubprocessManager:
    """Manages a single subprocess with output piping and heartbeat monitoring.

    Args:
        command: Command to run as a subprocess.
        log_dir: Directory for log files.
        cwd: Working directory for the subprocess (defaults to current dir).
        stop_timeout: Seconds to wait for graceful shutdown before killing.
        heartbeat_file: Path to heartbeat timestamp file (optional).
        heartbeat_stale_seconds: Kill process if heartbeat older than this.
    """

    def __init__(
        self,
        command: list[str],
        log_dir: str,
        cwd: str | None = None,
        stop_timeout: int = 30,
        heartbeat_file: str | None = None,
        heartbeat_stale_seconds: int = 120,
    ):
        self.command = command
        self.log_dir = Path(log_dir)
        self.cwd = cwd
        self.stop_timeout = stop_timeout
        self.heartbeat_file = heartbeat_file
        self.heartbeat_stale_seconds = heartbeat_stale_seconds

        self.process: asyncio.subprocess.Process | None = None
        self.running: bool = False
        self.start_time: float | None = None
        self.last_exit_code: int | None = None
        self.stderr_tail: deque[str] = deque(maxlen=50)
        self._stop_requested: bool = False

    async def start(self) -> None:
        """Launch the subprocess."""
        if self.process and self.process.returncode is None:
            return

        self.process = None
        self.stderr_tail.clear()
        self._stop_requested = False

        self.log_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if sys.platform == "win32":
            import subprocess as _sp
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.CREATE_NO_WINDOW
            )

        logger.info("Starting subprocess: %s", " ".join(self.command))
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                cwd=self.cwd,
                **kwargs,
            )
        except Exception as e:
            logger.error("Failed to spawn subprocess: %s", e)
            self.process = None
            self.running = False
            return

        self.running = True
        self.start_time = time.time()

        # Reset heartbeat so hung detector doesn't see stale timestamp
        if self.heartbeat_file:
            try:
                with open(self.heartbeat_file, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass

        asyncio.create_task(self._pipe_output(self.process.stdout, "stdout"))
        asyncio.create_task(self._pipe_output(self.process.stderr, "stderr"))

    async def stop(self, timeout: int | None = None) -> None:
        """Send SIGINT and wait for graceful shutdown."""
        if not self.process or self.process.returncode is not None:
            return
        self._stop_requested = True
        timeout = timeout or self.stop_timeout
        logger.info("Sending shutdown signal...")
        try:
            self.process.send_signal(signal.SIGINT)
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timed out, killing...")
            self.process.kill()
            await self.process.wait()
        self.running = False

    async def wait_for_exit(self) -> int:
        """Wait for the subprocess to exit. Returns exit code (-1 if hung/killed)."""
        if not self.process:
            self.running = False
            return -1
        if self.process.returncode is not None:
            code = self.process.returncode
            self.process = None
            self.running = False
            self.last_exit_code = code
            return code

        while True:
            try:
                code = await asyncio.wait_for(self.process.wait(), timeout=30)
                self.process = None
                self.running = False
                self.last_exit_code = code
                return code
            except asyncio.TimeoutError:
                if self.is_heartbeat_stale():
                    logger.error(
                        "Heartbeat stale >%ds — killing hung process",
                        self.heartbeat_stale_seconds,
                    )
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    self.process = None
                    self.running = False
                    self.last_exit_code = -1
                    return -1

    def is_heartbeat_stale(self) -> bool:
        """Check if the heartbeat file is older than the stale threshold."""
        if not self.heartbeat_file:
            return False
        try:
            with open(self.heartbeat_file, "r") as f:
                last_beat = float(f.read().strip())
            return (time.time() - last_beat) > self.heartbeat_stale_seconds
        except (FileNotFoundError, ValueError):
            return False

    def is_heartbeat_healthy(self, healthy_seconds: int = 90) -> bool:
        """Check if heartbeat is fresh (within healthy_seconds)."""
        if not self.heartbeat_file:
            return False
        try:
            ts = float(Path(self.heartbeat_file).read_text().strip())
            return (time.time() - ts) < healthy_seconds
        except (FileNotFoundError, ValueError, OSError):
            return False

    def heartbeat_age(self) -> float | None:
        """Return seconds since last heartbeat, or None if no heartbeat."""
        if not self.heartbeat_file:
            return None
        try:
            ts = float(Path(self.heartbeat_file).read_text().strip())
            return time.time() - ts
        except (FileNotFoundError, ValueError, OSError):
            return None

    async def _pipe_output(self, stream, name: str) -> None:
        """Read subprocess output line by line, print to console and save."""
        log_file = self.log_dir / "guard.log"
        while True:
            try:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except ValueError:
                try:
                    await stream.readuntil(b"\n")
                except Exception:
                    try:
                        stream._buffer.clear()
                    except Exception:
                        pass
                continue
            except Exception:
                break

            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode())
            except Exception:
                pass

            if name == "stderr":
                self.stderr_tail.append(line)
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{name}] {line}\n")
            except Exception:
                pass
