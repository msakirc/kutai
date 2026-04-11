"""Heartbeat protocol — shared between guard and managed app.

The guard (wrapper) reads heartbeat files to detect hung processes.
The managed app writes them periodically to prove it's alive.
This module provides helpers for both sides.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("yasar_usta.heartbeat")

# ── Exit code protocol ────────────────────────────────────────────────
# Shared constants so both sides agree on the meaning of exit codes.

EXIT_RESTART = 42  # Managed app requests restart (no backoff)
EXIT_STOP = 0      # Clean shutdown (guard waits for /start)


def write_heartbeat(*paths: str) -> None:
    """Write current timestamp to one or more heartbeat files.

    Args:
        *paths: Heartbeat file paths to write.
    """
    ts = str(time.time())
    for path in paths:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(ts)
        except Exception:
            pass


class HeartbeatWriter:
    """Background async task that writes heartbeat files periodically.

    Usage in managed app::

        writer = HeartbeatWriter("logs/heartbeat", "logs/orchestrator.heartbeat")
        task = asyncio.create_task(writer.run())
        # ... later ...
        task.cancel()

    Args:
        *paths: Heartbeat file paths to write.
        interval: Seconds between writes (default 15).
    """

    def __init__(self, *paths: str, interval: float = 15.0):
        self.paths = paths
        self.interval = interval

    async def run(self) -> None:
        """Write heartbeat files in a loop until cancelled."""
        # Write immediately on start
        write_heartbeat(*self.paths)
        while True:
            try:
                await asyncio.sleep(self.interval)
                write_heartbeat(*self.paths)
            except asyncio.CancelledError:
                return
            except Exception:
                pass
