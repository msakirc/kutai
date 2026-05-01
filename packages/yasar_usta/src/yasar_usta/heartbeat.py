"""Heartbeat protocol — shared between guard and managed app.

The guard (wrapper) reads heartbeat files to detect hung processes.
The managed app writes them periodically to prove it's alive.
This module provides helpers for both sides.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable

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


def write_state_snapshot(path: str, state: dict) -> None:
    """Write a JSON state snapshot to ``path``. Never raises.

    Used by managed app to record context (active DB ops, in-flight tasks,
    etc.) at heartbeat tick. If the app freezes, the guard reads this
    file to log WHAT was happening at the moment of the last heartbeat.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(state)
        payload.setdefault("snapshot_ts", time.time())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def read_state_snapshot(path: str) -> dict | None:
    """Read a state snapshot file. Returns None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class HeartbeatWriter:
    """Background async task that writes heartbeat files periodically.

    Usage in managed app::

        writer = HeartbeatWriter(
            "logs/heartbeat", "logs/orchestrator.heartbeat",
            state_path="logs/orchestrator.state.json",
            state_provider=lambda: {"aux_active": [...], "in_flight": [...]},
        )
        task = asyncio.create_task(writer.run())
        # ... later ...
        task.cancel()

    Args:
        *paths: Heartbeat file paths to write.
        interval: Seconds between writes (default 15).
        state_path: Optional path for a JSON state snapshot file. Written
            alongside the heartbeat. If the app freezes, the guard reads
            this file to log the last-known state.
        state_provider: Callable returning a JSON-serialisable dict. Run
            on every heartbeat tick. Should be cheap (no async, no DB).
    """

    def __init__(
        self,
        *paths: str,
        interval: float = 15.0,
        state_path: str | None = None,
        state_provider: Callable[[], dict] | None = None,
    ):
        self.paths = paths
        self.interval = interval
        self.state_path = state_path
        self.state_provider = state_provider

    def _write_state(self) -> None:
        if not (self.state_path and self.state_provider):
            return
        try:
            state = self.state_provider() or {}
        except Exception:
            return
        write_state_snapshot(self.state_path, state)

    async def run(self) -> None:
        """Write heartbeat files in a loop until cancelled."""
        # Write immediately on start
        write_heartbeat(*self.paths)
        self._write_state()
        while True:
            try:
                await asyncio.sleep(self.interval)
                write_heartbeat(*self.paths)
                self._write_state()
            except asyncio.CancelledError:
                return
            except Exception:
                pass
