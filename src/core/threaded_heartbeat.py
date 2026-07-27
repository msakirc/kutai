"""Daemon-thread liveness heartbeat with a loop-tick watchdog.

Background (2026-07-27): KutAI's liveness heartbeat used to be written by an
asyncio task on the main event loop. But the loop *legitimately* blocks for
minutes — e.g. the first agent task after a cold restart imports LiteLLM + grpc
+ protobuf + opentelemetry + tool modules whose ``.pyc`` files are cold on disk
(~250s of disk-read + compile that hogs the GIL and the import lock). A
loop-based heartbeat can't write during that block, so Yaşar Usta read it stale
and false-killed the process at the startup-grace boundary — every manual
restart, then the warm-cache auto-restart ran clean.

Fix: write the heartbeat from a **daemon OS thread** so any loop block (cold
imports, CPU spikes) can't starve it — ``write_heartbeat`` is a bare file write
that never touches the import lock, and the thread gets GIL slices during the
block's disk-wait windows. To still catch a *genuinely* wedged loop, the event
loop bumps a tick via ``tick_loop``; if it hasn't ticked within
``wedge_threshold`` (chosen well above the worst legitimate block), the thread
withholds the heartbeat so the hub restarts it.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

from yasar_usta import write_heartbeat
from yasar_usta.heartbeat import write_state_snapshot

from src.infra.logging_config import get_logger

logger = get_logger("core.threaded_heartbeat")


class ThreadedHeartbeat:
    """Write liveness heartbeats from a daemon thread, gated by loop liveness.

    Usage:
        hb = ThreadedHeartbeat(paths, interval=15.0)
        hb.start_thread()                       # daemon writer
        tick_task = asyncio.create_task(hb.tick_loop())  # loop-liveness proof
        ...
        hb.stop()
    """

    def __init__(
        self,
        paths,
        interval: float = 15.0,
        state_path: str | None = None,
        state_provider: Callable[[], dict] | None = None,
        wedge_threshold: float = 480.0,
        tick_interval: float = 5.0,
        _clock: Callable[[], float] = time.time,
    ):
        self.paths = list(paths)
        self.interval = interval
        self.state_path = state_path
        self.state_provider = state_provider
        # A block up to ~250s (cold first-agent imports) is legitimate; only
        # declare the loop wedged well beyond that.
        self.wedge_threshold = wedge_threshold
        self.tick_interval = tick_interval
        self._clock = _clock
        self._last_tick = _clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── loop-liveness ────────────────────────────────────────────────────
    def bump(self) -> None:
        """Record that the event loop is still pumping. Called from the loop."""
        self._last_tick = self._clock()

    def loop_alive(self) -> bool:
        """True while the loop has ticked within ``wedge_threshold``."""
        return (self._clock() - self._last_tick) < self.wedge_threshold

    # ── writing ──────────────────────────────────────────────────────────
    def _write_once(self) -> None:
        # Bare liveness first (never touches the import lock, always reliable).
        write_heartbeat(*self.paths)
        # State snapshot is best-effort diagnostics; never let it block the beat.
        if self.state_path and self.state_provider is not None:
            try:
                state = self.state_provider() or {}
                write_state_snapshot(self.state_path, state)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.loop_alive():
                    self._write_once()
                # else: withhold — a genuinely wedged loop must go stale so the
                # hub restarts it.
            except Exception:
                pass
            self._stop.wait(self.interval)

    # ── lifecycle ────────────────────────────────────────────────────────
    def start_thread(self) -> None:
        # Write immediately so the file is fresh from t0.
        try:
            if self.loop_alive():
                self._write_once()
        except Exception:
            pass
        self._thread = threading.Thread(
            target=self._run, name="heartbeat", daemon=True
        )
        self._thread.start()

    async def tick_loop(self) -> None:
        """Bump the loop-liveness tick on a cadence. Run as an asyncio task."""
        while not self._stop.is_set():
            self.bump()
            try:
                await asyncio.sleep(self.tick_interval)
            except asyncio.CancelledError:
                return

    def stop(self) -> None:
        self._stop.set()
