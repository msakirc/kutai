"""Watchdog tasks — HealthWatchdog and IdleUnloader background asyncio loops."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Callable

from dallama.config import DaLLaMaConfig, ServerConfig

logger = logging.getLogger(__name__)


async def _call(fn, *args):
    """Call fn(*args); await the result if it is a coroutine."""
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


class HealthWatchdog:
    """Periodically checks server liveness and HTTP health, triggers swap on failure.

    Detection flow (every ``health_check_interval_seconds``):
    - No config loaded → skip (nothing to watch)
    - Swap in progress → skip, reset failure counter
    - Server not alive → crash detected, trigger swap immediately
    - Server alive but HTTP status not 200/503 → increment failure counter;
      once threshold reached, stop server and trigger swap for restart
    - HTTP 200 or 503 (busy loading) → reset failure counter
    """

    def __init__(self, config: DaLLaMaConfig, server: object, swap: object) -> None:
        self._config = config
        self._server = server
        self._swap = swap
        self._fail_count: int = 0

    async def run(self, get_current_config: Callable[[], ServerConfig | None]) -> None:
        """Run forever; cancel the task to stop."""
        while True:
            await asyncio.sleep(self._config.health_check_interval_seconds)
            try:
                await self._tick(get_current_config)
            except Exception:
                logger.exception("HealthWatchdog tick raised an unexpected exception")

    async def _tick(self, get_current_config: Callable[[], ServerConfig | None]) -> None:
        current_config = get_current_config()
        if current_config is None:
            return  # No model loaded — nothing to watch.

        if self._swap.swap_in_progress:
            self._fail_count = 0
            return

        if not await _call(self._server.is_alive):
            logger.warning("HealthWatchdog: server process died — triggering recovery swap")
            await self._swap.swap(self._server, current_config)
            self._fail_count = 0
            return

        # Server is alive — check HTTP health endpoint.
        try:
            status = await self._server._health_check_status()
        except Exception:
            logger.exception("HealthWatchdog: error calling _health_check_status")
            status = 0

        if status in (200, 503):
            # 200 = healthy, 503 = busy loading (not a hang)
            self._fail_count = 0
        else:
            self._fail_count += 1
            logger.warning(
                f"HealthWatchdog: unhealthy status {status} "
                f"(fail {self._fail_count}/{self._config.health_fail_threshold})"
            )
            if self._fail_count >= self._config.health_fail_threshold:
                logger.error(
                    "HealthWatchdog: failure threshold reached — stopping server and swapping"
                )
                await self._server.stop()
                await self._swap.swap(self._server, current_config)
                self._fail_count = 0


class IdleUnloader:
    """Unloads the model after a period of inactivity to free VRAM.

    ``reset_timer()`` must be called on every inference entry and exit (and
    keep_alive heartbeats) so the idle clock is accurate.

    The poll loop fires every ``_poll_interval`` seconds and checks whether
    ``idle_timeout_seconds`` has elapsed with no in-flight work.
    """

    def __init__(
        self,
        config: DaLLaMaConfig,
        server: object,
        swap: object,
        _poll_interval: float | None = None,
    ) -> None:
        self._config = config
        self._server = server
        self._swap = swap
        self._last_activity: float = 0.0
        # Default 30s in production; tests pass a shorter value.
        if _poll_interval is None:
            self._poll_interval = min(30.0, config.idle_timeout_seconds / 2)
        else:
            self._poll_interval = _poll_interval

    # ── Public API ───────────────────────────────────────────────────────────

    def reset_timer(self) -> None:
        """Record activity now. Called on infer() entry/exit and keep_alive()."""
        self._last_activity = time.time()

    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity; 0 if reset_timer() was never called."""
        if self._last_activity == 0.0:
            return 0.0
        return time.time() - self._last_activity

    # ── Background loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run forever; cancel the task to stop."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._tick()
            except Exception:
                logger.exception("IdleUnloader tick raised an unexpected exception")

    async def _tick(self) -> None:
        if not await _call(self._server.is_alive):
            return  # Nothing loaded.

        if self._swap.swap_in_progress:
            return

        if self._last_activity == 0.0:
            return  # Timer never started — don't unload on startup.

        if self.idle_seconds <= self._config.idle_timeout_seconds:
            return  # Not idle yet.

        if self._swap.has_inflight:
            return  # Active inference — wait.

        logger.info(
            f"IdleUnloader: model idle for {self.idle_seconds:.1f}s "
            f"(timeout {self._config.idle_timeout_seconds}s) — unloading"
        )
        await self._server.stop()

        cb = self._config.on_ready
        if cb is not None:
            try:
                cb(None, "idle_unload")
            except Exception:
                logger.exception("on_ready callback raised an exception during idle unload")
