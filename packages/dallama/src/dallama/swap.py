"""Swap manager — model transition orchestration with lock, drain, circuit breaker, and VRAM check."""
from __future__ import annotations

import asyncio
import logging
import time

from dallama.config import DaLLaMaConfig, ServerConfig

logger = logging.getLogger(__name__)


class SwapManager:
    """Orchestrates safe model transitions.

    Guarantees:
    - Only one swap at a time (asyncio.Lock)
    - In-flight requests drain before server stops (inference tracking + Event)
    - Restart loops prevented (circuit breaker with per-model failure counting)
    - Resource checked before loading (optional VRAM callback)
    """

    def __init__(self, config: DaLLaMaConfig) -> None:
        self._config = config

        # Swap exclusion lock
        self._lock = asyncio.Lock()

        # Inference tracking
        self._inflight_count: int = 0
        self._inflight_idle: asyncio.Event = asyncio.Event()
        self._inflight_idle.set()  # initially idle
        self._generation: int = 0

        # Circuit breaker
        self._fail_count: int = 0
        self._fail_model: str | None = None
        self._cooldown_until: float = 0.0

        # Watchdog coordination flag
        self.swap_in_progress: bool = False

    # ── Inference tracking ──────────────────────────────────────────────────

    def mark_inference_start(self) -> int:
        """Mark that an inference request is starting.

        Returns the current generation token. Callers MUST pass this back to
        mark_inference_end() so that orphaned completions from a pre-swap
        generation don't corrupt the counter.
        """
        self._inflight_count += 1
        self._inflight_idle.clear()
        return self._generation

    def mark_inference_end(self, generation: int) -> None:
        """Mark that an inference request has finished.

        If generation doesn't match the current one (a force-swap happened),
        the decrement is skipped — the counter was already reset.
        """
        if generation != self._generation:
            return
        self._inflight_count = max(0, self._inflight_count - 1)
        if self._inflight_count == 0:
            self._inflight_idle.set()

    @property
    def has_inflight(self) -> bool:
        """True if there are in-flight inference requests."""
        return self._inflight_count > 0

    def force_reset_inflight(self) -> None:
        """Bump generation and reset counter.

        Called when drain times out. Pre-swap inferences will see a mismatched
        generation on completion and silently skip decrement.
        """
        self._generation += 1
        self._inflight_count = 0
        self._inflight_idle.set()

    # ── Circuit breaker ─────────────────────────────────────────────────────

    def _is_blocked(self, model_name: str) -> bool:
        """Return True if the circuit breaker should block loading this model."""
        if self._cooldown_until <= 0:
            return False
        if time.time() >= self._cooldown_until:
            # Cooldown expired — auto-reset
            self._fail_count = 0
            self._cooldown_until = 0.0
            self._fail_model = None
            return False
        if self._fail_model == model_name:
            return True
        # Different model — not blocked
        return False

    def _record_failure(self, model_name: str) -> None:
        """Record a failed load attempt. Trips the breaker after threshold."""
        if self._fail_model != model_name:
            self._fail_model = model_name
            self._fail_count = 0
        self._fail_count += 1
        if self._fail_count >= self._config.circuit_breaker_threshold:
            self._cooldown_until = time.time() + self._config.circuit_breaker_cooldown_seconds
            logger.error(
                f"Circuit breaker tripped: {model_name} failed {self._fail_count} times — "
                f"refusing loads for {self._config.circuit_breaker_cooldown_seconds:.0f}s"
            )

    def _record_success(self) -> None:
        """Record a successful load. Resets all circuit breaker state."""
        self._fail_count = 0
        self._fail_model = None
        self._cooldown_until = 0.0

    # ── on_ready notification ───────────────────────────────────────────────

    def _notify(self, model_name: str | None, reason: str) -> None:
        """Fire on_ready callback, swallowing any exception."""
        cb = self._config.on_ready
        if cb is None:
            return
        try:
            cb(model_name, reason)
        except Exception:
            logger.exception("on_ready callback raised an exception")

    # ── Main swap entry point ───────────────────────────────────────────────

    async def swap(self, server: object, config: ServerConfig) -> bool:
        """Transition the server to config.

        Flow:
        1. Circuit breaker check
        2. Acquire lock (one swap at a time)
        3. Drain in-flight requests (with timeout → force reset)
        4. Stop current server if alive, sleep 2s for CUDA VRAM release
        5. VRAM check
        6. Start server with new config
        7. Record outcome, notify, return

        Returns True if the new model is loaded and healthy.
        """
        model_name = config.model_name

        # Check circuit breaker before acquiring the lock so blocked calls
        # return quickly without queuing behind an ongoing swap.
        if self._is_blocked(model_name):
            logger.warning(
                f"Circuit breaker active — refusing to load {model_name} "
                f"(cooldown {self._cooldown_until - time.time():.0f}s remaining)"
            )
            self._notify(None, "circuit_breaker_active")
            return False

        async with self._lock:
            self.swap_in_progress = True
            try:
                return await self._do_swap(server, config)
            finally:
                self.swap_in_progress = False

    async def _do_swap(self, server: object, config: ServerConfig) -> bool:
        """Inner swap logic, executed under the lock."""
        model_name = config.model_name

        # ── Drain in-flight inference ──────────────────────────────────────
        if self._inflight_count > 0:
            logger.info(
                f"Waiting for {self._inflight_count} in-flight inference(s) to drain "
                f"before swap to {model_name}…"
            )
            drain_timeout = self._config.inference_drain_timeout_seconds
            try:
                await asyncio.wait_for(self._inflight_idle.wait(), timeout=drain_timeout)
                logger.info("Inference drained, proceeding with swap")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Inference drain timed out after {drain_timeout}s "
                    f"({self._inflight_count} still active). "
                    f"Force-swapping — active requests will receive errors."
                )
                self.force_reset_inflight()

        # ── Stop running server ────────────────────────────────────────────
        if server.is_alive():
            await server.stop()
            # CUDA VRAM release lags behind process exit — give the driver time
            # to reclaim memory before checking free VRAM for the next model.
            await asyncio.sleep(2)

        # ── VRAM check ─────────────────────────────────────────────────────
        get_vram = self._config.get_vram_free_mb
        if get_vram is not None:
            free_mb = get_vram()
            if free_mb < self._config.min_free_vram_mb:
                logger.error(
                    f"Insufficient VRAM to load {model_name}: "
                    f"{free_mb}MB free, need {self._config.min_free_vram_mb}MB"
                )
                self._notify(None, "insufficient_vram")
                return False

        # ── Start server with new config ───────────────────────────────────
        success = await server.start(config)
        if success:
            logger.info(f"Model loaded: {model_name}")
            self._record_success()
            self._notify(model_name, "model_loaded")
        else:
            logger.error(f"Failed to load model: {model_name}")
            self._record_failure(model_name)
            self._notify(None, "load_failed")

        return success
