"""Swap manager — model transition orchestration with lock, drain, circuit breaker, and VRAM check."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace

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

        # Watchdog coordination flags
        self.swap_in_progress: bool = False
        self.intentional_unload: bool = False

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

    # ── GPU-layer override recheck ──────────────────────────────────────────

    def _recheck_gpu_override(self, config: ServerConfig) -> ServerConfig:
        """Strip --n-gpu-layers when live VRAM can no longer honour the pin.

        Only engages when:
          * extra_flags contains ``--n-gpu-layers`` (i.e. models.yaml override)
          * config.required_vram_mb > 0
          * a VRAM probe is configured (DaLLaMaConfig.get_vram_free_mb)

        If live free VRAM is below ``required_vram_mb`` we log a warning and
        return a new ServerConfig with the --n-gpu-layers flag pair removed.
        llama-server's --fit default then picks a size that actually fits.
        Never raises; falls back silently on any unexpected condition.
        """
        flags = list(config.extra_flags or [])
        if "--n-gpu-layers" not in flags:
            return config
        if config.required_vram_mb <= 0:
            return config
        get_vram = self._config.get_vram_free_mb
        if get_vram is None:
            return config
        try:
            free_mb = int(get_vram())
        except Exception:
            logger.debug("get_vram_free_mb raised — skipping override recheck", exc_info=True)
            return config
        if free_mb >= config.required_vram_mb:
            logger.debug(
                "GPU-layer override fits: %dMB free >= %dMB required for %s",
                free_mb, config.required_vram_mb, config.model_name,
            )
            return config
        # Doesn't fit — drop the flag pair and let --fit decide.
        try:
            idx = flags.index("--n-gpu-layers")
            removed = flags[idx:idx + 2]
            del flags[idx:idx + 2]
        except (ValueError, IndexError):
            return config
        logger.warning(
            "Dropping %s for %s: %dMB free < %dMB required — "
            "falling back to --fit",
            " ".join(removed), config.model_name, free_mb, config.required_vram_mb,
        )
        return replace(config, extra_flags=flags)

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

    async def swap(self, server: object, config: ServerConfig, load_timeout: float = 0.0) -> bool:
        """Transition the server to config.

        Flow:
        1. Circuit breaker check
        2. Acquire lock (one swap at a time)
        3. Drain in-flight requests (with timeout → force reset)
        4. Stop current server if alive, sleep 2s for CUDA VRAM release
        5. VRAM check
        6. Start server with new config
        7. Record outcome, notify, return

        Parameters
        ----------
        load_timeout:
            Caller-provided ceiling for the health-wait timeout (seconds).
            If >0, ``server.start()`` uses ``min(own_estimate, load_timeout)``
            instead of the internal estimate alone.  Passed from Fatih Hoca
            via the dispatcher so slow-loading models can be rejected earlier.

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
                return await self._do_swap(server, config, load_timeout=load_timeout)
            finally:
                self.swap_in_progress = False

    async def _do_swap(self, server: object, config: ServerConfig, load_timeout: float = 0.0) -> bool:
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

        # ── VRAM check (advisory) ──────────────────────────────────────────
        # The old server is already stopped at this point — refusing here
        # would leave us with nothing loaded. Log a warning but proceed;
        # if VRAM is truly insufficient server.start() will fail and the
        # circuit breaker handles repeated failures.
        get_vram = self._config.get_vram_free_mb
        if get_vram is not None:
            free_mb = get_vram()
            if free_mb < self._config.min_free_vram_mb:
                logger.warning(
                    f"Low VRAM for {model_name}: "
                    f"{free_mb}MB free, want {self._config.min_free_vram_mb}MB "
                    f"— attempting load anyway"
                )

        # ── Override-fit recheck ───────────────────────────────────────────
        # When models.yaml pinned an explicit --n-gpu-layers override, the
        # override was sized at registry-build time against VRAM that may
        # have been more generous than what is free right now (thinking-mode
        # activation buffer, fragmentation, another process grabbing VRAM).
        # --fit is the safe path: it sizes from *live* VRAM. So when the
        # pinned value no longer fits, strip the override and let --fit
        # decide. Leaves bare --fit loads (required_vram_mb == 0) alone.
        config = self._recheck_gpu_override(config)

        # ── Start server with new config ───────────────────────────────────
        success = await server.start(config, load_timeout=load_timeout)

        # Fallback path: --fit can underbudget compute buffer + CUDA
        # overhead on tight GPUs (observed 2026-04-23 on 8GB GPU running
        # 9B models, cudaMalloc 501 MiB fails at sched_reserve). If the
        # failure signature is OOM and we have a calculated gpu_layers
        # value, retry once forcing partial offload.
        if (not success
                and config.fallback_gpu_layers > 0
                and self._stderr_shows_oom(server)):
            logger.warning(
                "Retrying %s with --n-gpu-layers=%d fallback (OOM signature "
                "detected from --fit path)",
                model_name, config.fallback_gpu_layers,
            )
            # Clone config with the fallback flag injected.
            fallback_flags = list(config.extra_flags or [])
            if "--n-gpu-layers" not in fallback_flags:
                fallback_flags.extend(["--n-gpu-layers", str(config.fallback_gpu_layers)])
            retry_config = replace(config, extra_flags=fallback_flags)
            # Give the driver a moment to fully reclaim VRAM from the
            # failed process.
            await asyncio.sleep(3)
            success = await server.start(retry_config, load_timeout=load_timeout)

        if success:
            logger.info(f"Model loaded: {model_name}")
            self._record_success()
            self._notify(model_name, "model_loaded")
        else:
            logger.error(f"Failed to load model: {model_name}")
            self._record_failure(model_name)
            self._notify(None, "load_failed")

        return success

    @staticmethod
    def _stderr_shows_oom(server: object) -> bool:
        """Check if the last stderr tail contains a cudaMalloc OOM signature."""
        try:
            tail = server.read_stderr_tail(60)
        except Exception:
            return False
        if not tail:
            return False
        t = tail.lower()
        # Canonical OOM signatures + ggml internal mem-buffer assertion
        # (triggered by the same underlying cause: allocator couldn't
        # secure a contiguous block; sometimes surfaces as GGML_ASSERT
        # instead of cudaMalloc depending on when the failure lands).
        return (
            "cudamalloc failed" in t
            or "out of memory" in t
            or "failed to allocate compute" in t
            or "mem_buffer != null" in t
            or "ggml_assert" in t and "mem_buffer" in t
        )
