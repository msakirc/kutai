# local_model_manager.py
"""
Backward-compatible shim wrapping the DaLLaMa package.

All existing import paths and function signatures are preserved.
Consumers see the same `get_local_manager()`, `get_runtime_state()`,
`LocalModelManager`, `ModelRuntimeState`, and `ModelSwapRequest`
they've always used — the implementation delegates to DaLLaMa.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("models.local_model_manager")


# ── Preserved dataclasses ────────────────────────────────────────────────────

@dataclass
class ModelSwapRequest:
    """Kept for backward compatibility (some tests import it)."""
    model_name: str
    reason: str
    priority: int = 5
    event: asyncio.Event = field(default_factory=asyncio.Event)
    success: bool = False


@dataclass
class ModelRuntimeState:
    """
    Actual runtime parameters of the currently loaded llama-server instance.

    These may differ from registry defaults because of dynamic context/gpu_layer
    recalculation at swap time (based on live VRAM readings).  The scorer in
    router.py uses these values instead of static registry values so that:

    - Hard context filter uses the *actual* loaded window, not the nominal one.
    - Speed scoring uses *measured* tok/s from /metrics rather than estimates.
    - Thinking-state mismatch reduces stickiness (1.10x) so tasks that need
      thinking will trigger a swap rather than silently getting a worse model.
    """
    model_name: str
    thinking_enabled: bool
    context_length: int       # actual ctx window loaded (post dynamic-calc)
    gpu_layers: int           # actual n-gpu-layers used
    measured_tps: float = 0.0  # updated from /metrics after first generation
    loaded_at: float = field(default_factory=time.time)


# ── Shim class ───────────────────────────────────────────────────────────────

class LocalModelManager:
    """Thin wrapper around DaLLaMa that preserves the old public surface."""

    def __init__(self) -> None:
        from dallama import DaLLaMa, DaLLaMaConfig

        port = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
        llama_path = os.environ.get("LLAMA_SERVER_PATH", "llama-server")
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
        )
        os.environ.setdefault("DALLAMA_LOG_DIR", log_dir)

        config = DaLLaMaConfig(
            llama_server_path=llama_path,
            port=port,
            on_ready=self._on_ready,
            get_vram_free_mb=self._get_vram_free_mb,
        )
        self._dallama = DaLLaMa(config)
        self._port = port
        self.api_base: str = f"http://127.0.0.1:{port}"

        # State that consumers read directly
        self.swap_started_at: float = 0.0
        self._swap_ready = asyncio.Event()
        self._swap_ready.set()
        self.runtime_state: ModelRuntimeState | None = None

        self._last_request_time: float = 0.0
        self._total_swaps: int = 0
        self._started = False
        self._thinking_enabled: bool = False
        self._vision_enabled: bool = False

        # GPU scheduler for acquire/release — same as before
        from .gpu_scheduler import get_gpu_scheduler
        self._scheduler = get_gpu_scheduler()

    # ── Lazy start (called once from first ensure_model) ──────────────────

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._dallama.start()
            self._started = True

    # ── Callbacks for DaLLaMa ─────────────────────────────────────────────

    def _on_ready(self, model_name: str | None, reason: str) -> None:
        """Fired by DaLLaMa after every swap attempt and idle unload."""
        if model_name is None and reason == "idle_unload":
            # Clear stale config so status reflects reality
            self._dallama._current_config = None
            self.runtime_state = None
            self._thinking_enabled = False
            self._vision_enabled = False
            logger.info("Idle unload: cleared model state")
            return
        if model_name is not None:
            try:
                from src.infra.db import accelerate_retries
                loop = asyncio.get_running_loop()
                loop.create_task(accelerate_retries(f"model_loaded:{model_name}"))
            except Exception:
                pass

    @staticmethod
    def _get_vram_free_mb() -> int:
        try:
            from .gpu_monitor import get_gpu_monitor
            state = get_gpu_monitor().get_state()
            return state.gpu.vram_free_mb
        except Exception:
            return 99999  # assume enough if monitor unavailable

    # ── Public API (same signatures as old LocalModelManager) ─────────────

    def keep_alive(self) -> None:
        """Reset idle-unload timer — call during active inference."""
        self._dallama.keep_alive()

    async def ensure_model(
        self,
        model_name: str,
        reason: str = "",
        enable_thinking: bool = False,
        enable_vision: bool = False,
        min_context: int = 0,
    ) -> bool:
        """Load a model by name, translating to DaLLaMa ServerConfig.

        Args:
            min_context: Minimum context window the task requires. Used as
                a floor — if dynamic calculation yields less, context is
                bumped up to this value.
        """
        await self._ensure_started()

        # Check if already loaded with same config
        status = self._dallama.status
        if status.model_name == model_name and status.healthy:
            self._dallama.keep_alive()
            return True

        # Look up ModelInfo from registry
        from .model_registry import get_registry
        registry = get_registry()
        info = registry.get(model_name)
        if not info or not info.is_local:
            logger.error("Model '%s' not found in registry or not local", model_name)
            return False

        # Dynamic context recalculation — use a LOCAL variable so we never
        # mutate the shared ModelInfo in the registry.  Previous code did
        # `info.context_length = new_ctx` which permanently poisoned the
        # registry entry with a stale dynamic value.
        from .model_registry import calculate_dynamic_context
        registry_overrides = registry.get_overrides(model_name)
        context_length = info.context_length  # start from registry default

        if "context_length" not in registry_overrides:
            from .gpu_monitor import get_gpu_monitor
            gpu_monitor = get_gpu_monitor()
            gpu_monitor.invalidate_cache()
            fresh_state = gpu_monitor.get_state()

            current_model_name = self._dallama.status.model_name
            if current_model_name is not None and fresh_state.gpu.available:
                # A model IS loaded — VRAM reading is polluted by its footprint.
                # Best estimate: total VRAM minus ~700MB baseline (CUDA + desktop).
                # Everything else currently used belongs to the loaded model and
                # will be reclaimed after stop().
                baseline_vram_mb = 700
                projected_vram_free = max(
                    fresh_state.gpu.vram_free_mb,
                    fresh_state.gpu.vram_total_mb - baseline_vram_mb,
                )
                logger.debug(
                    "VRAM projection: total=%dMB - baseline=%dMB = projected_free=%dMB "
                    "(current_free=%dMB)",
                    fresh_state.gpu.vram_total_mb, baseline_vram_mb,
                    projected_vram_free, fresh_state.gpu.vram_free_mb,
                )
            else:
                # No model loaded → VRAM reading is accurate as-is
                projected_vram_free = fresh_state.gpu.vram_free_mb

            new_ctx = calculate_dynamic_context(
                file_size_mb=info.file_size_mb,
                n_layers=info.total_layers,
                gpu_layers=info.gpu_layers,
                available_ram_mb=fresh_state.ram_available_mb,
                available_vram_mb=projected_vram_free,
                family_key=info.family,
            )
            if new_ctx != context_length:
                logger.info(
                    "Dynamic context: %d -> %d (RAM free: %dMB, VRAM projected free: %dMB)",
                    context_length, new_ctx,
                    fresh_state.ram_available_mb, projected_vram_free,
                )
                context_length = new_ctx

        # Use task's min_context as a floor — if dynamic calc returned less
        # than the task needs, bump up.  llama-server will try to fit it;
        # if VRAM is truly insufficient it OOMs and the circuit breaker
        # handles it, same outcome as refusing but with a chance of success.
        if min_context > 0 and context_length < min_context:
            logger.info(
                "Bumping context %d -> %d to meet task min_context",
                context_length, min_context,
            )
            context_length = min_context

        # gpu_layers override handling — local var, don't mutate registry
        gpu_layers = info.gpu_layers
        gpu_layers_overridden = False
        if "gpu_layers" in registry_overrides:
            gpu_layers = registry_overrides["gpu_layers"]
            gpu_layers_overridden = True

        # Build ServerConfig
        from dallama import ServerConfig
        extra = []
        if hasattr(info, 'extra_server_flags') and info.extra_server_flags:
            extra = list(info.extra_server_flags)

        # Only pass gpu_layers if explicitly overridden
        if gpu_layers_overridden and gpu_layers > 0:
            extra.extend(["--n-gpu-layers", str(gpu_layers)])

        sc = ServerConfig(
            model_path=info.path,
            model_name=model_name,
            context_length=context_length,
            thinking=enable_thinking if info.thinking_model else False,
            vision_projector=(info.mmproj_path or "") if enable_vision and info.has_vision else "",
            extra_flags=extra,
        )

        # Signal swap in progress
        old_model = self._dallama.status.model_name
        self.swap_started_at = time.monotonic()
        self._swap_ready.clear()
        try:
            success = await self._dallama._swap.swap(self._dallama._server, sc)
            if success:
                self._dallama._current_config = sc
        finally:
            self.swap_started_at = 0.0
            self._swap_ready.set()

        if success:
            self._total_swaps += 1
            self._started_at = time.time()
            self._last_request_time = time.time()
            self._thinking_enabled = sc.thinking
            self._vision_enabled = bool(sc.vision_projector)
            self._dallama.keep_alive()  # Start idle-unload timer after swap

            # Seed measured_tps from persisted speed cache
            _seed_tps = info.tokens_per_second if info.tokens_per_second > 0 else 0.0
            self.runtime_state = ModelRuntimeState(
                model_name=model_name,
                thinking_enabled=enable_thinking,
                context_length=context_length,
                gpu_layers=gpu_layers,
                measured_tps=_seed_tps,
            )
            registry.mark_loaded(model_name, self.api_base)
            logger.info(
                "Model %s loaded (swap #%d)", model_name, self._total_swaps
            )

            # Record swap in budget
            try:
                from src.core.llm_dispatcher import get_dispatcher
                get_dispatcher().swap_budget.record_swap()
            except Exception as _e:
                logger.warning("Failed to record swap in budget: %s", _e)

            # Notify dispatcher for deferred grade draining
            try:
                from src.core.llm_dispatcher import get_dispatcher
                old_litellm = None
                new_litellm = None
                if old_model:
                    old_info = registry.get(old_model)
                    old_litellm = old_info.litellm_name if old_info else None
                new_info = registry.get(model_name)
                new_litellm = new_info.litellm_name if new_info else None
                asyncio.ensure_future(
                    get_dispatcher().on_model_swap(old_litellm, new_litellm)
                )
            except Exception as _e:
                logger.debug("Dispatcher swap notification failed: %s", _e)

            return True
        else:
            self.runtime_state = None
            if old_model:
                registry.mark_unloaded(old_model)
            registry.mark_unloaded(model_name)
            registry.demote_model(model_name, duration=300)
            logger.error("Failed to load %s (demoted for 5 min)", model_name)
            return False

    async def acquire_inference_slot(
        self,
        priority: int = 5,
        task_id: str = "?",
        agent_type: str = "",
        timeout: float = 120,
    ) -> bool:
        """Acquire GPU inference slot via priority scheduler."""
        from .gpu_scheduler import GPURequest, get_gpu_scheduler

        scheduler = get_gpu_scheduler()
        if priority >= 10:
            timeout = min(timeout, 30)

        request = GPURequest.make(
            priority=priority,
            task_id=task_id,
            agent_type=agent_type,
            model_needed=self.current_model or "",
        )
        granted = await scheduler.acquire(request, timeout=timeout)
        if granted:
            self._last_request_time = time.time()
        return granted

    def release_inference_slot(self) -> None:
        """Release GPU slot."""
        from .gpu_scheduler import get_gpu_scheduler
        get_gpu_scheduler().release()

    def mark_inference_start(self) -> int:
        """Delegate to DaLLaMa swap manager."""
        return self._dallama._swap.mark_inference_start()

    def mark_inference_end(self, generation: int) -> None:
        """Delegate to DaLLaMa swap manager."""
        self._dallama._swap.mark_inference_end(generation)

    def get_status(self) -> dict:
        """Return status dict for diagnostics / Telegram reporting."""
        from .model_registry import get_registry
        registry = get_registry()
        model = self.current_model
        model_info = registry.get(model) if model else None
        status = self._dallama.status
        return {
            "loaded_model": model,
            "model_type": model_info.model_type if model_info else None,
            "healthy": status.healthy,
            "port": self._port,
            "idle_seconds": round(self.idle_seconds, 1),
            "total_swaps": self._total_swaps,
            "uptime_seconds": round(
                time.time() - self._started_at, 1
            ) if hasattr(self, '_started_at') and self._started_at else 0,
            "inference_busy": self._scheduler.is_busy,
        }

    async def get_metrics(self) -> dict:
        """Fetch live metrics from llama-server."""
        result = {
            **self.get_status(),
            "prompt_tokens_total": 0,
            "generation_tokens_total": 0,
            "prompt_seconds_total": 0.0,
            "generation_seconds_total": 0.0,
            "prompt_tokens_per_second": 0.0,
            "generation_tokens_per_second": 0.0,
            "requests_processing": 0,
            "requests_pending": 0,
            "kv_cache_usage_percent": 0.0,
        }
        if not self.is_loaded:
            return result

        try:
            snap = await self._dallama._metrics.fetch(
                self._dallama._server.api_base
            )
            result["prompt_tokens_per_second"] = snap.prompt_tokens_per_second
            result["generation_tokens_per_second"] = snap.generation_tokens_per_second
            result["kv_cache_usage_percent"] = snap.kv_cache_usage_percent
            result["requests_processing"] = snap.requests_processing
            result["requests_pending"] = snap.requests_pending
            result["prompt_tokens_total"] = snap.prompt_tokens_total
            result["generation_tokens_total"] = snap.generation_tokens_total

            # Update runtime_state.measured_tps from live /metrics
            tps = snap.generation_tokens_per_second
            if tps > 0 and self.runtime_state is not None:
                self.runtime_state.measured_tps = tps
        except Exception as e:
            logger.debug("Failed to fetch llama-server metrics: %s", e)

        return result

    @property
    def current_model(self) -> Optional[str]:
        return self._dallama.status.model_name

    @current_model.setter
    def current_model(self, value: Optional[str]) -> None:
        # Some old code sets current_model = None on failure.
        # DaLLaMa tracks this internally via _current_config; the setter
        # is a no-op since DaLLaMa is the source of truth.
        pass

    @property
    def is_loaded(self) -> bool:
        s = self._dallama.status
        return s.model_name is not None and s.healthy

    @property
    def idle_seconds(self) -> float:
        if self._last_request_time == 0:
            return 0.0
        return time.time() - self._last_request_time

    async def run_idle_unloader(
        self,
        check_interval: float = 30,
        max_idle_minutes: float = 1,
    ) -> None:
        """No-op — DaLLaMa runs its own idle unloader internally."""
        # Block forever so create_task() callers don't get a completed task
        await asyncio.Event().wait()

    async def run_health_watchdog(self, check_interval: float = 30) -> None:
        """No-op — DaLLaMa runs its own watchdog internally."""
        await asyncio.Event().wait()

    async def _health_check(self) -> bool:
        """Check if the llama-server is healthy (delegates to DaLLaMa)."""
        if not self._started or not self._dallama._server:
            return False
        return await self._dallama._server.health_check()

    async def shutdown(self) -> None:
        """Graceful shutdown — called from atexit or orchestrator."""
        if self._started:
            await self._dallama.stop()
            self._started = False


# ── Singleton ────────────────────────────────────────────────────────────────

_manager: LocalModelManager | None = None


def _atexit_cleanup() -> None:
    """Last-resort cleanup when the Python process exits."""
    global _manager
    if _manager is None:
        return
    # DaLLaMa's server process is managed by platform helper with
    # Windows Job Object — it will be auto-killed. But try graceful
    # stop if an event loop is available.
    proc = _manager._dallama._server.process
    if proc is not None and proc.poll() is None:
        logger.info("atexit: killing llama-server (PID %d)", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=3)
        except Exception as e:
            logger.warning("atexit: llama-server kill failed: %s", e)
            import subprocess
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass


def get_local_manager() -> LocalModelManager:
    global _manager
    if _manager is None:
        _manager = LocalModelManager()
        atexit.register(_atexit_cleanup)
    return _manager


def get_runtime_state() -> ModelRuntimeState | None:
    """Return the runtime state of the currently loaded model, or None."""
    return _manager.runtime_state if _manager is not None else None
