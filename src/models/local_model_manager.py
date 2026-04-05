# local_model_manager.py
"""
Local Model Manager — controls llama-server lifecycle on a single GPU.

Responsibilities:
  - Start/stop llama-server with specific GGUF models
  - Model swapping with minimal downtime
  - Health checking and auto-restart
  - Work queue batching (minimize swaps)
  - Mutex: only one model loaded at a time
"""

from __future__ import annotations

import os
import asyncio
import subprocess
import time
from dataclasses import dataclass, field

from src.infra.logging_config import get_logger
from pathlib import Path
from typing import Optional

import httpx

from .gpu_monitor import get_gpu_monitor
from .model_registry import ModelInfo, get_registry
from .gpu_scheduler import get_gpu_scheduler

logger = get_logger("models.local_model_manager")

# Path to llama-server executable — set via env var or auto-detect
LLAMA_SERVER_PATH = Path(
    os.environ.get("LLAMA_SERVER_PATH", "llama-server")
)
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))


@dataclass
class ModelSwapRequest:
    """A request to load a specific model, with an event to signal completion."""
    model_name: str
    reason: str
    priority: int = 5              # higher = more urgent
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


# Restart circuit breaker thresholds
_RESTART_FAIL_THRESHOLD: int = 2   # failures before cooldown
_RESTART_COOLDOWN_S: float = 300.0  # 5 minutes


class LocalModelManager:
    """
    Manages a single llama-server process.

    Key design choices:
    - asyncio.Lock ensures only one swap happens at a time
    - Swap requests are queued and deduplicated
    - Health checks run periodically
    - If the process dies, auto-restart with the same model
    - Circuit breaker prevents restart loops on persistent failures
    """

    def __init__(self):
        self.current_model: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.port: int = LLAMA_SERVER_PORT
        self.api_base: str = f"http://127.0.0.1:{self.port}"

        self._swap_lock = asyncio.Lock()
        self._swap_queue: asyncio.Queue[ModelSwapRequest] = asyncio.Queue()
        # Swap state: 0.0 = no swap in progress, >0 = monotonic time swap started.
        # Using a timestamp instead of a boolean lets callers detect stale reads:
        # if swap_started_at changed between their read and their action, the
        # state they read is stale.
        self.swap_started_at: float = 0.0
        self._last_request_time: float = 0.0
        self._started_at: float = 0.0
        self._total_swaps: int = 0
        self._thinking_enabled: bool = False  # tracks server-side thinking state
        self._vision_enabled: bool = False    # tracks whether --mmproj is loaded

        # Tracks how many inference requests are currently in-flight.
        # Model swaps wait for this to reach zero before killing the server.
        self._active_inference_count: int = 0
        self._inference_idle = asyncio.Event()
        self._inference_idle.set()  # starts idle
        # Generation counter — bumped on force-swap so orphaned inferences
        # from pre-swap generations don't corrupt the count when they finish.
        self._inference_generation: int = 0

        self.runtime_state: ModelRuntimeState | None = None  # populated after successful swap

        # ── Restart circuit breaker ──────────────────────────────
        # Prevents restart loops when a model consistently fails to load.
        # After _RESTART_FAIL_THRESHOLD consecutive failures for the SAME
        # model, further load attempts are refused for _RESTART_COOLDOWN_S.
        self._restart_fail_count: int = 0
        self._restart_fail_model: str | None = None
        self._restart_cooldown_until: float = 0.0

        # Set by idle unloader before stopping, cleared after.
        # Prevents the health watchdog from misinterpreting a deliberate
        # idle unload as a crash.
        self._idle_unload_in_progress: bool = False

        self._scheduler = get_gpu_scheduler()
        self._job_object = self._create_job_object()

        # Kill any orphaned llama-server processes from prior runs
        self._kill_orphaned_servers()

    @staticmethod
    def _create_job_object():
        """Create a Windows Job Object with KILL_ON_JOB_CLOSE.

        When the parent process dies (even ungracefully), Windows closes
        all handles — including this job — which auto-kills every process
        assigned to it.  Returns None on non-Windows or on failure.
        """
        import platform
        if platform.system() != "Windows":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            # CreateJobObjectW(lpJobAttributes, lpName)
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                logger.debug("Failed to create Job Object")
                return None

            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            # Affinity is ULONG_PTR (pointer-width integer), not a pointer.
            # Using c_size_t which is the correct width on both 32- and 64-bit.
            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            ok = kernel32.SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                logger.debug("Failed to set KILL_ON_JOB_CLOSE on Job Object")
                kernel32.CloseHandle(handle)
                return None

            logger.info("Windows Job Object created (KILL_ON_JOB_CLOSE)")
            return handle
        except Exception as e:
            logger.debug(f"Job Object setup failed: {e}")
            return None

    def _assign_to_job(self, process: subprocess.Popen) -> None:
        """Assign a child process to the Job Object so it dies with us."""
        if self._job_object is None:
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_process = kernel32.OpenProcess(0x1F0FFF, False, process.pid)  # PROCESS_ALL_ACCESS
            if h_process:
                ok = kernel32.AssignProcessToJobObject(self._job_object, h_process)
                kernel32.CloseHandle(h_process)
                if ok:
                    logger.info(f"llama-server (PID {process.pid}) assigned to Job Object")
                else:
                    logger.warning(f"Failed to assign PID {process.pid} to Job Object")
        except Exception as e:
            logger.debug(f"Job assignment failed: {e}")

    def _kill_orphaned_servers(self) -> None:
        """Kill leftover llama-server processes that survived a prior crash/restart."""
        import platform
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning(
                        f"Killed orphaned llama-server(s): "
                        f"{result.stdout.strip()}"
                    )
            else:
                result = subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning("Killed orphaned llama-server process(es)")
        except Exception as e:
            logger.debug(f"Orphan cleanup skipped: {e}")

    # ── Inference tracking ─────────────────────────────────────

    def mark_inference_start(self) -> int:
        """Mark that an inference request is starting against the current server.

        Returns the current inference generation. Callers MUST pass this value
        back to mark_inference_end() so that orphaned inferences from a
        pre-force-swap generation don't corrupt the counter.
        """
        self._active_inference_count += 1
        self._inference_idle.clear()
        return self._inference_generation

    def mark_inference_end(self, generation: int) -> None:
        """Mark that an inference request has finished.

        Args:
            generation: The generation returned by mark_inference_start().
                        If the generation doesn't match the current one
                        (a force-swap happened), the decrement is skipped
                        because the counter was already reset.
        """
        if generation != self._inference_generation:
            # This inference started before a force-swap — the counter was
            # already reset when the generation was bumped. Decrementing
            # would make the counter go negative (or undercount new inferences).
            return
        self._active_inference_count = max(0, self._active_inference_count - 1)
        if self._active_inference_count == 0:
            self._inference_idle.set()

    # ── Restart circuit breaker ─────────────────────────────────

    def _record_restart_failure(self, model_name: str) -> None:
        """Record a failed load attempt. Triggers cooldown after threshold."""
        if self._restart_fail_model != model_name:
            # Different model — reset counter
            self._restart_fail_model = model_name
            self._restart_fail_count = 0
        self._restart_fail_count += 1
        if self._restart_fail_count >= _RESTART_FAIL_THRESHOLD:
            self._restart_cooldown_until = time.time() + _RESTART_COOLDOWN_S
            logger.error(
                f"Circuit breaker: {model_name} failed {self._restart_fail_count} "
                f"times — refusing loads for {_RESTART_COOLDOWN_S:.0f}s"
            )
            # Schedule sleeping queue wake when cooldown expires
            try:
                import asyncio as _aio

                async def _wake_on_cooldown():
                    await _aio.sleep(_RESTART_COOLDOWN_S)
                    try:
                        from src.infra.db import accelerate_retries
                        await accelerate_retries("circuit_breaker_reset")
                    except Exception:
                        pass

                _aio.ensure_future(_wake_on_cooldown())
            except RuntimeError:
                pass  # no running loop

    def _record_restart_success(self, model_name: str) -> None:
        """Record a successful load. Resets the circuit breaker."""
        self._restart_fail_count = 0
        self._restart_fail_model = None
        self._restart_cooldown_until = 0.0

    def _is_restart_blocked(self, model_name: str) -> bool:
        """Check if the circuit breaker blocks loading this model."""
        if self._restart_cooldown_until <= 0:
            return False
        if time.time() >= self._restart_cooldown_until:
            # Cooldown expired — reset
            self._restart_fail_count = 0
            self._restart_cooldown_until = 0.0
            return False
        if self._restart_fail_model == model_name:
            return True
        # Different model — not blocked
        return False

    # ── Public API ──────────────────────────────────────────────

    async def ensure_model(
        self,
        model_name: str,
        reason: str = "",
        enable_thinking: bool = False,
        enable_vision: bool = False,
    ) -> bool:
        """
        Ensure the specified model is loaded and healthy.
        If already loaded with the same thinking state, returns immediately.
        If different model or thinking state differs, swaps (blocks until ready).
        Returns False if the circuit breaker is blocking this model.

        enable_vision: if True AND the model has an mmproj file, restart
            the server with --mmproj. Vision is off by default to save
            ~876MB VRAM. Only toggled on for actual vision tasks.
        """
        # ── Circuit breaker: refuse loads of a model that keeps failing ──
        if self._is_restart_blocked(model_name):
            logger.warning(
                f"Circuit breaker active — refusing to load {model_name} "
                f"(cooldown until {self._restart_cooldown_until - time.time():.0f}s)"
            )
            return False

        if self.current_model == model_name:
            if await self._health_check():
                # Vision toggle: restart needed if vision requested but not loaded
                if enable_vision and not self._vision_enabled:
                    logger.info(
                        f"Restarting {model_name} with vision projector "
                        f"(mmproj) for vision task"
                    )
                    # Fall through to _swap_model which will restart with mmproj
                else:
                    # Model is loaded and healthy — skip restart even if thinking
                    # state differs.  Restarting the server just to toggle thinking
                    # causes swap storms that freeze the entire system.
                    if self._thinking_enabled != enable_thinking:
                        logger.debug(
                            f"Ignoring thinking state change "
                            f"{self._thinking_enabled} -> {enable_thinking} "
                            f"for {model_name} (avoiding swap storm)"
                        )
                    return True
            else:
                # Loaded but unhealthy — restart
                logger.warning(f"Model {model_name} unhealthy, restarting")

        return await self._swap_model(
            model_name, reason,
            enable_thinking=enable_thinking,
            enable_vision=enable_vision,
        )

    async def acquire_inference_slot(
        self,
        priority: int = 5,
        task_id: str = "?",
        agent_type: str = "",
        timeout: float = 120,
    ) -> bool:
        """
        Acquire GPU inference slot via priority scheduler.

        Returns True if granted, False if timed out.
        Callers MUST call release_inference_slot() in a finally block.
        """
        from .gpu_scheduler import GPURequest, get_gpu_scheduler

        scheduler = get_gpu_scheduler()

        # Adjust timeout for high-priority tasks
        if priority >= 10:
            timeout = min(timeout, 30)  # critical tasks fail fast to try cloud

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
        """Release GPU slot — grants to next highest-priority waiter."""
        from .gpu_scheduler import get_gpu_scheduler
        scheduler = get_gpu_scheduler()
        scheduler.release()

    @property
    def is_loaded(self) -> bool:
        return self.current_model is not None and self.process is not None

    @property
    def idle_seconds(self) -> float:
        if self._last_request_time == 0:
            return 0.0
        return time.time() - self._last_request_time

    def get_status(self) -> dict:
        """Return status dict for diagnostics / Telegram reporting."""
        registry = get_registry()
        model_info = registry.get(self.current_model) if self.current_model else None
        return {
            "loaded_model": self.current_model,
            "model_type": model_info.model_type if model_info else None,
            "healthy": self.process is not None and self.process.poll() is None,
            "port": self.port,
            "idle_seconds": round(self.idle_seconds, 1),
            "total_swaps": self._total_swaps,
            "uptime_seconds": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "inference_busy": self._scheduler.is_busy,
        }

    # ── Model Swapping ─────────────────────────────────────────

    async def _swap_model(self, model_name: str, reason: str = "",
                          enable_thinking: bool = False, enable_vision: bool = False) -> bool:
        """
        Stop current model, start new one. Protected by lock.
        Returns True if the new model is healthy.
        """
        async with self._swap_lock:
            # Re-check after acquiring lock — another task may have already
            # loaded the model while we were waiting.
            if self.current_model == model_name and await self._health_check():
                # Still need to check vision state — we may have been called
                # specifically to add mmproj
                if not (enable_vision and not self._vision_enabled):
                    logger.debug(f"Model {model_name} already healthy (resolved under lock)")
                    return True

            self.swap_started_at = time.monotonic()
            try:
                return await self._do_swap(model_name, reason, enable_thinking, enable_vision)
            finally:
                self.swap_started_at = 0.0

    async def _do_swap(self, model_name: str, reason: str = "",
                       enable_thinking: bool = False, enable_vision: bool = False) -> bool:
        """Inner swap logic, called under _swap_lock with swap_started_at set."""
        # ── Wait for in-flight inference to finish before killing server ──
        # This prevents killing llama-server while an HTTP request is mid-stream,
        # which causes the HTTP client to hang on a dead TCP connection.
        if self._active_inference_count > 0:
            logger.info(
                f"Waiting for {self._active_inference_count} in-flight inference(s) "
                f"to drain before swap to {model_name}..."
            )
            try:
                await asyncio.wait_for(self._inference_idle.wait(), timeout=30)
                logger.info("Inference drained, proceeding with swap")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Inference drain timed out after 30s "
                    f"({self._active_inference_count} still active). "
                    f"Force-swapping — active requests will receive errors."
                )
                # Bump the generation so orphaned mark_inference_end() calls
                # from pre-swap inferences are ignored. Then reset the counter
                # cleanly — any future decrements from the old generation will
                # see a mismatched generation and skip.
                self._inference_generation += 1
                self._active_inference_count = 0
                self._inference_idle.set()

        registry = get_registry()
        model_info = registry.get(model_name)

        if not model_info or not model_info.is_local:
            logger.error(f"Model '{model_name}' not found in registry or not local")
            return False

        if not model_info.path or not Path(model_info.path).is_file():
            logger.error(f"GGUF file not found: {model_info.path}")
            return False

        # Pre-swap checks
        gpu_monitor = get_gpu_monitor()
        state = gpu_monitor.get_state()
        if not state.can_load_model:
            logger.error(
                f"Insufficient RAM to load {model_name}: "
                f"{state.ram_available_mb}MB available, need >4096MB free"
            )
            return False

        old_model = self.current_model
        logger.info(
            f"🔄 Model swap: {old_model or '(none)'} → {model_name} "
            f"(reason: {reason})"
        )
        swap_start = time.time()

        # Stop existing server
        await self._stop_server()

        # ── Recalculate context + gpu_layers with live memory state ──
        # Server is stopped, so readings reflect true free memory.
        gpu_monitor.invalidate_cache()  # force fresh reading
        fresh_state = gpu_monitor.get_state()

        from .model_registry import calculate_dynamic_context

        # Check for user override in models.yaml
        registry_overrides = registry.get_overrides(model_name)

        if "context_length" not in registry_overrides:
            new_ctx = calculate_dynamic_context(
                file_size_mb=model_info.file_size_mb,
                n_layers=model_info.total_layers,
                gpu_layers=model_info.gpu_layers,
                available_ram_mb=fresh_state.ram_available_mb,
                available_vram_mb=fresh_state.gpu.vram_free_mb,
                family_key=model_info.family,
            )
            if new_ctx != model_info.context_length:
                logger.info(
                    f"📐 Dynamic context: {model_info.context_length} → {new_ctx} "
                    f"(RAM free: {fresh_state.ram_available_mb}MB, "
                    f"VRAM free: {fresh_state.gpu.vram_free_mb}MB)"
                )
                model_info.context_length = new_ctx

        # Only apply gpu_layers if user explicitly overrides in models.yaml.
        # Otherwise, llama-server --fit auto-calculates optimal layers.
        if "gpu_layers" in registry_overrides:
            model_info.gpu_layers = registry_overrides["gpu_layers"]
            model_info._gpu_layers_from_override = True
        else:
            model_info._gpu_layers_from_override = False

        # Start new server
        success = await self._start_server(model_info, enable_thinking=enable_thinking,
                                           enable_vision=enable_vision)

        swap_duration = time.time() - swap_start
        self._total_swaps += 1

        if success:
            self._record_restart_success(model_name)
            self.current_model = model_name
            self._thinking_enabled = enable_thinking
            self._vision_enabled = enable_vision
            self._started_at = time.time()
            self.runtime_state = ModelRuntimeState(
                model_name=model_name,
                thinking_enabled=enable_thinking,
                context_length=model_info.context_length,
                gpu_layers=model_info.gpu_layers,
            )
            registry.mark_loaded(model_name, self.api_base)
            logger.info(
                f"✅ Model {model_name} loaded in {swap_duration:.1f}s "
                f"(swap #{self._total_swaps})"
            )
            # Record swap in budget SYNCHRONOUSLY — this must never be
            # lost even if the async notification below fails to schedule.
            # Grade draining and sleeping queue signaling happen asynchronously
            # via on_model_swap, but the budget record is critical.
            try:
                from src.core.llm_dispatcher import get_dispatcher
                get_dispatcher().swap_budget.record_swap()
            except Exception as _e:
                logger.warning(f"Failed to record swap in budget: {_e}")

            # Notify dispatcher for deferred grade draining & sleeping queue wake
            try:
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
                logger.debug(f"Dispatcher swap notification failed: {_e}")
            return True
        else:
            self._record_restart_failure(model_name)
            self.current_model = None
            registry.mark_unloaded(model_name)
            # Demote failed model so the router skips it on retry
            registry.demote_model(model_name, duration=300)
            logger.error(
                f"❌ Failed to load {model_name} after {swap_duration:.1f}s "
                f"(demoted for 5 min, circuit_breaker={self._restart_fail_count}/"
                f"{_RESTART_FAIL_THRESHOLD})"
            )
            return False

    @staticmethod
    def _get_inference_threads() -> int:
        """Use physical core count for inference threads (not hyperthreads)."""
        try:
            import psutil
            # physical cores only — hyperthreads hurt llama.cpp performance
            physical = psutil.cpu_count(logical=False) or 4
            # Reserve 2 cores for the orchestrator + OS
            return max(2, physical - 2)
        except ImportError:
            import os
            # Fallback: logical cores / 2 as rough estimate of physical
            return max(2, (os.cpu_count() or 4) // 2 - 1)

    async def _start_server(self, model: ModelInfo, enable_thinking: bool = False,
                            enable_vision: bool = False) -> bool:
        """
        Launch llama-server process and wait for it to become healthy.
        """
        cmd = [
            str(LLAMA_SERVER_PATH),
            "--model", model.path,
            "--alias", "local-model",  # stable name for Perplexica/Vane integration
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--ctx-size", str(model.context_length),
            "--flash-attn", "auto",
            "--metrics",
            # ── Performance flags ──
            "--threads", str(self._get_inference_threads()),
            "--batch-size", "2048", # prompt processing batch size (higher = faster prefill)
            "--ubatch-size", "512", # micro-batch for generation
        ]

        # Always pass explicit --n-gpu-layers for deterministic VRAM usage.
        # --fit auto-calculates but allocates fewer layers under runtime VRAM
        # pressure (e.g. 1013MB baseline vs 760MB in clean benchmark), causing
        # 12.9 tok/s instead of 25.4 tok/s.  Passing 99 forces maximum offload
        # — llama-server caps at the model's actual layer count.
        if getattr(model, '_gpu_layers_from_override', False) and model.gpu_layers > 0:
            cmd.extend(["--n-gpu-layers", str(model.gpu_layers)])
        else:
            cmd.extend(["--n-gpu-layers", "99"])

        # Control thinking via chat_template_kwargs (server-level flag,
        # not supported per-request by llama-server).
        if model.thinking_model:
            import json as _json
            cmd.extend([
                "--chat-template-kwargs",
                _json.dumps({"enable_thinking": enable_thinking}),
            ])

        # Vision projector (mmproj) — only loaded when a vision task needs it.
        # Saves ~876MB VRAM for the 99% of requests that are text-only.
        if enable_vision and model.has_vision and getattr(model, 'mmproj_path', None):
            cmd.extend(["--mmproj", model.mmproj_path])

        # Per-model server flags (MoE override-kv, Apriel --no-jinja, etc.)
        if hasattr(model, 'extra_server_flags') and model.extra_server_flags:
            cmd.extend(model.extra_server_flags)

        logger.info(f"Starting llama-server: {' '.join(cmd)}...")

        try:
            # Use CREATE_NO_WINDOW on Windows to hide the console
            import platform
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            self._assign_to_job(self.process)
        except FileNotFoundError:
            logger.error(
                f"llama-server not found at '{LLAMA_SERVER_PATH}'. "
                f"Set LLAMA_SERVER_PATH environment variable."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False

        # Wait for health endpoint — poll with backoff.
        # Large models (>20GB) can take 90s+ to load on first start after
        # reboot when VRAM needs initial allocation. The 2x multiplier on
        # estimated load time handles most cases, but the ceiling must be
        # high enough for worst-case cold starts.
        max_wait = model.load_time_seconds * 2.5  # 2.5x estimated time
        max_wait = max(max_wait, 30)               # at least 30s
        max_wait = min(max_wait, 180)              # at most 180s

        healthy = await self._wait_for_healthy(timeout=max_wait)

        if not healthy:
            logger.error(f"llama-server failed to become healthy within {max_wait}s")
            await self._stop_server()
            return False

        return True

    async def _stop_server(self) -> None:
        """Gracefully stop the llama-server process (fully async — never blocks the event loop)."""
        if self.process is None:
            return

        old_model = self.current_model
        logger.info(f"Stopping llama-server (model: {old_model})...")

        loop = asyncio.get_running_loop()

        try:
            self.process.terminate()
            # Wait up to 10s for graceful shutdown (poll is non-blocking)
            for _ in range(20):
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            else:
                # Force kill if still running — use run_in_executor so
                # the synchronous process.wait() never blocks the event loop.
                logger.warning("llama-server didn't stop gracefully, killing...")
                self.process.kill()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self.process.wait),
                        timeout=5,
                    )
                except asyncio.TimeoutError:
                    logger.warning("llama-server did not exit within 5s after kill")
        except Exception as e:
            logger.warning(f"Error stopping llama-server: {e}")
            try:
                self.process.kill()
            except Exception:
                pass

        self.process = None
        self.runtime_state = None
        if old_model:
            get_registry().mark_unloaded(old_model)

    async def _wait_for_healthy(self, timeout: float = 60) -> bool:
        """Poll /health endpoint until server is ready."""
        start = time.time()
        check_interval = 1.0

        async with httpx.AsyncClient() as client:
            while (time.time() - start) < timeout:
                # Check if process died
                if self.process and self.process.poll() is not None:
                    logger.error(
                        f"llama-server process exited with code "
                        f"{self.process.returncode}"
                    )
                    return False

                try:
                    resp = await client.get(
                        f"{self.api_base}/health",
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status", "")
                        if status == "ok" or status == "no slot available":
                            # "no slot available" means loaded but busy = healthy
                            return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass  # server still starting
                except Exception as e:
                    logger.debug(f"Health check error: {e}")

                await asyncio.sleep(check_interval)
                # Increase interval after first few checks
                if check_interval < 3.0:
                    check_interval += 0.5

        return False

    async def get_metrics(self) -> dict:
        """
        Fetch live metrics from llama-server's /metrics (Prometheus format).
        Returns parsed dict with key performance indicators.
        """
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
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.api_base}/metrics", timeout=3.0)
                if resp.status_code != 200:
                    return result

                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    # Parse Prometheus lines: metric_name{labels} value
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    metric = parts[0].split("{")[0]
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        continue

                    # Normalize metric name: llama.cpp uses either colons or
                    # underscores depending on version (e.g., llamacpp:foo or llamacpp_foo)
                    m = metric.replace(":", "_")

                    if m == "llamacpp_prompt_tokens_total":
                        result["prompt_tokens_total"] = int(val)
                    elif m == "llamacpp_tokens_predicted_total":
                        result["generation_tokens_total"] = int(val)
                    elif m == "llamacpp_prompt_seconds_total":
                        result["prompt_seconds_total"] = round(val, 2)
                    elif m == "llamacpp_tokens_predicted_seconds_total":
                        result["generation_seconds_total"] = round(val, 2)
                    elif m == "llamacpp_prompt_tokens_seconds":
                        result["prompt_tokens_per_second"] = round(val, 1)
                    elif m == "llamacpp_tokens_predicted_seconds":
                        result["generation_tokens_per_second"] = round(val, 1)
                    elif m == "llamacpp_requests_processing":
                        result["requests_processing"] = int(val)
                    elif m == "llamacpp_requests_pending":
                        result["requests_pending"] = int(val)
                    elif m == "llamacpp_kv_cache_usage_ratio":
                        result["kv_cache_usage_percent"] = round(val * 100, 1)

                # Update runtime_state.measured_tps from live /metrics so
                # router.py can use actual tok/s for speed scoring.
                tps = result.get("generation_tokens_per_second", 0.0)
                if tps > 0 and self.runtime_state is not None:
                    self.runtime_state.measured_tps = tps

        except Exception as e:
            logger.debug(f"Failed to fetch llama-server metrics: {e}")

        return result

    async def _health_check(self) -> bool:
        """Quick health check — is the server responding?"""
        return (await self._health_check_status()) == 200

    async def _health_check_status(self) -> int:
        """Health check returning HTTP status code. Returns 0 on connection failure."""
        if self.process is None or self.process.poll() is not None:
            return 0
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/health",
                    timeout=3.0,
                )
                return resp.status_code
        except Exception:
            return 0

    # ── Background Tasks ────────────────────────────────────────

    async def run_idle_unloader(
        self,
        check_interval: float = 30,
        max_idle_minutes: float = 1,
    ) -> None:
        """
        Background task: unload model if idle for too long.
        Frees VRAM when no inference requests arrive within the idle window.
        Run as: asyncio.create_task(manager.run_idle_unloader())
        """
        max_idle = max_idle_minutes * 60

        while True:
            await asyncio.sleep(check_interval)

            if not self.is_loaded:
                continue

            if self.idle_seconds > max_idle:
                # Don't unload if tasks are actively processing — the agent
                # may be between LLM calls (tool execution, file I/O, etc.)
                try:
                    from src.infra.db import get_db
                    db = await get_db()
                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status = 'processing'"
                    )
                    processing = (await cursor.fetchone())[0]
                    if processing > 0:
                        logger.debug(
                            f"Idle unload skipped: {processing} task(s) still processing"
                        )
                        continue
                except Exception:
                    pass  # DB error → safe to proceed with unload

                logger.info(
                    f"Model {self.current_model} idle for "
                    f"{self.idle_seconds:.0f}s (>{max_idle}s), unloading"
                )
                self._idle_unload_in_progress = True
                try:
                    async with self._swap_lock:
                        await self._stop_server()
                        self.current_model = None
                finally:
                    self._idle_unload_in_progress = False

    async def run_health_watchdog(self, check_interval: float = 30) -> None:
        """
        Background task: detect crashed *or hung* server and restart.

        Detects two failure modes:
          1. Process exit (crash) — via process.poll()
          2. Process alive but unresponsive (hang) — via /health HTTP check.
             Three consecutive /health failures trigger a restart.

        Run as: asyncio.create_task(manager.run_health_watchdog())
        """
        consecutive_health_failures = 0
        HEALTH_FAIL_THRESHOLD = 3  # restart after this many consecutive failures

        while True:
            await asyncio.sleep(check_interval)

            if not self.current_model:
                consecutive_health_failures = 0
                continue

            # ── Crash detection (process exited) ──
            if self.process and self.process.poll() is not None:
                # If the idle unloader is deliberately stopping the server,
                # don't treat it as a crash — it will clear current_model itself.
                if self._idle_unload_in_progress:
                    logger.debug(
                        "Watchdog: process exited during idle unload, not a crash"
                    )
                    consecutive_health_failures = 0
                    continue

                model_name = self.current_model
                logger.error(
                    f"llama-server crashed! Restarting {model_name}..."
                )
                # Close pipes before discarding the process reference
                self.process = None
                self.current_model = None
                consecutive_health_failures = 0
                if self._is_restart_blocked(model_name):
                    logger.warning(
                        f"Watchdog: circuit breaker active for {model_name} "
                        f"— skipping crash recovery"
                    )
                    continue
                await self._swap_model(model_name, reason="crash recovery")
                continue

            # ── Hang detection (process alive but /health unresponsive) ──
            if self.process and self.process.poll() is None:
                # Skip if idle unloader or swap is in progress — server
                # may be stopping or restarting, not hung.
                if self._idle_unload_in_progress or self.swap_started_at > 0:
                    consecutive_health_failures = 0
                    continue

                status = await self._health_check_status()
                if status == 200:
                    consecutive_health_failures = 0
                elif status == 503:
                    # 503 = server alive but busy loading model. NOT a hang.
                    # Reset counter — the server is responsive, just occupied.
                    consecutive_health_failures = 0
                else:
                    consecutive_health_failures += 1
                    logger.warning(
                        f"llama-server /health failed "
                        f"({consecutive_health_failures}/{HEALTH_FAIL_THRESHOLD})"
                    )
                    if consecutive_health_failures >= HEALTH_FAIL_THRESHOLD:
                        model_name = self.current_model
                        logger.error(
                            f"llama-server hung ({HEALTH_FAIL_THRESHOLD} "
                            f"consecutive /health failures). Restarting "
                            f"{model_name}..."
                        )
                        consecutive_health_failures = 0
                        if self._is_restart_blocked(model_name):
                            logger.warning(
                                f"Watchdog: circuit breaker active for "
                                f"{model_name} — skipping hang recovery"
                            )
                        else:
                            await self._stop_server()
                            await self._swap_model(model_name, reason="hang recovery")


# ─── Singleton ───────────────────────────────────────────────
import os
import atexit

_manager: LocalModelManager | None = None


def _atexit_kill_llama_server():
    """Last-resort cleanup: kill llama-server when the Python process exits.

    This runs even on sys.exit() and unhandled exceptions (but NOT os._exit()).
    It uses synchronous subprocess calls because the event loop is gone by now.
    """
    global _manager
    if _manager is None:
        return

    proc = _manager.process
    if proc is not None and proc.poll() is None:
        logger.info("atexit: killing llama-server (PID %d)", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        except Exception as e:
            logger.warning("atexit: llama-server kill failed: %s", e)
            # Fallback: taskkill by name
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
        atexit.register(_atexit_kill_llama_server)
    return _manager


def get_runtime_state() -> ModelRuntimeState | None:
    """Return the runtime state of the currently loaded model, or None."""
    return _manager.runtime_state if _manager is not None else None
