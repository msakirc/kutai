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


class LocalModelManager:
    """
    Manages a single llama-server process.

    Key design choices:
    - asyncio.Lock ensures only one swap happens at a time
    - Swap requests are queued and deduplicated
    - Health checks run periodically
    - If the process dies, auto-restart with the same model
    """

    def __init__(self):
        self.current_model: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.port: int = LLAMA_SERVER_PORT
        self.api_base: str = f"http://127.0.0.1:{self.port}"

        self._swap_lock = asyncio.Lock()
        self._swap_queue: asyncio.Queue[ModelSwapRequest] = asyncio.Queue()
        self._last_request_time: float = 0.0
        self._started_at: float = 0.0
        self._total_swaps: int = 0
        self._thinking_enabled: bool = False  # tracks server-side thinking state

        self._scheduler = get_gpu_scheduler()

    # ── Public API ──────────────────────────────────────────────

    async def ensure_model(
        self,
        model_name: str,
        reason: str = "",
        enable_thinking: bool = False,
    ) -> bool:
        """
        Ensure the specified model is loaded and healthy.
        If already loaded with the same thinking state, returns immediately.
        If different model or thinking state differs, swaps (blocks until ready).
        """
        if self.current_model == model_name:
            if self._thinking_enabled == enable_thinking and await self._health_check():
                return True
            if self._thinking_enabled != enable_thinking:
                logger.info(
                    f"Thinking state change: {self._thinking_enabled} -> {enable_thinking}, "
                    f"restarting {model_name}"
                )
            else:
                # Loaded but unhealthy — restart
                logger.warning(f"Model {model_name} unhealthy, restarting")

        return await self._swap_model(model_name, reason, enable_thinking=enable_thinking)

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

    async def _swap_model(self, model_name: str, reason: str = "", enable_thinking: bool = False) -> bool:
        """
        Stop current model, start new one. Protected by lock.
        Returns True if the new model is healthy.
        """
        async with self._swap_lock:
            # Re-check after acquiring lock — another task may have already
            # loaded the model while we were waiting.
            if self.current_model == model_name and await self._health_check():
                logger.debug(f"Model {model_name} already healthy (resolved under lock)")
                return True

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

            from .model_registry import calculate_dynamic_context, calculate_gpu_layers

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

            if "gpu_layers" not in registry_overrides:
                new_layers = calculate_gpu_layers(
                    file_size_mb=model_info.file_size_mb,
                    n_layers=model_info.total_layers,
                    available_vram_mb=fresh_state.gpu.vram_free_mb,
                    context_length=model_info.context_length,
                )
                if new_layers != model_info.gpu_layers:
                    logger.info(
                        f"📐 Dynamic GPU layers: {model_info.gpu_layers} → {new_layers}"
                    )
                    model_info.gpu_layers = new_layers

            # Start new server
            success = await self._start_server(model_info, enable_thinking=enable_thinking)

            swap_duration = time.time() - swap_start
            self._total_swaps += 1

            if success:
                self.current_model = model_name
                self._thinking_enabled = enable_thinking
                self._started_at = time.time()
                registry.mark_loaded(model_name, self.api_base)
                logger.info(
                    f"✅ Model {model_name} loaded in {swap_duration:.1f}s "
                    f"(swap #{self._total_swaps})"
                )
                return True
            else:
                self.current_model = None
                registry.mark_unloaded(model_name)
                # Demote failed model so the router skips it on retry
                registry.demote_model(model_name, duration=300)
                logger.error(
                    f"❌ Failed to load {model_name} after {swap_duration:.1f}s "
                    f"(demoted for 5 min)"
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

    async def _start_server(self, model: ModelInfo, enable_thinking: bool = False) -> bool:
        """
        Launch llama-server process and wait for it to become healthy.
        """
        cmd = [
            str(LLAMA_SERVER_PATH),
            "--model", model.path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--n-gpu-layers", str(model.gpu_layers),
            "--ctx-size", str(model.context_length),
            "--flash-attn", "auto",
            "--metrics",
            # ── Performance flags ──
            "--mlock",              # lock model weights in RAM (prevent swap to disk)
            "--threads", str(self._get_inference_threads()),
            "--batch-size", "2048", # prompt processing batch size (higher = faster prefill)
            "--ubatch-size", "512", # micro-batch for generation
        ]

        # Control thinking via chat_template_kwargs (server-level flag,
        # not supported per-request by llama-server).
        if model.thinking_model:
            import json as _json
            cmd.extend([
                "--chat-template-kwargs",
                _json.dumps({"enable_thinking": enable_thinking}),
            ])

        # MoE models benefit from these flags
        if model.model_type == "moe":
            cmd.extend(["--override-kv", "tokenizer.ggml.eos_token_id=int:151645"])

        logger.info(f"Starting llama-server: {' '.join(cmd)}...")

        try:
            # Use CREATE_NO_WINDOW on Windows to hide the console
            import platform
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
        except FileNotFoundError:
            logger.error(
                f"llama-server not found at '{LLAMA_SERVER_PATH}'. "
                f"Set LLAMA_SERVER_PATH environment variable."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False

        # Wait for health endpoint — poll with backoff
        max_wait = model.load_time_seconds * 2  # double the estimated time
        max_wait = max(max_wait, 30)             # at least 30s
        max_wait = min(max_wait, 120)            # at most 120s

        healthy = await self._wait_for_healthy(timeout=max_wait)

        if not healthy:
            logger.error(f"llama-server failed to become healthy within {max_wait}s")
            await self._stop_server()
            return False

        return True

    async def _stop_server(self) -> None:
        """Gracefully stop the llama-server process."""
        if self.process is None:
            return

        old_model = self.current_model
        logger.info(f"Stopping llama-server (model: {old_model})...")

        try:
            self.process.terminate()
            # Wait up to 10s for graceful shutdown
            for _ in range(20):
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            else:
                # Force kill if still running
                logger.warning("llama-server didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error stopping llama-server: {e}")
            try:
                self.process.kill()
            except Exception:
                pass

        # Close stdout/stderr pipes to avoid ResourceWarning about unclosed files
        try:
            if self.process.stdout:
                self.process.stdout.close()
        except Exception:
            pass
        try:
            if self.process.stderr:
                self.process.stderr.close()
        except Exception:
            pass

        self.process = None
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
                    stderr = ""
                    try:
                        stderr = self.process.stderr.read().decode()[-500:]
                    except Exception:
                        pass
                    logger.error(
                        f"llama-server process exited with code "
                        f"{self.process.returncode}: {stderr}"
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
        except Exception as e:
            logger.debug(f"Failed to fetch llama-server metrics: {e}")

        return result

    async def _health_check(self) -> bool:
        """Quick health check — is the server responding?"""
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/health",
                    timeout=3.0,
                )
                return resp.status_code == 200
        except Exception:
            return False

    # ── Background Tasks ────────────────────────────────────────

    async def run_idle_unloader(
        self,
        check_interval: float = 60,
        max_idle_minutes: float = 10,
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
                logger.info(
                    f"Model {self.current_model} idle for "
                    f"{self.idle_seconds:.0f}s (>{max_idle}s), unloading"
                )
                async with self._swap_lock:
                    await self._stop_server()
                    self.current_model = None

    async def run_health_watchdog(self, check_interval: float = 30) -> None:
        """
        Background task: detect crashed server and restart.
        Run as: asyncio.create_task(manager.run_health_watchdog())
        """
        while True:
            await asyncio.sleep(check_interval)

            if not self.current_model:
                continue

            if self.process and self.process.poll() is not None:
                logger.error(
                    f"llama-server crashed! Restarting {self.current_model}..."
                )
                model_name = self.current_model
                # Close pipes before discarding the process reference
                try:
                    if self.process.stdout:
                        self.process.stdout.close()
                except Exception:
                    pass
                try:
                    if self.process.stderr:
                        self.process.stderr.close()
                except Exception:
                    pass
                self.process = None
                self.current_model = None
                await self._swap_model(model_name, reason="crash recovery")


# ─── Singleton ───────────────────────────────────────────────
import os

_manager: LocalModelManager | None = None


def get_local_manager() -> LocalModelManager:
    global _manager
    if _manager is None:
        _manager = LocalModelManager()
    return _manager
