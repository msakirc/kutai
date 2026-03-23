# gpu_monitor.py
"""
GPU & system resource monitoring.
Uses pynvml (NVIDIA) for GPU stats, psutil for RAM/CPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from src.infra.logging_config import get_logger

logger = get_logger("models.gpu_monitor")


@dataclass
class GPUState:
    available: bool
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: int = 0      # 0-100
    temperature_c: int = 0
    power_draw_w: float = 0.0
    timestamp: float = 0.0

    @property
    def vram_usage_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    @property
    def is_thermal_throttling(self) -> bool:
        return self.temperature_c > 85

    @property
    def is_busy(self) -> bool:
        """GPU is busy if utilization > 80% (model is actively inferencing)."""
        return self.gpu_utilization_pct > 80


@dataclass
class SystemState:
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cpu_percent: float = 0.0
    gpu: GPUState = None

    def __post_init__(self):
        if self.gpu is None:
            self.gpu = GPUState(available=False)

    @property
    def can_load_model(self) -> bool:
        """Conservative check: enough RAM headroom for model loading."""
        return self.ram_available_mb > 4096  # keep 4GB free for OS


@dataclass
class ExternalGPUUsage:
    """Result of external GPU process detection."""
    detected: bool = False
    external_vram_mb: int = 0
    external_process_count: int = 0
    our_vram_mb: int = 0
    total_vram_mb: int = 0

    @property
    def external_vram_fraction(self) -> float:
        if self.total_vram_mb == 0:
            return 0.0
        return self.external_vram_mb / self.total_vram_mb


class GPUMonitor:
    """Polls GPU/system state. Caches for 2 seconds to avoid spam."""

    def __init__(self):
        self._nvml_initialized = False
        self._handle = None
        self._last_state: SystemState | None = None
        self._last_poll: float = 0.0
        self._cache_ttl: float = 2.0
        self._our_pids: set[int] | None = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialized = True
            name = pynvml.nvmlDeviceGetName(self._handle)
            logger.info(f"GPU monitor initialized: {name}")
        except Exception as e:
            logger.warning(f"GPU monitoring unavailable: {e}")
            self._nvml_initialized = False

    def get_state(self) -> SystemState:
        """Get current system state (cached for 2s)."""
        now = time.time()
        if self._last_state and (now - self._last_poll) < self._cache_ttl:
            return self._last_state

        gpu = self._poll_gpu()
        ram_total, ram_available = self._poll_ram()

        state = SystemState(
            ram_total_mb=ram_total,
            ram_available_mb=ram_available,
            cpu_percent=self._poll_cpu(),
            gpu=gpu,
        )
        self._last_state = state
        self._last_poll = now
        return state

    def invalidate_cache(self) -> None:
        """Force next get_state() to poll fresh values."""
        self._last_poll = 0.0

    def _get_our_pids(self) -> set[int]:
        """Get PIDs belonging to our process tree (self + children)."""
        if self._our_pids is not None:
            return self._our_pids
        pids = {os.getpid()}
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            for child in proc.children(recursive=True):
                pids.add(child.pid)
            # Also include parent if it's python (e.g. launched via wrapper)
            parent = proc.parent()
            if parent and "python" in (parent.name() or "").lower():
                pids.add(parent.pid)
        except Exception:
            pass
        self._our_pids = pids
        return pids

    def detect_external_gpu_usage(self) -> ExternalGPUUsage:
        """Detect non-orchestrator processes using the GPU via pynvml."""
        if not self._nvml_initialized:
            return ExternalGPUUsage()
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_mb = mem.total // (1024 * 1024)

            # Refresh our PID set periodically (children may spawn)
            self._our_pids = None
            our_pids = self._get_our_pids()

            # Get all processes using the GPU
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            except Exception:
                procs = []

            # Also check graphics processes (games, desktop compositors)
            try:
                gfx_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self._handle)
                procs = list(procs) + list(gfx_procs)
            except Exception:
                pass

            external_vram = 0
            our_vram = 0
            external_count = 0
            seen_pids: set[int] = set()

            for proc in procs:
                pid = proc.pid
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                vram_mb = (proc.usedGpuMemory or 0) // (1024 * 1024)
                if pid in our_pids:
                    our_vram += vram_mb
                else:
                    external_vram += vram_mb
                    external_count += 1

            detected = external_vram > 2048 or (total_mb > 0 and external_vram / total_mb > 0.30)

            return ExternalGPUUsage(
                detected=detected,
                external_vram_mb=external_vram,
                external_process_count=external_count,
                our_vram_mb=our_vram,
                total_vram_mb=total_mb,
            )
        except Exception as e:
            logger.debug(f"External GPU detection failed: {e}")
            return ExternalGPUUsage()

    def _poll_gpu(self) -> GPUState:
        if not self._nvml_initialized:
            return GPUState(available=False)
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

            return GPUState(
                available=True,
                vram_total_mb=mem.total // (1024 * 1024),
                vram_used_mb=mem.used // (1024 * 1024),
                vram_free_mb=mem.free // (1024 * 1024),
                gpu_utilization_pct=util.gpu,
                temperature_c=temp,
                power_draw_w=power,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.debug(f"GPU poll failed: {e}")
            return GPUState(available=False)

    def _poll_ram(self) -> tuple[int, int]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total // (1024 * 1024), mem.available // (1024 * 1024)
        except ImportError:
            return 32768, 16384  # fallback estimate

    def _poll_cpu(self) -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=0)
        except ImportError:
            return 0.0


# ─── Singleton ───────────────────────────────────────────────
_monitor: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    global _monitor
    if _monitor is None:
        _monitor = GPUMonitor()
    return _monitor
