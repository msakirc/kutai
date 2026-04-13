"""GPU & system resource collector. Uses pynvml + psutil."""
from __future__ import annotations

import os
import time

from prometheus_client import Gauge

from yazbunu import get_logger

from nerd_herd.types import GPUState, SystemState, ExternalGPUUsage
from nerd_herd.platform import get_process_tree_pids

logger = get_logger("nerd_herd.gpu")

# Try to import pynvml — graceful degradation if unavailable
try:
    import pynvml as _pynvml
except ImportError:
    _pynvml = None

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

# Prometheus gauges
_g_vram_used = Gauge("nerd_herd_gpu_vram_used_mb", "GPU VRAM used in MB")
_g_vram_free = Gauge("nerd_herd_gpu_vram_free_mb", "GPU VRAM free in MB")
_g_vram_total = Gauge("nerd_herd_gpu_vram_total_mb", "GPU VRAM total in MB")
_g_util = Gauge("nerd_herd_gpu_utilization_pct", "GPU utilization percent")
_g_temp = Gauge("nerd_herd_gpu_temperature_c", "GPU temperature Celsius")
_g_power = Gauge("nerd_herd_gpu_power_draw_w", "GPU power draw watts")
_g_ext_vram = Gauge("nerd_herd_gpu_external_vram_mb", "External process VRAM MB")
_g_ext_procs = Gauge("nerd_herd_gpu_external_processes", "External GPU process count")
_g_ram = Gauge("nerd_herd_system_ram_available_mb", "System RAM available MB")
_g_cpu = Gauge("nerd_herd_system_cpu_percent", "System CPU usage percent")

_ALL_GAUGES = [
    _g_vram_used, _g_vram_free, _g_vram_total, _g_util, _g_temp, _g_power,
    _g_ext_vram, _g_ext_procs, _g_ram, _g_cpu,
]


class GPUCollector:
    """Polls GPU/system state. Caches for 2 seconds to avoid spam."""

    name = "gpu"

    def __init__(self, cache_ttl: float = 2.0) -> None:
        self._nvml_ok = False
        self._handle = None
        self._last_state: SystemState | None = None
        self._last_poll: float = 0.0
        self._cache_ttl = cache_ttl
        self._our_pids: set[int] | None = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        if _pynvml is None:
            logger.warning("pynvml not installed — GPU monitoring disabled")
            return
        try:
            _pynvml.nvmlInit()
            self._handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
            name = _pynvml.nvmlDeviceGetName(self._handle)
            logger.info("GPU monitor initialized", gpu=str(name))
        except Exception as e:
            logger.warning("GPU monitoring unavailable", error=str(e))

    def gpu_state(self) -> GPUState:
        return self._get_state().gpu

    def get_state(self) -> SystemState:
        """Get full system state (cached). Backward-compatible name."""
        return self._get_state()

    def system_state(self) -> SystemState:
        return self._get_state()

    def detect_external_gpu_usage(self) -> ExternalGPUUsage:
        if not self._nvml_ok:
            return ExternalGPUUsage()
        try:
            mem = _pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_mb = mem.total // (1024 * 1024)

            self._our_pids = None
            our_pids = get_process_tree_pids()

            try:
                procs = list(_pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle))
            except Exception:
                procs = []
            try:
                gfx = list(_pynvml.nvmlDeviceGetGraphicsRunningProcesses(self._handle))
                procs += gfx
            except Exception:
                pass

            external_vram = 0
            our_vram = 0
            external_count = 0
            seen: set[int] = set()

            for proc in procs:
                if proc.pid in seen:
                    continue
                seen.add(proc.pid)
                vram_mb = (proc.usedGpuMemory or 0) // (1024 * 1024)
                if proc.pid in our_pids:
                    our_vram += vram_mb
                else:
                    external_vram += vram_mb
                    external_count += 1

            detected = external_vram > 2048 or (
                total_mb > 0 and external_vram / total_mb > 0.30
            )
            return ExternalGPUUsage(
                detected=detected,
                external_vram_mb=external_vram,
                external_process_count=external_count,
                our_vram_mb=our_vram,
                total_vram_mb=total_mb,
            )
        except Exception as e:
            logger.debug("External GPU detection failed", error=str(e))
            return ExternalGPUUsage()

    def invalidate_cache(self) -> None:
        self._last_poll = 0.0

    def collect(self) -> dict[str, float | int | str]:
        s = self._get_state()
        ext = self.detect_external_gpu_usage()
        return {
            "gpu_available": int(s.gpu.available),
            "vram_used_mb": s.gpu.vram_used_mb,
            "vram_free_mb": s.gpu.vram_free_mb,
            "vram_total_mb": s.gpu.vram_total_mb,
            "gpu_utilization_pct": s.gpu.gpu_utilization_pct,
            "temperature_c": s.gpu.temperature_c,
            "power_draw_w": s.gpu.power_draw_w,
            "ram_available_mb": s.ram_available_mb,
            "cpu_percent": s.cpu_percent,
            "external_vram_mb": ext.external_vram_mb,
            "external_processes": ext.external_process_count,
        }

    def prometheus_metrics(self) -> list:
        s = self._get_state()
        ext = self.detect_external_gpu_usage()
        _g_vram_used.set(s.gpu.vram_used_mb)
        _g_vram_free.set(s.gpu.vram_free_mb)
        _g_vram_total.set(s.gpu.vram_total_mb)
        _g_util.set(s.gpu.gpu_utilization_pct)
        _g_temp.set(s.gpu.temperature_c)
        _g_power.set(s.gpu.power_draw_w)
        _g_ext_vram.set(ext.external_vram_mb)
        _g_ext_procs.set(ext.external_process_count)
        _g_ram.set(s.ram_available_mb)
        _g_cpu.set(s.cpu_percent)
        return list(_ALL_GAUGES)

    def _get_state(self) -> SystemState:
        now = time.time()
        if self._last_state and (now - self._last_poll) < self._cache_ttl:
            return self._last_state

        gpu = self._poll_gpu()
        ram_total, ram_avail = self._poll_ram()

        state = SystemState(
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
            cpu_percent=self._poll_cpu(),
            gpu=gpu,
        )
        self._last_state = state
        self._last_poll = now
        return state

    def _poll_gpu(self) -> GPUState:
        if not self._nvml_ok:
            return GPUState(available=False)
        try:
            mem = _pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = _pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            temp = _pynvml.nvmlDeviceGetTemperature(
                self._handle, _pynvml.NVML_TEMPERATURE_GPU
            )
            power = _pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
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
            logger.debug("GPU poll failed", error=str(e))
            return GPUState(available=False)

    def _poll_ram(self) -> tuple[int, int]:
        if _psutil is None:
            return 32768, 16384
        try:
            mem = _psutil.virtual_memory()
            return mem.total // (1024 * 1024), mem.available // (1024 * 1024)
        except Exception:
            return 32768, 16384

    def _poll_cpu(self) -> float:
        if _psutil is None:
            return 0.0
        try:
            return _psutil.cpu_percent(interval=0)
        except Exception:
            return 0.0
