# gpu_monitor.py
"""GPU monitor shim — standalone GPUCollector.

With NerdHerd running as a sidecar, the orchestrator uses a local
GPUCollector for GPU state queries. NerdHerd handles metrics/load.
"""
from __future__ import annotations

from nerd_herd.types import GPUState, SystemState, ExternalGPUUsage  # noqa: F401
from nerd_herd.gpu import GPUCollector as GPUMonitor  # noqa: F401

_instance: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    """Return a standalone GPUCollector."""
    global _instance
    if _instance is None:
        _instance = GPUMonitor()
    return _instance
