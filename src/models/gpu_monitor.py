# gpu_monitor.py
"""GPU monitor shim — delegates to nerd_herd.GPUCollector.

All import paths preserved for backward compatibility:
    from src.models.gpu_monitor import get_gpu_monitor, GPUState, ExternalGPUUsage
"""
from __future__ import annotations

from nerd_herd.types import GPUState, SystemState, ExternalGPUUsage  # noqa: F401

# Re-export the GPUCollector as GPUMonitor for backward compat
from nerd_herd.gpu import GPUCollector as GPUMonitor  # noqa: F401


_fallback: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    """Return the GPUCollector from the running NerdHerd instance."""
    try:
        from src.app.run import get_nerd_herd
        nh = get_nerd_herd()
        if nh is not None:
            return nh.registry.get("gpu")
    except Exception:
        pass
    # Fallback: create a standalone GPUCollector (e.g. during early startup)
    global _fallback
    if _fallback is None:
        _fallback = GPUMonitor()
    return _fallback
