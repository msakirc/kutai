"""S14 — machine-contention pressure (local pool only).

Negative-only. Two inputs, both already collected elsewhere:
  - external-GPU fraction (cached from the 30s auto-detect loop)
  - RAM pressure (psutil via SystemState)
Can fire while the user is away (e.g. an overnight render owns the GPU).
Tuning constants are starting guesses (spec §7).
"""
from __future__ import annotations

from typing import Any

EXTERNAL_GPU_VETO_FRACTION = 0.30   # another process owns >=30% VRAM → veto local
EXTERNAL_GPU_VETO = -10.0           # sentinel
RAM_USED_FLOOR = 0.80               # below this used-fraction → no pressure
RAM_USED_CAP = 0.95                 # at/above → full -1.0


def s14_contention(
    model: Any,
    *,
    ram_available_mb: int,
    ram_total_mb: int,
    external_gpu_fraction: float,
) -> float:
    if not getattr(model, "is_local", False):
        return 0.0
    if external_gpu_fraction >= EXTERNAL_GPU_VETO_FRACTION:
        return EXTERNAL_GPU_VETO
    if ram_total_mb <= 0:
        return 0.0
    used_frac = 1.0 - (ram_available_mb / ram_total_mb)
    if used_frac < RAM_USED_FLOOR:
        return 0.0
    if used_frac >= RAM_USED_CAP:
        return -1.0
    intensity = (used_frac - RAM_USED_FLOOR) / (RAM_USED_CAP - RAM_USED_FLOOR)
    return max(-1.0, min(0.0, -intensity))
