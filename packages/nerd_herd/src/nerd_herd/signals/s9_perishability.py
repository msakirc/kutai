"""S9 — universal perishability signal.

Equilibrium core. Same signal across pool types; per-pool computation differs.
Returns "how strongly does NOT using this model right now waste capacity
we'll lose."
"""
from __future__ import annotations

import math
import time
from typing import Any

from nerd_herd.types import LocalModelState, RateLimitMatrix


LOCAL_IDLE_SAT_SECS = 60.0
LOCAL_IDLE_MAX = 0.5
LOCAL_BUSY_PENALTY = -0.10
COLD_LOCAL_VRAM_OK = 0.4
COLD_LOCAL_NO_VRAM = -0.5
TIME_DECAY_SCALE_SECS = 86400.0
PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD = 7
FLUSH_THRESHOLD = 0.7


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def s9_perishability(
    model: Any,
    *,
    local: LocalModelState | None,
    vram_avail_mb: int,
    matrix: RateLimitMatrix,
    task_difficulty: int,
    now: float | None = None,
) -> float:
    ts = now if now is not None else time.time()

    # ── Local branches ─────────────────────────────────────────────
    if getattr(model, "is_local", False):
        loaded_name = (local.model_name or "") if local else ""
        if getattr(model, "is_loaded", False) and loaded_name == getattr(model, "name", ""):
            if int(getattr(local, "requests_processing", 0) or 0) > 0:
                return LOCAL_BUSY_PENALTY
            idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
            return _clamp(min(1.0, idle / LOCAL_IDLE_SAT_SECS) * LOCAL_IDLE_MAX, 0, LOCAL_IDLE_MAX)
        # Cold local
        size_mb = int(getattr(model, "size_mb", 0) or 0)
        if size_mb <= 0 or vram_avail_mb >= size_mb:
            return COLD_LOCAL_VRAM_OK
        return COLD_LOCAL_NO_VRAM

    # ── Cloud branches ─────────────────────────────────────────────
    # Find the strongest perishability cell across populated request-axis cells
    strongest = 0.0
    if getattr(model, "is_free", False):
        for _, rl in matrix.request_cells():
            if rl.limit is None or rl.limit <= 0:
                continue
            effective = max(0, (rl.remaining or 0) - rl.in_flight)
            frac = effective / rl.limit
            if frac < FLUSH_THRESHOLD:
                continue
            if rl.reset_at is None or rl.reset_at <= ts:
                continue
            secs_to_reset = max(0.0, rl.reset_at - ts)
            time_weight = math.exp(-secs_to_reset / TIME_DECAY_SCALE_SECS)
            v = _clamp(frac * time_weight, 0.0, 1.0)
            if v > strongest:
                strongest = v
        return strongest

    # Paid cloud — right-tool perishability when budget flush + hard task
    if task_difficulty < PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD:
        return 0.0
    for _, rl in matrix.request_cells():
        if rl.limit is None or rl.limit <= 0:
            continue
        effective = max(0, (rl.remaining or 0) - rl.in_flight)
        frac = effective / rl.limit
        if frac >= FLUSH_THRESHOLD:
            return 1.0
    return 0.0
