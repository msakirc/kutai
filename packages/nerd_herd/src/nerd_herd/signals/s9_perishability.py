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
# Local serial-only: llama-server runs --parallel 1 and the GPU hosts at
# most one model. ANY in-flight local task means a second local admission
# would either queue behind the first or trigger a swap. Both outcomes
# are bad — surface as a hard veto via -1.0 (admission threshold for any
# urgency clamps to >= -1.0, so this guarantees rejection).
LOCAL_BUSY_PENALTY = -1.0
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
    in_flight_calls: list | None = None,
) -> float:
    ts = now if now is not None else time.time()

    # ── Local branches ─────────────────────────────────────────────
    if getattr(model, "is_local", False):
        # Hard busy gate: any in-flight local call (admitted-not-yet-running
        # OR mid-call OR between iterations) blocks ALL local admissions
        # until release. Without this, the admission→prompt-processing gap
        # (~5-15s while agent runs RAG + chain-context + file-tree before
        # the first dispatcher.request lands) leaves requests_processing=0
        # at the metrics endpoint and a second admission squeaks through,
        # producing the swap storm or duplicate-llama-server pattern.
        # Production triage 2026-04-30: tasks #4464+#4457 admitted 15s
        # apart on the same loaded model.
        if in_flight_calls:
            for c in in_flight_calls:
                if getattr(c, "is_local", False):
                    return LOCAL_BUSY_PENALTY
        loaded_name = (local.model_name or "") if local else ""
        if getattr(model, "is_loaded", False) and loaded_name == getattr(model, "name", ""):
            if int(getattr(local, "requests_processing", 0) or 0) > 0:
                return LOCAL_BUSY_PENALTY
            idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
            return _clamp(min(1.0, idle / LOCAL_IDLE_SAT_SECS) * LOCAL_IDLE_MAX, 0, LOCAL_IDLE_MAX)
        # Cold local — same GPU veto as above. Even cold local can't be
        # admitted while another local call is mid-flight: claiming it
        # would either trigger an immediate swap (kicking out the running
        # model) or wait at the queue head.
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

    # Paid cloud — right-tool boost when budget remains AND task is hard.
    # Earlier this gated on frac>=FLUSH_THRESHOLD copied from the free
    # perishability arm, but paid pools don't reset, so "near reset"
    # semantics don't apply. The signal here is "right tool for the task,"
    # not "use-it-or-lose-it." Any remaining budget on a hard task should
    # boost paid cloud (S1 depletion arm handles low-frac warning).
    if task_difficulty < PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD:
        return 0.0
    for _, rl in matrix.request_cells():
        if rl.limit is None or rl.limit <= 0:
            continue
        effective = max(0, (rl.remaining or 0) - rl.in_flight)
        if effective > 0:
            return 1.0
    return 0.0
