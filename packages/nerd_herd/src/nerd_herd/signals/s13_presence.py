"""S13 — user-presence pressure (local pool only).

Negative-only "yield the machine to the human" signal. Cloud models are
unaffected (zero desktop impact). Tuning constants are starting guesses —
revisit against real kutai.jsonl idle distributions (spec §7).
"""
from __future__ import annotations

from typing import Any

ACTIVE_IDLE_S = 30.0
PRESENT_IDLE_S = 300.0          # >= this → away → 0.0
PRESENT_PENALTY = -0.6          # strongest graded penalty while actively present
PRESENT_PENALTY_FLOOR = -0.3    # penalty near the active/away boundary
FULLSCREEN_VETO = -10.0         # sentinel — survives M3/M4 weights, pegs scalar -1.0


def s13_presence(model: Any, *, user_idle_s: float, foreground_fullscreen: bool) -> float:
    if not getattr(model, "is_local", False):
        return 0.0
    if user_idle_s >= PRESENT_IDLE_S:
        return 0.0
    if foreground_fullscreen:
        return FULLSCREEN_VETO
    if user_idle_s <= ACTIVE_IDLE_S:
        return PRESENT_PENALTY
    span = PRESENT_IDLE_S - ACTIVE_IDLE_S
    decayed = PRESENT_PENALTY * (1.0 - (user_idle_s - ACTIVE_IDLE_S) / span)
    return max(PRESENT_PENALTY, min(0.0, decayed))
