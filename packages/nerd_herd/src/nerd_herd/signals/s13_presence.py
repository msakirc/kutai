"""S13 — user-presence pressure (local pool only).

Negative-only "yield the machine to the human" signal. Cloud models are
unaffected (zero desktop impact). Graded decay: present-active → −0.6,
decaying linearly to 0 as idle approaches the away threshold (300s).
FULLSCREEN_VETO (−10.0) overrides the gradient when a fullscreen window
is detected. Tuning constants are starting guesses — revisit against real
kutai.jsonl idle distributions (spec §7).
"""
from __future__ import annotations

from typing import Any

ACTIVE_IDLE_S = 30.0
PRESENT_IDLE_S = 300.0          # >= this → away → 0.0
PRESENT_PENALTY = -0.6          # strongest graded penalty while actively present
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
