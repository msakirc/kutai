"""Modifiers — reshape signal values without computing their own pressure."""
from __future__ import annotations

import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── M1: Capacity amplifier ─────────────────────────────────────────
def M1_capacity_amplifier(*, limit: int) -> float:
    """Small pools amplify negative signals; large pools dampen them.

    factor = clip(2.0 - 0.5 * log10(limit), 0.5, 2.0)
      limit=10  → 1.5
      limit=100 → 1.0
      limit=1000→ 0.5
    """
    if limit <= 0:
        return 1.0
    return _clamp(2.0 - 0.5 * math.log10(limit), 0.5, 2.0)


# ── M2: Perishability-conditional fit-excess dampener ──────────────
def M2_perishability_dampener(*, fit_excess: float, s9_value: float) -> float:
    """Reduce positive signals when model overqualified, UNLESS perishability fires.

    s9_value > 0.5  → no damp (1.0)
    s9_value > 0.2  → partial damp (clip(1 - fit_excess * 0.25, 0.5, 1.0))
    else            → full damp   (clip(1 - fit_excess * 0.5,  0.0, 1.0))
    """
    excess = max(0.0, fit_excess)
    if s9_value > 0.5:
        return 1.0
    if s9_value > 0.2:
        return _clamp(1.0 - excess * 0.25, 0.5, 1.0)
    return _clamp(1.0 - excess * 0.5, 0.0, 1.0)


# ── M3: Difficulty re-weights ──────────────────────────────────────
def M3_difficulty_weights(*, difficulty: int, model_is_paid: bool = False) -> dict[str, float]:
    """Per-signal weights driven by task difficulty.

    Easy (d≤3):   down-weight burden, up-weight queue & abundance
    Hard (d≥7):   up-weight burden, down-weight queue, S9 inverts on paid
    Mid:          all 1.0
    """
    if difficulty <= 3:
        s9_w = 0.7 if model_is_paid else 1.5
        # S12 (pool balance) up-weighted on easy: spreading cheap/easy work
        # across free providers is exactly where load-balancing belongs.
        return {
            "S1": 1.0, "S2": 0.5, "S3": 0.5,
            "S4": 1.5, "S5": 1.5, "S6": 1.5,
            "S7": 1.0, "S9": s9_w,
            "S10": 1.0, "S11": 1.5, "S12": 1.5,
        }
    if difficulty >= 7:
        s9_w = 1.5 if model_is_paid else 0.7
        # S7 burn-conservation must not divert HARD work off a PAID right-tool
        # (you'd rather spend quota than fail the task). Free/local keep S7=1.0:
        # hard work shouldn't ride a weak free model regardless. (2026-06-05,
        # exposed once the de-blinded S7 went live in the sim.)
        s7_w = 0.0 if model_is_paid else 1.0
        # S12 down-weighted on hard: don't route hard work to a weak free
        # model just because it's under-used — capability must win.
        return {
            "S1": 1.0, "S2": 1.5, "S3": 1.5,
            "S4": 0.7, "S5": 0.7, "S6": 0.7,
            "S7": s7_w, "S9": s9_w,
            "S10": 1.0, "S11": 0.7, "S12": 0.5,
        }
    return {k: 1.0 for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S10", "S11", "S12")}


# ── M4: Load-mode weights on desktop signals (S13/S14) ─────────────
# Re-expresses "yük modu" as per-signal weights, replacing the dead
# VRAM-% cap. Minimal is handled upstream by selector eligibility
# (load_mode_minimal), so it doesn't need a veto weight here — passthrough.
_M4_BY_MODE: dict[str, float] = {
    "full": 0.0,      # ignore the user — desktop signals silenced
    "heavy": 1.5,     # cloud-bias strength: amplify desktop penalty
    "shared": 2.0,    # stronger cloud bias
    "minimal": 1.0,   # local already vetoed at eligibility; passthrough
}


def M4_load_mode_weights(*, mode: str) -> dict[str, float]:
    """Per-signal weights for S13/S14 driven by load mode. Multiplied into
    the M3 weight dict before the fold. Unknown mode → passthrough (1.0)."""
    factor = _M4_BY_MODE.get(mode, 1.0)
    return {"S13": factor, "S14": factor}
