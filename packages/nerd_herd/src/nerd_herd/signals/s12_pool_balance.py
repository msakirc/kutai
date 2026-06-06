"""S12 — fleet-relative pool balance (load-spreading / anti-monopoly).

Positive (abundance) signal, continuous and always-on. Pulls work toward the
FREE provider that has taken the fewest ABSOLUTE calls so far, so daily-reset
quota across the fleet gets used before it perishes and no single provider
becomes the lynchpin — the "everything collapses onto one provider, then 'no
models available' when it exhausts" failure (2026-06-04 diagnosis).

Why ABSOLUTE consumption, not frac
----------------------------------
frac-of-own-limit is scale-invariant: a giant tank (groq 14400/day) sits near
frac=1.0 even after absorbing every task, so frac-based signals (S1) cannot
tell it apart from a genuinely idle small tank (gemini 20/day). Absolute
consumed (`limit - remaining`) does: groq 182 vs gemini 0. S12 balances
absolute call counts across providers toward an equal share; S1's depletion
arm independently caps a small tank before it over-drains (water-fill emerges
from the two signals composing — no capacity term needed here).

Shape
-----
    fair      = total_consumed / n_free_providers_with_capacity
    deficit_p = (fair - consumed_p) / fair          # >0 ⇒ under fair share
    S12_p     = smoothstep(clamp(deficit_p, 0, 1))   # continuous 0→1

At cycle start (total_consumed=0) every deficit is 0 → S12=0 (no spurious
activation). As one provider accumulates picks, the untouched provider's
deficit climbs continuously toward 1 and the pull strengthens — no gate, no
threshold, no "until too late" cliff.

Paid / local models return 0: their quota does not perish at a reset, so the
use-it-or-lose-it balance does not apply (paid right-tool pull lives in S9).
"""
from __future__ import annotations

from typing import Any, Mapping

from nerd_herd.signals._curves import smoothstep as _smoothstep


def s12_pool_balance(
    model: Any,
    *,
    fleet_consumed: Mapping[str, float] | None,
) -> float:
    """Fleet-relative under-use pull for free-cloud `model`.

    fleet_consumed: {provider -> absolute calls consumed this cycle}, built by
        the ranking layer over FREE providers that still have capacity. None or
        a single-provider map ⇒ no balancing signal (nothing to spread across).
    """
    if not getattr(model, "is_free", False):
        return 0.0
    if not fleet_consumed or len(fleet_consumed) <= 1:
        return 0.0
    provider = getattr(model, "provider", "")
    if provider not in fleet_consumed:
        return 0.0
    total = float(sum(fleet_consumed.values()))
    if total <= 0.0:
        return 0.0
    fair = total / len(fleet_consumed)
    if fair <= 0.0:
        return 0.0
    deficit = (fair - float(fleet_consumed[provider])) / fair
    return _smoothstep(deficit)
