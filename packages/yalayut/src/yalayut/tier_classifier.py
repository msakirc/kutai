"""Tier classifier — combine source/owner trust caps with auto-check caps.

Tier integers: 0 (T0, best) .. 3 (T3, worst).

Spec intent (Tier classifier section): the owner can ELEVATE a source — a
trusted owner rescues a sketchy source. With 0=best, elevation = pick the
lower (better) integer => trust_cap = min(source_max, owner_max). Auto-checks
ALWAYS cap and can never be elevated past => final = max(trust_cap, *checks)
(worst wins for checks). The spec's literal `max(source,owner)` is corrected
to `min` here because every worked example requires it; see the plan's Task 7
ambiguity note.
"""
from __future__ import annotations


def classify(
    source_max: int,
    owner_max: int,
    check_maxes: dict[str, int],
) -> tuple[int, dict]:
    """Return (final_tier, audit_dict).

    audit_dict carries each contribution for the per-vetting-decision audit
    row stored on yalayut_index (source_max + check_max_json columns).
    """
    trust_cap = min(source_max, owner_max)          # owner elevates source
    check_max = max(check_maxes.values()) if check_maxes else 0
    final_tier = max(trust_cap, check_max)          # checks always cap
    final_tier = max(0, min(3, final_tier))
    audit = {
        "source_max": source_max,
        "owner_max": owner_max,
        "trust_cap": trust_cap,
        "check_max": check_max,
        "check_maxes": dict(check_maxes),
        "final_tier": final_tier,
    }
    return final_tier, audit
