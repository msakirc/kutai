"""Swap-budget policy — owned by Fatih Hoca.

nerd_herd stores the raw event stream (recent_swap_count, record_swap).
This module decides whether a proposed swap is allowed given that state.
"""
from __future__ import annotations

MAX_SWAPS_PER_WINDOW: int = 3
# Exempt ranks — never blocked regardless of budget.
HIGH_PRIORITY_FLOOR: int = 9


def can_swap(
    recent_count: int,
    *,
    local_only: bool = False,
    priority: int = 5,
    max_swaps: int = MAX_SWAPS_PER_WINDOW,
) -> bool:
    """Return True when a swap should be allowed.

    Exemptions:
      * `local_only=True` — caller wants to reuse a loaded local, no swap risk.
      * `priority >= HIGH_PRIORITY_FLOOR` — critical work bypasses the budget.

    Otherwise the per-window cap applies.
    """
    if local_only or priority >= HIGH_PRIORITY_FLOOR:
        return True
    return recent_count < max_swaps
