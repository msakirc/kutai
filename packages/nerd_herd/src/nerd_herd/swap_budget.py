"""Swap-budget state owned by nerd_herd. Hoca reads; dispatcher writes."""
from __future__ import annotations

import time


class SwapBudget:
    """Rate-limits model swaps. Max N swaps per window.

    Exemptions: local_only requests, priority >= 9.
    """

    def __init__(self, max_swaps: int = 3, window_seconds: int = 300) -> None:
        self._max = max_swaps
        self._window = window_seconds
        self._timestamps: list[float] = []

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def can_swap(self, local_only: bool, priority: int) -> bool:
        if local_only or priority >= 9:
            return True
        self._prune()
        return len(self._timestamps) < self._max

    def record_swap(self) -> None:
        self._timestamps.append(time.monotonic())

    def recent_count(self) -> int:
        self._prune()
        return len(self._timestamps)

    @property
    def remaining(self) -> int:
        self._prune()
        return max(0, self._max - len(self._timestamps))

    @property
    def exhausted(self) -> bool:
        self._prune()
        return len(self._timestamps) >= self._max
