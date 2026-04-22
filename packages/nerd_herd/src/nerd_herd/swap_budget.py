"""Swap event stream owned by nerd_herd.

Data layer only — stores the recent-swap timeline and exposes read
accessors. All policy decisions (is this swap allowed?) live in
fatih_hoca.swap_policy; the dispatcher writes events here after a
successful swap.
"""
from __future__ import annotations

import time


class SwapBudget:
    """Sliding-window counter of recent model swaps."""

    def __init__(self, window_seconds: int = 300) -> None:
        self._window = window_seconds
        self._timestamps: list[float] = []

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def record_swap(self) -> None:
        self._prune()
        self._timestamps.append(time.monotonic())

    def recent_count(self) -> int:
        self._prune()
        return len(self._timestamps)
