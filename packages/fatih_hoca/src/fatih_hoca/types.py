"""Shared types for Fatih Hoca model selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Pick:
    model: object  # ModelInfo — typed loosely here, fully typed in __init__
    min_time_seconds: float
    estimated_load_seconds: float = 0.0  # 0 if already loaded, else expected swap time


@dataclass
class Failure:
    model: str              # litellm_name that failed
    reason: str             # "timeout", "rate_limit", "context_overflow",
                            # "quality_failure", "server_error", "loading"
    latency: float | None = None


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

    @property
    def remaining(self) -> int:
        self._prune()
        return max(0, self._max - len(self._timestamps))

    @property
    def exhausted(self) -> bool:
        self._prune()
        return len(self._timestamps) >= self._max
