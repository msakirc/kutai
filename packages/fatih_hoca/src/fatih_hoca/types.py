"""Shared types for Fatih Hoca model selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Pick:
    model: object  # ModelInfo — typed loosely here, fully typed in __init__
    min_time_seconds: float
    estimated_load_seconds: float = 0.0  # 0 if already loaded, else expected swap time
    # Composite score from ranking layer (post-utilization adjust). 0.0
    # default keeps tests/legacy callers working. Source: ScoredModel.score
    # at the moment select() picks the winner. The dispatcher persists
    # this into model_pick_log.picked_score so offline analysis can
    # correlate "what we picked" with "how confident we were".
    score: float = 0.0
    # Top-N candidate summary, one line. Format:
    #   "model1=8.4, model2=7.2, ..." up to 5 entries.
    # Persists into model_pick_log.snapshot_summary so offline analysis
    # can see what *else* was on the table when each pick fired (e.g.
    # "did the runner-up score nearly as high?"). Empty when select()
    # didn't compute alternatives (eligibility-only path).
    top_summary: str = ""


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
