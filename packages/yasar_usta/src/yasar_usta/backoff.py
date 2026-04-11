"""Escalating backoff with stability reset."""

from __future__ import annotations

import time


class BackoffTracker:
    """Tracks crash count and computes escalating backoff delays.

    Args:
        steps: List of delay seconds, indexed by crash count.
        reset_after: Reset crash count if process runs longer than this (seconds).
    """

    def __init__(self, steps: list[int] | None = None, reset_after: int = 600):
        self.steps = steps or [5, 15, 60, 300]
        self.reset_after = reset_after
        self.crash_count: int = 0
        self.total_crashes: int = 0
        self._start_time: float | None = None
        self.last_crash_time: float = 0

    def get_delay(self) -> int:
        """Return the current backoff delay in seconds."""
        idx = min(self.crash_count, len(self.steps) - 1)
        return self.steps[idx]

    def record_crash(self) -> None:
        """Record a crash event."""
        self.crash_count += 1
        self.total_crashes += 1
        self.last_crash_time = time.time()

    def mark_started(self) -> None:
        """Mark that the managed process has started."""
        self._start_time = time.time()

    def maybe_reset(self) -> None:
        """Reset crash count if process has been stable long enough."""
        if self._start_time and (time.time() - self._start_time) > self.reset_after:
            self.crash_count = 0
