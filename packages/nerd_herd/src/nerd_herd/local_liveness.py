"""Process-level local-inference liveness tracker.

The per-model circuit breaker (DaLLaMa swap) cannot express "the whole local
server is down" — it resets its counter whenever a *different* model is
attempted, so a server-wide outage (port collision, crashed/unbootable
llama-server, VRAM wall, missing GGUF, driver fault) slips past it and every
task keeps getting admitted against a dead server (live 2026-06-16: hours with
not a single task processed, the local load failing once per task forever).

This tracker counts consecutive local load failures across ANY model and trips
a process-level ``down`` flag. While down, the selector lays off ALL local at
eligibility so tasks route to cloud instead of re-attempting a dead server.

Recovery is half-open: once the cooldown elapses, ``is_down`` returns False so
a single probe task may try local — but the failure streak is preserved, so a
single further failure re-trips immediately (no need to burn the full
threshold again), while a success fully recovers. Net cost during a sustained
outage: ~one wasted local attempt per cooldown window, not one per task.
"""
from __future__ import annotations

import time
from typing import Callable


class LocalLivenessTracker:
    def __init__(
        self,
        threshold: int = 5,
        cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._threshold = max(1, int(threshold))
        self._cooldown = float(cooldown_s)
        self._clock = clock
        self._streak = 0
        self._down_until = 0.0

    def record_load(self, ok: bool) -> None:
        """Record the outcome of a local model load attempt."""
        if ok:
            self._streak = 0
            self._down_until = 0.0
            return
        self._streak += 1
        if self._streak >= self._threshold:
            self._down_until = self._clock() + self._cooldown

    def is_down(self) -> bool:
        """True while local inference should be laid off entirely.

        On cooldown expiry this flips to False (half-open) WITHOUT resetting the
        streak, so the next single failure re-trips immediately.
        """
        if self._down_until <= 0.0:
            return False
        if self._clock() >= self._down_until:
            self._down_until = 0.0  # half-open; keep streak for instant re-trip
            return False
        return True
