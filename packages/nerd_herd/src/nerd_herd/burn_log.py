"""Rolling burn-rate log per (provider, model). Used by S7."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class BurnRate:
    tokens_per_min: float
    calls_per_min: float


class BurnLog:
    def __init__(self, window_secs: float = 300.0):
        self._window = window_secs
        self._entries: dict[tuple[str, str], deque] = {}
        self._lock = Lock()

    def record(self, *, provider: str, model: str, tokens: int, calls: int = 1, now: float | None = None):
        ts = now if now is not None else time.time()
        key = (provider, model)
        with self._lock:
            d = self._entries.setdefault(key, deque())
            d.append((ts, tokens, calls))
            self._evict(d, ts)

    def _evict(self, d: deque, now: float):
        cutoff = now - self._window
        while d and d[0][0] < cutoff:
            d.popleft()

    def rate(self, *, provider: str, model: str, now: float | None = None) -> BurnRate:
        ts = now if now is not None else time.time()
        key = (provider, model)
        with self._lock:
            d = self._entries.get(key)
            if not d:
                return BurnRate(0.0, 0.0)
            self._evict(d, ts)
            if not d:
                return BurnRate(0.0, 0.0)
            tot_tok = sum(e[1] for e in d)
            tot_calls = sum(e[2] for e in d)
        # rate = total observed in window / window length, normalized per-minute.
        # Cold-start with only 1 min of data on a 5-min window yields 1/5 rate —
        # conservative on bursty/short history, prevents runaway extrapolation.
        return BurnRate(
            tokens_per_min=(tot_tok * 60.0) / self._window,
            calls_per_min=(tot_calls * 60.0) / self._window,
        )


# Module-level singleton (process-scoped); main.py wires up before snapshot use
_GLOBAL_BURN_LOG: BurnLog | None = None


def get_burn_log() -> BurnLog:
    global _GLOBAL_BURN_LOG
    if _GLOBAL_BURN_LOG is None:
        _GLOBAL_BURN_LOG = BurnLog(window_secs=300.0)
    return _GLOBAL_BURN_LOG
