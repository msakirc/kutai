"""Fixed-size ring buffer for rate pre-computation."""
from __future__ import annotations

import threading
import time


class RingBuffer:
    """Circular buffer of (timestamp, value) samples with rate computation.

    Thread-safe. Used to pre-compute rate() so Grafana doesn't need a TSDB.
    """

    def __init__(self, capacity: int = 60):
        self._capacity = capacity
        self._buf: list[tuple[float, float]] = []
        self._pos: int = 0
        self._full: bool = False
        self._lock = threading.Lock()

    def append(self, ts: float, value: float) -> None:
        with self._lock:
            if self._full:
                self._buf[self._pos] = (ts, value)
            elif len(self._buf) < self._capacity:
                self._buf.append((ts, value))
            else:
                self._buf[self._pos] = (ts, value)
                self._full = True
            self._pos = (self._pos + 1) % self._capacity

    def rate(self, window_seconds: float, now: float | None = None) -> float:
        """Compute rate of change over the given time window.

        Returns (newest_value - oldest_in_window_value) / elapsed.
        Returns 0.0 if fewer than 2 samples in window or counter reset detected.
        """
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            samples = self._sorted()

        if now is None:
            now = samples[-1][0]

        cutoff = now - window_seconds
        in_window = [(t, v) for t, v in samples if t >= cutoff]
        if len(in_window) < 2:
            return 0.0

        oldest_t, oldest_v = in_window[0]
        newest_t, newest_v = in_window[-1]
        elapsed = newest_t - oldest_t
        if elapsed <= 0:
            return 0.0

        delta = newest_v - oldest_v
        if delta < 0:
            return 0.0  # counter reset

        return delta / elapsed

    def latest(self) -> float | None:
        """Return the most recent value, or None if empty."""
        with self._lock:
            if not self._buf:
                return None
            samples = self._sorted()
            return samples[-1][1]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def _sorted(self) -> list[tuple[float, float]]:
        """Return samples sorted by timestamp. Caller must hold lock."""
        return sorted(self._buf, key=lambda s: s[0])
