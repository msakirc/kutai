"""Per-provider circuit breaker — track failures, temporarily disable."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Track failures per provider and temporarily disable."""

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 300,
        cooldown_seconds: float = 600,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.failures: list[float] = []
        self.degraded_until: float = 0.0

    def record_failure(self) -> None:
        now = time.time()
        self.failures.append(now)
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        if len(self.failures) >= self.failure_threshold:
            self.degraded_until = now + self.cooldown_seconds
            logger.warning(
                "circuit breaker tripped, cooldown_seconds=%s",
                self.cooldown_seconds,
            )

    def record_success(self) -> None:
        self.failures.clear()
        self.degraded_until = 0.0

    @property
    def is_degraded(self) -> bool:
        if time.time() >= self.degraded_until:
            if self.degraded_until > 0:
                self.degraded_until = 0.0
                self.failures.clear()
            return False
        return True

    # ── Persistence ──────────────────────────────────────────────────────
    def snapshot_state(self) -> dict:
        return {
            "failures": list(self.failures),
            "degraded_until": self.degraded_until,
        }

    def restore_state(self, snap: dict) -> None:
        # Drop failure timestamps that fell outside the window during downtime.
        now = time.time()
        cutoff = now - self.window_seconds
        self.failures = [t for t in snap.get("failures", []) if t > cutoff]
        # If the cooldown clock already expired during downtime, leave it 0.
        deg = float(snap.get("degraded_until", 0.0))
        self.degraded_until = deg if deg > now else 0.0
