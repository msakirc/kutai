"""Per-source circuit breaker.

Implements the classic three-state circuit breaker pattern to prevent
hammering sources that are consistently failing:

- **CLOSED** -- normal operation, requests flow through.
- **OPEN** -- source is failing, all requests are rejected immediately.
- **HALF_OPEN** -- cooldown expired, allow *one* test request.

Thresholds:
- 5 consecutive failures -> OPEN
- 60 s cooldown -> HALF_OPEN
- 1 success in HALF_OPEN -> CLOSED
"""

from __future__ import annotations

import time
from enum import Enum

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.circuit_breaker")

# ─── Constants ──────────────────────────────────────────────────────────────

FAILURE_THRESHOLD = 5       # consecutive failures before opening
COOLDOWN_SECONDS = 60       # seconds before half-open test
HALF_OPEN_SUCCESSES = 1     # successes needed to close again


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Per-domain circuit breaker with in-memory state."""

    def __init__(
        self,
        failure_threshold: int = FAILURE_THRESHOLD,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        # Per-domain state
        self._states: dict[str, CircuitState] = {}
        self._failure_counts: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}
        self._half_open_successes: dict[str, int] = {}

    def _ensure_domain(self, domain: str) -> None:
        """Initialise tracking for a domain on first encounter."""
        if domain not in self._states:
            self._states[domain] = CircuitState.CLOSED
            self._failure_counts[domain] = 0
            self._last_failure_time[domain] = 0.0
            self._half_open_successes[domain] = 0

    def is_allowed(self, domain: str) -> bool:
        """Check whether a request to *domain* is allowed.

        Returns ``True`` for CLOSED and HALF_OPEN, ``False`` for OPEN
        (unless cooldown has elapsed, in which case it transitions to
        HALF_OPEN).
        """
        self._ensure_domain(domain)
        state = self._states[domain]

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time[domain]
            if elapsed >= self.cooldown_seconds:
                self._states[domain] = CircuitState.HALF_OPEN
                self._half_open_successes[domain] = 0
                logger.info("Circuit for %s -> HALF_OPEN after %.0fs cooldown", domain, elapsed)
                return True
            return False

        # HALF_OPEN: allow one test request
        return True

    def on_success(self, domain: str) -> None:
        """Record a successful request to *domain*."""
        self._ensure_domain(domain)
        state = self._states[domain]

        if state == CircuitState.HALF_OPEN:
            self._half_open_successes[domain] += 1
            if self._half_open_successes[domain] >= HALF_OPEN_SUCCESSES:
                self._states[domain] = CircuitState.CLOSED
                self._failure_counts[domain] = 0
                logger.info("Circuit for %s -> CLOSED (recovered)", domain)
        else:
            # Reset failure counter on any success in CLOSED state
            self._failure_counts[domain] = 0

    def on_failure(self, domain: str) -> None:
        """Record a failed request to *domain*."""
        self._ensure_domain(domain)
        self._failure_counts[domain] += 1
        self._last_failure_time[domain] = time.time()

        if self._states[domain] == CircuitState.HALF_OPEN:
            # Failed during test request, go back to OPEN
            self._states[domain] = CircuitState.OPEN
            logger.warning("Circuit for %s -> OPEN (half-open test failed)", domain)
        elif self._failure_counts[domain] >= self.failure_threshold:
            self._states[domain] = CircuitState.OPEN
            logger.warning(
                "Circuit for %s -> OPEN after %d consecutive failures",
                domain, self._failure_counts[domain],
            )

    def get_state(self, domain: str) -> str:
        """Return the current circuit state name for *domain*."""
        self._ensure_domain(domain)
        # Re-evaluate OPEN -> HALF_OPEN transition on read
        if self._states[domain] == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time[domain]
            if elapsed >= self.cooldown_seconds:
                self._states[domain] = CircuitState.HALF_OPEN
                self._half_open_successes[domain] = 0
        return self._states[domain].value

    def get_all_states(self) -> dict[str, str]:
        """Return a dict of all tracked domains and their circuit states."""
        return {domain: self.get_state(domain) for domain in self._states}


# ─── Global Instance ────────────────────────────────────────────────────────

_breaker = CircuitBreaker()


async def check_circuit(domain: str) -> bool:
    """Check whether a request to *domain* is allowed by the circuit breaker.

    Returns ``True`` if the request can proceed, ``False`` if the circuit
    is open and the domain should be skipped.
    """
    return _breaker.is_allowed(domain)


async def record_success(domain: str) -> None:
    """Record a successful request to *domain*."""
    _breaker.on_success(domain)


async def record_failure(domain: str) -> None:
    """Record a failed request to *domain*."""
    _breaker.on_failure(domain)


def get_circuit_status() -> dict[str, str]:
    """Return all tracked circuit states.

    Returns
    -------
    Dict mapping domain name to state string (``"CLOSED"``, ``"OPEN"``,
    ``"HALF_OPEN"``).
    """
    return _breaker.get_all_states()
