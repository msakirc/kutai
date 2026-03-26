"""Anti-detection monitoring per domain.

Tracks rolling success rates for scraped Turkish market domains and
automatically applies a cooldown when the rate drops below threshold
(indicating the site may be blocking or rate-limiting requests).

Domains monitored by default:
  trendyol.com, hepsiburada.com, akakce.com, n11.com, ciceksepeti.com,
  sahibinden.com, gittigidiyor.com, pazarama.com

Behaviour:
- A sliding window of the last ``WINDOW_SIZE`` request outcomes is kept per
  domain using a ``deque``.
- If the rolling success rate falls below ``SUCCESS_RATE_THRESHOLD``, the
  domain enters cooldown for ``COOLDOWN_SECONDS``.
- After cooldown expires, the domain resumes at a reduced effective rate to
  allow gradual recovery (``GRADUAL_RESUME_FACTOR``).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.detection_monitor")

# ─── Constants ───────────────────────────────────────────────────────────────

WINDOW_SIZE: int = 50              # rolling window of outcomes per domain
SUCCESS_RATE_THRESHOLD: float = 0.6  # below this -> enter cooldown
COOLDOWN_SECONDS: float = 300.0    # 5 minutes hard cooldown
GRADUAL_RESUME_FACTOR: float = 0.5  # success-rate target after cooldown ends


# ─── Domain State ────────────────────────────────────────────────────────────

@dataclass
class _DomainStats:
    """Mutable per-domain tracking state."""

    outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    total_requests: int = 0
    successful_requests: int = 0
    last_request_time: float = 0.0
    cooldown_until: float = 0.0   # epoch seconds; 0 means not in cooldown


# ─── Monitor Class ───────────────────────────────────────────────────────────

class DetectionMonitor:
    """In-memory anti-detection monitor with per-domain rolling stats."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        success_rate_threshold: float = SUCCESS_RATE_THRESHOLD,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        gradual_resume_factor: float = GRADUAL_RESUME_FACTOR,
    ) -> None:
        self.window_size = window_size
        self.success_rate_threshold = success_rate_threshold
        self.cooldown_seconds = cooldown_seconds
        self.gradual_resume_factor = gradual_resume_factor
        self._domains: dict[str, _DomainStats] = {}

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_or_create(self, domain: str) -> _DomainStats:
        if domain not in self._domains:
            self._domains[domain] = _DomainStats(
                outcomes=deque(maxlen=self.window_size)
            )
        return self._domains[domain]

    def _rolling_rate(self, stats: _DomainStats) -> float:
        """Compute success rate over the current window."""
        if not stats.outcomes:
            return 1.0  # no data yet — assume healthy
        return sum(stats.outcomes) / len(stats.outcomes)

    def _maybe_trigger_cooldown(self, domain: str, stats: _DomainStats) -> None:
        """Enter cooldown if the rolling success rate has fallen too low."""
        if stats.cooldown_until > time.time():
            return  # already in cooldown

        rate = self._rolling_rate(stats)
        if len(stats.outcomes) >= self.window_size // 2 and rate < self.success_rate_threshold:
            stats.cooldown_until = time.time() + self.cooldown_seconds
            logger.warning(
                "Detection cooldown triggered for %s "
                "(rolling success rate %.1f%% < %.0f%% threshold). "
                "Cooldown for %.0fs.",
                domain,
                rate * 100,
                self.success_rate_threshold * 100,
                self.cooldown_seconds,
            )

    # ── Public API ───────────────────────────────────────────────────────────

    async def record_request(self, domain: str, success: bool) -> None:
        """Record the outcome of a request to *domain*.

        Parameters
        ----------
        domain:
            Bare domain name, e.g. ``"trendyol.com"``.
        success:
            ``True`` if the request succeeded, ``False`` otherwise.
        """
        stats = self._get_or_create(domain)
        stats.outcomes.append(success)
        stats.total_requests += 1
        if success:
            stats.successful_requests += 1
        stats.last_request_time = time.time()

        self._maybe_trigger_cooldown(domain, stats)

    async def is_domain_cooled_down(self, domain: str) -> bool:
        """Return ``True`` if *domain* is currently in a cooldown period.

        Once cooldown expires the method returns ``False`` and logs the
        transition so callers can resume gradually.
        """
        stats = self._get_or_create(domain)
        now = time.time()

        if stats.cooldown_until == 0.0:
            return False

        if now < stats.cooldown_until:
            return True

        # Cooldown has elapsed — clear it and log resumption
        remaining_until = stats.cooldown_until
        stats.cooldown_until = 0.0
        elapsed = now - (remaining_until - self.cooldown_seconds)
        logger.info(
            "Cooldown expired for %s after %.0fs. Resuming gradually "
            "(target success rate ≥ %.0f%%).",
            domain,
            elapsed,
            self.gradual_resume_factor * 100,
        )
        return False

    async def get_success_rate(self, domain: str) -> float:
        """Return the rolling success rate for *domain* (0.0–1.0).

        Returns ``1.0`` if fewer than one window's worth of requests have
        been recorded (optimistic default).
        """
        if domain not in self._domains:
            return 1.0
        return self._rolling_rate(self._domains[domain])

    async def get_detection_metrics(self) -> dict[str, dict]:
        """Return a summary dict of all tracked domains.

        Returns
        -------
        Dict mapping domain name to::

            {
                "total_requests":      int,
                "successful_requests": int,
                "rolling_success_rate": float,   # 0.0–1.0
                "window_size":         int,       # current window fill
                "in_cooldown":         bool,
                "cooldown_until":      float,     # epoch; 0 if not in cooldown
                "seconds_remaining":   float,     # 0 if not in cooldown
            }
        """
        now = time.time()
        metrics: dict[str, dict] = {}

        for domain, stats in self._domains.items():
            in_cooldown = now < stats.cooldown_until
            seconds_remaining = max(0.0, stats.cooldown_until - now) if in_cooldown else 0.0
            metrics[domain] = {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "rolling_success_rate": self._rolling_rate(stats),
                "window_size": len(stats.outcomes),
                "in_cooldown": in_cooldown,
                "cooldown_until": stats.cooldown_until,
                "seconds_remaining": seconds_remaining,
            }

        return metrics


# ─── Global Instance ─────────────────────────────────────────────────────────

_monitor = DetectionMonitor()


async def record_request(domain: str, success: bool) -> None:
    """Record the outcome of a request to *domain* on the global monitor."""
    await _monitor.record_request(domain, success)


async def is_domain_cooled_down(domain: str) -> bool:
    """Check whether *domain* is in cooldown on the global monitor."""
    return await _monitor.is_domain_cooled_down(domain)


async def get_success_rate(domain: str) -> float:
    """Get the rolling success rate for *domain* from the global monitor."""
    return await _monitor.get_success_rate(domain)


async def get_detection_metrics() -> dict[str, dict]:
    """Return detection metrics for all tracked domains."""
    return await _monitor.get_detection_metrics()
