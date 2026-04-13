"""Health registry — track capability availability."""
from __future__ import annotations

from datetime import datetime, timezone

from prometheus_client import Gauge

from nerd_herd.types import HealthStatus

_g_healthy = Gauge(
    "nerd_herd_capability_healthy",
    "Whether a capability is healthy (1) or degraded (0)",
    ["name"],
)


class HealthRegistry:
    name = "health"

    def __init__(self) -> None:
        self._capabilities: dict[str, bool] = {}
        self._boot_time = datetime.now(tz=timezone.utc).isoformat()

    def mark_degraded(self, capability: str) -> None:
        self._capabilities[capability] = False

    def mark_healthy(self, capability: str) -> None:
        self._capabilities[capability] = True

    def is_healthy(self, capability: str) -> bool:
        return self._capabilities.get(capability, True)

    def get_status(self) -> HealthStatus:
        return HealthStatus(
            boot_time=self._boot_time,
            capabilities=dict(self._capabilities),
        )

    def collect(self) -> dict[str, float | int | str]:
        return {k: int(v) for k, v in self._capabilities.items()}

    def prometheus_metrics(self) -> list:
        for name, healthy in self._capabilities.items():
            _g_healthy.labels(name=name).set(1 if healthy else 0)
        return [_g_healthy]
