"""Configuration dataclasses for Kuleden Dönen Var."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class KuledenConfig:
    """Engine settings — configured once at startup."""
    circuit_breaker_threshold: int = 3
    circuit_breaker_window_seconds: float = 300.0
    circuit_breaker_cooldown_seconds: float = 600.0
    on_capacity_change: Callable[[CapacityEvent], None] | None = None


@dataclass
class ModelStatus:
    """Per-model capacity state."""
    model_id: str
    utilization_pct: float = 0.0
    has_capacity: bool = True
    daily_exhausted: bool = False
    rpm_remaining: int | None = None
    tpm_remaining: int | None = None
    rpd_remaining: int | None = None


@dataclass
class ProviderStatus:
    """Per-provider capacity state."""
    provider: str
    circuit_breaker_open: bool = False
    utilization_pct: float = 0.0
    rpm_remaining: int | None = None
    tpm_remaining: int | None = None
    rpd_remaining: int | None = None
    reset_in_seconds: float | None = None
    models: dict[str, ModelStatus] = field(default_factory=dict)


@dataclass
class CapacityEvent:
    """Fired on meaningful capacity state changes."""
    provider: str
    model_id: str | None
    event_type: str
    snapshot: ProviderStatus


@dataclass
class PreCallResult:
    """Result of pre_call check.

    When `allowed=False`, `reason` indicates the binding constraint and
    `wait_seconds` the calculated time until recovery:
        rpm        — sliding-window saturation (recovers as oldest entry ages)
        tpm        — token bucket too tight for this call's estimate
        rpd        — daily allotment exhausted (recovers at rpd_reset_at)
        provider_rpm/tpm/rpd — provider-aggregate equivalent
        circuit_breaker — too many recent failures, in cooldown
        daily      — explicit daily exhausted (also sets daily_exhausted=True)
    """
    allowed: bool
    wait_seconds: float
    daily_exhausted: bool
    reason: str = ""
    binding_provider: bool = False  # True if provider-aggregate was binding
