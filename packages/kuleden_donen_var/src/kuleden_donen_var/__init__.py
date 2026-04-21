"""Kuleden Dönen Var — cloud LLM provider capacity tracker."""
from .config import (
    CapacityEvent,
    KuledenConfig,
    ModelStatus,
    PreCallResult,
    ProviderStatus,
)
from .header_parser import RateLimitSnapshot, parse_rate_limit_headers
from .in_flight import InFlightHandle, InFlightTracker
from .kdv import KuledenDonenVar

__all__ = [
    "KuledenDonenVar",
    "KuledenConfig",
    "CapacityEvent",
    "ModelStatus",
    "PreCallResult",
    "ProviderStatus",
    "RateLimitSnapshot",
    "parse_rate_limit_headers",
    "InFlightHandle",
    "InFlightTracker",
    "begin_call",
    "end_call",
    "in_flight_count",
]

_in_flight_tracker: InFlightTracker | None = None


def _get_tracker() -> InFlightTracker:
    global _in_flight_tracker
    if _in_flight_tracker is None:
        # Production wiring: nerd_herd + state_getter are None for now.
        # A later task will wire a real state_getter once KDV exposes
        # its internal provider state as CloudProviderState objects.
        _in_flight_tracker = InFlightTracker()
    return _in_flight_tracker


def begin_call(provider: str, model: str, ttl_s: float | None = None) -> InFlightHandle:
    kwargs = {}
    if ttl_s is not None:
        kwargs["ttl_s"] = ttl_s
    return _get_tracker().begin_call(provider, model, **kwargs)


def end_call(handle: InFlightHandle) -> None:
    _get_tracker().end_call(handle)


def in_flight_count(provider: str, model: str) -> int:
    return _get_tracker().count(provider, model)
