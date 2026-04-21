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

# Module-level singleton used by the dispatcher to bracket cloud calls.
_in_flight_tracker = InFlightTracker()


def begin_call(
    provider: str,
    model: str,
    ttl_s: float | None = None,
) -> InFlightHandle:
    kwargs = {}
    if ttl_s is not None:
        kwargs["ttl_s"] = ttl_s
    return _in_flight_tracker.begin_call(provider, model, **kwargs)


def end_call(handle: InFlightHandle) -> None:
    _in_flight_tracker.end_call(handle)


def in_flight_count(provider: str, model: str) -> int:
    return _in_flight_tracker.count(provider, model)
