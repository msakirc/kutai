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
    "configure_in_flight_push",
]

_in_flight_tracker: InFlightTracker | None = None


def _get_tracker() -> InFlightTracker:
    global _in_flight_tracker
    if _in_flight_tracker is None:
        # Default construction: no push target. configure_in_flight_push()
        # below rewires the singleton once an adapter converting KDV's
        # internal ProviderStatus -> nerd_herd.CloudProviderState is ready.
        _in_flight_tracker = InFlightTracker()
    return _in_flight_tracker


def configure_in_flight_push(nerd_herd, state_getter) -> None:
    """Wire the in-flight tracker to nerd_herd for push-on-boundary.

    Call once at app startup once an adapter exists that yields a
    CloudProviderState from KDV's internal provider records. Until
    configured, begin_call/end_call track handles in-process only and
    never reach nerd_herd — meaning `pressure_for()` sees in_flight=0.
    """
    global _in_flight_tracker
    _in_flight_tracker = InFlightTracker(
        nerd_herd=nerd_herd,
        state_getter=state_getter,
    )


def begin_call(provider: str, model: str, ttl_s: float | None = None) -> InFlightHandle:
    kwargs = {}
    if ttl_s is not None:
        kwargs["ttl_s"] = ttl_s
    return _get_tracker().begin_call(provider, model, **kwargs)


def end_call(handle: InFlightHandle) -> None:
    _get_tracker().end_call(handle)


def in_flight_count(provider: str, model: str) -> int:
    return _get_tracker().count(provider, model)
