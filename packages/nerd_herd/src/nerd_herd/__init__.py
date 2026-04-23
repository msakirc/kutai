"""Nerd Herd — standalone observability package."""

from nerd_herd.nerd_herd import NerdHerd
from nerd_herd.client import NerdHerdClient, GPUStateProxy
from nerd_herd.types import (
    GPUState,
    SystemState,
    ExternalGPUUsage,
    HealthStatus,
    InFlightCall,
    RateLimit,
    RateLimits,
    CloudModelState,
    CloudProviderState,
    LocalModelState,
    SystemSnapshot,
    QueueProfile,
)
from nerd_herd.registry import CollectorRegistry, Collector
from nerd_herd.gpu import GPUCollector
from nerd_herd.load import LoadManager
from nerd_herd.health import HealthRegistry
from nerd_herd.inference import InferenceCollector
from nerd_herd.ring_buffer import RingBuffer
from nerd_herd.health_summary import health_summary  # noqa: F401
from nerd_herd.swap_budget import SwapBudget

__all__ = [
    "NerdHerd",
    "NerdHerdClient",
    "GPUStateProxy",
    "GPUState",
    "SystemState",
    "ExternalGPUUsage",
    "HealthStatus",
    "InFlightCall",
    "RateLimit",
    "RateLimits",
    "CloudModelState",
    "CloudProviderState",
    "LocalModelState",
    "SystemSnapshot",
    "CollectorRegistry",
    "Collector",
    "GPUCollector",
    "LoadManager",
    "HealthRegistry",
    "InferenceCollector",
    "RingBuffer",
    "health_summary",
    "SwapBudget",
    "record_swap",
    "push_queue_profile",
    "push_in_flight",
    "snapshot",
    "refresh_snapshot",
    "QueueProfile",
]

# Module-level singleton for module-level API (dispatcher uses this).
_singleton: NerdHerd | None = None


def _get_singleton() -> NerdHerd:
    global _singleton
    if _singleton is None:
        _singleton = NerdHerd()
    return _singleton


def record_swap(model_name: str = "") -> None:
    """Record that a model swap occurred. Called by dispatcher after ensure_local_model."""
    _get_singleton().record_swap(model_name)


def push_queue_profile(profile: QueueProfile) -> None:
    """Store the latest queue profile. Called by Beckman on queue-change events."""
    _get_singleton().push_queue_profile(profile)


def snapshot() -> SystemSnapshot:
    """Module-level snapshot access — returns the NerdHerdClient's cached snapshot.

    Used by Fatih Hoca (sync). Falls back to an empty SystemSnapshot if
    the orchestrator hasn't wired a NerdHerdClient yet.
    """
    from nerd_herd.client import get_default
    client = get_default()
    if client is None:
        return SystemSnapshot()
    return client.snapshot()


async def refresh_snapshot() -> SystemSnapshot:
    """Fetch a fresh SystemSnapshot from the sidecar and return it.

    Beckman awaits this per admission candidate so in-flight pushes from
    just-admitted tasks are reflected before the next pressure_for() call.
    """
    from nerd_herd.client import get_default
    client = get_default()
    if client is None:
        return SystemSnapshot()
    return await client.refresh_snapshot()


async def push_in_flight(calls: list[InFlightCall]) -> None:
    """Push the current in-flight call list to the NerdHerd sidecar.

    Called by the LLM dispatcher on every begin/end. Full-list replacement.
    No-op if the orchestrator hasn't wired a NerdHerdClient yet.
    """
    from nerd_herd.client import get_default
    client = get_default()
    if client is None:
        return
    await client.push_in_flight(calls)
