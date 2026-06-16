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
    RateLimitMatrix,
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
from nerd_herd.breakdown import PressureBreakdown

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
    "RateLimitMatrix",
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
    "record_image_server_state",
    "push_queue_profile",
    "push_in_flight",
    "snapshot",
    "refresh_snapshot",
    "QueueProfile",
    "PressureBreakdown",
]

# Module-level singleton for module-level API (dispatcher uses this).
_singleton: NerdHerd | None = None


def _get_singleton() -> NerdHerd:
    global _singleton
    if _singleton is None:
        _singleton = NerdHerd()
    return _singleton


def gpu_vram_free_mb(*, invalidate: bool = False) -> int:
    """Cheap free-VRAM read off the in-process singleton's GPU collector
    (2s-cached — does NOT rebuild a full SystemSnapshot). Pass
    ``invalidate=True`` to force a fresh poll first (e.g. immediately after
    freeing VRAM). Returns 0 when no GPU is available."""
    nh = _get_singleton()
    if invalidate:
        nh.invalidate_gpu_cache()
    gpu = nh.gpu_state()
    if not getattr(gpu, "available", False):
        return 0
    return int(getattr(gpu, "vram_free_mb", 0) or 0)


def record_swap(model_name: str = "") -> None:
    """Record that a model swap occurred. Called by dispatcher after ensure_local_model.

    MIRROR pattern: fans out to BOTH the in-process singleton AND the default
    NerdHerdClient's local swap budget. The text selector's hard gate reads
    ``client.recent_swap_count()`` and ranking stickiness reads
    ``snapshot.recent_swap_count`` off the client's snapshot — without the
    mirror, both read 0 forever in prod (NerdHerd runs as a sidecar; the
    sidecar has no swap write path).
    """
    _get_singleton().record_swap(model_name)
    try:
        from nerd_herd.client import get_default
        client = get_default()
        if client is not None:
            client.record_swap(model_name)
    except Exception:
        pass  # best-effort mirror — singleton write already landed


def record_local_load(ok: bool) -> None:
    """Record the outcome of a local model load attempt (called by the
    dispatcher after ensure_local_model).

    MIRROR pattern (see record_swap): fans out to BOTH the in-process singleton
    AND the default NerdHerdClient. The text selector reads
    ``client.is_local_inference_down()`` off the client cache — the sidecar has
    no load-outcome write path, so without the mirror the selector would read
    "up" forever and keep admitting tasks against a dead local server.
    """
    _get_singleton().record_local_load(ok)
    try:
        from nerd_herd.client import get_default
        client = get_default()
        if client is not None:
            client.record_local_load(ok)
    except Exception:
        pass  # best-effort mirror — singleton write already landed


def is_local_inference_down() -> bool:
    """True while local inference is structurally unavailable. Reads the default
    client cache when present (what the selector sees), else the singleton."""
    try:
        from nerd_herd.client import get_default
        client = get_default()
        if client is not None:
            return client.is_local_inference_down()
    except Exception:
        pass
    return _get_singleton().is_local_inference_down()


def record_image_server_state(*, resident: bool, vram_mb: int) -> None:
    """Record clair_obscur (local image-server) residency. Called by
    clair_obscur on start/stop. Read by fatih_hoca.image_select via the
    snapshot.

    MIRROR pattern: also stores on the default NerdHerdClient, which
    overlays the values onto sidecar-parsed snapshots (no transport
    delivers them to the sidecar, so parsed values were permanently
    False/0). fatih_hoca's image path reads the singleton directly via
    _effective_snapshot and stays correct either way; the mirror makes
    the client snapshot truthful for other consumers.
    """
    _get_singleton().push_image_server_state(resident=resident, vram_mb=vram_mb)
    try:
        from nerd_herd.client import get_default
        client = get_default()
        if client is not None:
            client.set_local_image_server_state(resident=resident, vram_mb=vram_mb)
    except Exception:
        pass  # best-effort mirror — singleton write already landed


def push_queue_profile(profile: QueueProfile) -> None:
    """Store the latest queue profile. Called by Beckman on queue-change events.

    MIRROR pattern: also stores the profile on the default NerdHerdClient,
    which overlays it onto sidecar-parsed snapshots. Without the mirror the
    text selector (which reads snapshots through the client) saw
    queue_profile=None forever — the sidecar has no queue_profile transport.
    """
    _get_singleton().push_queue_profile(profile)
    try:
        from nerd_herd.client import get_default
        client = get_default()
        if client is not None:
            client.set_local_queue_profile(profile)
    except Exception:
        pass  # best-effort mirror — singleton write already landed


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
