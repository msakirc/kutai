"""Nerd Herd — standalone observability package."""

from nerd_herd.nerd_herd import NerdHerd
from nerd_herd.client import NerdHerdClient, GPUStateProxy
from nerd_herd.types import (
    GPUState,
    SystemState,
    ExternalGPUUsage,
    HealthStatus,
    RateLimit,
    RateLimits,
    CloudModelState,
    CloudProviderState,
    LocalModelState,
    SystemSnapshot,
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
]
