"""Shared dataclasses for Nerd Herd collectors."""
from __future__ import annotations

from dataclasses import dataclass, field

from nerd_herd.pool_pressure import PoolPressure, compute_pool_pressure


@dataclass
class GPUState:
    available: bool
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: int = 0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    timestamp: float = 0.0

    @property
    def vram_usage_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    @property
    def is_throttling(self) -> bool:
        return self.temperature_c > 85

    @property
    def is_busy(self) -> bool:
        return self.gpu_utilization_pct > 80


@dataclass
class SystemState:
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cpu_percent: float = 0.0
    gpu: GPUState = field(default_factory=lambda: GPUState(available=False))

    @property
    def can_load_model(self) -> bool:
        return self.ram_available_mb > 4096


@dataclass
class ExternalGPUUsage:
    detected: bool = False
    external_vram_mb: int = 0
    external_process_count: int = 0
    our_vram_mb: int = 0
    total_vram_mb: int = 0

    @property
    def external_vram_fraction(self) -> float:
        if self.total_vram_mb == 0:
            return 0.0
        return self.external_vram_mb / self.total_vram_mb


@dataclass
class HealthStatus:
    boot_time: str
    capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def degraded(self) -> list[str]:
        return [k for k, v in self.capabilities.items() if not v]


@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None        # absolute epoch seconds
    in_flight: int = 0                 # calls dispatched but not yet confirmed


@dataclass
class RateLimits:
    rpm: RateLimit = field(default_factory=RateLimit)
    tpm: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)


@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimits = field(default_factory=RateLimits)
    pool_pressure: PoolPressure | None = None


@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None   # epoch seconds
    limits: RateLimits = field(default_factory=RateLimits)
    models: dict[str, CloudModelState] = field(default_factory=dict)


@dataclass
class LocalModelState:
    model_name: str | None = None
    thinking_enabled: bool = False
    vision_enabled: bool = False
    measured_tps: float = 0.0
    context_length: int = 0
    is_swapping: bool = False
    kv_cache_ratio: float = 0.0
    idle_seconds: float = 0.0   # seconds since last completed local inference; 0 while a call is in-flight or before first inference


@dataclass
class QueueProfile:
    hard_tasks_count: int = 0
    total_ready_count: int = 0


@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)
    queue_profile: QueueProfile | None = None

    def pressure_for(self, model) -> float:
        if getattr(model, "is_local", False):
            return self._local_pressure()
        # Pool profile: free-tier cloud = time_bucketed; paid cloud = per_call.
        if getattr(model, "is_free", False) is True:
            kwargs = dict(
                depletion_threshold=0.30,
                depletion_max=-0.5,
                abundance_mode="time_decay",
                exhausted_neutral=True,
            )
        else:
            kwargs = dict(
                depletion_threshold=0.15,
                depletion_max=-1.0,
                abundance_mode="flat",
                exhausted_neutral=False,
            )
        provider = getattr(model, "provider", "")
        prov = self.cloud.get(provider)
        if prov is None:
            return 0.0
        model_id = getattr(model, "name", "")
        m = prov.models.get(model_id)
        if m is None:
            if prov.limits.rpd.limit:
                return compute_pool_pressure(
                    remaining=prov.limits.rpd.remaining,
                    limit=prov.limits.rpd.limit,
                    reset_at=prov.limits.rpd.reset_at,
                    in_flight_count=prov.limits.rpd.in_flight,
                    **kwargs,
                ).value
            return 0.0
        # Per-snapshot memoization keyed on rpd fields AND the pool-profile
        # kwargs (value depends on both).
        key = (
            m.limits.rpd.remaining,
            m.limits.rpd.limit,
            m.limits.rpd.reset_at,
            m.limits.rpd.in_flight,
            kwargs["depletion_threshold"],
            kwargs["depletion_max"],
            kwargs["abundance_mode"],
            kwargs["exhausted_neutral"],
        )
        cached = m.pool_pressure
        if cached is None or getattr(cached, "_key", None) != key:
            pp = compute_pool_pressure(
                remaining=m.limits.rpd.remaining,
                limit=m.limits.rpd.limit,
                reset_at=m.limits.rpd.reset_at,
                in_flight_count=m.limits.rpd.in_flight,
                **kwargs,
            )
            object.__setattr__(pp, "_key", key)
            m.pool_pressure = pp
            cached = pp
        return cached.value

    def _local_pressure(self) -> float:
        if self.local is None or self.local.model_name is None:
            return 0.0
        if self.local.is_swapping:
            return -0.5
        idle = self.local.idle_seconds or 0.0
        if idle <= 0:
            return -0.2
        return min(0.3, idle / 60.0 * 0.3)
