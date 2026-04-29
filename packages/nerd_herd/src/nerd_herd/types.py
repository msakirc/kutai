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
class RateLimitMatrix:
    # Request-axis cells
    rpm: RateLimit = field(default_factory=RateLimit)
    rph: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)
    rpw: RateLimit = field(default_factory=RateLimit)
    rpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (total)
    tpm: RateLimit = field(default_factory=RateLimit)
    tph: RateLimit = field(default_factory=RateLimit)
    tpd: RateLimit = field(default_factory=RateLimit)
    tpw: RateLimit = field(default_factory=RateLimit)
    tpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (split)
    itpm: RateLimit = field(default_factory=RateLimit)
    itpd: RateLimit = field(default_factory=RateLimit)
    otpm: RateLimit = field(default_factory=RateLimit)
    otpd: RateLimit = field(default_factory=RateLimit)

    # Cost-axis cells
    cpd: RateLimit = field(default_factory=RateLimit)
    cpmonth: RateLimit = field(default_factory=RateLimit)

    def populated_cells(self):
        for name, rl in vars(self).items():
            if isinstance(rl, RateLimit) and rl.limit is not None and rl.limit > 0:
                yield name, rl

    def token_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith(("tp", "itp", "otp")):
                yield name, rl

    def request_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith("rp"):
                yield name, rl

    def cost_cells(self):
        for name, rl in self.populated_cells():
            if name.startswith("cp"):
                yield name, rl


@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimitMatrix = field(default_factory=RateLimitMatrix)
    pool_pressure: PoolPressure | None = None


@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None   # epoch seconds
    limits: RateLimitMatrix = field(default_factory=RateLimitMatrix)
    models: dict[str, CloudModelState] = field(default_factory=dict)


@dataclass
class LocalModelState:
    model_name: str | None = None
    thinking_enabled: bool = False
    vision_enabled: bool = False
    measured_tps: float = 0.0
    pp_tps: float = 0.0  # prompt-processing tokens/sec (llamacpp:prompt_tokens_seconds)
    context_length: int = 0
    is_swapping: bool = False
    kv_cache_ratio: float = 0.0
    idle_seconds: float = 0.0   # seconds since last completed local inference; 0 while a call is in-flight or before first inference
    requests_processing: int = 0  # live in-flight call count from llama-server /metrics


@dataclass
class InFlightCall:
    call_id: str
    task_id: int | None
    category: str   # "main_work" | "overhead"
    model: str
    provider: str
    is_local: bool
    started_at: float


@dataclass
class QueueProfile:
    hard_tasks_count: int = 0
    total_ready_count: int = 0
    # New fields (2026-04-29 — pressure utilization equilibrium)
    by_difficulty: dict[int, int] = field(default_factory=dict)
    by_capability: dict[str, int] = field(default_factory=dict)
    projected_tokens: int = 0
    projected_calls: int = 0


@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)
    queue_profile: QueueProfile | None = None
    in_flight_calls: list[InFlightCall] = field(default_factory=list)

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
        model_id = getattr(model, "name", "")
        # Cloud in-flight count is authored by dispatcher via push_in_flight.
        # rpd.in_flight (pushed by KDV) is no longer consulted for pressure —
        # the InFlightCall list is the single source of truth for "running now".
        in_flight_n = sum(
            1 for c in self.in_flight_calls
            if not c.is_local and c.provider == provider and c.model == model_id
        )
        prov = self.cloud.get(provider)
        if prov is None:
            return 0.0
        m = prov.models.get(model_id)
        if m is None:
            if prov.limits.rpd.limit:
                return compute_pool_pressure(
                    remaining=prov.limits.rpd.remaining,
                    limit=prov.limits.rpd.limit,
                    reset_at=prov.limits.rpd.reset_at,
                    in_flight_count=in_flight_n,
                    **kwargs,
                ).value
            return 0.0
        # Per-snapshot memoization keyed on rpd fields + in-flight count +
        # pool-profile kwargs (value depends on all).
        key = (
            m.limits.rpd.remaining,
            m.limits.rpd.limit,
            m.limits.rpd.reset_at,
            in_flight_n,
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
                in_flight_count=in_flight_n,
                **kwargs,
            )
            object.__setattr__(pp, "_key", key)
            m.pool_pressure = pp
            cached = pp
        return cached.value

    def _local_pressure(self) -> float:
        # Signed scarcity in [-1, +1]: positive = abundant capacity,
        # negative = depleted. The admission gate is `pressure >=
        # threshold(urgency)` where threshold = 0.5 - urgency.
        #
        # Ordering by descending abundance:
        #   loaded + warm-idle  →  +0.5 .. +1.0   (best: ready, no swap cost)
        #   cold (no model)     →  +0.5           (good: capacity but pays load cost)
        #   loaded, busy now    →   0.0           (just received a request, no headroom yet)
        #   swapping            →  -0.5           (transient depletion)
        #   in-flight reserved  →  -1.0           (saturated; --parallel 1)
        #
        # Both cold and the lowest warm-idle tier sit at +0.5, which
        # clears the maximum possible threshold (0.5 - 0.0 = 0.5) for
        # any task. That fixes the cold-start deadlock that froze
        # mission 46 task 2939 (priority=4, threshold=0.05) when cold
        # used to return 0.0. Loaded+warm-idle still ranks ABOVE cold
        # at peak idle (+1.0 vs +0.5), so Fatih Hoca's selection
        # weights still correctly prefer no-swap continuation.
        if any(c.is_local for c in self.in_flight_calls):
            return -1.0

        if self.local is None or self.local.model_name is None:
            # Cold local: full slot, zero contention, but a swap cost
            # ahead. Above any positive threshold, below warm-idle.
            return 0.5

        if self.local.is_swapping:
            return -0.5

        # If model is loaded AND nothing is in-flight, there's NO
        # contention — slot is available now. idle_seconds telemetry
        # is unreliable (stays at 0.0 immediately after load even when
        # no work is happening, observed mission 46 2026-04-26: pressure
        # stuck at 0.0 with loaded local + empty in_flight, blocking
        # priority-4 admission for 5+ minutes). Trust in_flight as the
        # ground truth for "is the slot free", and use idle_seconds
        # only as a "warmer is better" tiebreaker on top.
        idle = self.local.idle_seconds or 0.0
        if idle <= 0:
            # Empty in_flight + zero idle telemetry. Either truly just-
            # loaded or a stale tracker. Either way, slot is free —
            # treat as cold-equivalent (+0.5) so admission can proceed.
            return 0.5
        # Linear scale from cold-equivalent (0.5) up to peak abundance
        # (1.0) over 60 seconds of measured idle. Loaded + warm beats
        # loaded + just-loaded.
        return min(1.0, 0.5 + (idle / 60.0) * 0.5)
