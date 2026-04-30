"""Shared dataclasses for Nerd Herd collectors."""
from __future__ import annotations

from dataclasses import dataclass, field


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

    def pressure_for(
        self,
        model,
        *,
        task_difficulty: int = 5,
        est_per_call_tokens: int = 0,
        est_per_task_tokens: int = 0,
        est_iterations: int = 1,
        est_call_cost: float = 0.0,
        cap_needed: float = 5.0,
        consecutive_failures: int = 0,
    ):
        """Compute pressure breakdown via 10 signals + 4 modifiers.

        Returns a PressureBreakdown (use .scalar for the scalar value).
        """
        from nerd_herd.breakdown import PressureBreakdown
        from nerd_herd.burn_log import get_burn_log
        from nerd_herd.combine import combine_signals
        from nerd_herd.modifiers import (
            M1_capacity_amplifier, M2_perishability_dampener, M3_difficulty_weights,
        )
        from nerd_herd.signals.s1_remaining import s1_remaining
        from nerd_herd.signals.s2_call_burden import s2_call_burden
        from nerd_herd.signals.s3_task_burden import s3_task_burden
        from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens
        from nerd_herd.signals.s5_queue_calls import s5_queue_calls
        from nerd_herd.signals.s6_capable_supply import s6_capable_supply
        from nerd_herd.signals.s7_burn_rate import s7_burn_rate
        from nerd_herd.signals.s9_perishability import s9_perishability
        from nerd_herd.signals.s10_failure import s10_failure
        from nerd_herd.signals.s11_cost import s11_cost

        # Resolve matrix for this model (model-specific cell wins; provider is fallback)
        provider = getattr(model, "provider", "")
        prov = self.cloud.get(provider)
        model_state = (prov.models.get(getattr(model, "name", "")) if prov else None)
        matrix = (model_state.limits if model_state else
                  prov.limits if prov else
                  RateLimitMatrix())

        # Profile selection (free vs paid)
        profile = "time_bucketed" if getattr(model, "is_free", False) else "per_call"

        # Time-to-reset for the model's RPD cell (fall back to provider rpd)
        rpd_cell = matrix.rpd if matrix.rpd.limit else (
            prov.limits.rpd if prov else RateLimit()
        )
        import time as _time
        now = _time.time()
        reset_in = max(0.0, (rpd_cell.reset_at - now)) if rpd_cell.reset_at else 0.0

        # Total in-flight count for the model's pool
        in_flight_n = sum(
            1 for c in self.in_flight_calls
            if not c.is_local and c.provider == provider and c.model == getattr(model, "name", "")
        )

        # Compute signals
        sig = {
            "S1": s1_remaining(matrix, reset_in_secs=reset_in, in_flight=in_flight_n, profile=profile),
            "S2": s2_call_burden(matrix, est_per_call_tokens=est_per_call_tokens),
            "S3": s3_task_burden(matrix, est_per_task_tokens=est_per_task_tokens),
            "S4": s4_queue_tokens(matrix, queue=self.queue_profile or QueueProfile()),
            "S5": s5_queue_calls(matrix, queue=self.queue_profile or QueueProfile()),
            "S6": s6_capable_supply(model, queue=self.queue_profile or QueueProfile(),
                                    eligible_models=[], iter_avg=float(est_iterations or 8)),
            "S7": s7_burn_rate(matrix, provider=provider, model=getattr(model, "name", ""),
                               burn_log=get_burn_log(), now=now),
            "S9": s9_perishability(model, local=self.local, vram_avail_mb=self.vram_available_mb,
                                   matrix=matrix, task_difficulty=task_difficulty, now=now,
                                   in_flight_calls=self.in_flight_calls),
            "S10": s10_failure(consecutive_failures=consecutive_failures),
            "S11": s11_cost(est_call_cost=est_call_cost,
                            daily_cost_remaining=(matrix.cpd.remaining or 0.0)),
        }

        # Modifiers
        weights = M3_difficulty_weights(
            difficulty=task_difficulty,
            model_is_paid=not getattr(model, "is_free", False) and not getattr(model, "is_local", False),
        )

        # Apply M1 to negative-arm signals: amplify by limit-aware factor
        # Use the smallest-limit populated cell as the amplifier basis (worst-axis-wins)
        smallest_limit = min(
            (rl.limit for _, rl in matrix.populated_cells() if rl.limit), default=100,
        )
        m1 = M1_capacity_amplifier(limit=smallest_limit)
        for k in ("S1", "S2", "S3", "S4", "S5"):
            if sig[k] < 0:
                sig[k] *= m1

        # Apply M2 to positive S9 (over-qualification dampener, perishability-conditional)
        fit_excess = max(0.0, getattr(model, "cap_score", 5.0) - cap_needed)
        m2 = M2_perishability_dampener(fit_excess=fit_excess, s9_value=sig["S9"])
        if sig["S9"] > 0:
            sig["S9"] *= m2

        breakdown = combine_signals(signals=sig, weights=weights)
        breakdown.modifiers = {"M1": m1, "M2": m2, "M3_difficulty": task_difficulty,
                               "weights": dict(weights)}
        return breakdown

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
