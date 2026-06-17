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


# Per-minute axes: refill continuously (~60s) — a PACING constraint, not a
# conservation one. Queue-conservation signals (S4/S5) exclude these via
# cycle_*_cells(); per-task fit (S2/S3) and per-minute pacing (lane caps +
# in-flight) still use the full token_cells()/request_cells(). Note rpm≠rpmonth.
_PER_MINUTE_AXES = frozenset({"rpm", "tpm", "itpm", "otpm"})


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

    def cycle_token_cells(self):
        """Token cells on a reset CYCLE (daily/hourly/…) — excludes per-minute.
        A per-minute window refills every ~60s: it paces, it does not conserve
        (a deep queue drains over minutes, never exhausting it). Queue-
        conservation (S4) reads these; per-task fit (S2/S3) reads token_cells.
        """
        for name, rl in self.token_cells():
            if name not in _PER_MINUTE_AXES:
                yield name, rl

    def cycle_request_cells(self):
        """Request cells on a reset CYCLE — excludes per-minute (rpm). See
        cycle_token_cells. Queue-conservation (S5) reads these."""
        for name, rl in self.request_cells():
            if name not in _PER_MINUTE_AXES:
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
    # Rolling success rate over the last ~30 calls / 24h. 1.0 == perfect
    # or no-data. Source: kuleden_donen_var rolling outcome window via
    # nerd_herd_adapter. Read by S10_failure pressure signal — flaky
    # models contribute a negative scalar through OTHER_BUCKET, ranking
    # naturally down-weights them. Replaced the prior post-composite
    # multiplier in ranking.py with a real signal (2026-05-03).
    recent_success_rate: float = 1.0
    # Sample count in the rolling outcome window. S10 uses this to
    # gate its signal: below min_samples (default 5) the signal is 0
    # (no data, no opinion). Without this, freshly-revived models
    # default to 1.0 success_rate and rank as if perfectly reliable,
    # producing the revival cycle (mark_dead expires → empty history
    # → ranks top → fails → ❌). Now: empty history → S10=0 (neutral),
    # provider-prior carries the signal until samples accumulate.
    recent_samples_n: int = 0
    # Provider-level success-rate prior, aggregated across siblings on
    # the same provider (or openrouter sub-vendor key — see adapter
    # _prior_key). S10 reads this as a fallback when the model's own
    # samples_n is below MIN_SAMPLES: under-sampled models otherwise
    # get a 0 (neutral) signal, leaving freshly-revived ids and brand-
    # new ids rank-blind during their warm-up window. Provider prior
    # carries the signal in that gap. None means "not enough data
    # across the aggregated set either" — S10 still returns 0.
    provider_prior_rate: float | None = None
    # KDV's daily-exhausted state for the per-model rpd cell. True when
    # KDV.pre_call would refuse with daily_exhausted reason — i.e. the
    # provider has signaled the model has hit its daily quota and won't
    # serve until the reset window. Selector eligibility must reject
    # such models BEFORE ranking — otherwise ranking computes positive
    # composites against rpd_remaining that's stale (gemini doesn't
    # return rpd headers, so KDV's body-derived daily-exhausted state
    # is the only authoritative signal). Production 2026-05-02 root:
    # selector kept admitting tasks on gemini ids that KDV.pre_call
    # would refuse — the two views of capacity diverged.
    daily_exhausted: bool = False
    # Per-minute cooldown installed by an RFC 7231 Retry-After header (or
    # x-ratelimit-reset-* on a 429). True when the provider has explicitly
    # told KDV "do not call this model before T" and T is still in the
    # future. Selector eligibility must reject — otherwise after the 5s
    # freshness window expires, KDV's rpm_remaining property reverts to
    # sliding-window math, surfacing fake capacity, and the next call
    # eats a guaranteed 429. Source: kuleden_donen_var.RateLimitManager
    # .is_rpm_cooldown() — reads raw `_header_rpm_reset_at` field directly.
    rpm_cooldown: bool = False


@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None   # epoch seconds
    # KDV's circuit breaker state. True when the provider's per-process
    # CircuitBreaker is in cooldown (3+ failures in 300s window). The
    # selector's existing consecutive_failures-based gate is dead in
    # production (no writer outside tests); circuit_breaker_open is the
    # authoritative signal. Without this field plumbed, selector kept
    # picking gemini variants while every gemini call fast-failed at
    # KDV.pre_call with "circuit_breaker" — production 2026-05-02 saw
    # 5+ tasks burn through their candidate pools this way.
    circuit_breaker_open: bool = False
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
    # Projected token consumption for THIS task on this model. Set by
    # Beckman at reserve_task time from estimate_for(); zero for legacy
    # callers / tests. Pool-pressure consumers read this to back-pressure
    # admission against a saturated cloud BEFORE the actual call lands
    # at KDV.record_attempt — closing the admission→call gap where
    # several parallel admissions on the same cloud model all see fresh
    # tpm_remaining and overshoot the budget. Production 2026-05-02
    # observed: 5 ticks × 3s admit 5 tasks on gemini before any one's
    # caller.py reaches KDV.record_attempt. Counted in nerd_herd.types
    # pressure_for via subtract-from-effective on S1.
    est_tokens: int = 0


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
    # Recent swap count within nerd_herd's swap-budget window (default
    # 300s / 3 swaps). Populated by nerd_herd.snapshot() so ranking can
    # apply anti-flap stickiness — when swaps have just happened, dial
    # up loaded-model stickiness so the next pick can't oscillate. Read
    # by ranking._apply_loaded_stickiness; defaults to 0 in tests that
    # build snapshots manually.
    recent_swap_count: int = 0
    # Image-server residency (clair_obscur). Read by fatih_hoca's
    # image_select._eviction_cost. Written via NerdHerd.push_image_server_state()
    # / module-level record_image_server_state(), driven by clair_obscur on
    # start()/stop() (Plan 2).
    image_server_resident: bool = False
    image_server_vram_mb: int = 0
    # ── Desktop-awareness fields (2026-06-09 resource-signals) ──────
    # Populated by NerdHerd.snapshot(). Defaults describe an absent user
    # on an idle machine in "full" mode, so a manually-built snapshot
    # (tests, sims) reads as "no desktop pressure" — identical to today.
    load_mode: str = "full"
    user_idle_s: float = 1e9          # seconds since last user input; large = away
    foreground_fullscreen: bool = False
    ram_available_mb: int = 0
    ram_total_mb: int = 0
    external_gpu_fraction: float = 0.0  # cached from the 30s auto-detect loop
    # Process-level local-inference liveness: True while llama-server is
    # structurally unbootable (≥5 consecutive load failures). Overlaid in-process
    # by the selector from the nerd_herd client cache (the sidecar has no
    # load-outcome write path); the selector lays off ALL local while True.
    local_inference_down: bool = False

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
        fleet_consumed: dict | None = None,
        now: float | None = None,
        burn_log=None,
        eligible_models: list | None = None,
    ):
        """Compute pressure breakdown via signals + modifiers.

        ``fleet_consumed`` ({free-provider -> absolute calls consumed this
        cycle}) feeds S12 pool-balance. Built once per tick by the ranking
        layer (it knows which models are free); None ⇒ S12 contributes 0,
        which keeps pressure-only unit tests (no fleet view) unaffected.

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
        from nerd_herd.signals.s12_pool_balance import s12_pool_balance
        from nerd_herd.signals.s13_presence import s13_presence
        from nerd_herd.signals.s14_contention import s14_contention
        from nerd_herd.modifiers import M4_load_mode_weights

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
        now = now if now is not None else _time.time()
        reset_in = max(0.0, (rpd_cell.reset_at - now)) if rpd_cell.reset_at else 0.0

        # In-flight count + projected tokens. Filter by provider, NOT by
        # model id — the matrix cells often carry provider-AGGREGATE limits
        # (gemini free-tier rpm/tpm are per-API-key, shared across every
        # gemini model id on that key). Pre-fix this counted only same-id
        # in_flight, so 5 different gemini ids admitted in the same tick
        # each saw full rpm/tpm headroom and overshot the shared bucket.
        # Per-model-only limits (rpd / tpd) over-subtract slightly under
        # this rule — that's the safe direction (fewer admissions, no
        # overshoot). The 2026-05-02 14:44 cascade and 2026-05-03 ❌ flood
        # both traced to the same-id filter.
        in_flight_n = sum(
            1 for c in self.in_flight_calls
            if not c.is_local and c.provider == provider
        )
        in_flight_est_tokens = sum(
            int(getattr(c, "est_tokens", 0) or 0)
            for c in self.in_flight_calls
            if not c.is_local and c.provider == provider
        )

        # Build an "effective" matrix that subtracts in-flight reservations
        # from rpm/tpm/rpd remaining. Burden / remaining signals (S1, S2,
        # S3) read this view instead of the raw KDV-fed matrix. The
        # original matrix is preserved for non-budget signals.
        if in_flight_est_tokens > 0 or in_flight_n > 0:
            from copy import copy as _copy
            eff = RateLimitMatrix()
            for axis_name, rl in matrix.populated_cells():
                new_rl = _copy(rl)
                if rl.remaining is not None:
                    if axis_name in ("tpm", "cpm", "tpd", "cpd"):
                        new_rl.remaining = max(0, rl.remaining - in_flight_est_tokens)
                    elif axis_name in ("rpm", "rpd"):
                        new_rl.remaining = max(0, rl.remaining - in_flight_n)
                setattr(eff, axis_name, new_rl)
            matrix_effective = eff
        else:
            matrix_effective = matrix

        # Compute signals
        sig = {
            "S1": s1_remaining(matrix_effective, reset_in_secs=reset_in, in_flight=in_flight_n, profile=profile),
            "S2": s2_call_burden(matrix_effective, est_per_call_tokens=est_per_call_tokens),
            "S3": s3_task_burden(matrix_effective, est_per_task_tokens=est_per_task_tokens),
            "S4": s4_queue_tokens(matrix, queue=self.queue_profile or QueueProfile()),
            "S5": s5_queue_calls(matrix, queue=self.queue_profile or QueueProfile()),
            "S6": s6_capable_supply(model, queue=self.queue_profile or QueueProfile(),
                                    eligible_models=eligible_models or [],
                                    iter_avg=float(est_iterations or 8)),
            "S7": s7_burn_rate(matrix, provider=provider, model=getattr(model, "name", ""),
                               burn_log=(burn_log if burn_log is not None else get_burn_log()),
                               now=now),
            "S9": s9_perishability(model, local=self.local, vram_avail_mb=self.vram_available_mb,
                                   matrix=matrix, task_difficulty=task_difficulty, now=now,
                                   in_flight_calls=self.in_flight_calls),
            "S10": s10_failure(
                success_rate=(
                    model_state.recent_success_rate if model_state else 1.0
                ),
                samples_n=(
                    getattr(model_state, "recent_samples_n", 0) if model_state else 0
                ),
                provider_prior_rate=(
                    getattr(model_state, "provider_prior_rate", None)
                    if model_state else None
                ),
                consecutive_failures=consecutive_failures,
            ),
            "S11": s11_cost(est_call_cost=est_call_cost,
                            daily_cost_remaining=(matrix.cpd.remaining or 0.0)),
            "S12": s12_pool_balance(model, fleet_consumed=fleet_consumed),
            "S13": s13_presence(
                model,
                user_idle_s=self.user_idle_s,
                foreground_fullscreen=self.foreground_fullscreen,
            ),
            "S14": s14_contention(
                model,
                ram_available_mb=self.ram_available_mb,
                ram_total_mb=self.ram_total_mb,
                external_gpu_fraction=self.external_gpu_fraction,
            ),
        }

        # Modifiers
        weights = M3_difficulty_weights(
            difficulty=task_difficulty,
            model_is_paid=not getattr(model, "is_free", False) and not getattr(model, "is_local", False),
        )

        # M4: load-mode weights on the desktop signals only. Merge over M3
        # (which never defines S13/S14), so the multiply is a clean overlay.
        weights.update(M4_load_mode_weights(mode=self.load_mode))

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
