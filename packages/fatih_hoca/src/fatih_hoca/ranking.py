# ranking.py
"""
Fatih Hoca — Ranking Module (Layers 2 & 3)

Takes a list of already-eligible ModelInfo objects (pre-filtered by selector.py
which handles Layer 1 eligibility) and returns them scored and sorted.

Scoring pipeline:
  Layer 2: Five-dimension composite scoring (capability, cost, availability,
            performance, speed), each 0-100, weighted by difficulty.
  Layer 3: Ranking adjustments — thinking fitness, specialty alignment,
            swap stickiness.
  Post:     Time gate rescue, sibling rebalancing (S7), failure adaptation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fatih_hoca.capabilities import TaskRequirements, score_model_for_task
from fatih_hoca.capability_curve import cap_needed_for_difficulty, CAP_NEEDED_BY_DIFFICULTY
from fatih_hoca.estimates import estimate_for
from fatih_hoca.grading import grading_perf_score
from fatih_hoca.pools import (
    Pool, classify_pool,
    UTILIZATION_K,
)
from fatih_hoca.requirements import get_quota_planner

if TYPE_CHECKING:
    pass

from fatih_hoca.registry import ModelInfo
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure
from nerd_herd.types import SystemSnapshot

logger = logging.getLogger("fatih_hoca.ranking")

# Weight for grading-derived score in the perf_score blend (Phase 2c).
# blended = GRADING_WEIGHT * grading + (1 - GRADING_WEIGHT) * tps_perf
GRADING_WEIGHT: float = 0.6


# ─── ScoredModel ─────────────────────────────────────────────────────────────

@dataclass
class ScoredModel:
    model: ModelInfo
    score: float
    capability_score: float = 0.0
    composite_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    pool: str = ""       # "local" | "time_bucketed" | "per_call"
    urgency: float = 0.0  # [0, 1]

    @property
    def litellm_name(self) -> str:
        return self.model.litellm_name


# ─── Failure Adaptation Helpers ──────────────────────────────────────────────

def _build_failure_index(failures: list[Failure]) -> dict[str, list[Failure]]:
    """Index failures by litellm_name for O(1) lookups."""
    idx: dict[str, list[Failure]] = {}
    for f in failures:
        idx.setdefault(f.model, []).append(f)
    return idx


def _provider_from_litellm(litellm_name: str) -> str:
    """Extract provider prefix from 'provider/model' litellm_name."""
    if "/" in litellm_name:
        return litellm_name.split("/")[0]
    return ""


def _failure_penalty(
    model: ModelInfo,
    failure_idx: dict[str, list[Failure]],
    all_failures: list[Failure],
    snapshot: SystemSnapshot,
) -> tuple[float, bool, list[str]]:
    """
    Compute a multiplier (0-1) and exclusion flag for failure adaptation.

    Rate-limit policy (narrowed 2026-04-17):
    - Per-model 429 → penalize ONLY that litellm_name (0.3×).
    - Provider-wide penalty (0.3× on siblings) applies ONLY when
      snapshot.cloud[provider].consecutive_failures >= 3. This prevents
      a single-model quota hit from poisoning healthy siblings.

    Returns (multiplier, exclude, reasons).
    """
    model_failures = failure_idx.get(model.litellm_name, [])
    reasons: list[str] = []
    multiplier = 1.0
    exclude = False

    for f in model_failures:
        if f.reason == "loading":
            exclude = True
            reasons.append("fail_loading")
        elif f.reason == "timeout":
            multiplier = min(multiplier, 0.2)
            reasons.append("fail_timeout")
        elif f.reason == "quality_failure":
            multiplier = min(multiplier, 0.5)
            reasons.append("fail_quality")
        elif f.reason == "server_error":
            multiplier = min(multiplier, 0.3)
            reasons.append("fail_server_error")
        elif f.reason == "rate_limit":
            multiplier = min(multiplier, 0.3)
            reasons.append("fail_rate_limit")

    # Provider-wide rate-limit penalty: ONLY when the circuit breaker trips
    # (consecutive_failures >= 3). Otherwise a single model's 429 does not
    # poison its siblings.
    model_provider = model.provider
    if model_provider and getattr(model, "location", None) == "cloud":
        prov_state = snapshot.cloud.get(model_provider) if snapshot else None
        consec = getattr(prov_state, "consecutive_failures", 0) if prov_state else 0
        if consec >= 3:
            for failure in all_failures:
                if failure.reason != "rate_limit":
                    continue
                if failure.model == model.litellm_name:
                    continue  # already counted as direct
                fp = _provider_from_litellm(failure.model)
                if fp == model_provider:
                    multiplier = min(multiplier, 0.3)
                    reasons.append(f"fail_provider_rate_limit({model_provider},consec={consec})")
                    break

    return multiplier, exclude, reasons


# ─── Utilization Layer Helper ────────────────────────────────────────────────

def _apply_utilization_layer(
    scored: list[ScoredModel],
    snapshot: SystemSnapshot,
    task_difficulty: int,
    reqs: "ModelRequirements | None" = None,
) -> None:
    """Apply Phase 2d unified utilization equation.

    For each ScoredModel:
        breakdown = snapshot.pressure_for(model, task_difficulty=d, ...)
        composite *= 1 + UTILIZATION_K * breakdown.scalar

    The fit dampener is absorbed inside pressure_for (M2 modifier).
    Queue state is read from snapshot.queue_profile (pushed by Beckman).
    Mutates each .score/.composite_score/.pool/.urgency in place.
    Does NOT re-sort — caller is responsible.
    """
    if not scored:
        return
    # Build estimate_for proxy: reqs already has agent_type; context is optional.
    task_proxy = reqs  # estimate_for reads task.agent_type and task.context
    for sm in scored:
        pool = classify_pool(sm.model)
        sm.pool = pool.value

        # btable empty-dict cold-start; populated by Beckman rollup cron (Task 26)
        estimates = estimate_for(task_proxy, btable={},
                                 model_is_thinking=getattr(sm.model, "is_thinking", False))
        prov_state = snapshot.cloud.get(getattr(sm.model, "provider", ""))
        breakdown = snapshot.pressure_for(
            sm.model,
            task_difficulty=task_difficulty,
            est_per_call_tokens=estimates.per_call_tokens,
            est_per_task_tokens=estimates.total_tokens,
            est_iterations=estimates.iterations,
            est_call_cost=getattr(sm.model, "estimated_cost",
                                  lambda *_: 0.0)(estimates.in_tokens, estimates.out_tokens),
            cap_needed=CAP_NEEDED_BY_DIFFICULTY.get(task_difficulty, 5.0),
            consecutive_failures=(
                getattr(prov_state, "consecutive_failures", 0) if prov_state else 0
            ),
        )
        scalar = breakdown.scalar
        # Reuse `urgency` column for pressure scalar — telemetry schema continuity.
        # Note: per_call positive abundance is suppressed at the source in
        # s1_remaining.py PROFILE_PARAMS (abundance_max=0.0). Paid abundance
        # for d>=7 comes from S9 right-tool-perishability — no ranking-layer
        # gate needed.
        sm.urgency = scalar

        if scalar == 0.0:
            continue
        adjustment = 1.0 + UTILIZATION_K * scalar
        if adjustment == 1.0:
            continue
        sm.score *= adjustment
        sm.composite_score = sm.score
        sm.reasons.append(
            f"util={pool.value}:s={scalar:+.2f}→{adjustment:.3f}"
        )


# ─── Core Ranking Function ───────────────────────────────────────────────────

def rank_candidates(
    candidates: list[ModelInfo],
    reqs: ModelRequirements,
    snapshot: SystemSnapshot,
    failures: list[Failure],
    remaining_budget: float = 0.0,
) -> list[ScoredModel]:
    """
    Score and rank an already-filtered list of ModelInfo objects.

    Parameters
    ----------
    candidates : list[ModelInfo]
        Models that have already passed Layer 1 eligibility filtering.
    reqs : ModelRequirements
        Task requirements driving the scoring.
    snapshot : SystemSnapshot
        Current system state (local model, cloud utilization).
    failures : list[Failure]
        Recent failure records used for penalty adaptation.

    Returns
    -------
    list[ScoredModel]
        Scored and sorted models, best first. May be empty.
    """
    effective_task = reqs.effective_task
    needed_ctx = reqs.effective_context_needed
    min_score = reqs.effective_min_score

    # Pre-build failure index for O(1) per-model lookup
    failure_idx = _build_failure_index(failures)

    scored: list[ScoredModel] = []
    time_gated: list[ScoredModel] = []

    # Runtime state for the loaded local model
    local_state = snapshot.local

    for model in candidates:
        reasons: list[str] = []

        # ── Failure adaptation: exclude or penalize failed models ──
        fail_mult, fail_exclude, fail_reasons = _failure_penalty(model, failure_idx, failures, snapshot)
        if fail_exclude:
            logger.debug(
                "model excluded by failure adaptation: model=%s reasons=%s",
                model.name, fail_reasons,
            )
            continue
        reasons.extend(fail_reasons)

        # ── Time gate: flag models too slow for the 300s budget ──
        is_time_gated = False
        if model.is_local and reqs.estimated_output_tokens > 0:
            # Use measured tps from snapshot when this is the loaded model
            if (
                model.is_loaded
                and local_state.model_name == model.name
                and local_state.measured_tps > 0
            ):
                gate_tps = local_state.measured_tps
            else:
                gate_tps = model.tokens_per_second

            if gate_tps > 0:
                gate_secs = reqs.estimated_output_tokens / gate_tps
                TIME_BUDGET = 300.0
                if gate_secs > TIME_BUDGET:
                    is_time_gated = True
                    reasons.append(
                        f"time_gated({gate_tps:.1f}tps"
                        f"×{reqs.estimated_output_tokens}tok"
                        f"={gate_secs:.0f}s>{TIME_BUDGET:.0f}s)"
                    )

        # ════════════════════════════════════════════════════════
        # LAYER 2: Five-Dimension Scoring
        # ════════════════════════════════════════════════════════

        # ── 1. Capability Fit (0–100) ──
        cap_score_raw = score_model_for_task(
            model_capabilities=model.capabilities,
            model_operational=model.operational_dict(),
            requirements=TaskRequirements(
                task_name=effective_task or reqs.primary_capability,
                min_context=needed_ctx,
                needs_function_calling=reqs.needs_function_calling,
                needs_json_mode=reqs.needs_json_mode,
                needs_vision=reqs.needs_vision,
                needs_thinking=reqs.needs_thinking,
                prefer_local=reqs.prefer_local or reqs.local_only,
                prefer_fast=reqs.prefer_speed,
            ),
        )

        if cap_score_raw < 0:
            logger.debug(
                "model rejected by capability score: model=%s cap_score_raw=%.2f",
                model.name, cap_score_raw,
            )
            continue
        # No min_score gate — ranking sorts by score, best model wins.
        # Filtering low scorers risks zero candidates (worse than a
        # mediocre pick) and triggers unnecessary swaps.

        # cap_score_raw is a 0–10 weighted mean of per-dim 0–10 scores.
        # Scale cleanly to 0–100 without a min() ceiling so that models scoring
        # above 10 raw (possible when specialty weights exceed 1.0) preserve signal.
        cap_score = cap_score_raw * 10.0
        reasons.append(f"cap={cap_score_raw:.2f}")
        if effective_task:
            reasons.append(f"task={effective_task}")

        # ── 2. Cost Efficiency (0–100) ──
        if model.is_local:
            cost_score = 95 if model.is_loaded else 90
            if not model.is_loaded:
                reasons.append("needs_swap")
            reasons.append("local")
            # Skip load mode penalty — snapshot doesn't carry vram_budget_fraction;
            # the selector already filtered models that exceed the VRAM budget.
        elif model.is_free:
            cost_score = 85
            reasons.append("free_cloud")
        else:
            est_cost = model.estimated_cost(
                reqs.estimated_input_tokens, reqs.estimated_output_tokens
            )
            if est_cost <= 0.001:
                cost_score = 75
            elif est_cost <= 0.01:
                cost_score = 50
            elif est_cost <= 0.05:
                cost_score = 30
            else:
                cost_score = 10
            reasons.append(f"cost=${est_cost:.4f}")

            # Quota planner penalty for paid models below threshold
            planner = get_quota_planner()
            if reqs.difficulty < planner.expensive_threshold:
                penalty = (planner.expensive_threshold - reqs.difficulty) * 8
                cost_score = max(0, cost_score - penalty)
                reasons.append(f"quota_pen=-{penalty}")

        # ── 3. Availability (0–100) ──
        if model.is_local:
            if model.is_loaded:
                avail_score = 100
                reasons.append("loaded")
            else:
                swap_time = model.load_time_seconds
                avail_score = 75 if swap_time < 10 else (55 if swap_time < 30 else 35)
                # Budget-aware penalty: loading that eats >50% of remaining
                # budget gets a steep discount so faster-loading models win.
                if remaining_budget > 0 and swap_time > remaining_budget * 0.5:
                    ratio = swap_time / remaining_budget
                    # ratio 0.5→penalty 0, ratio 1.0→penalty 30
                    budget_penalty = min(30, int((ratio - 0.5) * 60))
                    avail_score = max(5, avail_score - budget_penalty)
                    reasons.append(f"load_budget({swap_time:.0f}s/{remaining_budget:.0f}s)")
                reasons.append(f"swap_{swap_time:.0f}s")
        else:
            # Use snapshot.cloud for utilization data
            prov_state = snapshot.cloud.get(model.provider)
            model_state = (
                prov_state.models.get(model.litellm_name) if prov_state else None
            )
            model_util = model_state.utilization_pct if model_state else 0.0
            provider_util = prov_state.utilization_pct if prov_state else 0.0

            # Daily exhaustion check — no graduated fallback
            daily_exhausted = getattr(model_state, "daily_exhausted", False) if model_state else False
            if daily_exhausted:
                avail_score = 0
                reasons.append("daily_exhausted")
            else:
                # Smooth curve: 0% util → 95, 100% util → 5
                effective_util = max(model_util, provider_util)
                avail_score = max(5, int(95 - effective_util * 0.90))
                if effective_util >= 50:
                    reasons.append(f"util={effective_util:.0f}%")

        # ── 4. Performance History (0–100) ──
        # Blends tps-derived (local speed signal) with grading-derived
        # (success_rate from model_stats). Falls back cleanly when either side
        # is missing. Phase 2c: replaces the flat perf=50 fallback for cloud.
        if model.is_local and model.is_loaded and \
           local_state.model_name == model.name and local_state.measured_tps > 0:
            tps = local_state.measured_tps
            tps_perf = min(95.0, 50.0 + (tps - 10) * 1.5) if tps >= 10 else max(20.0, 20.0 + tps * 3.0)
        elif model.is_local and model.tokens_per_second > 0:
            tps = model.tokens_per_second
            tps_perf = min(90.0, 45.0 + (tps - 10) * 1.2) if tps >= 10 else max(15.0, 15.0 + tps * 3.0)
        else:
            tps_perf = 50.0

        grading = grading_perf_score(model.name)
        if grading is not None:
            perf_score = GRADING_WEIGHT * grading + (1.0 - GRADING_WEIGHT) * tps_perf
            reasons.append(f"perf={perf_score:.0f}(g={grading:.0f},tps={tps_perf:.0f})")
        else:
            perf_score = tps_perf
            reasons.append(f"perf={perf_score:.0f}")

        # ── 5. Speed (0–100) ──
        if model.is_local:
            # Use measured tps from snapshot when this is the loaded model
            if (
                model.is_loaded
                and local_state.model_name == model.name
                and local_state.measured_tps > 0
            ):
                tps = local_state.measured_tps
            else:
                tps = model.tokens_per_second

            active = getattr(model, "active_params_b", 0) or model.total_params_b

            if tps >= 50:
                speed_score = 100
            elif tps >= 20:
                speed_score = 80
            elif tps >= 10:
                speed_score = 60
            elif tps >= 5:
                speed_score = 40
            elif tps >= 2:
                speed_score = 20
            elif tps > 0:
                speed_score = 10
            else:
                # No measured tps — estimate from model size
                if active < 5:
                    speed_score = 75
                elif active < 10:
                    speed_score = 55
                elif active < 20:
                    speed_score = 35
                else:
                    speed_score = 15

            # Output-length penalty for slow local models
            est_out = reqs.estimated_output_tokens
            effective_tps = tps if tps > 0 else (
                25 if active < 5
                else 12 if active < 15
                else 5 if active < 30
                else 3
            )
            est_generation_secs = est_out / effective_tps if effective_tps > 0 else 0
            if est_generation_secs > 300:
                speed_score = max(0, speed_score - 50)
                reasons.append(f"very_slow({est_generation_secs:.0f}s)")
            elif est_generation_secs > 120:
                speed_score = max(0, speed_score - 30)
                reasons.append(f"slow({est_generation_secs:.0f}s)")
            elif est_generation_secs > 60:
                speed_score = max(0, speed_score - 15)
                reasons.append(f"moderate({est_generation_secs:.0f}s)")

            # Amplify speed score when prefer_speed is set
            if reqs.prefer_speed and tps > 0:
                tps_boost = min(1.0, tps / 50.0)
                speed_score = speed_score * (0.5 + tps_boost * 0.5)
        else:
            speed_map = {
                "groq": 95,
                "cerebras": 95,
                "sambanova": 80,
                "gemini": 70,
                "openai": 60,
                "anthropic": 50,
            }
            speed_score = speed_map.get(model.provider, 50)

        # ── Composite Weighting ──
        d = reqs.difficulty
        if d <= 3:
            weights = {"capability": 20, "cost": 35, "availability": 20, "performance": 10, "speed": 15}
        elif d <= 5:
            weights = {"capability": 30, "cost": 20, "availability": 20, "performance": 15, "speed": 15}
        elif d <= 7:
            weights = {"capability": 35, "cost": 15, "availability": 15, "performance": 15, "speed": 20}
        else:
            weights = {"capability": 45, "cost": 5, "availability": 10, "performance": 20, "speed": 20}

        # Modifier adjustments
        if reqs.prefer_speed:
            weights["speed"] += 15
            weights["cost"] -= 10
        if reqs.prefer_quality:
            weights["capability"] += 10
            weights["cost"] -= 10
        if reqs.prefer_local:
            weights["cost"] += 10
            weights["speed"] -= 5
            weights["availability"] -= 5
        if reqs.local_only:
            weights["availability"] += 10
            weights["cost"] -= 10

        total_weight = sum(weights.values())
        composite = (
            cap_score * weights["capability"]
            + cost_score * weights["cost"]
            + avail_score * weights["availability"]
            + perf_score * weights["performance"]
            + speed_score * weights["speed"]
        ) / total_weight

        # ════════════════════════════════════════════════════════
        # LAYER 3: Ranking Adjustments
        # ════════════════════════════════════════════════════════

        # Group A: Thinking fitness — reward thinking models when CoT requested
        if reqs.needs_thinking and model.thinking_model:
            composite *= 1.20
            reasons.append("thinking_bonus")

        # Group B: Specialty alignment (observability only, no composite effect).
        # The 1.15× multiplier was removed in Phase 2a: after Phase 1 blended AA
        # benchmark signal into ModelInfo.capabilities, a coding-specialty model
        # on a coder task already gets its code_* dims heavily boosted via the
        # profile dot-product; multiplying again caused specialty models to beat
        # general models with objectively stronger benchmark coder scores.
        # Hard filtering for coding specialty on non-coder tasks still lives
        # in selector._check_eligibility, which is unchanged.
        if model.specialty and effective_task:
            _specialty_tasks = {
                "coding": {"coder", "implementer", "fixer", "test_generator"},
                "reasoning": {"planner", "architect", "analyst"},
                "vision": {"visual_reviewer"},
            }
            matched = _specialty_tasks.get(model.specialty, set())
            if effective_task in matched:
                reasons.append(f"specialty={model.specialty}")

        # Group C: Swap stickiness — tiebreaker between close-cap locals,
        # not a capability override (2026-04-20).
        #
        # Original 1.4× magnitude was strong enough to crush cloud options
        # entirely, including qualified cloud on hard tasks. Stickiness's
        # real job is narrow: when two local models have similar cap
        # scores, prefer the loaded one to avoid a 30s+ GPU swap. It was
        # never meant to overcome capability gaps.
        #
        # Scaling:
        #   main_work: 1.10× (10% bias — enough to edge a close local peer,
        #             not enough to beat cloud that's +20% on composite)
        #   overhead:  1.50× (stronger — swap for a tiny classifier call is
        #             wasteful; but still not 2.0×)
        #
        # Additionally fades when loaded model is under-qualified:
        #   fit_excess >=  0.00 → full stickiness (qualified or better)
        #   fit_excess  = -0.10 → 0.5× factor
        #   fit_excess <= -0.20 → 0.0× factor (no stickiness, we're wrong tool)
        _is_overhead = reqs.call_category == "overhead"
        if model.is_local and model.is_loaded:
            _thinking_mismatch = (
                reqs.needs_thinking
                and local_state.model_name == model.name
                and not local_state.thinking_enabled
            )
            if _thinking_mismatch:
                composite *= 1.05
                reasons.append("thinking_mismatch")
            else:
                # Anti-flap: when recent swaps have occurred, dial up the
                # stickiness multiplier so a 35B↔9B oscillation can't keep
                # winning by a hair. Production triage 2026-05-01: 8 swaps
                # in 3 min, ~96% force-kill rate. Each task picked
                # independently, easy task→9B wins, hard task→35B wins,
                # mixed queue → ping-pong. Hard veto in swap_policy fires
                # only at 3/window — between firings, swaps were free.
                # Now: stickiness ramps with recent_swap_count so the 2nd
                # and 3rd swaps require a much bigger cap delta to win.
                # 0 swaps: 1.10× (default)
                # 1 swap: 1.25× (cold model needs +25% composite)
                # 2 swaps: 1.50× (cold model needs +50% composite)
                # Hard veto at 3 still owned by swap_policy.can_swap.
                _base_stick = 1.50 if _is_overhead else 1.10
                _recent_swaps = int(getattr(snapshot, "recent_swap_count", 0) or 0)
                if _recent_swaps >= 2:
                    _antiflap = 1.50 if not _is_overhead else 1.80
                elif _recent_swaps == 1:
                    _antiflap = 1.25 if not _is_overhead else 1.65
                else:
                    _antiflap = _base_stick
                _stick_raw = _antiflap
                _fit_excess = (cap_score - cap_needed_for_difficulty(reqs.difficulty)) / 100.0
                if _fit_excess >= 0:
                    _qual_factor = 1.0
                else:
                    _qual_factor = max(0.0, 1.0 + _fit_excess * 5.0)
                _stick = 1.0 + (_stick_raw - 1.0) * _qual_factor
                composite *= _stick
                if _qual_factor < 1.0:
                    reasons.append(
                        f"loaded_qual({_qual_factor:.2f}→{_stick:.2f})"
                        + ("_overhead" if _is_overhead else "")
                    )
                elif _recent_swaps >= 1:
                    reasons.append(f"loaded_antiflap({_recent_swaps}swaps→{_stick:.2f}x)")
                else:
                    reasons.append("loaded" if not _is_overhead else "loaded_overhead")
        elif model.is_local and not model.is_loaded:
            # Symmetric softening on the penalty side — swap cost is real
            # but ~30s on a typical task, not a composite-crushing factor.
            _swap_penalty = 0.60 if _is_overhead else 0.92
            composite *= _swap_penalty
            reasons.append("needs_swap" if not _is_overhead else "needs_swap_overhead")

        # Apply failure penalty multiplier (from adaptation above)
        if fail_mult < 1.0:
            composite *= fail_mult

        sm = ScoredModel(
            model=model,
            score=composite,
            capability_score=cap_score_raw,
            composite_score=composite,
            reasons=reasons,
        )
        if is_time_gated:
            time_gated.append(sm)
        else:
            scored.append(sm)

    # ── Time Gate Rescue ──
    # If ALL candidates are time-gated, rescue the fastest to avoid empty result.
    if not scored and time_gated:
        time_gated.sort(key=lambda c: -(c.model.tokens_per_second or 0))
        rescued = time_gated[0]
        rescued.reasons.append("rescued(only_option)")
        scored.append(rescued)
        logger.warning(
            "time_gate_rescue: all candidates were too slow, rescued fastest: model=%s tps=%.1f",
            rescued.model.name, rescued.model.tokens_per_second,
        )

    scored.sort(key=lambda c: -c.score)

    # ── Phase 2d: Unified utilization layer ──
    # snapshot.queue_profile is the sole source of queue state (pushed by
    # Beckman). When tests/sims set profiles on QuotaPlanner but not on
    # snapshot, mirror planner.queue_profile onto snapshot.
    planner = get_quota_planner()
    if getattr(snapshot, "queue_profile", None) is None and planner.queue_profile is not None:
        # Single QueueProfile type now (collapsed 2026-04-29) — direct mirror,
        # no shape check needed.
        snapshot.queue_profile = planner.queue_profile
    _apply_utilization_layer(
        scored,
        snapshot,
        task_difficulty=reqs.difficulty,
        reqs=reqs,
    )
    scored.sort(key=lambda c: -c.score)

    # ── S7: Sibling Rebalancing ──
    # When one cloud model in a provider has >70% util and a sibling has <30%,
    # nudge the sibling's score +8% to distribute load.
    try:
        prov_groups: dict[str, list[ScoredModel]] = {}
        for c in scored:
            if not c.model.is_local:
                prov_groups.setdefault(c.model.provider, []).append(c)

        rebalanced = False
        for provider, group in prov_groups.items():
            if len(group) < 2:
                continue

            prov_state = snapshot.cloud.get(provider)
            group_utils = [
                (
                    sc,
                    (
                        prov_state.models.get(sc.model.litellm_name).utilization_pct
                        if (
                            prov_state
                            and prov_state.models.get(sc.model.litellm_name)
                        )
                        else 0.0
                    ),
                )
                for sc in group
            ]

            max_util = max(u for _, u in group_utils)
            if max_util < 70:
                continue

            congested_name = next(
                sc.model.name for sc, u in group_utils if u == max_util
            )

            for sc, sib_util in group_utils:
                if sib_util < 30:
                    old = sc.score
                    sc.score *= 1.08
                    sc.composite_score = sc.score
                    sc.reasons.append(
                        f"sibling_rebal({provider}:{max_util:.0f}%>{sib_util:.0f}%)"
                    )
                    logger.debug(
                        "sibling rebalancing nudge: sibling=%s congested=%s provider=%s "
                        "max_util=%.0f%% sib_util=%.0f%% score %.1f->%.1f",
                        sc.model.name, congested_name, provider,
                        max_util, sib_util, old, sc.score,
                    )
                    rebalanced = True

        if rebalanced:
            scored.sort(key=lambda c: -c.score)
    except Exception as e:
        logger.debug("sibling rebalancing failed: %s", str(e))

    # Logging
    if scored:
        top3 = scored[:3]
        task_str = effective_task or reqs.primary_capability
        log_parts = [
            f"{c.model.name}({c.score:.1f}|cap={c.capability_score:.1f}|{','.join(c.reasons[:2])})"
            for c in top3
        ]
        extra = f" (+{len(scored) - 3} more)" if len(scored) > 3 else ""
        logger.info(
            "rank_candidates result: task=%s count=%d top=%s",
            task_str, len(scored), " > ".join(log_parts) + extra,
        )
    else:
        logger.warning(
            "rank_candidates: no models passed scoring: task=%s input_candidates=%d",
            effective_task or reqs.primary_capability, len(candidates),
        )

    return scored
