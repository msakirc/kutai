# router.py
"""
Model Router — owns select_model() over the legacy registry plus
get_kdv()/ModelCallFailed. Selection types live in fatih_hoca; import
ModelRequirements / ScoredModel / AGENT_REQUIREMENTS / CAPABILITY_TO_TASK
from fatih_hoca.requirements (or .ranking) directly.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field


class ModelCallFailed(RuntimeError):
    """All model candidates exhausted — no backpressure, no retry.

    Raised by the dispatcher when every candidate fails. The caller
    (process_task) catches this and puts the task to sleep, waiting
    for a signal that capacity has changed.
    """

    def __init__(self, call_id: str, last_error: str, error_category: str):
        super().__init__(f"All models failed for '{call_id}': {last_error}")
        self.call_id = call_id
        self.last_error = last_error
        self.error_category = error_category


from src.infra.logging_config import get_logger
from kuleden_donen_var import KuledenDonenVar, KuledenConfig, CapacityEvent
from fatih_hoca.requirements import (
    ModelRequirements,
    get_quota_planner,
)
from fatih_hoca.ranking import ScoredModel
from fatih_hoca.capabilities import ALL_CAPABILITIES, Cap, TASK_PROFILES, \
  TaskRequirements as CapabilityTaskReqs, score_model_for_task
from src.models.model_registry import ModelInfo, get_registry

logger = get_logger("core.router")


_kdv: KuledenDonenVar | None = None


def get_kdv() -> KuledenDonenVar:
    global _kdv
    if _kdv is None:
        from src.models.rate_limiter import _INITIAL_PROVIDER_LIMITS
        from src.models.model_registry import get_registry as _get_registry

        def _on_capacity_change(evt: CapacityEvent) -> None:
            planner = get_quota_planner()
            snap = evt.snapshot
            if snap.utilization_pct > 0:
                reset_in = snap.reset_in_seconds or 3600
                planner.update_paid_utilization(evt.provider, snap.utilization_pct, reset_in)

            if evt.event_type in ("capacity_restored", "circuit_breaker_reset"):
                try:
                    from src.infra.db import schedule_accelerate_retries
                    schedule_accelerate_retries("capacity_restored")
                except Exception:
                    pass

        cfg = KuledenConfig(on_capacity_change=_on_capacity_change)
        _kdv = KuledenDonenVar(cfg)

        try:
            registry = _get_registry()
            for model in registry.cloud_models():
                agg = _INITIAL_PROVIDER_LIMITS.get(model.provider, {})
                _kdv.register(
                    model_id=model.litellm_name,
                    provider=model.provider,
                    rpm=model.rate_limit_rpm,
                    tpm=model.rate_limit_tpm,
                    provider_aggregate_rpm=agg.get("rpm"),
                    provider_aggregate_tpm=agg.get("tpm"),
                )
                # Propagate the daily-axis quota when known. KDV.register
                # only accepts rpm/tpm; rpd lives on RateLimitState as a
                # separate field. Static seeds (Gemini free tier per AI
                # Studio quota table) populate this on every registration
                # — without it, S1's time_bucketed depletion arm has no
                # rpd cell to compute frac on, and exhausted models stay
                # invisible to pool pressure.
                if model.rate_limit_rpd is not None:
                    state = _kdv._rate_limiter.model_limits.get(model.litellm_name)
                    if state is not None:
                        state.rpd_limit = int(model.rate_limit_rpd)
                        state.rpd_remaining = int(model.rate_limit_rpd)
            # Mark each cloud provider as enabled so KDV can surface
            # "no observations after Nh" warnings later.
            for provider in {m.provider for m in registry.cloud_models()}:
                _kdv.mark_provider_enabled(provider)
        except Exception:
            pass

        # Wire the in-flight tracker so begin_call / end_call push a
        # CloudProviderState (with overlaid in_flight) into nerd_herd.
        # Without this, the tracker counts handles in-process but the
        # signal never reaches pool_pressure computation.
        try:
            import nerd_herd
            from kuleden_donen_var import configure_in_flight_push
            from kuleden_donen_var.nerd_herd_adapter import make_state_getter
            configure_in_flight_push(nerd_herd, make_state_getter(_kdv))
        except Exception:
            pass

        # Restore persisted KDV state synchronously here so the first
        # pre_call after boot sees real adapted limits / 429 history /
        # daily counters / header reset clocks. Uses plain sqlite3 (not
        # aiosqlite) so it works whether or not an event loop is active.
        # Best-effort: failures degrade to cold-start state. Skipped
        # silently when DB_PATH is unset (CLI tools, tests).
        try:
            import os
            db_path = os.environ.get("DB_PATH")
            if db_path:
                from src.infra import kdv_persistence
                kdv_persistence.load_sync(_kdv, db_path)
        except Exception:
            pass
    return _kdv


def select_model(reqs: ModelRequirements) -> list[ScoredModel]:
    """
    Select models matching requirements, ranked by composite score.

    Scoring (all 0-100, then weighted):
    1. CAPABILITY FIT (35)  — 15-dimension weighted dot product
    2. COST EFFICIENCY (25) — local > free cloud > cheap paid > expensive
    3. AVAILABILITY (20)    — rate limit headroom, loaded status
    4. PERFORMANCE (15)     — historical success rate + quality grades
    5. SPEED (5)            — tps, provider speed class
    """
    registry = get_registry()

    task_profile = reqs.task_profile
    effective_task = reqs.effective_task
    min_score = reqs.effective_min_score

    candidates: list[ScoredModel] = []
    _time_gated: list[ScoredModel] = []  # models that passed all other filters but are too slow

    # Fetch actual runtime state for the currently loaded local model.
    # Used to apply runtime-aware scoring adjustments inside the loop.
    try:
        from src.models.local_model_manager import get_runtime_state
        _loaded_runtime = get_runtime_state()
    except Exception:
        _loaded_runtime = None

    for name, model in registry.models.items():
        reasons: list[str] = []

        # ╔══════════════════════════════════════════╗
        # ║  LAYER 1: Eligibility (hard pass/fail)  ║
        # ╚══════════════════════════════════════════╝

        def _skip(reason: str) -> bool:
            logger.debug("model filtered", model_name=name, reason=reason,
                         task=effective_task)
            return True

        if registry.is_demoted(name):
            _skip("demoted (recent load failure)")
            continue

        if model.litellm_name in reqs.exclude_models:
            _skip("excluded"); continue
        if model.demoted:
            _skip("demoted"); continue
        if reqs.local_only and not model.is_local:
            _skip("local_only"); continue

        needed_ctx = reqs.effective_context_needed
        if needed_ctx > 0 and model.context_length < needed_ctx:
            _skip(f"ctx({model.context_length}<{needed_ctx})"); continue

        # Runtime context check: if this IS the loaded model, use the actual
        # loaded ctx window (may be smaller than registry due to dynamic calc).
        if (model.is_local and model.is_loaded and needed_ctx > 0
                and _loaded_runtime is not None
                and _loaded_runtime.model_name == name
                and _loaded_runtime.context_length < needed_ctx):
            _skip(
                f"runtime_ctx({_loaded_runtime.context_length}<{needed_ctx})"
            )
            continue
        if reqs.needs_function_calling and not model.supports_function_calling:
            _skip("no_function_calling"); continue
        if reqs.needs_json_mode and not model.supports_json_mode:
            _skip("no_json_mode"); continue
        # needs_thinking is a SOFT preference, not a hard filter.
        # Thinking models get a bonus in scoring below, but we don't
        # exclude capable non-thinking cloud models (e.g. groq-llama-70b
        # is perfectly fine for planning, just without CoT).
        # Only needs_vision stays hard — you can't do vision without it.
        if reqs.needs_vision and not model.has_vision:
            _skip("no_vision"); continue
        # Exclude vision variants from non-vision tasks — loading mmproj
        # wastes RAM for no benefit, and the base model is identical.
        if not reqs.needs_vision and "vision" in getattr(model, "variant_flags", set()):
            _skip("vision_variant_not_needed"); continue

        if reqs.max_cost > 0 and not model.is_free:
            est_cost = model.estimated_cost(
                reqs.estimated_input_tokens, reqs.estimated_output_tokens
            )
            if est_cost > reqs.max_cost:
                _skip(f"cost({est_cost:.4f}>{reqs.max_cost:.4f})"); continue

        # Coding-specialty models use XML function-call format incompatible with
        # non-code tasks (classification, research, conversation, …).  A 0.50x
        # multiplier in the ranking section was too weak — the model could still
        # win if its capability score was high enough.  Hard-filter instead.
        if (model.specialty == "coding" and effective_task
                and effective_task not in {"coder", "implementer", "fixer", "test_generator"}):
            _skip(f"coding_specialty_mismatch(task={effective_task})"); continue

        if not model.is_local:
            kdv = get_kdv()
            prov_status = kdv.status.get(model.provider)
            if prov_status and prov_status.circuit_breaker_open:
                _skip(f"circuit_breaker({model.provider})"); continue

        # ── Load mode enforcement ──
        if model.is_local:
            from src.infra.load_manager import is_local_inference_allowed, get_vram_budget_fraction
            if not is_local_inference_allowed():
                _skip("load_mode_minimal"); continue
            _vram_budget = get_vram_budget_fraction()
            if 0 < _vram_budget < 1.0:
                _model_vram = getattr(model, 'vram_required_mb', 0) or 0
                if _model_vram > 0:
                    from src.models.gpu_monitor import get_gpu_monitor
                    _gpu = get_gpu_monitor().get_state().gpu
                    if _gpu.available:
                        _budget_mb = int(_gpu.vram_total_mb * _vram_budget)
                        if _model_vram > _budget_mb:
                            _skip(f"vram_budget({_model_vram}>{_budget_mb}MB)"); continue

        # ── Time gate flag: mark models too slow for the timeout budget ──
        # Checked after scoring; if ALL candidates would be time-gated,
        # the least-slow one is rescued so we never return empty.
        # Uses the dispatcher's hard cap (300s) as ceiling: when TPS is
        # known, the dispatcher computes min(300, est_gen*2), so generation
        # can never exceed 300s without timing out.
        _is_time_gated = False
        if model.is_local and reqs.estimated_output_tokens > 0:
            _gate_tps = (
                _loaded_runtime.measured_tps
                if (model.is_loaded
                    and _loaded_runtime is not None
                    and _loaded_runtime.model_name == name
                    and _loaded_runtime.measured_tps > 0)
                else model.tokens_per_second
            )
            if _gate_tps > 0:
                _gate_secs = reqs.estimated_output_tokens / _gate_tps
                _TIME_BUDGET = 300.0  # dispatcher hard cap
                if _gate_secs > _TIME_BUDGET:
                    _is_time_gated = True
                    reasons.append(
                        f"time_gated({_gate_tps:.1f}tps×"
                        f"{reqs.estimated_output_tokens}tok"
                        f"={_gate_secs:.0f}s>{_TIME_BUDGET:.0f}s)"
                    )

        # ╔══════════════════════════════════════════╗
        # ║  LAYER 2: Capability gate              ║
        # ╚══════════════════════════════════════════╝

        # ─── 1. Capability fit (0–100) ───────────────────────────────

        cap_score_raw = score_model_for_task(
            model_capabilities=model.capabilities,
            model_operational=model.operational_dict(),
            requirements=CapabilityTaskReqs(
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
            _skip("capability_reject"); continue
        if min_score > 0 and cap_score_raw < min_score:
            _skip(f"below_min_score({cap_score_raw:.1f}<{min_score})"); continue

        cap_score = min(cap_score_raw * 10, 100)
        reasons.append(f"cap={cap_score_raw:.1f}")
        if effective_task:
            reasons.append(f"task={effective_task}")

        # ═════════════════════════════════════════
        # 2. COST EFFICIENCY (0-100)
        # ═════════════════════════════════════════

        if model.is_local:
            cost_score = 95 if model.is_loaded else 90  # still free even when unloaded
            if not model.is_loaded:
                reasons.append("needs_swap")
            reasons.append("local")
            # Load mode penalty: reduce local preference in shared/heavy modes
            _vb = get_vram_budget_fraction()
            if _vb < 1.0:
                cost_score = int(cost_score * (0.5 + _vb * 0.5))
                reasons.append(f"load_pen={_vb:.1f}")
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

            # Quota planner: penalize paid models when below threshold
            planner = get_quota_planner()
            if reqs.difficulty < planner.expensive_threshold:
                penalty = (planner.expensive_threshold - reqs.difficulty) * 8
                cost_score = max(0, cost_score - penalty)
                reasons.append(f"quota_pen=-{penalty}")

        # ═════════════════════════════════════════
        # 3. AVAILABILITY (0-100)
        # ═════════════════════════════════════════

        if model.is_local:
            if model.is_loaded:
                avail_score = 100
                reasons.append("loaded")
            else:
                swap_time = model.load_time_seconds
                avail_score = 75 if swap_time < 10 else (55 if swap_time < 30 else 35)
                reasons.append(f"swap_{swap_time:.0f}s")
        else:
            # ── Graduated headroom scoring (S3b) ──
            # Replaced binary has_capacity() gate with continuous scoring.
            # Uses utilization percentage to compute a smooth availability
            # score so models near their limit gracefully degrade rather
            # than cliff-dropping from 50→25 when they cross the capacity
            # threshold.
            kdv = get_kdv()
            prov_status = kdv.status.get(model.provider)
            model_status = prov_status.models.get(model.litellm_name) if prov_status else None
            model_util = model_status.utilization_pct if model_status else 0.0
            provider_util = prov_status.utilization_pct if prov_status else 0.0
            # Check daily limit exhaustion (hard gate — no graduated
            # fallback; if daily limit is gone, the model is useless).
            _daily_exhausted = model_status.daily_exhausted if model_status else False
            if _daily_exhausted:
                avail_score = 0
                reasons.append("daily_exhausted")
            else:
                # Smooth curve: 100% util → 5, 0% util → 95
                # Blends model-level and provider-level utilization (worst wins)
                _effective_util = max(model_util, provider_util)
                avail_score = max(5, int(95 - _effective_util * 0.90))
                if _effective_util >= 80:
                    reasons.append(f"util={_effective_util:.0f}%")
                elif _effective_util >= 50:
                    reasons.append(f"util={_effective_util:.0f}%")

        # ═════════════════════════════════════════
        # 4. PERFORMANCE HISTORY (0-100)
        # ═════════════════════════════════════════
        # TODO: wire up performance cache (refresh from DB stats)
        # so historical success rate and grade influence scoring.
        perf_score = 50

        # ═════════════════════════════════════════
        # 5. SPEED (0-100)
        # ═════════════════════════════════════════

        if model.is_local:
            # Prefer measured tok/s from live /metrics over static registry value.
            # This reflects current GPU state (thermal throttle, context fill, etc.)
            tps = (
                _loaded_runtime.measured_tps
                if (model.is_loaded
                    and _loaded_runtime is not None
                    and _loaded_runtime.model_name == name
                    and _loaded_runtime.measured_tps > 0)
                else model.tokens_per_second
            )
            active = getattr(model, 'active_params_b', 0) or model.total_params_b
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
                speed_score = 10  # < 2 tok/s is basically unusable
            else:
                # No measured tps yet — estimate from model size
                # MoE models (active_params < total_params * 0.5) are much faster
                if active < 5:
                    speed_score = 75
                elif active < 10:
                    speed_score = 55
                elif active < 20:
                    speed_score = 35
                else:
                    speed_score = 15

            # ── Output-length penalty for slow local models ──
            # A 2 tok/s model generating 2000 tokens takes 17 minutes.
            # Penalize proportionally so long-output tasks route to
            # smaller (faster) local models or fall through to cloud.
            est_out = reqs.estimated_output_tokens
            effective_tps = tps if tps > 0 else (
                25 if active < 5
                else 12 if active < 15
                else 5 if active < 30
                else 3
            )
            est_generation_secs = est_out / effective_tps
            if est_generation_secs > 300:        # >5 min: severe penalty
                speed_score = max(0, speed_score - 50)
                reasons.append(f"very_slow({est_generation_secs:.0f}s)")
            elif est_generation_secs > 120:      # >2 min: moderate penalty
                speed_score = max(0, speed_score - 30)
                reasons.append(f"slow({est_generation_secs:.0f}s)")
            elif est_generation_secs > 60:       # >1 min: mild penalty
                speed_score = max(0, speed_score - 15)
                reasons.append(f"moderate({est_generation_secs:.0f}s)")

            # Amplify speed score when prefer_speed is set — measured TPS directly boosts score
            if reqs.prefer_speed and tps > 0:
                # Normalize TPS: 50+ tok/s → 1.0 boost, 5 tok/s → 0.1 boost
                tps_boost = min(1.0, tps / 50.0)
                speed_score = speed_score * (0.5 + tps_boost * 0.5)
        else:
            speed_map = {
                "groq": 95, "cerebras": 95, "sambanova": 80,
                "gemini": 70, "openai": 60, "anthropic": 50,
            }
            speed_score = speed_map.get(model.provider, 50)

        # ═════════════════════════════════════════
        # Composite
        # ═════════════════════════════════════════

        # Base weights from difficulty
        # Speed matters more than before — a 0.8 tok/s model is useless
        # for any real task regardless of capability.
        d = reqs.difficulty
        if d <= 3:
            weights = {"capability": 20, "cost": 35, "availability": 20, "performance": 10, "speed": 15}
        elif d <= 5:
            weights = {"capability": 30, "cost": 20, "availability": 20, "performance": 15, "speed": 15}
        elif d <= 7:
            weights = {"capability": 35, "cost": 15, "availability": 15, "performance": 15, "speed": 20}
        else:
            weights = {"capability": 45, "cost": 5, "availability": 10, "performance": 20, "speed": 20}

        # Modifiers
        if reqs.prefer_speed:
            weights["speed"] += 15
            weights["cost"] -= 10
        if reqs.prefer_quality:
            weights["capability"] += 10
            weights["cost"] -= 10
        if reqs.prefer_local:
            weights["cost"] += 10          # amplify local's cost advantage
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

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  LAYER 3: Ranking adjustments (3 conceptual groups)         ║
        # ║                                                              ║
        # ║  Group A — Thinking fitness (+20% capability boost)         ║
        # ║  Group B — Specialty alignment (+15% match)                 ║
        # ║  Group C — Swap stickiness (+40% loaded / -25% unloaded,   ║
        # ║             or +10% on thinking-state mismatch)             ║
        # ║                                                              ║
        # ║  NOTE: coding_model mismatch was here as 0.50x but has been ║
        # ║  moved to Layer 1 (hard filter) — see eligibility section.  ║
        # ║  prefer_local boost (was 1.15x) is now fully expressed by   ║
        # ║  weight modifiers in the composite formula above.           ║
        # ╚══════════════════════════════════════════════════════════════╝

        # Group A: Thinking fitness
        # Reward thinking models when the task explicitly requests CoT.
        # Note: loaded-model thinking *mismatch* is handled by Group C.
        if reqs.needs_thinking and model.thinking_model:
            composite *= 1.20
            reasons.append("thinking_bonus")

        # Group B: Specialty alignment
        # Specialty-matched models have architecturally better outputs for
        # that task type (e.g. code-tuned models for code tasks).
        # Mismatch is now a hard filter (Layer 1), so only the bonus remains.
        if model.specialty and effective_task:
            _specialty_tasks = {
                "coding": {"coder", "implementer", "fixer", "test_generator"},
                "reasoning": {"planner", "architect", "analyst"},
                "vision": {"visual_reviewer"},
            }
            _matched = _specialty_tasks.get(model.specialty, set())
            if effective_task in _matched:
                composite *= 1.15
                reasons.append(f"specialty={model.specialty}")

        # Group C: Swap stickiness
        # Strongly prefer the already-loaded model (avoids 25s+ GPU warm-up).
        # Only a substantially better model should justify triggering a swap.
        # Thinking-state mismatch reduces — but does not eliminate — stickiness.
        if model.is_local and model.is_loaded:
            _thinking_mismatch = (
                reqs.needs_thinking
                and _loaded_runtime is not None
                and _loaded_runtime.model_name == name
                and not _loaded_runtime.thinking_enabled
            )
            if _thinking_mismatch:
                composite *= 1.10
                reasons.append("thinking_mismatch")
            else:
                composite *= 1.40
                reasons.append("loaded")
        elif model.is_local and not model.is_loaded:
            composite *= 0.75
            reasons.append("needs_swap")

        _scored = ScoredModel(
            model=model,
            score=composite,
            capability_score=cap_score_raw,
            composite_score=composite,
            reasons=reasons,
        )
        if _is_time_gated:
            _time_gated.append(_scored)
        else:
            candidates.append(_scored)

    # ── Rescue: if time gate filtered all candidates, let the fastest through ──
    if not candidates and _time_gated:
        _time_gated.sort(key=lambda c: -(c.model.tokens_per_second or 0))
        rescued = _time_gated[0]
        rescued.reasons.append("rescued(only_option)")
        candidates.append(rescued)
        logger.warning(
            "time_gate_rescue: all candidates were too slow, rescued fastest",
            model=rescued.model.name,
            tps=rescued.model.tokens_per_second,
        )

    candidates.sort(key=lambda c: -c.score)

    # ── S7: Multi-model provider sibling rebalancing ──────────────────────────
    # When one model in a provider is heavily congested (>70% utilization) and
    # a sibling from the same provider has low utilization (<30%), nudge the
    # sibling's score upward so load distributes across provider capacity rather
    # than hammering a single endpoint.  The nudge is modest (+8%) so it only
    # changes ordering when candidates are close in score — it never overrides a
    # genuinely superior model.
    # Only applies to cloud models (local models have a single GPU slot, so
    # sibling rebalancing is irrelevant there).
    try:
        _kdv_s7 = get_kdv()
        _prov_groups: dict[str, list[ScoredModel]] = {}
        for _c in candidates:
            if not _c.model.is_local:
                _p = _c.model.provider
                _prov_groups.setdefault(_p, []).append(_c)

        _rebalanced = False
        for _provider, _group in _prov_groups.items():
            if len(_group) < 2:
                continue
            # Compute utilization for every model in this provider group once
            _prov_s7 = _kdv_s7.status.get(_provider)
            _group_utils = [
                (
                    _sc,
                    (
                        _prov_s7.models.get(_sc.model.litellm_name).utilization_pct
                        if (_prov_s7 and _prov_s7.models.get(_sc.model.litellm_name))
                        else 0.0
                    ),
                )
                for _sc in _group
            ]
            # Is there at least one heavily-congested model in this group?
            _max_util = max(u for _, u in _group_utils)
            if _max_util < 70:
                continue  # no congestion in this provider — skip
            # Find the name of the most congested model for logging
            _congested_name = next(
                sc.model.name for sc, u in _group_utils if u == _max_util
            )
            # Nudge all underutilized siblings (<30% util)
            for _sc, _sib_util in _group_utils:
                if _sib_util < 30:
                    _old = _sc.score
                    _sc.score *= 1.08
                    _sc.composite_score = _sc.score
                    _sc.reasons.append(
                        f"sibling_rebal({_provider}:{_max_util:.0f}%>"
                        f"{_sib_util:.0f}%)"
                    )
                    logger.debug(
                        "sibling rebalancing nudge",
                        sibling=_sc.model.name,
                        congested_model=_congested_name,
                        provider=_provider,
                        max_util=f"{_max_util:.0f}%",
                        sibling_util=f"{_sib_util:.0f}%",
                        score_change=f"{_old:.1f}->{_sc.score:.1f}",
                    )
                    _rebalanced = True
        if _rebalanced:
            candidates.sort(key=lambda c: -c.score)
    except Exception as _e7:
        logger.debug("sibling rebalancing failed", error=str(_e7))

    # Logging
    if candidates:
        top3 = candidates[:3]
        task_str = effective_task or reqs.primary_capability
        log_parts = [
            f"{c.model.name}({c.score:.1f}|cap={c.capability_score:.1f}|{','.join(c.reasons[:2])})"
            for c in top3
        ]
        extra = f" (+{len(candidates)-3} more)" if len(candidates) > 3 else ""
        logger.info(
            "route chosen",
            task=task_str,
            local_only=reqs.local_only,
            prefer_quality=reqs.prefer_quality,
            prefer_speed=reqs.prefer_speed,
            top_models=" > ".join(log_parts) + extra,
            candidate_count=len(candidates),
        )
    else:
        logger.warning(
            "no models matched",
            task=effective_task or reqs.primary_capability,
            min_score=min_score,
            context_needed=reqs.effective_context_needed,
            needs_function_calling=reqs.needs_function_calling,
            needs_vision=reqs.needs_vision,
            needs_thinking=reqs.needs_thinking,
            local_only=reqs.local_only,
        )

    return candidates


# ─── Convenience Selectors ───────────────────────────────────────────────────

def select_for_task(task: str, **kwargs) -> list[ScoredModel]:
    """Simplified selection by task name."""
    return select_model(ModelRequirements(task=task, **kwargs))



# ─── Cost Budget ─────────────────────────────────────────────────────────────

async def check_cost_budget() -> dict:
    try:
        from ..infra.db import check_budget
        return await check_budget("daily")
    except Exception as e:
        return {"ok": True, "reason": f"check failed: {e}"}
