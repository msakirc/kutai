# selector.py
"""
Fatih Hoca — Selector (Layer 1 Eligibility + select())

The Selector class is the main entry point for model selection:
  1. Gets system snapshot from Nerd Herd.
  2. Filters models by eligibility (Layer 1 hard gates).
  3. Calls rank_candidates() for Layer 2/3 scoring.
  4. Returns the top Pick or None.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fatih_hoca.ranking import rank_candidates
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure, Pick

logger = logging.getLogger("fatih_hoca.selector")


# Pick telemetry moved to dispatcher post-iteration write (src/infra/pick_log.py).
# Selector is now side-effect free w.r.t. model_pick_log — it only returns a Pick.


class Selector:
    """
    Model selector — owns swap budget, applies eligibility filtering,
    delegates scoring to rank_candidates(), returns a Pick.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        nerd_herd: object,
        available_providers: set[str] | None = None,
    ) -> None:
        self._registry = registry
        self._nerd_herd = nerd_herd
        # Providers with API keys configured — cloud models without a key are filtered
        self._available_providers: set[str] | None = available_providers

    # ─── Public API ──────────────────────────────────────────────────────────

    def set_available_providers(self, providers: set[str] | None) -> None:
        """Update the set of providers eligible for selection. Called by the
        boot caller / refresh handler; mutating ``_available_providers``
        directly is private API and must not be done from outside the package."""
        self._available_providers = providers

    def select(
        self,
        task: str = "",
        agent_type: str = "",
        difficulty: int = 5,
        needs_function_calling: bool = False,
        needs_vision: bool = False,
        needs_thinking: bool = True,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        min_context_length: int = 0,
        max_cost: float = 0.0,
        prefer_speed: bool = False,
        prefer_local: bool = False,
        prefer_quality: bool = False,
        priority: int = 5,
        local_only: bool = False,
        failures: list[Failure] | None = None,
        exclude_models: list[str] | None = None,
        remaining_budget: float = 0.0,
        call_category: str = "main_work",
    ) -> Pick | None:
        """
        Select the best model for a task.

        Returns a Pick with the chosen model and estimated min_time_seconds,
        or None if no eligible model was found.
        """
        failures = failures or []
        exclude_models = exclude_models or []

        # ── Get system snapshot ──────────────────────────────────────────────
        snapshot = self._nerd_herd.snapshot()

        # ── Build requirements ───────────────────────────────────────────────
        reqs = ModelRequirements(
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            needs_function_calling=needs_function_calling,
            needs_vision=needs_vision,
            needs_thinking=needs_thinking,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            min_context_length=min_context_length,
            max_cost=max_cost,
            prefer_speed=prefer_speed,
            prefer_local=prefer_local,
            prefer_quality=prefer_quality,
            priority=priority,
            local_only=local_only,
            exclude_models=exclude_models,
            call_category=call_category,
        )

        # Build failed model set for quick lookup
        failed_models: set[str] = {f.model for f in failures}

        # ── Layer 1: Eligibility filtering ───────────────────────────────────
        # Track filter-reason histogram for visibility. When the eligible
        # set comes up empty, the warning carries the histogram so the
        # operator can see WHICH filter killed every candidate (vs.
        # spelunking through DEBUG lines for each model). Pre-2026-04-30
        # production ran with a registry full of dead models that all
        # filtered out for the same reason — but with only a generic
        # "no eligible candidates" warning, triage took hours.
        from collections import Counter
        filter_reasons: Counter[str] = Counter()
        candidates: list[ModelInfo] = []
        for model in self._registry.all_models():
            reason = self._check_eligibility(
                model=model,
                reqs=reqs,
                failed_models=failed_models,
                snapshot=snapshot,
            )
            if reason is not None:
                filter_reasons[reason] += 1
                logger.debug(
                    "model filtered: name=%s reason=%s task=%s",
                    model.name, reason, task,
                )
                continue
            candidates.append(model)

        if not candidates:
            # Histogram in descending order of frequency. Prefix common
            # reasons with their count so log greppers can spot patterns.
            hist = ", ".join(
                f"{count}×{reason}"
                for reason, count in filter_reasons.most_common()
            )
            logger.warning(
                "selector: no eligible candidates: task=%s local_only=%s "
                "filtered=%d reasons=[%s]",
                task, local_only, sum(filter_reasons.values()), hist,
            )
            return None

        # ── Layer 2/3: Ranking ───────────────────────────────────────────────
        scored = rank_candidates(
            candidates=candidates,
            reqs=reqs,
            snapshot=snapshot,
            failures=failures,
            remaining_budget=remaining_budget,
        )

        if not scored:
            logger.warning(
                "selector: rank_candidates returned empty: task=%s candidates=%d",
                task, len(candidates),
            )
            return None

        best = scored[0]

        # ── Swap budget enforcement ──────────────────────────────────────────
        # Policy lives in fatih_hoca; nerd_herd just holds the event stream.
        from fatih_hoca.swap_policy import can_swap as _can_swap
        if best.model.is_local and not best.model.is_loaded:
            recent = self._nerd_herd.recent_swap_count()
            if not _can_swap(recent, local_only=local_only, priority=priority):
                # Budget exhausted — try to find an already-loaded or cloud model
                logger.info(
                    "swap budget exhausted — looking for loaded or cloud alternative: "
                    "best=%s task=%s",
                    best.model.name, task,
                )
                alternative = self._find_no_swap_alternative(scored)
                if alternative is not None:
                    best = alternative
                else:
                    # No alternative — return None to signal capacity wait
                    logger.warning(
                        "swap budget exhausted and no loaded/cloud alternative: task=%s",
                        task,
                    )
                    return None
            else:
                # Swap will be recorded by the dispatcher after successful execution (Task 4).
                logger.info(
                    "model swap approved: model=%s recent_swaps=%d task=%s",
                    best.model.name, self._nerd_herd.recent_swap_count(), task,
                )

        min_time = self._calc_min_time(
            best.model, estimated_output_tokens, needs_thinking
        )
        load_time = 0.0 if (best.model.is_loaded or not best.model.is_local) else best.model.load_time_seconds

        logger.info(
            "selector pick: model=%s score=%.1f min_time=%.1fs load=%.0fs task=%s",
            best.model.name, best.score, min_time, load_time, task,
        )

        # Pick telemetry — structured log only. DB persistence is now the
        # dispatcher's job (fires post-iteration with real outcome). This
        # keeps select() pure.
        try:
            effective_task = reqs.effective_task or task
            top_n = min(len(scored), 5)
            top_summary = ", ".join(
                f"{r.model.name}={r.score:.1f}"
                for r in scored[:top_n]
            )
            logger.info(
                "picked=%s score=%.1f task=%s diff=%d category=%s candidates=[%s]",
                best.model.name, best.score,
                effective_task, reqs.difficulty, call_category,
                top_summary,
            )
        except Exception as e:
            logger.debug("pick telemetry log failed: %s", e)

        return Pick(model=best.model, min_time_seconds=min_time, estimated_load_seconds=load_time)

    # ─── Eligibility Check (Layer 1) ─────────────────────────────────────────

    def _check_eligibility(
        self,
        model: ModelInfo,
        reqs: ModelRequirements,
        failed_models: set[str],
        snapshot: object,
    ) -> str | None:
        """
        Return a rejection reason string if the model is ineligible,
        or None if it passes all hard gates.
        """
        name = model.name

        # Demoted flag (load failure recorded in ModelInfo)
        if model.demoted:
            return "demoted"

        # Failed model list (from this call's failure history)
        if model.litellm_name in failed_models:
            return "in_failed_list"

        # Exclusion list
        if model.litellm_name in reqs.exclude_models or name in reqs.exclude_models:
            return "excluded"

        # Runtime dead-model set: 404'd at call-time, provider retired the id.
        # Same id won't resurrect — exclude until restart or rediscovery.
        if self._registry.is_dead(name) or self._registry.is_dead(model.litellm_name):
            return "model_not_found"

        # local_only — reject cloud models
        if reqs.local_only and not model.is_local:
            return "local_only"

        # Cloud provider API key check — reject cloud models without a key
        if not model.is_local and self._available_providers is not None:
            if model.provider not in self._available_providers:
                return f"no_api_key({model.provider})"

        # Context length check — static trained ceiling only. The dynamic
        # `effective_context_at_current_vram` method exists on ModelInfo
        # for future scoring use, but MUST NOT be used as a hard filter:
        # snapshot VRAM is transient (a prior swap holds VRAM that gets
        # freed the moment DaLLaMa unloads to make room), and the
        # calculated effective value collapses to 4096 during tight
        # windows — filtering there rejected the entire local fleet for
        # analyst tasks (2026-04-24 incident). llama-server's --fit plus
        # the circuit-breaker OOM retry path are the real load-time gates;
        # selection should not pre-empt them.
        needed_ctx = reqs.effective_context_needed
        if needed_ctx > 0 and model.context_length < needed_ctx:
            return f"ctx({model.context_length}<{needed_ctx})"

        # Function calling requirement
        if reqs.needs_function_calling and not model.supports_function_calling:
            return "no_function_calling"

        # JSON mode requirement
        if reqs.needs_json_mode and not model.supports_json_mode:
            return "no_json_mode"

        # Vision requirement (hard filter)
        if reqs.needs_vision and not model.has_vision:
            return "no_vision"

        # Vision variant not needed — loading mmproj wastes RAM
        if not reqs.needs_vision and "vision" in getattr(model, "variant_flags", set()):
            return "vision_variant_not_needed"

        # Cost constraint
        if reqs.max_cost > 0 and not model.is_free:
            est_cost = model.estimated_cost(
                reqs.estimated_input_tokens, reqs.estimated_output_tokens
            )
            if est_cost > reqs.max_cost:
                return f"cost({est_cost:.4f}>{reqs.max_cost:.4f})"

        # Coding specialty mismatch — XML function-call format incompatible with
        # non-code tasks. Hard-filter to avoid the model winning on raw capability.
        effective_task = reqs.effective_task
        if (
            model.specialty == "coding"
            and effective_task
            and effective_task not in {"coder", "implementer", "fixer", "test_generator"}
        ):
            return f"coding_specialty_mismatch(task={effective_task})"

        # Cloud circuit breaker check
        if not model.is_local:
            prov_state = getattr(snapshot, "cloud", {}).get(model.provider)
            if prov_state is not None:
                if getattr(prov_state, "consecutive_failures", 0) >= 5:
                    return f"circuit_breaker({model.provider})"

        # Local inference allowed check — use snapshot.vram_available_mb > 0
        if model.is_local:
            vram_available = getattr(snapshot, "vram_available_mb", 0)
            if vram_available == 0:
                # Only block if there are alternatives; track this as soft hint
                # For now, treat vram_available_mb == 0 as no GPU available for
                # local inference.
                return "no_vram_available"

        return None

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _find_no_swap_alternative(self, scored: list) -> object | None:
        """
        Find the best already-loaded local model or cloud model among scored
        candidates. Used when swap budget is exhausted.
        """
        for candidate in scored:
            model = candidate.model
            # Loaded local: no swap needed
            if model.is_local and model.is_loaded:
                return candidate
            # Cloud: no swap needed
            if not model.is_local:
                return candidate
        return None

    @staticmethod
    def _calc_min_time(
        model: ModelInfo,
        estimated_output_tokens: int,
        needs_thinking: bool,
    ) -> float:
        """Calculate the minimum expected generation time in seconds.

        When the model is a thinking variant, it generates thinking tokens
        regardless of the ``needs_thinking`` flag (server-level setting).
        Apply a multiplier even for overhead calls that reuse a loaded
        thinking model — otherwise the timeout is too short.
        """
        tps = model.tokens_per_second or 10.0
        est_output = estimated_output_tokens or 500
        min_time = est_output / tps
        if model.thinking_model:
            # 3× when thinking is requested, 1.5× when it's not but the
            # server-side thinking is still active (generates think tokens).
            min_time *= 3.0 if needs_thinking else 1.5
        return min_time


# --- Test-only helper (Phase 2d scenarios) -----------------------------------

from dataclasses import dataclass


@dataclass
class _SimPickResult:
    model_name: str
    pool: str
    cap_score_100: float
    tokens_per_second: float


def select_for_simulation(
    *,
    task_name: str,
    difficulty: int,
    estimated_output_tokens: int,
    snapshot: Any,
    providers_cfg: dict,
    queue_profile: Any = None,
) -> "_SimPickResult":
    """Test-only adapter: build ModelInfo stubs from providers_cfg, call
    rank_candidates, return a light pick result.

    Used by Phase 2d stateful simulator. Not intended for production code.
    """
    from types import SimpleNamespace
    from fatih_hoca import ranking as _ranking_mod
    from fatih_hoca.ranking import rank_candidates
    from fatih_hoca.requirements import ModelRequirements, QueueProfile, get_quota_planner

    # Install queue profile on the (module-global) QuotaPlanner so that
    # rank_candidates' utilization layer sees live queue pressure. Reset
    # to empty if caller didn't provide one — sequential sim ticks must
    # not leak state from prior runs.
    planner = get_quota_planner()
    if queue_profile is not None:
        planner.set_queue_profile(queue_profile)
    else:
        planner.set_queue_profile(QueueProfile())

    candidates: list[Any] = []
    cap_overrides: dict[str, float] = {"loaded-local": 55.0}

    # Local stub — one loaded local (always present)
    local_model = SimpleNamespace(
        name="loaded-local",
        litellm_name="loaded-local",
        is_local=True,
        is_loaded=True,
        is_free=False,
        provider="local",
        capabilities=SimpleNamespace(),
        tokens_per_second=20.0,
        load_time_seconds=0.0,
        total_params_b=7,
        active_params_b=7,
        specialty=None,
        thinking_model=False,
        operational_dict=lambda: {"context_window": 32000},
        estimated_cost=lambda inp, out: 0.0,
        location="local",
    )

    # Cloud stubs
    for provider, cfg in providers_cfg.items():
        is_free = cfg.get("is_free", False)
        for model_id, model_cfg in cfg.get("models", {}).items():
            cap_overrides[model_id] = float(model_cfg.get("cap_score_100", 50.0))
            candidates.append(SimpleNamespace(
                name=model_id,
                litellm_name=model_id,
                is_local=False,
                is_loaded=False,
                is_free=is_free,
                provider=provider,
                capabilities=SimpleNamespace(),
                tokens_per_second=0.0,
                load_time_seconds=0.0,
                total_params_b=0,
                active_params_b=0,
                specialty=None,
                thinking_model=False,
                operational_dict=lambda: {"context_window": 128000},
                estimated_cost=(lambda inp, out, _free=is_free: 0.0 if _free else 0.005),
                location="cloud",
            ))
    candidates.append(local_model)

    # Monkey-patch score_model_for_task to return per-model overrides
    # (ranking.py expects a 0-10 raw score — we divide cap_100 by 10).
    real_score = _ranking_mod.score_model_for_task

    def _fake_score_0_10(model_capabilities, model_operational, requirements):
        for c in candidates:
            if c.capabilities is model_capabilities:
                return cap_overrides.get(c.name, 50.0) / 10.0
        return 5.0

    _ranking_mod.score_model_for_task = _fake_score_0_10
    try:
        reqs = ModelRequirements(
            task=task_name or "generic",
            difficulty=difficulty,
            estimated_output_tokens=estimated_output_tokens,
        )
        scored = rank_candidates(
            candidates=candidates,
            reqs=reqs,
            snapshot=snapshot,
            failures=[],
            remaining_budget=300.0,
        )
    finally:
        _ranking_mod.score_model_for_task = real_score

    if not scored:
        return _SimPickResult(
            model_name="loaded-local", pool="local",
            cap_score_100=55.0, tokens_per_second=20.0,
        )

    top = scored[0]
    return _SimPickResult(
        model_name=top.model.name,
        pool=top.pool or "local",
        cap_score_100=top.capability_score * 10.0,
        tokens_per_second=top.model.tokens_per_second or 20.0,
    )
