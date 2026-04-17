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

import logging

from fatih_hoca.ranking import rank_candidates
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure, Pick, SwapBudget

logger = logging.getLogger("fatih_hoca.selector")


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
        self._swap_budget = SwapBudget(max_swaps=3, window_seconds=300)
        # Providers with API keys configured — cloud models without a key are filtered
        self._available_providers: set[str] | None = available_providers

    # ─── Public API ──────────────────────────────────────────────────────────

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
        candidates: list[ModelInfo] = []
        for model in self._registry.all_models():
            reason = self._check_eligibility(
                model=model,
                reqs=reqs,
                failed_models=failed_models,
                snapshot=snapshot,
            )
            if reason is not None:
                logger.debug(
                    "model filtered: name=%s reason=%s task=%s",
                    model.name, reason, task,
                )
                continue
            candidates.append(model)

        if not candidates:
            logger.warning(
                "selector: no eligible candidates: task=%s local_only=%s",
                task, local_only,
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
        # If the best model requires a swap (local but not loaded), check budget.
        if best.model.is_local and not best.model.is_loaded:
            if not self._swap_budget.can_swap(local_only=local_only, priority=priority):
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
                # Record the swap
                self._swap_budget.record_swap()
                logger.info(
                    "model swap recorded: model=%s budget_remaining=%d task=%s",
                    best.model.name, self._swap_budget.remaining, task,
                )

        min_time = self._calc_min_time(
            best.model, estimated_output_tokens, needs_thinking
        )
        load_time = 0.0 if (best.model.is_loaded or not best.model.is_local) else best.model.load_time_seconds

        logger.info(
            "selector pick: model=%s score=%.1f min_time=%.1fs load=%.0fs task=%s",
            best.model.name, best.score, min_time, load_time, task,
        )
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

        # local_only — reject cloud models
        if reqs.local_only and not model.is_local:
            return "local_only"

        # Cloud provider API key check — reject cloud models without a key
        if not model.is_local and self._available_providers is not None:
            if model.provider not in self._available_providers:
                return f"no_api_key({model.provider})"

        # Context length check
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
