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

from fatih_hoca.need_ctx import compute_need_ctx
from fatih_hoca.ranking import rank_candidates
from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.requirements import ModelRequirements
from fatih_hoca.types import Failure, Pick, SelectionFailure

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
        needs_json_mode: bool = False,
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
        urgency: float = 0.5,
        remaining_budget_usd: float | None = None,
        diag_out: dict | None = None,
    ) -> Pick | SelectionFailure | None:
        """
        Select the best model for a task.

        Returns a Pick with the chosen model and estimated min_time_seconds,
        or None if no eligible model was found.
        """
        failures = failures or []
        exclude_models = exclude_models or []

        # ── Get system snapshot ──────────────────────────────────────────────
        snapshot = self._nerd_herd.snapshot()
        # In-flight overlay: nerd_herd.client's cached snapshot is sourced
        # from the sidecar HTTP cache, which lags the in-process registry
        # by one refresh cycle (~1s). Beckman overlays in_flight in its
        # next_task(), but the dispatcher's retry recursion calls
        # fatih_hoca.select() directly — without an overlay here it sees
        # stale in_flight and S9's hard local-busy veto silently passes.
        # Production triage 2026-05-01: tasks #7712 + #7755 both ended
        # up on local 9B/35B because dispatcher recursion's snap was
        # stale between admission and the retry's select. Pull
        # in-process truth from src.core.in_flight here so EVERY
        # selector entry path (admission + retry recursion) sees the
        # same authoritative list. Fail-open on import error (test
        # environments without the runtime in_flight module).
        try:
            from src.core.in_flight import in_flight_snapshot as _ifs
            from nerd_herd.types import InFlightCall as _IFC
            _local_ifs = _ifs()
            if _local_ifs:
                snapshot.in_flight_calls = [
                    _IFC(
                        call_id=e.call_id, task_id=e.task_id,
                        category=e.category, model=e.model,
                        provider=e.provider, is_local=e.is_local,
                        started_at=e.started_at,
                        # Preserve admission-time token reservation —
                        # pressure_for subtracts this from effective tpm
                        # to back-pressure parallel admissions. Without
                        # this projection, est_tokens=0 propagates and
                        # 5+ cloud admissions in the same window each
                        # see fresh tpm headroom and overshoot the
                        # shared bucket. Field exists on _InFlightEntry
                        # (set by Beckman.reserve_task and dispatcher's
                        # begin_call) but was being dropped at this
                        # type-conversion boundary.
                        est_tokens=int(getattr(e, "est_tokens", 0) or 0),
                    )
                    for e in _local_ifs
                ]
        except Exception:
            pass

        # ── Build requirements ───────────────────────────────────────────────
        # Floor needs_function_calling against the canonical agent profile.
        # llm_dispatcher derives this flag from `bool(tools)` on the request
        # — but ReAct-style agents may call between iterations without
        # tools=, and constrained_emit / grader paths never carry tools.
        # Without this floor, models flagged supports_function_calling=False
        # (groq/compound{,-mini}, gpt-oss-safeguard-20b, llama-prompt-guard-2-*,
        # allam-2-7b) leak past eligibility for tool-using agents. Production
        # 2026-05-01 task #7532: prompt-guard-2-22m picked for test_generator
        # (which has needs_function_calling=True in AGENT_REQUIREMENTS),
        # immediate "This model does not support JSON output" → DLQ.
        # Profile is the source of truth; caller can only escalate False→True,
        # never relax True→False.
        from fatih_hoca.requirements import AGENT_REQUIREMENTS
        _profile = AGENT_REQUIREMENTS.get(task) or AGENT_REQUIREMENTS.get(agent_type)
        if _profile is not None:
            if _profile.needs_function_calling and not needs_function_calling:
                needs_function_calling = True
            if _profile.needs_vision and not needs_vision:
                needs_vision = True

        reqs = ModelRequirements(
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            needs_function_calling=needs_function_calling,
            needs_vision=needs_vision,
            needs_thinking=needs_thinking,
            needs_json_mode=needs_json_mode,
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
        # WS-1 forensics (handoff 2026-05-25): when the pool comes up empty,
        # the downstream admission_violations row must say WHICH filter killed
        # it — not just "no_candidates". Capture the per-reason histogram and,
        # specifically, every FUNCTION-CALLING-capable model that was rejected
        # for a NON-FC reason (rate cap / daily-exhausted / pressure / vram).
        # That is the exact signal the researcher-starvation analysis needs:
        # "which tool-capable model COULD have served, and why it didn't".
        fc_rejected: dict[str, str] = {}

        def _emit_diag(stage: str | None, **extra) -> None:
            """Populate the caller's diag dict in place (no-op when absent).
            ``stage`` is the empty-pool stage (eligibility/rank/pressure/
            swap_budget/budget) or None when a Pick was served."""
            if diag_out is None:
                return
            diag_out["empty_stage"] = stage
            diag_out["filter_reasons"] = dict(filter_reasons)
            diag_out["eligible_count"] = len(candidates)
            diag_out["fc_capable_rejected"] = dict(fc_rejected)
            diag_out.update(extra)

        for model in self._registry.all_models():
            reason = self._check_eligibility(
                model=model,
                reqs=reqs,
                failed_models=failed_models,
                snapshot=snapshot,
            )
            if reason is not None:
                filter_reasons[reason] += 1
                # An FC-capable model rejected for something OTHER than
                # lacking FC is the high-value forensic: it means a tool
                # agent's pool was emptied by rate/pressure, not by a
                # structural capability gap. Cap the map so a 300-model
                # openrouter registry can't blow the 1KB snapshot budget.
                if (reason != "no_function_calling"
                        and getattr(model, "supports_function_calling", False)
                        and len(fc_rejected) < 25):
                    key = getattr(model, "litellm_name", None) or model.name
                    fc_rejected[key] = reason
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
            _emit_diag("eligibility")
            return None

        # Visibility: per-provider eligible-candidate count. When a provider
        # like openrouter has 300+ registered models but ZERO get picked,
        # this surfaces whether the issue is at eligibility (provider
        # filtered out entirely) or at ranking (scores too low). Pre-this,
        # no way to tell from logs without setting DEBUG on every model
        # filtered line.
        prov_counts: Counter[str] = Counter()
        for c in candidates:
            prov = "local" if getattr(c, "is_local", False) else getattr(c, "provider", "?")
            prov_counts[prov] += 1
        prov_str = ", ".join(f"{p}={n}" for p, n in prov_counts.most_common())
        # Note also which providers were FULLY filtered (had registrations
        # but every model was filtered). Catches the "openrouter present
        # in registry but always filtered" case.
        filtered_provs: Counter[str] = Counter()
        for m in self._registry.all_models():
            if m.is_local:
                continue
            if m not in candidates:
                filtered_provs[m.provider] += 1
        fully_filtered = [p for p in filtered_provs
                          if p not in prov_counts and filtered_provs[p] > 0]
        logger.debug(
            "selector eligibility: task=%s candidates=%d providers=[%s]%s",
            task, len(candidates), prov_str,
            (f" fully_filtered=[{','.join(fully_filtered)}]" if fully_filtered else ""),
        )

        # ── Budget filter (hard cap on per-call cost) ────────────────────────
        # Applied AFTER eligibility (hard gates), BEFORE scoring (Layer 2/3).
        # remaining_budget_usd=None means no filter. 0.0 means only free
        # models ($0 estimated cost) pass.
        if remaining_budget_usd is not None:
            before = len(candidates)
            candidates = [
                m for m in candidates
                if (getattr(m, "estimated_cost_usd", 0.0) or 0.0) <= remaining_budget_usd
            ]
            logger.info(
                "budget filter: %d/%d eligible at remaining=$%.4f",
                len(candidates), before, remaining_budget_usd,
            )
            if not candidates:
                _emit_diag("budget",
                           budget_remaining_usd=remaining_budget_usd)
                return SelectionFailure(
                    reason="budget",
                    detail=f"no model fits remaining ${remaining_budget_usd:.4f}",
                )

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
            _emit_diag("rank")
            return None

        # ── Pool-pressure gate (single source of truth) ──────────────────────
        # rank_candidates stamps `urgency` on each ScoredModel as the pool-
        # pressure scalar. Filter out models whose pressure is below the
        # task's urgency-derived admission threshold. This is the SINGLE
        # mechanism — Beckman's admission-time pressure check and the
        # dispatcher's recursion-time pressure check both delegate here.
        # Without consolidating, three separate gates with three separate
        # thresholds drifted out of sync (production triage 2026-04-30).
        #
        threshold = max(-1.0, -0.5 - 0.5 * urgency)
        # Strict-greater-than at the floor: scalar=-1.0 means "literally
        # depleted" (S1 hit depletion_max with full intensity, or S9
        # local busy hard-veto). No urgency level should admit a model
        # that's structurally unable to serve. With `>= threshold` and
        # threshold=-1.0 (urgency=1.0 case), -1.0 admitted → defeated
        # the gate's purpose.
        scored_after = [
            s for s in scored
            if getattr(s, "urgency", 0.0) >= threshold
            and getattr(s, "urgency", 0.0) > -1.0
        ]
        if scored_after:
            scored = scored_after
        else:
            # Every candidate fell below threshold — return None so caller
            # can either back off or escalate urgency. Important: do NOT
            # silently relax the threshold — that's how we got back into
            # the "selector keeps picking dead models" loop.
            logger.info(
                "selector: all candidates below pressure threshold "
                "task=%s urgency=%.2f threshold=%+.2f scalars=[%s]",
                task, urgency, threshold,
                ", ".join(f"{s.model.name}={getattr(s, 'urgency', 0.0):+.2f}"
                          for s in scored[:5]),
            )
            _emit_diag(
                "pressure",
                pressure_threshold=round(threshold, 3),
                pressure_scalars={
                    s.model.name: round(getattr(s, "urgency", 0.0), 3)
                    for s in scored[:10]
                },
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
                    _emit_diag("swap_budget")
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

        # Pick telemetry — structured log + Pick fields the dispatcher
        # persists into model_pick_log. select() stays pure (no DB
        # writes); the dispatcher fires the actual row post-iteration
        # with the real outcome.
        top_summary = ""
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

        _emit_diag(None, picked=best.model.name)
        need_ctx = 0
        if getattr(best.model, "is_local", False):
            try:
                _model_ctx = int(getattr(best.model, "context_length", 0) or 0)
            except (TypeError, ValueError):
                _model_ctx = 0
            need_ctx = compute_need_ctx(
                min_context=reqs.effective_context_needed or min_context_length,
                est_in=estimated_input_tokens,
                est_out=estimated_output_tokens,
                model_ctx=_model_ctx,
            )
        return Pick(
            model=best.model,
            min_time_seconds=min_time,
            estimated_load_seconds=load_time,
            score=float(best.score),
            top_summary=top_summary,
            need_ctx=need_ctx,
        )

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
        # Same id won't resurrect until rediscovery or per-cause TTL expiry.
        if self._registry.is_dead(name) or self._registry.is_dead(model.litellm_name):
            return "model_not_found"

        # Provider-level dead (auth failure, key cap). Replaces the
        # legacy per-model mass-mark loop — single row excludes every
        # cloud model on the affected provider until operator /revive.
        if (not model.is_local
                and self._registry.is_provider_dead(model.provider)):
            return f"provider_dead({model.provider})"

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

        # Per-request input-token cap (hard). Some cloud tiers gate the
        # single-request input below the advertised context_window —
        # e.g. Groq free-tier compound/compound-mini accept context=131K
        # but reject single requests over ~6K with HTTP 413
        # `request_too_large`. Provider-side enforced; no slack possible.
        # Production triage 2026-05-06: planner spec_review (~8K tokens)
        # repeatedly picked groq/compound-mini, hit 413, retry classifier
        # mapped to `unknown`, selector re-picked, exhausted attempts.
        max_in = getattr(model, "max_input_tokens", None)
        if (max_in is not None
                and reqs.estimated_input_tokens > 0
                and reqs.estimated_input_tokens > max_in):
            return f"per_request_too_large(est={reqs.estimated_input_tokens}>cap={max_in})"

        # Per-call TPM hard filter. Cloud models with a TPM ceiling that
        # cannot fit a single call are STRUCTURALLY ineligible — the
        # provider will 429 with "Request too large for model" regardless
        # of how empty the bucket is. Pool pressure's S2 burden signal
        # warns about this softly; this hard gate prevents the model
        # from even reaching ranking. Production triage 2026-05-01:
        # groq/openai/gpt-oss-20b (TPM=8K) kept getting picked for
        # implementer tasks (~12K estimate) and bouncing off Groq's
        # per-call ceiling.
        #
        # Skip for local (TPM-based limits don't apply — llama-server
        # ctx-size is the binding constraint, handled above) and for
        # models without a known per-minute cap.
        if (not model.is_local
                and getattr(model, "rate_limit_tpm", 0) > 0
                and reqs.estimated_input_tokens + reqs.estimated_output_tokens > 0):
            est = reqs.estimated_input_tokens + reqs.estimated_output_tokens
            tpm_cap = int(model.rate_limit_tpm)
            # Allow some slack — actual provider ceiling is usually a bit
            # above the documented TPM (per-call vs per-minute). Reject
            # only when estimate exceeds tpm_cap by 10%+.
            if est > tpm_cap * 1.1:
                return f"per_call_too_large(est={est}>tpm={tpm_cap})"

        # Coding specialty mismatch — XML function-call format incompatible with
        # non-code tasks. Hard-filter to avoid the model winning on raw capability.
        effective_task = reqs.effective_task
        if (
            model.specialty == "coding"
            and effective_task
            and effective_task not in {"coder", "implementer", "fixer", "test_generator"}
        ):
            return f"coding_specialty_mismatch(task={effective_task})"

        # Cloud circuit breaker check.
        # Two paths converge here:
        #   1. KDV's per-process CircuitBreaker (3 failures / 300s window,
        #      600s cooldown). Authoritative — KDV's pre_call already
        #      refuses calls while degraded. Plumbed via
        #      CloudProviderState.circuit_breaker_open from the
        #      kuleden_donen_var/nerd_herd_adapter (Beckman cloud overlay).
        #   2. Legacy consecutive_failures threshold. Field exists on
        #      CloudProviderState but has no production writer — kept for
        #      backward compatibility with tests + future signals.
        # Production 2026-05-02: gemini circuit breaker tripped after the
        # free-tier 20-req daily quota got eaten. KDV refused every gemini
        # pre_call with reason=circuit_breaker, but selector didn't see the
        # signal and kept ranking gemini ids first. Each one fast-failed,
        # got added to the failures list, retry recursion exhausted the
        # pool. With this gate, gemini's entire model set drops out of
        # eligibility while the breaker is open — selector picks a
        # different provider instead of cycling through dead gemini ids.
        if not model.is_local:
            prov_state = getattr(snapshot, "cloud", {}).get(model.provider)
            if prov_state is not None:
                if getattr(prov_state, "circuit_breaker_open", False):
                    return f"circuit_breaker({model.provider})"
                if getattr(prov_state, "consecutive_failures", 0) >= 5:
                    return f"circuit_breaker({model.provider})"
                # Per-model daily-exhausted: KDV's body-derived rpd-out
                # state. Selector's matrix-based S1 can't see this when
                # the provider doesn't return rpd headers (gemini). KDV
                # is the only authoritative source. Drop the model from
                # candidates so retry recursion doesn't keep cycling
                # through ids the caller will refuse on first call.
                # Production 2026-05-02 14:54: tasks admitted on
                # gemini/* with positive composites; every call refused
                # with daily_exhausted; reselect picked another gemini
                # variant; same refusal; pool exhausted → DLQ at 10/10.
                mstate = getattr(prov_state, "models", {}).get(
                    getattr(model, "litellm_name", "")
                ) or getattr(prov_state, "models", {}).get(
                    getattr(model, "name", "")
                )
                if mstate is not None and getattr(
                    mstate, "daily_exhausted", False
                ):
                    return f"daily_exhausted({model.provider})"
                # Per-model rpm cooldown: KDV recorded a Retry-After /
                # x-ratelimit-reset floor with remaining=0. Without this
                # gate the model stays eligible after the 5s header
                # freshness window expires (rpm_remaining property reverts
                # to sliding-window math, exposes fake capacity). Selector
                # picks it; KDV.pre_call refuses; retry recursion cycles
                # through the same provider's other ids; pool exhausts.
                # Same failure mode as daily_exhausted, just on the
                # per-minute axis. Production hardening 2026-05-03 after
                # retry-after wiring.
                if mstate is not None and getattr(
                    mstate, "rpm_cooldown", False
                ):
                    return f"rpm_cooldown({model.provider})"

        # Local inference allowed check — use snapshot.vram_available_mb > 0
        if model.is_local:
            # Minimal load mode = cloud-only. Local is structurally
            # ineligible — clearer than a pressure veto and gives a
            # named diag reason. (resource-signals 2026-06-09)
            if getattr(snapshot, "load_mode", "full") == "minimal":
                return "load_mode_minimal"
            vram_available = getattr(snapshot, "vram_available_mb", 0)
            if vram_available == 0:
                # Only block if there are alternatives; track this as soft hint
                # For now, treat vram_available_mb == 0 as no GPU available for
                # local inference.
                return "no_vram_available"

        return None

    # ─── Continuation gate (RC-A, mission 74) ────────────────────────────────

    def is_servable(self, *, model: ModelInfo, reqs: ModelRequirements) -> bool:
        """Can a task CONTINUE on the model it already holds, right now?

        Runs the hard-eligibility chain for this ONE model against the
        current snapshot — WITHOUT the pool-pressure scalar/threshold gate.
        Pressure decides whether to start *new* load; it must not evict a
        model a task already reserved (that race is the no_candidates
        mechanism). Used by coulson's ``pick_for_iter`` to reuse the
        admitted pick across no-failure iterations instead of re-racing
        the live pool every turn.

        A held local model that's already loaded survives
        ``vram_available_mb == 0``: its own residency consumed the VRAM —
        that is continuation, not contention.
        """
        snapshot = self._nerd_herd.snapshot()
        reason = self._check_eligibility(
            model=model, reqs=reqs, failed_models=set(), snapshot=snapshot,
        )
        if reason is None:
            return True
        if (reason == "no_vram_available"
                and getattr(model, "is_local", False)
                and getattr(model, "is_loaded", False)):
            return True
        return False

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
    provider: str = ""


def select_for_simulation(
    *,
    task_name: str,
    difficulty: int,
    estimated_output_tokens: int,
    snapshot: Any,
    providers_cfg: dict,
    queue_profile: Any = None,
    now: float | None = None,
    burn_log=None,
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
        capabilities=set(),
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
            _caps = set(model_cfg.get("capabilities", []))
            _ms = None
            _ps = snapshot.cloud.get(provider) if hasattr(snapshot, "cloud") else None
            if _ps is not None:
                _ms = _ps.models.get(model_id)
            _rpd_rem = 0
            if _ms is not None and _ms.limits.rpd is not None:
                _rpd_rem = _ms.limits.rpd.remaining or 0
            candidates.append(SimpleNamespace(
                name=model_id,
                litellm_name=model_id,
                is_local=False,
                is_loaded=False,
                is_free=is_free,
                provider=provider,
                capabilities=_caps,
                rpd_remaining=_rpd_rem,
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
            now=now,
            burn_log=burn_log,
        )
    finally:
        _ranking_mod.score_model_for_task = real_score

    if not scored:
        return _SimPickResult(
            model_name="loaded-local", pool="local",
            cap_score_100=55.0, tokens_per_second=20.0, provider="local",
        )

    top = scored[0]
    return _SimPickResult(
        model_name=top.model.name,
        pool=top.pool or "local",
        cap_score_100=top.capability_score * 10.0,
        tokens_per_second=top.model.tokens_per_second or 20.0,
        provider=getattr(top.model, "provider", "") or "local",
    )
