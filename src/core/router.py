# router.py
"""
Model Router v2 — 14-dimension task-aware model selection,
rate limiting, retries, cross-provider fallback, GPU-aware scheduling.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass, field

import litellm

litellm.suppress_debug_info = True
litellm.return_response_headers = True

from src.infra.logging_config import get_logger
from src.models.rate_limiter import get_rate_limit_manager
from src.models.header_parser import parse_rate_limit_headers
from src.models.quota_planner import get_quota_planner
from src.models.capabilities import ALL_CAPABILITIES, Cap, TASK_PROFILES, \
  TaskRequirements as CapabilityTaskReqs, score_model_for_task
from src.models.model_registry import ModelInfo, get_registry

logger = get_logger("core.router")


# ─── Capability ↔ Task Mapping ───────────────────────────────────────────────

CAPABILITY_TO_TASK: dict[str, str] = {
    # Map primary_capability values to TASK_PROFILES keys
    "reasoning":             "planner",
    "planning":              "planner",
    "analysis":              "analyst",
    "code_generation":       "coder",
    "code_reasoning":        "fixer",
    "system_design":         "architect",
    "prose_quality":         "writer",
    "instruction_adherence": "executor",
    "domain_knowledge":      "researcher",
    "context_utilization":   "summarizer",
    "structured_output":     "executor",
    "tool_use":              "executor",
    "vision":                "visual_reviewer",
    "conversation":          "assistant",
    "general":               "assistant",
}


def _make_adhoc_profile(primary_cap: str) -> dict[str, float]:
    """Create a task profile on the fly for unknown capability requests."""
    profile = {cap: 0.3 for cap in ALL_CAPABILITIES}
    for c in Cap:
        if c.value == primary_cap or primary_cap in c.value:
            profile[c.value] = 1.0
            return profile
    profile[Cap.REASONING.value] = 0.8
    profile[Cap.INSTRUCTION_ADHERENCE.value] = 0.8
    return profile


# ─── Model Requirements ─────────────────────────────────────────────────────

@dataclass
class ModelRequirements:
    """
    Structured description of what a task needs from a model.

    Uses difficulty (1-10) to express how capable the model must be.
    """
    # ── Task identity (preferred path) ──
    task: str = ""                            # Key into TASK_PROFILES

    # ── Capability path (auto-maps to task) ──
    primary_capability: str = "general"
    secondary_capabilities: list[str] = field(default_factory=list)

    # ── Difficulty (1-10) — drives model quality selection ──
    difficulty: int = 5
    min_score: float = 0.0                     # Override; if 0, computed from difficulty

    # ── Context requirements ──
    estimated_input_tokens: int = 2000
    estimated_output_tokens: int = 1000
    min_context_length: int = 0

    # ── Feature requirements ──
    needs_function_calling: bool = False
    needs_json_mode: bool = False
    needs_thinking: bool = False
    needs_vision: bool = False

    # ── Constraints ──
    local_only: bool = False
    prefer_speed: bool = False
    prefer_quality: bool = False
    prefer_local: bool = False
    max_cost: float = 0.0

    # ── Priority ──
    priority: int = 5

    # ── Exclusion / pinning ──
    exclude_models: list[str] = field(default_factory=list)
    model_override: str | None = None

    # ── Agent context ──
    agent_type: str = ""

    @property
    def effective_task(self) -> str:
        if self.task and self.task in TASK_PROFILES:
            return self.task
        mapped = CAPABILITY_TO_TASK.get(self.primary_capability)
        if mapped and mapped in TASK_PROFILES:
            return mapped
        return ""

    @property
    def task_profile(self) -> dict[str, float]:
        task = self.effective_task
        if task:
            return TASK_PROFILES[task]
        return _make_adhoc_profile(self.primary_capability)

    @property
    def effective_context_needed(self) -> int:
        if self.min_context_length > 0:
            return self.min_context_length
        return int((self.estimated_input_tokens + self.estimated_output_tokens) * 1.3)

    @property
    def effective_min_score(self) -> float:
        if self.min_score > 0:
            return self.min_score
        return self.difficulty * 0.7

    def escalate(self) -> "ModelRequirements":
        """
        Return a copy with escalated quality requirements.
        Used by base.py for mid-task escalation.
        """
        escalated = copy.copy(self)
        escalated.difficulty = min(10, self.difficulty + 2)
        escalated.min_score = 0.0  # reset so it recomputes from new difficulty
        escalated.prefer_quality = True
        return escalated


# ─── Per-Provider Rate Limiting ──────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter tracking both RPM and TPM."""

    def __init__(self, rpm: int = 30, tpm: int = 100000):
        self.rpm = rpm
        self.tpm = tpm
        self._request_timestamps: list[float] = []
        self._token_log: list[tuple[float, int]] = []

    @property
    def current_rpm_usage(self) -> int:
        now = time.time()
        return len([t for t in self._request_timestamps if now - t < 60])

    @property
    def current_tpm_usage(self) -> int:
        now = time.time()
        return sum(tc for ts, tc in self._token_log if now - ts < 60)

    @property
    def rpm_headroom(self) -> int:
        return self.rpm - self.current_rpm_usage

    @property
    def tpm_headroom(self) -> int:
        return self.tpm - self.current_tpm_usage

    def has_capacity(self, estimated_tokens: int = 0) -> bool:
        return self.rpm_headroom > 2 and self.tpm_headroom > estimated_tokens

    async def wait(self) -> None:
        now = time.time()
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]
        if len(self._request_timestamps) >= self.rpm:
            wait_time = 60 - (now - self._request_timestamps[0]) + 0.5
            logger.info("rate limiter wait", wait_time_seconds=wait_time)
            await asyncio.sleep(wait_time)
        self._request_timestamps.append(time.time())

    def record_tokens(self, token_count: int) -> None:
        now = time.time()
        self._token_log.append((now, token_count))
        self._token_log = [(t, c) for t, c in self._token_log if now - t < 60]


class CircuitBreaker:
    """Track failures per provider and temporarily disable."""

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 300,
        cooldown_seconds: float = 600,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.failures: list[float] = []
        self.degraded_until: float = 0.0

    def record_failure(self) -> None:
        now = time.time()
        self.failures.append(now)
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        if len(self.failures) >= self.failure_threshold:
            self.degraded_until = now + self.cooldown_seconds
            logger.warning("circuit breaker tripped", cooldown_seconds=self.cooldown_seconds)

    def record_success(self) -> None:
        self.failures.clear()
        self.degraded_until = 0.0

    @property
    def is_degraded(self) -> bool:
        if time.time() >= self.degraded_until:
            if self.degraded_until > 0:
                self.degraded_until = 0.0
                self.failures.clear()
            return False
        return True


_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_circuit_breaker(provider: str) -> CircuitBreaker:
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker()
    return _circuit_breakers[provider]


_limiters_initialized = False


# ─── Performance Cache ──────────────────────────────────────────────────────

_perf_cache: dict[str, dict[str, dict]] = {}
_perf_cache_ready: bool = False
_perf_cache_last_refresh: float = 0.0
_PERF_CACHE_TTL: float = 300.0


async def refresh_perf_cache() -> None:
    global _perf_cache, _perf_cache_ready, _perf_cache_last_refresh
    now = time.time()
    if _perf_cache_ready and (now - _perf_cache_last_refresh) < _PERF_CACHE_TTL:
        return
    try:
        from ..infra.db import get_model_stats
        stats = await get_model_stats()
        cache: dict[str, dict[str, dict]] = {}
        for s in stats:
            at = s["agent_type"]
            m = s["model"]
            if at not in cache:
                cache[at] = {}
            cache[at][m] = s
        _perf_cache = cache
        _perf_cache_ready = True
        _perf_cache_last_refresh = now
    except Exception as e:
        logger.debug("performance cache refresh failed", error=str(e))


# ─── Model Selection ────────────────────────────────────────────────────────

@dataclass
class ScoredModel:
    """A model candidate with its selection score and reasoning."""
    model: ModelInfo
    score: float
    capability_score: float = 0.0
    composite_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def litellm_name(self) -> str:
        return self.model.litellm_name


def select_model(reqs: ModelRequirements) -> list[ScoredModel]:
    """
    Select models matching requirements, ranked by composite score.

    Scoring (all 0-100, then weighted):
    1. CAPABILITY FIT (35)  — 14-dimension weighted dot product
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

    for name, model in registry.models.items():
        reasons: list[str] = []

        # ═════════════════════════════════════════
        # Hard Filters
        # ═════════════════════════════════════════

        if model.litellm_name in reqs.exclude_models:
            continue
        if reqs.local_only and not model.is_local:
            continue

        needed_ctx = reqs.effective_context_needed
        if needed_ctx > 0 and model.context_length < needed_ctx:
            continue
        if reqs.needs_function_calling and not model.supports_function_calling:
            continue
        if reqs.needs_json_mode and not model.supports_json_mode:
            continue
        if reqs.needs_thinking and not model.thinking_model:
            continue
        if reqs.needs_vision and not model.has_vision:
            continue

        if reqs.max_cost > 0 and not model.is_free:
            est_cost = model.estimated_cost(
                reqs.estimated_input_tokens, reqs.estimated_output_tokens
            )
            if est_cost > reqs.max_cost:
                continue

        if not model.is_local:
            cb = _get_circuit_breaker(model.provider)
            if cb.is_degraded:
                continue

        # ═════════════════════════════════════════
        # 1. CAPABILITY FIT (0-100)
        # ═════════════════════════════════════════

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
            continue
        if min_score > 0 and cap_score_raw < min_score:
            continue

        cap_score = min(cap_score_raw * 10, 100)
        reasons.append(f"cap={cap_score_raw:.1f}")
        if effective_task:
            reasons.append(f"task={effective_task}")

        # ═════════════════════════════════════════
        # 2. COST EFFICIENCY (0-100)
        # ═════════════════════════════════════════

        if model.is_local:
            cost_score = 95 if model.is_loaded else 70
            if not model.is_loaded:
                reasons.append("needs_swap")
            reasons.append("local")
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
                avail_score = 60 if swap_time < 10 else (40 if swap_time < 30 else 20)
                reasons.append(f"swap_{swap_time:.0f}s")
        else:
            rl_manager = get_rate_limit_manager()
            total_tokens = (
                reqs.estimated_input_tokens + reqs.estimated_output_tokens
            )

            # Check BOTH model-level and provider-level capacity
            has_model_cap = rl_manager.has_capacity(
                model.litellm_name, model.provider, total_tokens,
            )
            model_util = rl_manager.get_utilization(model.litellm_name)
            provider_util = rl_manager.get_provider_utilization(
                model.provider,
            )

            if has_model_cap and model_util < 50:
                avail_score = 95
            elif has_model_cap and model_util < 80:
                avail_score = 75
                reasons.append(f"util={model_util:.0f}%")
            elif has_model_cap:
                avail_score = 50
                reasons.append(f"util={model_util:.0f}%")
            elif provider_util < 100:
                avail_score = 25
                reasons.append("model_limited")
            else:
                avail_score = 5
                reasons.append("rate_limited")

        # ═════════════════════════════════════════
        # 4. PERFORMANCE HISTORY (0-100)
        # ═════════════════════════════════════════

        perf_score = 50
        if reqs.agent_type and _perf_cache_ready:
            agent_perf = _perf_cache.get(reqs.agent_type, {})
            model_perf = agent_perf.get(model.litellm_name)
            if model_perf and model_perf.get("total_calls", 0) >= 3:
                sr = model_perf["success_rate"]
                grade = model_perf.get("avg_grade", 3.0)
                perf_score = max(0, min(100, (sr * grade / 5.0) * 100))
                reasons.append(
                    f"perf(sr={sr:.2f},g={grade:.1f},n={model_perf['total_calls']})"
                )
                # Penalize unreliable models
                if sr < 0.5:
                    avail_score = int(avail_score * 0.3)
                    reasons.append("unreliable")

        # ═════════════════════════════════════════
        # 5. SPEED (0-100)
        # ═════════════════════════════════════════

        if model.is_local:
            tps = model.tokens_per_second
            if tps >= 50:
                speed_score = 100
            elif tps >= 20:
                speed_score = 70
            elif tps > 0:
                speed_score = 40
            else:
                speed_score = 80 if model.total_params_b < 5 else (50 if model.total_params_b < 15 else 30)
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
        d = reqs.difficulty
        if d <= 3:
            weights = {"capability": 15, "cost": 50, "availability": 20, "performance": 10, "speed": 5}
        elif d <= 5:
            weights = {"capability": 30, "cost": 30, "availability": 20, "performance": 15, "speed": 5}
        elif d <= 7:
            weights = {"capability": 40, "cost": 20, "availability": 20, "performance": 15, "speed": 5}
        else:
            weights = {"capability": 55, "cost": 5, "availability": 15, "performance": 20, "speed": 5}

        # Modifiers
        if reqs.prefer_speed:
            weights["speed"] += 15
            weights["cost"] -= 10
        if reqs.prefer_quality:
            weights["capability"] += 10
            weights["cost"] -= 10
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

        # Swap stickiness: prefer already-loaded local model to avoid swap cost
        if model.is_local and model.is_loaded:
            composite *= 1.10

        candidates.append(ScoredModel(
            model=model,
            score=composite,
            capability_score=cap_score_raw,
            composite_score=composite,
            reasons=reasons,
        ))

    candidates.sort(key=lambda c: -c.score)

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



def _update_limits_from_response(
    response,
    model: ModelInfo,
    rl_manager,
) -> None:
    """Parse rate limit headers from a litellm response and update state."""
    try:
        hidden = getattr(response, "_hidden_params", None)
        if not hidden:
            return
        headers = hidden.get("additional_headers") or hidden.get("headers")
        if not headers:
            return

        snapshot = parse_rate_limit_headers(model.provider, dict(headers))
        if snapshot is None:
            return

        rl_manager.update_from_headers(
            model.litellm_name, model.provider, snapshot,
        )

        # Update quota planner utilization from header data
        planner = get_quota_planner()
        if snapshot.rpm_limit and snapshot.rpm_remaining is not None:
            util_pct = (1.0 - snapshot.rpm_remaining / snapshot.rpm_limit) * 100
            reset_in = (snapshot.rpm_reset_at - time.time()) if snapshot.rpm_reset_at else 3600
            planner.update_paid_utilization(model.provider, util_pct, max(0, reset_in))

            # Detect quota restoration
            prev_util = planner._paid_utilization.get(model.provider, 100.0)
            if prev_util > 80 and util_pct < 30:
                planner.on_quota_restored(model.provider, snapshot.rpm_remaining / snapshot.rpm_limit * 100)

    except Exception as e:
        logger.debug("header parsing failed", model_name=model.name, error=str(e))


# ─── Main API: call_model ────────────────────────────────────────────────────

async def call_model(
    reqs: ModelRequirements,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """
    Call the best available model matching requirements.

    Usage:
        await call_model(ModelRequirements(...), messages=[...], tools=[...])
    """

    await refresh_perf_cache()

    # ── Direct model override ──
    if reqs.model_override:
        registry = get_registry()
        pinned = registry.find_by_litellm_name(reqs.model_override)
        if pinned:
            candidates = [ScoredModel(model=pinned, score=999, reasons=["pinned"])]
        else:
            candidates = [ScoredModel(
                model=ModelInfo(
                    name="override",
                    location="cloud",
                    provider="unknown",
                    litellm_name=reqs.model_override,
                    capabilities={cap: 5.0 for cap in ALL_CAPABILITIES},
                    context_length=128000,
                    max_tokens=4096,
                ),
                score=999,
                reasons=["pinned_raw"],
            )]
    else:
        if tools:
            reqs.needs_function_calling = True
        candidates = select_model(reqs)

    if not candidates:
        fallback_reqs = ModelRequirements(
            task="assistant",
            primary_capability="general",
            difficulty=1,
            min_score=0,
            agent_type=reqs.agent_type,
        )
        candidates = select_model(fallback_reqs)

    if not candidates:
        raise RuntimeError("No models available!")

    last_error: str | None = None

    for scored in candidates[:5]:
        model = scored.model

        # ── Local model: ensure loaded ──
        if model.is_local and model.location != "ollama":
            from ..models.local_model_manager import get_local_manager
            manager = get_local_manager()
            if not model.is_loaded:
                success = await manager.ensure_model(
                    model.name,
                    reason=f"{reqs.agent_type}:{reqs.effective_task or reqs.primary_capability}",
                )
                if not success:
                    last_error = f"Failed to load local model {model.name}"
                    continue

        # ── Build completion kwargs ──
        is_thinking = model.thinking_model
        if is_thinking:
            temperature = None   # thinking models control sampling internally
        else:
            from ..models.model_profiles import get_task_params
            _task = reqs.effective_task or reqs.primary_capability
            temperature = get_task_params(_task).get("temperature", 0.3)
            logger.debug("task params applied", task=_task, temperature=temperature)

        timeout_val = 60
        if model.is_local:
            timeout_val = 120
        if is_thinking:
            timeout_val = max(timeout_val, 180)

        use_tools = None
        if tools and model.supports_function_calling:
            use_tools = tools

        completion_kwargs = dict(
            model=model.litellm_name,
            messages=messages,
            max_tokens=min(reqs.estimated_output_tokens * 2, model.max_tokens),
        )

        if temperature is not None:
            completion_kwargs["temperature"] = temperature
        if model.api_base:
            completion_kwargs["api_base"] = model.api_base

        if use_tools:
            completion_kwargs["tools"] = use_tools
            completion_kwargs["tool_choice"] = "auto"
        elif tools and not model.supports_function_calling and model.supports_json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        # ── Rate limiting (two-tier) ──
        if not model.is_local:
            rl_manager = get_rate_limit_manager()
            estimated_tokens = (
                reqs.estimated_input_tokens + reqs.estimated_output_tokens
            )
            wait_time = await rl_manager.wait_and_acquire(
                litellm_name=model.litellm_name,
                provider=model.provider,
                estimated_tokens=estimated_tokens,
            )
            if wait_time < 0:
                logger.warning(
                    "daily limit exhausted",
                    model_name=model.name,
                )
                last_error = f"Daily limit exhausted for {model.name}"
                continue
            if wait_time > 0:
                logger.info(
                    "rate limiter waited",
                    model_name=model.name,
                    wait_time_seconds=wait_time,
                )

        # ── GPU semaphore ──
        local_manager = None
        if model.is_local and model.location != "ollama":
            from ..models.local_model_manager import get_local_manager
            local_manager = get_local_manager()

            granted = await local_manager.acquire_inference_slot(
                priority=reqs.priority,
                task_id=reqs.agent_type,  # or actual task_id if available
                agent_type=reqs.agent_type,
                timeout=120 if reqs.priority < 10 else 30,
            )

            if not granted:
                # GPU busy and timed out — skip to next candidate (likely cloud)
                logger.warning(
                    "gpu access denied",
                    model_name=model.name,
                    priority=reqs.priority,
                )
                last_error = f"GPU queue timeout for {model.name}"
                continue

        max_retries = 2 if model.is_local else 3

        try:
            for attempt in range(max_retries):
                try:
                    call_start = time.time()
                    task_label = reqs.effective_task or reqs.primary_capability

                    logger.info(
                        "calling model",
                        model_name=model.name,
                        task=task_label,
                        capability_score=scored.capability_score,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        thinking=is_thinking,
                        vision=model.has_vision,
                    )

                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=timeout_val,
                    )

                    call_latency = time.time() - call_start

                    try:
                        cost = litellm.completion_cost(completion_response=response)
                    except Exception:
                        cost = 0.0
                    if model.is_local:
                        cost = 0.0

                    # Record actual token usage
                    if not model.is_local and response.usage:
                        total_tokens = (
                            (response.usage.prompt_tokens or 0)
                            + (response.usage.completion_tokens or 0)
                        )
                        rl_manager.record_tokens(
                            model.litellm_name,
                            model.provider,
                            total_tokens,
                        )

                    # Update rate limits from response headers
                    if not model.is_local:
                        _update_limits_from_response(response, model, rl_manager)

                    # Update measured speed
                    if model.is_local and response.usage:
                        output_tokens = response.usage.completion_tokens or 0
                        if output_tokens > 0 and call_latency > 0:
                            registry = get_registry()
                            registry.update_speed(model.name, output_tokens / call_latency)

                    # Extract tool calls
                    msg = response.choices[0].message
                    tool_calls = None
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls = []
                        for tc in msg.tool_calls:
                            fn = tc.function
                            try:
                                args = json.loads(fn.arguments) if fn.arguments else {}
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            tool_calls.append({
                                "id": tc.id,
                                "name": fn.name,
                                "arguments": args,
                            })

                    thinking_content = _extract_thinking(msg) if is_thinking else None

                    if not model.is_local:
                        _get_circuit_breaker(model.provider).record_success()

                    return {
                        "content": msg.content or "",
                        "model": model.litellm_name,
                        "model_name": model.name,
                        "cost": cost or 0.0,
                        "usage": dict(response.usage) if response.usage else {},
                        "tool_calls": tool_calls,
                        "latency": call_latency,
                        "thinking": thinking_content,
                        "is_local": model.is_local,
                        "task": task_label,
                        "capability_score": scored.capability_score,
                        "difficulty": reqs.difficulty,
                    }

                except asyncio.TimeoutError:
                    if not model.is_local:
                        _get_circuit_breaker(model.provider).record_failure()
                    last_error = f"Timeout on {model.name}"
                    logger.warning("model timeout", model_name=model.name, attempt=attempt + 1)
                    continue

                except Exception as e:
                    error_str = str(e).lower()
                    last_error = str(e)

                    if not model.is_local:
                        _get_circuit_breaker(model.provider).record_failure()

                    if any(kw in error_str for kw in [
                        "api key", "authentication", "unauthorized",
                        "billing", "credit", "quota",
                    ]):
                        logger.error("auth/billing error", model_name=model.name, error=str(e))
                        break

                    is_rate_limit = any(kw in error_str for kw in [
                        "rate limit", "rate_limit", "429",
                        "too many requests", "tokens per minute",
                        "resource_exhausted",
                    ])
                    if is_rate_limit:
                        if not model.is_local:
                            rl_manager.record_429(
                                model.litellm_name,
                                model.provider,
                            )
                            if not model.is_free:
                                get_quota_planner().record_429(model.provider)
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            logger.warning(
                                "model rate limited",
                                model_name=model.name,
                                wait_time_seconds=wait_time,
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        break

                    if attempt < 1:
                        await asyncio.sleep(2)
                        continue
                    break

        finally:
            if local_manager:
                local_manager.release_inference_slot()

    # ── All candidates exhausted — try backpressure queue ──
    from ..infra.backpressure import get_backpressure_queue

    bp_queue = get_backpressure_queue()
    call_id = f"{reqs.agent_type}:{reqs.primary_capability}"

    logger.warning(
        "all models failed fallback to backpressure",
        call_id=call_id,
        last_error=last_error,
    )


    # Create a retry callable that re-runs selection + call
    async def _retry_call():
        # On retry, refresh perf cache and try again
        await refresh_perf_cache()
        # Recursive call — but with a flag to prevent infinite backpressure
        reqs._is_retry = True
        return await call_model(reqs, messages, tools)

    # Check if this is already a retry (prevent infinite recursion)
    if getattr(reqs, '_is_retry', False):
        raise RuntimeError(
            f"All models failed after backpressure retry for "
            f"'{call_id}'. Last error: {last_error}"
        )

    try:
        return await bp_queue.enqueue(
            call_id=call_id,
            priority=reqs.priority,
            last_error=last_error or "Unknown",
            call_func=_retry_call,
        )
    except RuntimeError:
        # Queue full or expired — propagate
        raise


# ─── Thinking Helpers ────────────────────────────────────────────────────────

def _extract_thinking(msg) -> str | None:
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    content = msg.content or ""
    match = re.search(
        r"<(?:thinking|think)>(.*?)</(?:thinking|think)>",
        content, re.DOTALL,
    )
    return match.group(1).strip() if match else None


# ─── Response Grading ────────────────────────────────────────────────────────

GRADING_PROMPT = """Rate this AI response on a scale of 1-5:
1 = Wrong/useless, 2 = Partially relevant, 3 = Adequate,
4 = Good and complete, 5 = Excellent

Task: {task_title}
Response to grade:
{response}

Respond with ONLY JSON: {{"score": N, "reason": "brief"}}"""


async def grade_response(
    task_title: str,
    task_description: str,
    response_text: str,
    generating_model: str = "",
    task_name: str = "",
) -> float | None:
    """Grade a response using a DIFFERENT model."""
    if not response_text or len(response_text.strip()) < 10:
        return None

    try:
        grading_reqs = ModelRequirements(
            task="reviewer",
            difficulty=3,
            estimated_input_tokens=800,
            estimated_output_tokens=50,
            prefer_speed=True,
            exclude_models=[generating_model] if generating_model else [],
        )

        result = await call_model(
            grading_reqs,
            messages=[{
                "role": "user",
                "content": GRADING_PROMPT.format(
                    task_title=task_title[:100],
                    response=response_text[:2000],
                ),
            }],
        )

        raw = result.get("content", "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        parsed = json.loads(raw.strip())
        score = float(parsed.get("score", 3))
        grade = max(1.0, min(5.0, score))

        # Feed grade back to registry
        if generating_model:
            registry = get_registry()
            model_name = generating_model.split("/")[-1] if "/" in generating_model else generating_model
            # Find dominant capability for the grading task context
            lookup = task_name or result.get("task", "")
            if lookup and lookup in TASK_PROFILES:
                profile = TASK_PROFILES[lookup]
                dominant = max(profile.items(), key=lambda x: x[1])
                cap_key = dominant[0].value if hasattr(dominant[0], 'value') else dominant[0]
                # Pass call count so EMA uses lower alpha for early samples
                call_count = 0
                if _perf_cache_ready:
                    for _agent_perf in _perf_cache.values():
                        mp = _agent_perf.get(generating_model)
                        if mp:
                            call_count = mp.get("total_calls", 0)
                            break
                registry.update_quality_from_grading(
                    model_name, cap_key, grade * 2.0, call_count=call_count,
                )

        return grade

    except Exception as e:
        logger.debug("response grading failed", error=str(e))
        return None


# ─── Cost Budget ─────────────────────────────────────────────────────────────

async def check_cost_budget() -> dict:
    try:
        from ..infra.db import check_budget
        return await check_budget("daily")
    except Exception as e:
        return {"ok": True, "reason": f"check failed: {e}"}


# ─── Backward Compatibility ─────────────────────────────────────────────────

# ─── Agent Requirement Templates ─────────────────────────────────────────────

AGENT_REQUIREMENTS: dict[str, ModelRequirements] = {
    "planner":        ModelRequirements(task="planner",        difficulty=7, estimated_output_tokens=2000, prefer_quality=True, needs_json_mode=True),
    "architect":      ModelRequirements(task="architect",      difficulty=7, estimated_output_tokens=3000, prefer_quality=True),
    "coder":          ModelRequirements(task="coder",          difficulty=6, estimated_output_tokens=4000, needs_function_calling=True),
    "implementer":    ModelRequirements(task="implementer",    difficulty=5, estimated_output_tokens=4000, needs_function_calling=True),
    "fixer":          ModelRequirements(task="fixer",          difficulty=6, estimated_output_tokens=3000, needs_function_calling=True),
    "test_generator": ModelRequirements(task="test_generator", difficulty=5, estimated_output_tokens=3000, needs_function_calling=True),
    "reviewer":       ModelRequirements(task="reviewer",       difficulty=6, estimated_output_tokens=2000),
    "researcher":     ModelRequirements(task="researcher",     difficulty=5, estimated_output_tokens=2000, needs_function_calling=True),
    "analyst":        ModelRequirements(task="analyst",        difficulty=6, estimated_output_tokens=3000, needs_function_calling=True),
    "writer":         ModelRequirements(task="writer",         difficulty=5, estimated_output_tokens=3000),
    "executor":       ModelRequirements(task="executor",       difficulty=3, estimated_output_tokens=1000, needs_function_calling=True, prefer_speed=True),
    "visual_reviewer": ModelRequirements(task="visual_reviewer", difficulty=5, estimated_output_tokens=2000, needs_vision=True),
    "assistant":       ModelRequirements(task="assistant",       difficulty=5, estimated_output_tokens=2000),
    "summarizer":      ModelRequirements(task="summarizer",      difficulty=4, estimated_output_tokens=2000, prefer_speed=True),
    "error_recovery":  ModelRequirements(task="error_recovery",  difficulty=6, estimated_output_tokens=2000, needs_function_calling=True),
}
