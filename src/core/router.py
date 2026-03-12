# router.py
"""
Model Router v2 — 14-dimension task-aware model selection,
rate limiting, retries, cross-provider fallback, GPU-aware scheduling.

Backward compatible: call_model() accepts both ModelRequirements objects
AND the old (tier_str, messages, ...) signature for gradual migration.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field

import litellm

litellm.suppress_debug_info = True

from src.models.rate_limiter import get_rate_limit_manager
from src.models.capabilities import ALL_CAPABILITIES, Cap, TASK_PROFILES, \
  TaskRequirements as CapabilityTaskReqs, score_model_for_task
from src.models.model_registry import ModelInfo, get_registry

logger = logging.getLogger(__name__)


# ─── Capability ↔ Task Mapping ───────────────────────────────────────────────

CAPABILITY_TO_TASK: dict[str, str] = {
    "coding":         "coder",
    "code":           "coder",
    "debugging":      "fixer",
    "debug":          "fixer",
    "fixing":         "fixer",
    "fix":            "fixer",
    "reasoning":      "planner",
    "planning":       "planner",
    "plan":           "planner",
    "architecture":   "architect",
    "design":         "architect",
    "writing":        "writer",
    "write":          "writer",
    "documentation":  "writer",
    "research":       "researcher",
    "analysis":       "reviewer",
    "review":         "reviewer",
    "testing":        "test_generator",
    "test":           "test_generator",
    "implementation": "implementer",
    "implement":      "implementer",
    "execution":      "executor",
    "execute":        "executor",
    "tool_use":       "executor",
    "routing":        "router",
    "classification": "router",
    "visual":         "visual_reviewer",
    "vision":         "visual_reviewer",
    "screenshot":     "visual_reviewer",
    "conversation":   "assistant",
    "chat":           "assistant",
    "assistant":      "assistant",
    "summarize":      "summarizer",
    "summary":        "summarizer",
    "general":        "assistant",
    "simple":         "executor",
}

# Legacy tier → task mapping (for backward compat)
TIER_TO_TASK: dict[str, str] = {
    "routing":   "router",
    "cheap":     "executor",
    "code":      "coder",
    "medium":    "assistant",
    "expensive": "planner",
}

# Legacy tier → quality floor
TIER_TO_MIN_QUALITY: dict[str, int] = {
    "routing":   1,
    "cheap":     3,
    "code":      5,
    "medium":    6,
    "expensive": 8,
}

# Tier escalation order (used by base.py, preserved for compat)
TIER_ESCALATION_ORDER: list[str] = ["cheap", "code", "medium", "expensive"]


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

    Supports both new task-based routing and legacy tier/capability routing.
    """
    # ── Task identity (preferred path) ──
    task: str = ""                            # Key into TASK_PROFILES

    # ── Legacy capability path (auto-maps to task) ──
    primary_capability: str = "general"
    secondary_capabilities: list[str] = field(default_factory=list)

    # ── Quality floor ──
    min_score: float = 0.0                     # Minimum weighted task score (0-10)
    min_quality: int = 1                       # Legacy: maps to min_score

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

    # ── Legacy tier (for backward compat — auto-maps to task) ──
    _tier: str = ""

    @property
    def effective_task(self) -> str:
        if self.task and self.task in TASK_PROFILES:
            return self.task
        mapped = CAPABILITY_TO_TASK.get(self.primary_capability)
        if mapped and mapped in TASK_PROFILES:
            return mapped
        if self._tier:
            tier_mapped = TIER_TO_TASK.get(self._tier)
            if tier_mapped and tier_mapped in TASK_PROFILES:
                return tier_mapped
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
        if self.min_quality > 1:
            return self.min_quality * 0.8
        return 0.0

    @classmethod
    def from_tier(
        cls,
        tier: str,
        agent_type: str = "",
        model_override: str | None = None,
        **kwargs,
    ) -> "ModelRequirements":
        """Create ModelRequirements from a legacy tier string."""
        task = TIER_TO_TASK.get(tier, "assistant")
        min_q = TIER_TO_MIN_QUALITY.get(tier, 5)
        return cls(
            task=task,
            _tier=tier,
            primary_capability=CAPABILITY_TO_TASK.get(task, "general"),
            min_quality=min_q,
            agent_type=agent_type,
            model_override=model_override,
            prefer_speed=(tier in ("routing", "cheap")),
            prefer_quality=(tier == "expensive"),
            needs_function_calling=(tier == "code"),
            **kwargs,
        )

    def escalate(self) -> "ModelRequirements":
        """
        Return a copy with escalated quality requirements.
        Used by base.py for mid-task escalation.
        """
        escalated = copy.copy(self)
        # Bump quality floor
        escalated.min_quality = min(10, self.min_quality + 2)
        escalated.min_score = min(10.0, self.effective_min_score + 1.5)
        escalated.prefer_quality = True
        # If we had a tier, escalate it
        if self._tier and self._tier in TIER_ESCALATION_ORDER:
            idx = TIER_ESCALATION_ORDER.index(self._tier)
            if idx < len(TIER_ESCALATION_ORDER) - 1:
                escalated._tier = TIER_ESCALATION_ORDER[idx + 1]
                escalated.task = TIER_TO_TASK.get(escalated._tier, escalated.task)
        return escalated

    @property
    def tier(self) -> str:
        """Legacy tier accessor for checkpoint compat."""
        if self._tier:
            return self._tier
        # Reverse-map from task
        for t, task_name in TIER_TO_TASK.items():
            if task_name == self.effective_task:
                return t
        if self.prefer_quality or self.min_quality >= 8:
            return "expensive"
        if self.min_quality >= 6:
            return "medium"
        if self.needs_function_calling:
            return "code"
        return "cheap"


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
            logger.info(f"Rate limiter: waiting {wait_time:.1f}s (RPM)")
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
            logger.warning(
                f"Circuit breaker TRIPPED — degraded for {self.cooldown_seconds:.0f}s"
            )

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
        logger.debug(f"Performance cache refresh failed: {e}")


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

        weights = {"capability": 35, "cost": 25, "availability": 20, "performance": 15, "speed": 5}

        if reqs.prefer_quality:
            weights = {"capability": 50, "cost": 10, "availability": 15, "performance": 20, "speed": 5}
        elif reqs.prefer_speed:
            weights = {"capability": 25, "cost": 15, "availability": 25, "performance": 10, "speed": 25}

        if reqs.local_only:
            weights["availability"] = 30
            weights["cost"] = 10

        if reqs.priority >= 10:
            weights = {"capability": 30, "cost": 5, "availability": 30, "performance": 10, "speed": 25}
        elif reqs.priority <= 2:
            weights = {"capability": 25, "cost": 40, "availability": 15, "performance": 15, "speed": 5}

        total_weight = sum(weights.values())
        composite = (
            cap_score * weights["capability"]
            + cost_score * weights["cost"]
            + avail_score * weights["availability"]
            + perf_score * weights["performance"]
            + speed_score * weights["speed"]
        ) / total_weight

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
            f"Model selection [{task_str}]"
            f"{'[local]' if reqs.local_only else ''}"
            f"{'[quality]' if reqs.prefer_quality else ''}"
            f"{'[fast]' if reqs.prefer_speed else ''}"
            f": {' > '.join(log_parts)}{extra}"
        )
    else:
        logger.warning(
            f"No models match: task={effective_task or reqs.primary_capability}, "
            f"min_score={min_score:.1f}, ctx={reqs.effective_context_needed}, "
            f"fc={reqs.needs_function_calling}, vision={reqs.needs_vision}, "
            f"thinking={reqs.needs_thinking}, local={reqs.local_only}"
        )

    return candidates


# ─── Convenience Selectors ───────────────────────────────────────────────────

def select_for_task(task: str, **kwargs) -> list[ScoredModel]:
    """Simplified selection by task name."""
    return select_model(ModelRequirements(task=task, **kwargs))


# ─── Task Classification ────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a task router for an AI agent system. Classify this task.
Respond ONLY with valid JSON, no markdown.

Available task types:
- "planner": goal decomposition, project planning, step ordering
- "architect": system design, API design, technology decisions
- "coder": writing new code from specs
- "implementer": following detailed implementation plans exactly
- "fixer": debugging, fixing errors, root cause analysis
- "test_generator": writing tests, edge case identification
- "reviewer": code review, quality analysis, critique
- "researcher": finding information, comparisons, documentation lookup
- "writer": prose, documentation, emails, reports
- "executor": running tools, file operations, simple transformations
- "router": simple classification, routing decisions
- "visual_reviewer": analyzing screenshots, UI review, diagram understanding
- "assistant": general conversation, Q&A, personal assistance
- "summarizer": condensing long content, extracting key points

Determine:
- task_type: best matching type from above
- "min_score" (1-10): minimum model quality needed.
  1-3: trivial (definitions, formatting, classification)
  4-6: moderate (standard code, summaries, Q&A)
  7-8: complex (multi-file refactoring, architecture, deep analysis)
  9-10: critical (production decisions, novel algorithms, security audits)
- needs_tools: does this need to execute actions (files, shell, search)?
- needs_vision: does this need to look at images/screenshots?
- needs_thinking: does this need deep multi-step reasoning?
- local_only: personal/sensitive data that shouldn't go to cloud?
- "priority": "critical" | "high" | "normal" | "low" | "background"
  critical = user actively waiting for immediate response
  high = important, should run soon
  normal = standard background work
  low = can wait, nice-to-have
  background = scheduled maintenance, optional
  
BIAS: Most tasks need min_score 4-6. Only use 8+ for genuinely complex work.
Default to needs_tools=false unless the task clearly requires execution.
Default to local_only=false unless personal data is explicitly mentioned.

Task: {task_description}

Respond as: {{"task_type": "coding", "min_score": 6, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "reasoning": "brief explanation"}}"""


async def classify_task(title: str, description: str) -> ModelRequirements:
    """Classify a task and return structured ModelRequirements."""
    classifier_reqs = ModelRequirements(
        task="router",
        min_quality=1,
        estimated_input_tokens=500,
        estimated_output_tokens=150,
        prefer_speed=True,
    )

    candidates = select_model(classifier_reqs)
    if not candidates:
        return _keyword_classify(title, description)

    classifier_model = candidates[0].model

    try:
        if classifier_model.is_local and not classifier_model.is_loaded:
            cloud_candidates = [c for c in candidates if not c.model.is_local]
            if cloud_candidates:
                classifier_model = cloud_candidates[0].model

        # Rate limiting via two-tier system
        if not classifier_model.is_local:
            rl_manager = get_rate_limit_manager()
            await rl_manager.wait_and_acquire(
                litellm_name=classifier_model.litellm_name,
                provider=classifier_model.provider,
                estimated_tokens=650,  # input + output estimate
            )

        completion_kwargs = dict(
            model=classifier_model.litellm_name,
            messages=[{
                "role": "user",
                "content": ROUTER_PROMPT.format(
                    task_description=f"{title}: {description[:500]}"
                ),
            }],
            max_tokens=200,
            temperature=0,
        )
        if classifier_model.api_base:
            completion_kwargs["api_base"] = classifier_model.api_base

        response = await asyncio.wait_for(
            litellm.acompletion(**completion_kwargs),
            timeout=30,
        )

        # Record token usage
        if not classifier_model.is_local and response.usage:
            total_tokens = (
                (response.usage.prompt_tokens or 0)
                + (response.usage.completion_tokens or 0)
            )
            get_rate_limit_manager().record_tokens(
                classifier_model.litellm_name,
                classifier_model.provider,
                total_tokens,
            )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        result = json.loads(raw)

        priority_map = {
            "critical": 10, "high": 8, "normal": 5, "low": 3, "background": 1,
        }

        task_type = result.get("task_type", "assistant")
        min_score = result.get("min_score", 5)

        reqs = ModelRequirements(
            task=result.get("task_type", "assistant"),
            primary_capability=result.get("task_type", "general"),
            min_score=min_score,
            min_quality=min_score,
            needs_function_calling=result.get("needs_tools", False),
            needs_vision=result.get("needs_vision", False),
            needs_thinking=result.get("needs_thinking", False),
            local_only=result.get("local_only", False),
            priority=priority_map.get(result.get("priority", "normal"), 5),
        )

        logger.info(
            f"Classified: task={task_type}, min_q={min_score}, "
            f"tools={reqs.needs_function_calling}, "
            f"vision={reqs.needs_vision}, "
            f"thinking={reqs.needs_thinking}, "
            f"local={reqs.local_only}, "
            f"priority={reqs.priority} — "
            f"{result.get('reasoning', '')[:60]}"
        )
        return reqs

    except Exception as e:
        # Record 429 for adaptive rate limiting
        error_str = str(e).lower()
        if any(kw in error_str for kw in [
            "rate limit", "429", "too many requests",
            "resource_exhausted",
        ]):
            try:
                get_rate_limit_manager().record_429(
                    classifier_model.litellm_name,
                    classifier_model.provider,
                )
            except Exception:
                pass

        logger.warning(
            f"Classification failed ({e}), using keyword fallback"
        )
        return _keyword_classify(title, description)


def _keyword_classify(title: str, description: str) -> ModelRequirements:
    """Fast keyword-based classification."""
    try:
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(title, description)
        category = result["category"]

        category_map = {
            "simple_qa":       ModelRequirements(task="executor",   min_quality=3, prefer_speed=True),
            "code_simple":     ModelRequirements(task="coder",      min_quality=5, needs_function_calling=True),
            "code_complex":    ModelRequirements(task="coder",      min_quality=7, needs_function_calling=True, prefer_quality=True),
            "research":        ModelRequirements(task="researcher", min_quality=6, needs_function_calling=True),
            "writing":         ModelRequirements(task="writer",     min_quality=6),
            "planning":        ModelRequirements(task="planner",    min_quality=7, prefer_quality=True),
            "action_required": ModelRequirements(task="executor",   min_quality=5, needs_function_calling=True),
            "sensitive":       ModelRequirements(task="assistant",  min_quality=5, local_only=True),
        }

        return category_map.get(category, ModelRequirements(task="assistant"))
    except ImportError:
        pass

    text = f"{title} {description}".lower()
    if any(kw in text for kw in ["fix", "bug", "error", "debug", "traceback"]):
        return ModelRequirements(task="fixer", min_quality=6, needs_function_calling=True)
    if any(kw in text for kw in ["implement", "create", "build", "write code"]):
        return ModelRequirements(task="coder", min_quality=6, needs_function_calling=True)
    if any(kw in text for kw in ["test", "spec", "coverage"]):
        return ModelRequirements(task="test_generator", min_quality=5, needs_function_calling=True)
    if any(kw in text for kw in ["review", "analyze", "audit"]):
        return ModelRequirements(task="reviewer", min_quality=6)
    if any(kw in text for kw in ["plan", "design", "architect"]):
        return ModelRequirements(task="planner", min_quality=7, prefer_quality=True)
    if any(kw in text for kw in ["screenshot", "image", "visual", "ui"]):
        return ModelRequirements(task="visual_reviewer", needs_vision=True, min_quality=5)
    if any(kw in text for kw in ["write", "document", "email", "report"]):
        return ModelRequirements(task="writer", min_quality=5)
    if any(kw in text for kw in ["search", "find", "research"]):
        return ModelRequirements(task="researcher", min_quality=5, needs_function_calling=True)
    if any(kw in text for kw in ["summarize", "tldr", "key points"]):
        return ModelRequirements(task="summarizer", min_quality=5)
    return ModelRequirements(task="assistant", min_quality=5)


# ─── Main API: call_model ────────────────────────────────────────────────────

async def call_model(
    reqs_or_tier,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    *,
    # Legacy kwargs (used when first arg is a tier string)
    agent_type: str = "",
    model_override: str | None = None,
    stream: bool = False,
    # New-style: reqs + messages as first two positional args
    **kwargs,
) -> dict:
    """
    Call the best available model matching requirements.

    Supports BOTH calling conventions:

    New style:
        await call_model(reqs=ModelRequirements(...), messages=[...], tools=[...])

    Legacy style (tier string):
        await call_model("medium", messages, agent_type="coder")
        await call_model("code", messages, agent_type="fixer", model_override="...")

    Legacy style auto-converts to ModelRequirements internally.
    """
    # ── Detect calling convention ──
    if isinstance(reqs_or_tier, str):
        # Legacy: call_model(tier, messages, agent_type=..., model_override=...)
        tier_str = reqs_or_tier
        reqs = ModelRequirements.from_tier(
            tier_str,
            agent_type=agent_type,
            model_override=model_override,
        )
        # messages is the second positional arg
        if messages is None:
            raise ValueError("messages required when using tier string")
    elif isinstance(reqs_or_tier, ModelRequirements):
        reqs = reqs_or_tier
        # messages may be second positional or keyword
        if messages is None:
            raise ValueError("messages required")
    else:
        raise TypeError(
            f"call_model() first argument must be ModelRequirements or tier string, "
            f"got {type(reqs_or_tier)}"
        )

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
            min_quality=1,
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
        temperature = None if is_thinking else 0.3

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
            if wait_time > 0:
                logger.info(
                    f"Rate limiter waited {wait_time:.1f}s for "
                    f"{model.name}"
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
                    f"GPU access denied for {model.name} "
                    f"(priority={reqs.priority}, queue too deep) "
                    f"— trying next candidate"
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
                        f"Calling {model.name} "
                        f"(task={task_label}, "
                        f"cap={scored.capability_score:.1f}, "
                        f"attempt={attempt+1}/{max_retries}"
                        f"{'|thinking' if is_thinking else ''}"
                        f"{'|vision' if model.has_vision else ''})"
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
                        # Legacy compat: agents read "tier" from response
                        "tier": reqs.tier,
                    }

                except asyncio.TimeoutError:
                    if not model.is_local:
                        _get_circuit_breaker(model.provider).record_failure()
                    last_error = f"Timeout on {model.name}"
                    logger.warning(f"Timeout on {model.name} (attempt {attempt+1})")
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
                        logger.error(f"Auth/billing error on {model.name}: {e}")
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
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            logger.warning(
                                f"Rate limited on {model.name}, "
                                f"waiting {wait_time}s"
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
        f"All models failed for {call_id} — "
        f"submitting to backpressure queue. Last error: {last_error}"
    )


    # Create a retry callable that re-runs selection + call
    async def _retry_call():
        # On retry, refresh perf cache and try again
        await refresh_perf_cache()
        # Recursive call — but with a flag to prevent infinite backpressure
        reqs._is_retry = True
        return await call_model(reqs, messages, tools, stream)

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
) -> float | None:
    """Grade a response using a DIFFERENT model."""
    if not response_text or len(response_text.strip()) < 10:
        return None

    try:
        grading_reqs = ModelRequirements(
            task="reviewer",
            min_quality=3,
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
            task_name = result.get("task", "")
            if task_name and task_name in TASK_PROFILES:
                profile = TASK_PROFILES[task_name]
                dominant = max(profile.items(), key=lambda x: x[1])
                cap_key = dominant[0].value if hasattr(dominant[0], 'value') else dominant[0]
                registry.update_quality_from_grading(model_name, cap_key, grade * 2.0)

        return grade

    except Exception as e:
        logger.debug(f"Response grading failed: {e}")
        return None


# ─── Cost Budget ─────────────────────────────────────────────────────────────

async def check_cost_budget() -> dict:
    try:
        from ..infra.db import check_budget
        return await check_budget("daily")
    except Exception as e:
        return {"ok": True, "reason": f"check failed: {e}"}


# ─── Backward Compatibility ─────────────────────────────────────────────────

_model_tiers_cache: dict | None = None
_classifier_model_cache: str | None = None


def _build_compat_tiers():
    global _model_tiers_cache, _classifier_model_cache
    if _model_tiers_cache is not None:
        return

    _model_tiers_cache = {}
    for tname in TIER_ESCALATION_ORDER + ["routing"]:
        treqs = ModelRequirements.from_tier(tname)
        tcandidates = select_model(treqs)
        if tcandidates:
            top = tcandidates[0]
            _model_tiers_cache[tname] = {
                "model": top.litellm_name,
                "fallbacks": [c.litellm_name for c in tcandidates[1:4]],
                "max_tokens": top.model.max_tokens,
                "temperature": 0.0 if tname == "routing" else 0.3,
                "description": tname,
            }

    classifier_candidates = select_model(
        ModelRequirements(task="router", min_quality=1, prefer_speed=True)
    )
    _classifier_model_cache = (
        classifier_candidates[0].litellm_name
        if classifier_candidates
        else "groq/llama-3.1-8b-instant"
    )


class _LazyTiers:
    """Lazy proxy for MODEL_TIERS backward compat."""
    def _ensure(self):
        _build_compat_tiers()

    def __getitem__(self, key):
        self._ensure()
        return _model_tiers_cache[key]

    def __contains__(self, key):
        self._ensure()
        return key in _model_tiers_cache

    def __iter__(self):
        self._ensure()
        return iter(_model_tiers_cache)

    def items(self):
        self._ensure()
        return _model_tiers_cache.items()

    def keys(self):
        self._ensure()
        return _model_tiers_cache.keys()

    def values(self):
        self._ensure()
        return _model_tiers_cache.values()

    def get(self, key, default=None):
        self._ensure()
        return _model_tiers_cache.get(key, default)


class _LazyClassifier:
    """Lazy proxy for CLASSIFIER_MODEL."""
    def __str__(self):
        _build_compat_tiers()
        return _classifier_model_cache

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def split(self, *args, **kwargs):
        return str(self).split(*args, **kwargs)

    def lower(self):
        return str(self).lower()

    def startswith(self, *args):
        return str(self).startswith(*args)

    def endswith(self, *args):
        return str(self).endswith(*args)

    def __contains__(self, item):
        return item in str(self)


MODEL_TIERS = _LazyTiers()
CLASSIFIER_MODEL = _LazyClassifier()


# ─── Agent Requirement Templates ─────────────────────────────────────────────

AGENT_REQUIREMENTS: dict[str, ModelRequirements] = {
    "planner":        ModelRequirements(task="planner",        min_quality=7, estimated_output_tokens=2000, prefer_quality=True, needs_json_mode=True),
    "architect":      ModelRequirements(task="architect",      min_quality=7, estimated_output_tokens=3000, prefer_quality=True),
    "coder":          ModelRequirements(task="coder",          min_quality=6, estimated_output_tokens=4000, needs_function_calling=True),
    "implementer":    ModelRequirements(task="implementer",    min_quality=5, estimated_output_tokens=4000, needs_function_calling=True),
    "fixer":          ModelRequirements(task="fixer",          min_quality=6, estimated_output_tokens=3000, needs_function_calling=True),
    "test_generator": ModelRequirements(task="test_generator", min_quality=5, estimated_output_tokens=3000, needs_function_calling=True),
    "reviewer":       ModelRequirements(task="reviewer",       min_quality=6, estimated_output_tokens=2000),
    "researcher":     ModelRequirements(task="researcher",     min_quality=5, estimated_output_tokens=2000, needs_function_calling=True),
    "writer":         ModelRequirements(task="writer",         min_quality=5, estimated_output_tokens=3000),
    "executor":       ModelRequirements(task="executor",       min_quality=3, estimated_output_tokens=1000, needs_function_calling=True, prefer_speed=True),
    "visual_reviewer": ModelRequirements(task="visual_reviewer", min_quality=5, estimated_output_tokens=2000, needs_vision=True),
    "assistant":      ModelRequirements(task="assistant",      min_quality=5, estimated_output_tokens=2000),
}


def get_agent_requirements(agent_type: str, **overrides) -> ModelRequirements:
    """Get pre-built requirements for an agent type with optional overrides."""
    template = AGENT_REQUIREMENTS.get(agent_type)
    if template:
        reqs = copy.copy(template)
    else:
        task = CAPABILITY_TO_TASK.get(agent_type, "assistant")
        reqs = ModelRequirements(task=task)
    reqs.agent_type = agent_type
    for key, value in overrides.items():
        if hasattr(reqs, key):
            setattr(reqs, key, value)
    return reqs


# Legacy import compat — AGENT_TIER_MAP maps agent names to tier strings
AGENT_TIER_MAP: dict[str, str] = {
    "planner":        "expensive",
    "architect":      "expensive",
    "coder":          "code",
    "implementer":    "code",
    "fixer":          "code",
    "test_generator": "code",
    "reviewer":       "medium",
    "researcher":     "medium",
    "writer":         "medium",
    "executor":       "cheap",
}
