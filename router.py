# router.py (rewritten)
"""
Model router — requirements-based model selection, rate limiting,
retries, cross-provider fallback, and GPU-aware scheduling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import litellm
litellm.suppress_debug_info = True

from config import THINKING_MODELS, COST_BUDGET_DAILY
from model_registry import ModelInfo, get_registry
from gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


# ─── Model Requirements ─────────────────────────────────────────────────────

@dataclass
class ModelRequirements:
    """
    Structured description of what a task needs from a model.
    Built by agents/pipeline, consumed by select_model.

    This replaces the flat 'tier' string with rich, queryable requirements.
    """
    # What the task needs to do
    primary_capability: str = "general"        # "coding", "reasoning", "planning", etc.
    secondary_capabilities: list[str] = field(default_factory=list)

    # Quality floor (1-10). Minimum acceptable quality for primary_capability.
    min_quality: int = 1

    # Context requirements
    estimated_input_tokens: int = 2000         # estimated prompt size
    estimated_output_tokens: int = 1000        # expected completion size
    min_context_length: int = 0                # 0 = auto-calculate from estimates

    # Feature requirements
    needs_function_calling: bool = False
    needs_json_mode: bool = False
    needs_thinking: bool = False

    # Privacy / location constraints
    local_only: bool = False                   # for personal/sensitive data

    # Speed vs quality preference
    prefer_speed: bool = False                 # True = prioritize fast models
    prefer_quality: bool = False               # True = prioritize best quality

    # Budget constraint for this call
    max_cost: float = 0.0                      # 0 = no limit

    # Task priority (from TASK_PRIORITY)
    priority: int = 5                          # 10=critical(user waiting), 1=background

    # Model diversity — avoid these litellm names (for review loops)
    exclude_models: list[str] = field(default_factory=list)

    # Agent context for performance lookup
    agent_type: str = ""

    # Direct model pin (escape hatch)
    model_override: str | None = None

    @property
    def effective_context_needed(self) -> int:
        """Calculate minimum context window needed."""
        if self.min_context_length > 0:
            return self.min_context_length
        # 1.3x multiplier for safety margin
        return int((self.estimated_input_tokens + self.estimated_output_tokens) * 1.3)


# ─── Per-Provider Rate Limiting ──────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter tracking both RPM and TPM."""

    def __init__(self, rpm: int = 30, tpm: int = 100000):
        self.rpm = rpm
        self.tpm = tpm
        self._request_timestamps: list[float] = []
        self._token_log: list[tuple[float, int]] = []  # (timestamp, token_count)

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
        """Check if there's capacity for a request without waiting."""
        return self.rpm_headroom > 2 and self.tpm_headroom > estimated_tokens

    async def wait(self) -> None:
        """Wait until rate limit allows a new request."""
        now = time.time()
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]

        if len(self._request_timestamps) >= self.rpm:
            wait_time = 60 - (now - self._request_timestamps[0]) + 0.5
            logger.info(f"Rate limiter: waiting {wait_time:.1f}s (RPM)")
            await asyncio.sleep(wait_time)

        self._request_timestamps.append(time.time())

    def record_tokens(self, token_count: int) -> None:
        """Record token usage after a call completes."""
        now = time.time()
        self._token_log.append((now, token_count))
        # Clean old entries
        self._token_log = [(t, c) for t, c in self._token_log if now - t < 60]


class CircuitBreaker:
    """Track failures per provider and temporarily disable."""

    def __init__(
        self, failure_threshold: int = 3,
        window_seconds: float = 300, cooldown_seconds: float = 600,
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


# Per-provider instances
_rate_limiters: dict[str, RateLimiter] = {}
_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_limiter(provider: str, rpm: int = 30, tpm: int = 100000) -> RateLimiter:
    if provider not in _rate_limiters:
        _rate_limiters[provider] = RateLimiter(rpm, tpm)
    return _rate_limiters[provider]


def _get_circuit_breaker(provider: str) -> CircuitBreaker:
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker()
    return _circuit_breakers[provider]


# Initialize limiters from registry at first use
_limiters_initialized = False


def _init_limiters():
    global _limiters_initialized
    if _limiters_initialized:
        return
    registry = get_registry()
    for m in registry.cloud_models():
        if m.provider not in _rate_limiters:
            _rate_limiters[m.provider] = RateLimiter(
                m.rate_limit_rpm, m.rate_limit_tpm
            )
    _limiters_initialized = True


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
        from db import get_model_stats
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


# ─── Model Selection (Core Redesign) ────────────────────────────────────────

@dataclass
class ScoredModel:
    """A model candidate with its selection score and reasoning."""
    model: ModelInfo
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def litellm_name(self) -> str:
        return self.model.litellm_name


def select_model(reqs: ModelRequirements) -> list[ScoredModel]:
    """
    Select models matching requirements, ranked by composite score.

    Scoring dimensions (all normalized to 0-100 range, then weighted):

    1. CAPABILITY FIT (weight: 35)
       How well the model matches the required capabilities.

    2. COST EFFICIENCY (weight: 25)
       Prefer free/local over paid. Prefer cheaper paid models.

    3. AVAILABILITY (weight: 20)
       Rate limit headroom, circuit breaker status, model loaded status.

    4. PERFORMANCE HISTORY (weight: 15)
       Success rate and quality grades from past calls with this agent_type.

    5. SPEED (weight: 5)
       Estimated tokens/second, latency class.
    """
    _init_limiters()
    registry = get_registry()

    candidates: list[ScoredModel] = []

    for name, model in registry.models.items():
        reasons: list[str] = []
        skip = False

        # ── Hard filters (instant rejection) ──

        # Excluded models (diversity enforcement)
        if model.litellm_name in reqs.exclude_models:
            continue

        # Local-only constraint
        if reqs.local_only and not model.is_local:
            continue

        # Context length
        needed_ctx = reqs.effective_context_needed
        if needed_ctx > 0 and model.context_length < needed_ctx:
            continue

        # Function calling required but not supported
        if reqs.needs_function_calling and not model.supports_function_calling:
            continue

        # Thinking required but not available
        if reqs.needs_thinking and not model.thinking_model:
            continue

        # Cost constraint
        if reqs.max_cost > 0 and not model.is_free:
            est_cost = model.estimated_cost(
                reqs.estimated_input_tokens, reqs.estimated_output_tokens
            )
            if est_cost > reqs.max_cost:
                continue

        # Primary capability minimum quality
        primary_q = model.quality_for(reqs.primary_capability)
        # Fall back to general quality if specific capability not listed
        if primary_q == 0:
            primary_q = model.quality_for("general")
        if primary_q < reqs.min_quality:
            continue

        # Circuit breaker (cloud only)
        if not model.is_local:
            cb = _get_circuit_breaker(model.provider)
            if cb.is_degraded:
                continue

        # Local model: must be loaded OR loadable
        if model.is_local and not model.is_loaded:
            # Check if we CAN load it (file exists, already verified by registry)
            # The actual loading happens in call_model, not here
            # But we penalize unloaded models in scoring
            pass

        # ── Scoring ──

        # 1. CAPABILITY FIT (0-100)
        cap_score = primary_q * 10  # 0-100
        # Bonus for secondary capabilities
        for sec_cap in reqs.secondary_capabilities:
            sec_q = model.quality_for(sec_cap)
            if sec_q > 0:
                cap_score += sec_q * 2  # up to +20 per secondary
                reasons.append(f"{sec_cap}={sec_q}")
        cap_score = min(cap_score, 100)
        reasons.insert(0, f"primary({reqs.primary_capability})={primary_q}")

        # 2. COST EFFICIENCY (0-100)
        if model.is_local:
            cost_score = 95  # local is almost free
            if not model.is_loaded:
                # Penalize unloaded: swap costs time
                cost_score = 70
                reasons.append("needs_swap")
            reasons.append("local")
        elif model.is_free:
            cost_score = 85  # free cloud is great
            reasons.append("free_cloud")
        else:
            # Paid: score inversely proportional to cost
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
            reasons.append(f"est_cost=${est_cost:.4f}")

        # 3. AVAILABILITY (0-100)
        if model.is_local:
            if model.is_loaded:
                avail_score = 100
                reasons.append("loaded")
            else:
                # Not loaded — estimate swap time penalty
                swap_time = model.load_time_seconds
                if swap_time < 10:
                    avail_score = 60
                elif swap_time < 30:
                    avail_score = 40
                else:
                    avail_score = 20
                reasons.append(f"swap_{swap_time:.0f}s")
        else:
            limiter = _get_limiter(
                model.provider, model.rate_limit_rpm, model.rate_limit_tpm
            )
            total_tokens = reqs.estimated_input_tokens + reqs.estimated_output_tokens
            if limiter.has_capacity(total_tokens):
                avail_score = 90
            elif limiter.rpm_headroom > 0:
                avail_score = 50
                reasons.append("low_headroom")
            else:
                avail_score = 10
                reasons.append("rate_limited")

        # 4. PERFORMANCE HISTORY (0-100)
        perf_score = 50  # neutral default
        if reqs.agent_type and _perf_cache_ready:
            agent_perf = _perf_cache.get(reqs.agent_type, {})
            model_perf = agent_perf.get(model.litellm_name)
            if model_perf and model_perf.get("total_calls", 0) >= 3:
                sr = model_perf["success_rate"]
                grade = model_perf.get("avg_grade", 3.0)
                perf_score = (sr * grade / 5.0) * 100
                perf_score = max(0, min(100, perf_score))
                reasons.append(
                    f"perf(sr={sr:.2f},g={grade:.1f},n={model_perf['total_calls']})"
                )

        # 5. SPEED (0-100)
        if model.is_local:
            tps = model.tokens_per_second
            if tps >= 50:
                speed_score = 100
            elif tps >= 20:
                speed_score = 70
            else:
                speed_score = 40
        else:
            # Cloud speed classes (based on provider reputation)
            speed_map = {
                "groq": 95, "cerebras": 95,       # inference-optimized
                "sambanova": 80,
                "gemini": 70,
                "openai": 60,
                "anthropic": 50,
            }
            speed_score = speed_map.get(model.provider, 50)

        # ── Composite score with weights ──
        weights = {
            "capability": 35,
            "cost": 25,
            "availability": 20,
            "performance": 15,
            "speed": 5,
        }

        # Adjust weights based on requirements
        if reqs.prefer_quality:
            weights["capability"] = 50
            weights["cost"] = 10
        elif reqs.prefer_speed:
            weights["speed"] = 25
            weights["capability"] = 25
            weights["availability"] = 25
            weights["cost"] = 15
            weights["performance"] = 10

        # Critical priority: maximize availability + quality
        if reqs.priority >= 10:
            weights["availability"] = 30
            weights["capability"] = 35
            weights["speed"] = 15
            weights["cost"] = 10
            weights["performance"] = 10

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
            reasons=reasons,
        ))

    # Sort descending by score
    candidates.sort(key=lambda c: -c.score)

    # Log selection
    if candidates:
        top3 = candidates[:3]
        log_parts = [
            f"{c.model.name}({c.score:.1f}|{','.join(c.reasons[:3])})"
            for c in top3
        ]
        logger.info(
            f"Model selection for {reqs.primary_capability}"
            f"{'[local_only]' if reqs.local_only else ''}: "
            f"{' > '.join(log_parts)}"
            f"{f' (+{len(candidates)-3} more)' if len(candidates) > 3 else ''}"
        )
    else:
        logger.warning(
            f"No models match requirements: cap={reqs.primary_capability}, "
            f"min_q={reqs.min_quality}, ctx={reqs.effective_context_needed}, "
            f"fc={reqs.needs_function_calling}, local={reqs.local_only}"
        )

    return candidates


# ─── Classification (Lightweight) ────────────────────────────────────────────

ROUTER_PROMPT = """You are a task router. Classify this task into requirements.
Respond ONLY with valid JSON, no markdown.

Categories for primary_capability:
- "routing": classification only
- "simple": factual questions, definitions, conversions
- "coding": write/debug/refactor code
- "reasoning": multi-step logic, math, analysis
- "planning": architecture, project planning, decomposition
- "writing": prose, documentation, emails
- "research": finding information, comparisons

Also determine:
- min_quality (1-10): how smart the model needs to be
- needs_tools (bool): does this task require executing actions?
- local_only (bool): does this involve personal/private data?

BIAS: Most tasks need min_quality 5-7. Only use 8+ for complex architecture,
multi-file refactoring, or critical decisions.

Task: {task_description}

Respond as: {{"primary_capability": "coding", "min_quality": 6, "needs_tools": true, "local_only": false, "reasoning": "brief"}}"""


async def classify_task(title: str, description: str) -> ModelRequirements:
    """
    Classify a task and return structured ModelRequirements.
    Uses cheapest available model. Falls back to keyword heuristic.
    """
    # Build minimal requirements for the classifier itself
    classifier_reqs = ModelRequirements(
        primary_capability="routing",
        min_quality=1,
        estimated_input_tokens=500,
        estimated_output_tokens=100,
        prefer_speed=True,
    )

    candidates = select_model(classifier_reqs)
    if not candidates:
        return _keyword_classify(title, description)

    classifier_model = candidates[0].model

    try:
        # If local model and not loaded, use cloud for classification
        # (don't swap just for routing)
        if classifier_model.is_local and not classifier_model.is_loaded:
            cloud_candidates = [c for c in candidates if not c.model.is_local]
            if cloud_candidates:
                classifier_model = cloud_candidates[0].model

        limiter = _get_limiter(
            classifier_model.provider,
            classifier_model.rate_limit_rpm,
            getattr(classifier_model, 'rate_limit_tpm', 100000),
        )
        await limiter.wait()

        completion_kwargs = dict(
            model=classifier_model.litellm_name,
            messages=[{
                "role": "user",
                "content": ROUTER_PROMPT.format(
                    task_description=f"{title}: {description[:500]}"
                ),
            }],
            max_tokens=150,
            temperature=0,
        )
        if classifier_model.api_base:
            completion_kwargs["api_base"] = classifier_model.api_base

        response = await asyncio.wait_for(
            litellm.acompletion(**completion_kwargs),
            timeout=30,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        result = json.loads(raw)

        reqs = ModelRequirements(
            primary_capability=result.get("primary_capability", "general"),
            min_quality=result.get("min_quality", 5),
            needs_function_calling=result.get("needs_tools", False),
            local_only=result.get("local_only", False),
        )

        logger.info(
            f"Classified: cap={reqs.primary_capability}, "
            f"min_q={reqs.min_quality}, "
            f"tools={reqs.needs_function_calling}, "
            f"local={reqs.local_only} — "
            f"{result.get('reasoning', '')[:60]}"
        )
        return reqs

    except Exception as e:
        logger.warning(f"Classification failed ({e}), using keyword fallback")
        return _keyword_classify(title, description)


def _keyword_classify(title: str, description: str) -> ModelRequirements:
    """Fast keyword-based classification — no LLM call needed."""
    text = f"{title} {description}".lower()

    # Import the keyword rules from task_classifier
    from task_classifier import _classify_by_keywords
    result = _classify_by_keywords(title, description)
    category = result["category"]

    # Map category → ModelRequirements
    category_map = {
        "simple_qa":        ModelRequirements(primary_capability="simple", min_quality=3),
        "code_simple":      ModelRequirements(primary_capability="coding", min_quality=5, needs_function_calling=True),
        "code_complex":     ModelRequirements(primary_capability="coding", min_quality=7, needs_function_calling=True, prefer_quality=True),
        "research":         ModelRequirements(primary_capability="research", min_quality=6, needs_function_calling=True),
        "writing":          ModelRequirements(primary_capability="writing", min_quality=6),
        "planning":         ModelRequirements(primary_capability="planning", min_quality=7, prefer_quality=True),
        "action_required":  ModelRequirements(primary_capability="general", min_quality=5, needs_function_calling=True),
        "sensitive":        ModelRequirements(primary_capability="general", min_quality=5, local_only=True),
    }

    reqs = category_map.get(category, ModelRequirements())
    logger.debug(f"Keyword classified: {category} → {reqs.primary_capability}")
    return reqs


# ─── Main API: call_model ────────────────────────────────────────────────────

async def call_model(
    reqs: ModelRequirements,
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = False,
) -> dict:
    """
    Call the best available model matching requirements.

    Handles: model selection, local model loading, rate limits,
    retries with fallback, GPU semaphore, thinking models,
    and function calling negotiation.
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
                    capabilities={"general": 5},
                    context_length=128000,
                    max_tokens=4096,
                ),
                score=999,
                reasons=["pinned_raw"],
            )]
    else:
        # If tools are provided, we either need FC support or JSON mode
        if tools:
            reqs.needs_function_calling = True
        candidates = select_model(reqs)

    if not candidates:
        # Last resort: drop constraints and try again
        fallback_reqs = ModelRequirements(
            primary_capability="general",
            min_quality=1,
            agent_type=reqs.agent_type,
        )
        candidates = select_model(fallback_reqs)

    if not candidates:
        raise RuntimeError("No models available!")

    last_error: str | None = None

    for scored in candidates[:5]:
        model = scored.model

        # ── Local model: ensure loaded ──
        if model.is_local:
            from local_model_manager import get_local_manager
            manager = get_local_manager()

            if not model.is_loaded:
                success = await manager.ensure_model(
                    model.name,
                    reason=f"{reqs.agent_type}:{reqs.primary_capability}",
                )
                if not success:
                    last_error = f"Failed to load local model {model.name}"
                    continue

        # ── Build completion kwargs ──
        is_thinking = model.thinking_model
        temperature = 0.3
        if is_thinking:
            temperature = None  # thinking models control their own temp

        timeout_val = 60
        if model.is_local:
            timeout_val = 120
        if is_thinking:
            timeout_val = max(timeout_val, 180)

        # Tool negotiation
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
        elif not model.supports_function_calling and model.supports_json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        # ── Rate limiting ──
        if not model.is_local:
            limiter = _get_limiter(
                model.provider, model.rate_limit_rpm,
                model.rate_limit_tpm,
            )
            await limiter.wait()

        # ── GPU semaphore for local models ──
        local_manager = None
        if model.is_local:
            from local_model_manager import get_local_manager
            local_manager = get_local_manager()
            await local_manager.acquire_inference_slot()

        max_retries = 2 if model.is_local else 3

        try:
            for attempt in range(max_retries):
                try:
                    call_start = time.time()

                    logger.info(
                        f"Calling {model.name} "
                        f"(cap={reqs.primary_capability}, "
                        f"q={model.quality_for(reqs.primary_capability)}, "
                        f"attempt={attempt+1}/{max_retries}"
                        f"{', thinking' if is_thinking else ''})"
                    )

                    response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=timeout_val,
                    )

                    call_latency = time.time() - call_start

                    # Calculate cost
                    try:
                        cost = litellm.completion_cost(completion_response=response)
                    except Exception:
                        cost = 0.0
                    if model.is_local:
                        cost = 0.0

                    # Record token usage for TPM tracking
                    if not model.is_local and response.usage:
                        total_tokens = (
                            (response.usage.prompt_tokens or 0)
                            + (response.usage.completion_tokens or 0)
                        )
                        limiter.record_tokens(total_tokens)

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

                    # Extract thinking content
                    thinking_content = _extract_thinking(msg) if is_thinking else None

                    # Success — reset circuit breaker
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
                        "capability_quality": model.quality_for(reqs.primary_capability),
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

                    # Auth/billing → skip provider
                    if any(kw in error_str for kw in [
                        "api key", "authentication", "unauthorized",
                        "billing", "credit", "quota",
                    ]):
                        logger.error(f"Auth/billing error on {model.name}: {e}")
                        break

                    # Rate limit → backoff or skip
                    if any(kw in error_str for kw in [
                        "rate limit", "429", "too many requests",
                        "resource_exhausted",
                    ]):
                        if attempt < max_retries - 1:
                            wait = (attempt + 1) * 5
                            logger.warning(f"Rate limited on {model.name}, waiting {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        break

                    # Unknown error → one retry
                    if attempt < 1:
                        await asyncio.sleep(2)
                        continue
                    break

        finally:
            # Always release GPU semaphore
            if local_manager:
                local_manager.release_inference_slot()

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ─── Thinking Model Helpers ─────────────────────────────────────────────────

def _extract_thinking(msg) -> str | None:
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    content = msg.content or ""
    import re
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
    """
    Grade a response using a DIFFERENT model than the one that generated it.
    Uses cheapest available model that ISN'T the generating model.
    """
    if not response_text or len(response_text.strip()) < 10:
        return None

    try:
        grading_reqs = ModelRequirements(
            primary_capability="reasoning",
            min_quality=3,
            estimated_input_tokens=800,
            estimated_output_tokens=50,
            prefer_speed=True,
            exclude_models=[generating_model] if generating_model else [],
        )

        result = await call_model(
            reqs=grading_reqs,
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
        return max(1.0, min(5.0, score))

    except Exception as e:
        logger.debug(f"Response grading failed: {e}")
        return None


# ─── Cost Budget ─────────────────────────────────────────────────────────────

async def check_cost_budget() -> dict:
    try:
        from db import check_budget
        return await check_budget("daily")
    except Exception as e:
        return {"ok": True, "reason": f"check failed: {e}"}


# ─── Backward Compatibility Layer ────────────────────────────────────────────
# These map old tier-based calls to new requirements-based calls.
# Remove once all callers are migrated.

_TIER_TO_REQUIREMENTS: dict[str, ModelRequirements] = {
    "routing": ModelRequirements(primary_capability="routing", min_quality=1, prefer_speed=True),
    "cheap": ModelRequirements(primary_capability="general", min_quality=3),
    "code": ModelRequirements(primary_capability="coding", min_quality=5, needs_function_calling=True),
    "medium": ModelRequirements(primary_capability="general", min_quality=6),
    "expensive": ModelRequirements(primary_capability="reasoning", min_quality=8, prefer_quality=True),
}

# Build MODEL_TIERS and CLASSIFIER_MODEL for modules that still import them
MODEL_TIERS: dict[str, dict] = {}
for _tname, _treqs in _TIER_TO_REQUIREMENTS.items():
    _tcandidates = select_model(_treqs)
    if _tcandidates:
        _top = _tcandidates[0]
        MODEL_TIERS[_tname] = {
            "model": _top.litellm_name,
            "fallbacks": [c.litellm_name for c in _tcandidates[1:4]],
            "max_tokens": _top.model.max_tokens,
            "temperature": 0.0 if _tname == "routing" else 0.3,
            "description": _tname,
        }

_classifier_candidates = select_model(ModelRequirements(
    primary_capability="routing", min_quality=1, prefer_speed=True,
))
CLASSIFIER_MODEL: str = (
    _classifier_candidates[0].litellm_name
    if _classifier_candidates
    else "groq/llama-3.1-8b-instant"
)
