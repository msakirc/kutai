# router.py
"""
Model router — task classification, smart model selection,
rate limiting, retries, and cross-provider fallback.
"""
import json
import asyncio
import logging
import time

import litellm
litellm.suppress_debug_info = True

from config import (
    MODEL_POOL,
    FALLBACK_ORDER,
    COST_BUDGET_DAILY,
    THINKING_MODELS,
)

logger = logging.getLogger(__name__)


# ─── Per-Provider Rate Limiting ──────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter for a single provider."""

    def __init__(self, calls_per_minute: int = 25):
        self.calls_per_minute = calls_per_minute
        self.timestamps: list[float] = []

    @property
    def current_usage(self) -> int:
        now = time.time()
        return len([t for t in self.timestamps if now - t < 60])

    async def wait(self) -> None:
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < 60]

        if len(self.timestamps) >= self.calls_per_minute:
            wait_time = 60 - (now - self.timestamps[0]) + 0.5
            logger.info(
                f"Rate limiter: waiting {wait_time:.1f}s "
                f"({len(self.timestamps)}/{self.calls_per_minute} rpm)"
            )
            await asyncio.sleep(wait_time)

        self.timestamps.append(time.time())


# One limiter per provider, initialized from MODEL_POOL
_rate_limiters: dict[str, RateLimiter] = {}


# ─── Per-Provider Circuit Breaker ────────────────────────────────────────────

class CircuitBreaker:
    """Track consecutive failures per provider and temporarily disable."""

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
        # Keep only failures within the window
        self.failures = [t for t in self.failures if now - t < self.window_seconds]
        if len(self.failures) >= self.failure_threshold:
            self.degraded_until = now + self.cooldown_seconds
            logger.warning(
                f"Circuit breaker TRIPPED — provider degraded for "
                f"{self.cooldown_seconds:.0f}s "
                f"({len(self.failures)} failures in "
                f"{self.window_seconds:.0f}s window)"
            )

    def record_success(self) -> None:
        """Reset failures on a successful call."""
        self.failures.clear()
        self.degraded_until = 0.0

    @property
    def is_degraded(self) -> bool:
        if time.time() >= self.degraded_until:
            if self.degraded_until > 0:
                # Cooldown expired, reset
                self.degraded_until = 0.0
                self.failures.clear()
                logger.info("Circuit breaker RESET — provider recovered")
            return False
        return True


_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create a circuit breaker for *provider*."""
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker()
    return _circuit_breakers[provider]

def _get_limiter(provider: str, default_rpm: int = 30) -> RateLimiter:
    """Get or create a rate limiter for *provider*."""
    if provider not in _rate_limiters:
        _rate_limiters[provider] = RateLimiter(default_rpm)
    return _rate_limiters[provider]


def _get_limiter_for_model(litellm_name: str) -> RateLimiter:
    """Find the rate limiter for a litellm model name."""
    for cfg in MODEL_POOL.values():
        if cfg["litellm_name"] == litellm_name:
            return _get_limiter(cfg["provider"], cfg.get("rate_limit", 30))
    # Infer provider from name prefix
    provider = litellm_name.split("/")[0] if "/" in litellm_name else "unknown"
    return _get_limiter(provider, 20)


# Pre-initialize limiters for known providers
for _pool_cfg in MODEL_POOL.values():
    _prov = _pool_cfg["provider"]
    _rpm = _pool_cfg.get("rate_limit", 30)
    if _prov not in _rate_limiters:
        _rate_limiters[_prov] = RateLimiter(_rpm)


# ─── Classification ─────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a task router. Classify the given task into a complexity tier.
Respond ONLY with valid JSON, no markdown.

Tiers:
- "cheap": Simple/quick tasks — factual questions, formatting, lookups, \
classification, translations, simple math, definitions. \
MOST tasks should be cheap. When in doubt, choose cheap.
- "code": Code-related tasks — writing code, debugging, creating scripts, \
building features, fixing bugs, writing tests.
- "medium": Moderate tasks — multi-paragraph summaries, content drafting, \
data analysis, detailed explanations, comparisons, planning.
- "expensive": Complex tasks ONLY — multi-step reasoning, full project \
architecture, critical business decisions, creative strategy requiring nuance.

IMPORTANT: Bias heavily toward "cheap". Code tasks go to "code". \
Only use "expensive" if truly necessary.

Also determine if human approval is needed:
- needs_approval: true ONLY if task involves external actions, spending money, \
sending communications, or irreversible operations.

Task: {task_description}

Respond as: {{"tier": "cheap", "needs_approval": false, "reasoning": "brief"}}"""


async def classify_task(title: str, description: str) -> dict:
    """Use the cheapest model to classify a task's complexity tier."""
    limiter = _get_limiter_for_model(CLASSIFIER_MODEL)
    await limiter.wait()

    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=CLASSIFIER_MODEL,
                messages=[{
                    "role": "user",
                    "content": ROUTER_PROMPT.format(
                        task_description=f"{title}: {description[:500]}"
                    ),
                }],
                max_tokens=150,
                temperature=0,
            ),
            timeout=30,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        result = json.loads(raw)
        tier = _resolve_tier(result.get("tier", "cheap"))

                # Bump action-heavy tasks out of the weakest tier
        if tier == "cheap":
            text = f"{title} {description[:200]}".lower()

            # Strong verbs — always bump
            strong_verbs = [
                "fetch", "download", "install", "deploy", "execute",
                "run ", "run:", "clone", "setup", "set up", "configure",
                "compile", "test ", "debug", "create", "build", "scan",
                "scrape", "start ", "stop ", "restart", "launch",
                "migrate", "import ", "export ", "delete", "remove",
            ]
            # Contextual verbs — bump only when paired with tech targets
            context_verbs = [
                "list", "show", "check", "update", "read",
                "write", "open", "search", "analyze", "monitor",
                "get ", "find ",
            ]
            tech_targets = [
                "repo", "file", "folder", "directory", "server",
                "database", "api", "endpoint", "package", "container",
                "docker", "service", "script", "code", "project",
                "branch", "commit", "log", "port", "process",
                "dependencies", "config", "workspace", "module",
            ]

            needs_bump = any(v in text for v in strong_verbs)
            if not needs_bump:
                has_verb = any(v in text for v in context_verbs)
                has_target = any(t in text for t in tech_targets)
                needs_bump = has_verb and has_target

            if needs_bump:
                tier = _resolve_tier("code")
                logger.info(
                    f"Tier bumped: 'cheap' → '{tier}' (action task detected)"
                )

        logger.info(
            f"Classified: '{tier}' — "
            f"{result.get('reasoning', 'N/A')[:80]}"
        )
        return {
            "tier": tier,
            "needs_approval": result.get("needs_approval", False),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        logger.warning(f"Classification failed ({e}), defaulting to cheap")
        return {
            "tier": _get_cheapest_available_tier(),
            "needs_approval": False,
            "reasoning": f"Classification failed, defaulting: {e}",
        }


# ─── Tier Resolution ────────────────────────────────────────────────────────

def _resolve_tier(requested: str) -> str:
    """
    Map a requested tier to the best available tier.
    Supports: routing, cheap, code, medium, expensive.
    Falls back down the priority chain if the requested tier has no models.
    """
    if requested in MODEL_TIERS:
        return requested

    # Priority order for downward fallback
    priority = ["expensive", "medium", "code", "cheap", "routing"]
    try:
        start_idx = priority.index(requested)
    except ValueError:
        start_idx = 0

    # Walk down from requested tier
    for i in range(start_idx, len(priority)):
        if priority[i] in MODEL_TIERS:
            if priority[i] != requested:
                logger.info(
                    f"Tier '{requested}' unavailable, falling back to "
                    f"'{priority[i]}'"
                )
            return priority[i]

    return _get_cheapest_available_tier()


def _get_cheapest_available_tier() -> str:
    """Return the cheapest tier that has models configured."""
    for tier in reversed(["routing", "cheap", "code", "medium", "expensive"]):
        if tier in MODEL_TIERS:
            return tier
    return "cheap"


# ─── Smart Model Selection ──────────────────────────────────────────────────

def select_model(
    tier: str,
    capability: str | None = None,
    prefer_local: bool = True,
    agent_type: str | None = None,
    required_capabilities: list[str] | None = None,
    min_context_length: int | None = None,
    sensitivity: str | None = None,
) -> list[dict]:
    """
    Return a ranked list of models suitable for *tier* + optional *capability*.

    Phase 10 enhancements:
      - required_capabilities: filter to models that have ALL listed capabilities
      - min_context_length: skip models with context_length below this threshold
      - sensitivity: "public" | "private" | "secret" — restrict to local for
        private/secret data

    Ranking:
      1. Provider class (local > free cloud > paid)
      2. Historical performance for agent_type (Phase 4)
      3. Matches capability (if specified)
      4. Within tier quality range
      5. Higher quality wins ties
    """
    tier_ranges = {
        "routing":   (0, 5),
        "cheap":     (0, 6),
        "code":      (0, 99),   # any quality, filtered by capability below
        "medium":    (4, 8),
        "expensive": (7, 99),
    }
    q_min, q_max = tier_ranges.get(tier, (0, 99))

    # Phase 10.3: Local-only providers for sensitive data
    _local_providers = {"ollama", "llamacpp", "custom_openai"}
    restrict_local = sensitivity in ("private", "secret")

    # Fetch performance stats if agent_type is known
    perf_by_model: dict[str, dict] = {}
    if agent_type and _perf_cache_ready:
        perf_by_model = _perf_cache.get(agent_type, {})

    candidates: list[dict] = []

    for key, cfg in MODEL_POOL.items():
        quality = cfg["quality"]

        # Phase 10.3: Skip non-local models for sensitive data
        if restrict_local and cfg["provider"] not in _local_providers:
            continue

        # Phase 10.4: Required capabilities filter
        if required_capabilities:
            model_caps = set(cfg.get("capabilities", []))
            if not all(rc in model_caps for rc in required_capabilities):
                continue

        # Phase 10.4: Context length filter
        if min_context_length:
            model_ctx = cfg.get("context_length", 8192)
            if model_ctx < min_context_length:
                continue

        # For "code" tier, require coding capability
        if tier == "code" and "coding" not in cfg.get("capabilities", []):
            continue

        # For "routing" tier, prefer routing/classification capable
        if tier == "routing":
            has_routing = any(
                c in cfg.get("capabilities", [])
                for c in ("routing", "classification")
            )
            if not has_routing and quality > 5:
                continue   # skip expensive models for routing

        # Quality range filter
        if quality < q_min or quality > q_max:
            continue

        # Capability match scoring
        quality_adj = quality
        if capability and capability not in cfg.get("capabilities", []):
            quality_adj = quality - 2

        # Rate headroom check
        limiter = _get_limiter(cfg["provider"], cfg.get("rate_limit", 30))
        headroom = limiter.calls_per_minute - limiter.current_usage
        rate_ok = headroom > 2

        # Circuit breaker check — skip degraded providers
        cb = _get_circuit_breaker(cfg["provider"])
        if cb.is_degraded:
            logger.debug(
                f"Skipping {key} — provider '{cfg['provider']}' is degraded"
            )
            continue

        # Provider priority class:
        FREE_CLOUD = {"groq", "cerebras", "sambanova", "gemini"}
        if cfg["provider"] == "ollama":
            provider_bonus = 200
        elif cfg["provider"] in FREE_CLOUD:
            provider_bonus = 100
        else:
            provider_bonus = 0

        score = provider_bonus + (quality_adj * 10)
        if rate_ok:
            score += 5
        else:
            score -= 30

        # ── Performance-aware bonus (Phase 4) ──
        # If we have historical stats for this model + agent_type,
        # adjust score based on success_rate and avg_grade.
        perf = perf_by_model.get(cfg["litellm_name"])
        if perf and perf.get("total_calls", 0) >= 3:
            # Composite: success_rate (0-1) * avg_grade (0-5) → 0-5
            perf_score = perf["success_rate"] * perf.get("avg_grade", 3.0)
            # Scale to ±20 points centered at 3.0 (neutral)
            perf_bonus = (perf_score - 3.0) * 8
            score += perf_bonus
            logger.debug(
                f"  {key}: perf_bonus={perf_bonus:+.1f} "
                f"(sr={perf['success_rate']:.2f}, "
                f"grade={perf.get('avg_grade', 0):.1f}, "
                f"calls={perf['total_calls']})"
            )

        candidates.append({
            "key": key,
            "litellm_name": cfg["litellm_name"],
            "provider": cfg["provider"],
            "max_tokens": cfg.get("max_tokens", 2048),
            "rate_limit": cfg.get("rate_limit", 30),
            "quality": quality,
            "score": score,
            "api_base": cfg.get("api_base"),
        })

    candidates.sort(key=lambda c: -c["score"])

    # Debug: show which models were selected and their scores
    if candidates:
        model_list = [
            f"{c['litellm_name']}(s={c['score']:.0f})"
            for c in candidates
        ]
        logger.debug(f"select_model for tier '{tier}': {model_list}")

    return candidates


# ─── Performance Cache ──────────────────────────────────────────────────────
# In-memory cache of model performance stats, refreshed periodically.
# Avoids hitting DB on every select_model call.

_perf_cache: dict[str, dict[str, dict]] = {}
_perf_cache_ready: bool = False
_perf_cache_last_refresh: float = 0.0
_PERF_CACHE_TTL: float = 300.0  # refresh every 5 minutes


async def refresh_perf_cache() -> None:
    """Refresh the in-memory performance cache from DB."""
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
        logger.debug(f"Performance cache refreshed: {len(stats)} entries")
    except Exception as e:
        logger.debug(f"Performance cache refresh failed: {e}")

def _get_fallback_models(tier: str) -> list[dict]:
    """
    Build a full fallback chain: tier candidates first, then adjacent tiers.
    """
    seen_names: set[str] = set()
    chain: list[dict] = []

    # Primary: models for the requested tier
    for c in select_model(tier):
        if c["litellm_name"] not in seen_names:
            chain.append(c)
            seen_names.add(c["litellm_name"])

    # Secondary: models from tier config fallbacks (if any)
    tier_cfg = MODEL_TIERS.get(tier, {})
    for fb_name in tier_cfg.get("fallbacks", []):
        if fb_name not in seen_names:
            # Find pool entry
            for cfg in MODEL_POOL.values():
                if cfg["litellm_name"] == fb_name:
                    chain.append({
                        "litellm_name": fb_name,
                        "provider": cfg["provider"],
                        "max_tokens": cfg.get("max_tokens", 2048),
                        "rate_limit": cfg.get("rate_limit", 30),
                        "quality": cfg["quality"],
                        "score": 0,
                        "api_base": cfg.get("api_base"),
                    })
                    seen_names.add(fb_name)
                    break

    # Tertiary: walk down FALLBACK_ORDER for adjacent tiers
    priority = FALLBACK_ORDER   # ["expensive", "medium", "code", "cheap", "routing"]
    try:
        tier_idx = priority.index(tier)
    except ValueError:
        tier_idx = 0

    for i in range(tier_idx + 1, len(priority)):
        for c in select_model(priority[i]):
            if c["litellm_name"] not in seen_names:
                chain.append(c)
                seen_names.add(c["litellm_name"])

    return chain

# ─── Build MODEL_TIERS and CLASSIFIER_MODEL ─────────────────────────────────

MODEL_TIERS: dict[str, dict] = {}

_tier_definitions = [
    ("routing",   0.0, "Task classification only"),
    ("cheap",     0.3, "Simple Q&A, formatting, lookups"),
    ("code",      0.1, "Code generation and debugging"),
    ("medium",    0.3, "Summaries, planning, moderate reasoning"),
    ("expensive", 0.3, "Complex analysis, full code gen, architecture"),
]

for _tname, _ttemp, _tdesc in _tier_definitions:
    _candidates = select_model(_tname)
    if _candidates:
        _primary = _candidates[0]
        MODEL_TIERS[_tname] = {
            "model":       _primary["litellm_name"],
            "fallbacks":   [c["litellm_name"] for c in _candidates[1:]],
            "max_tokens":  _primary.get("max_tokens", 2048),
            "temperature": _ttemp,
            "description": _tdesc,
        }

_routing_candidates = select_model("routing")
if _routing_candidates:
    CLASSIFIER_MODEL: str = _routing_candidates[0]["litellm_name"]
elif MODEL_POOL:
    _cheapest = min(MODEL_POOL, key=lambda k: MODEL_POOL[k]["quality"])
    CLASSIFIER_MODEL = MODEL_POOL[_cheapest]["litellm_name"]
else:
    CLASSIFIER_MODEL = "groq/llama-3.1-8b-instant"


# ─── Main API ────────────────────────────────────────────────────────────────

async def call_model(
    tier: str,
    messages: list,
    capability: str | None = None,
    task_context: str = "",
    tools: list[dict] | None = None,
    agent_type: str | None = None,
    stream: bool = False,
    model_override: str | None = None,
) -> dict:
    """
    Call the best available model for *tier*.

    Handles rate limits, retries with backoff, auth/billing errors,
    timeouts, automatic fallback across providers, thinking model
    detection, and optional streaming.

    Phase 10.5: If *model_override* is provided, skip tier selection
    entirely and use that exact litellm model name.
    """
    tier = _resolve_tier(tier)
    tier_cfg = MODEL_TIERS.get(tier, {})

    # Temperature from tier config (default 0.3)
    temperature = tier_cfg.get("temperature", 0.3)

    # Refresh performance cache periodically (Phase 4)
    await refresh_perf_cache()

    # ── Phase 10.5: Model pinning / override ──
    if model_override:
        # Find the model in MODEL_POOL by litellm_name
        pinned_cfg = None
        for _pk, _pcfg in MODEL_POOL.items():
            if _pcfg["litellm_name"] == model_override:
                pinned_cfg = _pcfg
                break

        if pinned_cfg:
            candidates = [{
                "key": _pk,
                "litellm_name": model_override,
                "provider": pinned_cfg["provider"],
                "max_tokens": pinned_cfg.get("max_tokens", 2048),
                "rate_limit": pinned_cfg.get("rate_limit", 30),
                "quality": pinned_cfg["quality"],
                "score": 999,
                "api_base": pinned_cfg.get("api_base"),
            }]
            logger.info(
                f"Model pinned: {model_override} (skipping tier selection)"
            )
        else:
            # Not in MODEL_POOL — try using as raw litellm name
            candidates = [{
                "litellm_name": model_override,
                "provider": "unknown",
                "max_tokens": 4096,
                "rate_limit": 30,
                "quality": 5,
                "score": 999,
                "api_base": None,
            }]
            logger.warning(
                f"Model override '{model_override}' not in MODEL_POOL — "
                f"attempting raw litellm call"
            )
    else:
        # Build ordered fallback chain
        candidates = _get_fallback_models(tier)

    if not candidates:
        # Absolute fallback: any model in the pool
        candidates = [
            {
                "litellm_name": cfg["litellm_name"],
                "provider": cfg["provider"],
                "max_tokens": cfg.get("max_tokens", 2048),
                "rate_limit": cfg.get("rate_limit", 30),
                "quality": cfg["quality"],
                "score": 0,
                "api_base": cfg.get("api_base"),
            }
            for cfg in MODEL_POOL.values()
        ]

    if not candidates:
        raise RuntimeError("No models available in MODEL_POOL!")

    last_error: str | None = None

    # Try up to 5 candidates
    for candidate in candidates[:5]:
        model_name = candidate["litellm_name"]
        provider = candidate.get("provider", "unknown")
        max_tokens = candidate.get("max_tokens", 2048)
        api_base = candidate.get("api_base")
        is_ollama = provider == "ollama"

        limiter = _get_limiter(provider, candidate.get("rate_limit", 30))
        max_retries = 2 if is_ollama else 3
        timeout_val = 120 if is_ollama else 60

        # ── Thinking model adjustments (Phase 4) ──
        is_thinking = _is_thinking_model(model_name)
        if is_thinking:
            timeout_val = max(timeout_val, 180)  # thinking models need more time
            temperature = 1  # many thinking models ignore or require temp=1

        for attempt in range(max_retries):
            try:
                await limiter.wait()

                logger.info(
                    f"Calling {model_name} "
                    f"(tier={tier}, attempt={attempt + 1}/{max_retries}"
                    f"{', thinking' if is_thinking else ''})"
                )

                # Decide whether to pass tools for function calling
                # or response_format for JSON mode
                use_tools = None
                model_supports_fc = False
                model_supports_rf = False
                for _pool_key, _pool_cfg in MODEL_POOL.items():
                    if _pool_cfg["litellm_name"] == model_name:
                        model_supports_fc = _pool_cfg.get(
                            "supports_function_calling", False
                        )
                        model_supports_rf = _pool_cfg.get(
                            "supports_response_format", False
                        )
                        break
                if tools and model_supports_fc:
                    use_tools = tools
                    logger.debug(
                        f"Using function calling for {model_name}"
                    )

                completion_kwargs = dict(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Phase 10.1: Pass api_base for custom endpoints
                if api_base:
                    completion_kwargs["api_base"] = api_base

                # Thinking models: don't set temperature (some reject it)
                if is_thinking:
                    completion_kwargs.pop("temperature", None)

                if use_tools:
                    completion_kwargs["tools"] = use_tools
                    completion_kwargs["tool_choice"] = "auto"
                elif model_supports_rf and not model_supports_fc:
                    completion_kwargs["response_format"] = {
                        "type": "json_object"
                    }
                    logger.debug(
                        f"Using response_format json_object for "
                        f"{model_name}"
                    )

                call_start = time.time()

                # ── Streaming path (Phase 4) ──
                if stream and not use_tools:
                    completion_kwargs["stream"] = True
                    chunks = []
                    stream_response = await asyncio.wait_for(
                        litellm.acompletion(**completion_kwargs),
                        timeout=timeout_val,
                    )
                    full_content = ""
                    async for chunk in stream_response:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            full_content += delta.content
                            chunks.append(chunk)

                    call_latency = time.time() - call_start

                    # Calculate cost from chunks
                    try:
                        cost = litellm.stream_chunk_builder(
                            chunks
                        ) if chunks else 0.0
                        # Actually cost is hard to get from stream,
                        # estimate from token count
                        cost = 0.0  # will be estimated below
                    except Exception:
                        cost = 0.0

                    if is_ollama:
                        cost = 0.0

                    _get_circuit_breaker(provider).record_success()

                    return {
                        "content": full_content,
                        "model": model_name,
                        "tier": tier,
                        "cost": cost,
                        "usage": {},
                        "tool_calls": None,
                        "latency": call_latency,
                    }

                # ── Standard (non-streaming) path ──
                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=timeout_val,
                )

                call_latency = time.time() - call_start

                # Calculate cost
                try:
                    cost = litellm.completion_cost(
                        completion_response=response
                    )
                except Exception:
                    cost = 0.0

                # Ollama is always free
                if is_ollama:
                    cost = 0.0

                # Extract tool_calls from function-calling response
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

                # ── Extract thinking content (Phase 4) ──
                thinking_content = None
                if is_thinking:
                    thinking_content = _extract_thinking(msg)

                # Success — reset circuit breaker for this provider
                _get_circuit_breaker(provider).record_success()

                return {
                    "content": msg.content or "",
                    "model": model_name,
                    "tier": tier,
                    "cost": cost or 0.0,
                    "usage": (
                        dict(response.usage) if response.usage else {}
                    ),
                    "tool_calls": tool_calls,
                    "latency": call_latency,
                    "thinking": thinking_content,
                }

            except asyncio.TimeoutError:
                _get_circuit_breaker(provider).record_failure()
                logger.warning(
                    f"Timeout on {model_name} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                last_error = f"Timeout on {model_name}"
                continue   # retry same model

            except Exception as e:
                _get_circuit_breaker(provider).record_failure()
                error_str = str(e).lower()
                last_error = str(e)

                # ── Auth / billing errors → skip provider entirely ──
                is_auth = any(kw in error_str for kw in [
                    "api key", "authentication", "unauthorized",
                    "invalid_api_key",
                ])
                is_billing = any(kw in error_str for kw in [
                    "credit", "billing", "quota", "insufficient",
                ])
                if is_auth or is_billing:
                    logger.error(
                        f"Auth/billing error on {model_name}: {e}"
                    )
                    break   # next candidate

                # ── Rate limit → backoff then retry or skip ──
                is_rate_limit = any(kw in error_str for kw in [
                    "rate limit", "rate_limit", "429",
                    "too many requests", "tokens per minute",
                    "resource_exhausted",
                ])
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        logger.warning(
                            f"Rate limited on {model_name}, "
                            f"waiting {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(
                            f"Rate limited on {model_name}, "
                            f"moving to next model"
                        )
                        break

                # ── Unknown error → one retry then move on ──
                if attempt < 1:
                    logger.warning(
                        f"Error on {model_name}: {e} — retrying once"
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning(f"Model {model_name} failed: {e}")
                break

    raise RuntimeError(
        f"All models failed for tier '{tier}'. Last error: {last_error}"
    )


# ─── Thinking Model Support (Phase 4) ───────────────────────────────────────

def _is_thinking_model(model_name: str) -> bool:
    """Detect if a model supports extended thinking/reasoning."""
    name_lower = model_name.lower()
    return any(p in name_lower for p in THINKING_MODELS)


def _extract_thinking(msg) -> str | None:
    """Extract thinking/reasoning content from model response.

    Different providers format thinking differently:
    - Anthropic: msg.thinking or <thinking> tags
    - OpenAI o1/o3: reasoning_content field
    - DeepSeek R1: <think> tags in content
    """
    # Check for dedicated thinking attribute
    if hasattr(msg, "thinking") and msg.thinking:
        return msg.thinking

    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content

    # Check for thinking tags in content
    content = msg.content or ""
    import re
    think_match = re.search(
        r"<(?:thinking|think)>(.*?)</(?:thinking|think)>",
        content, re.DOTALL
    )
    if think_match:
        return think_match.group(1).strip()

    return None


# ─── Response Grading (Phase 4) ─────────────────────────────────────────────

GRADING_PROMPT = """Rate this AI response on a scale of 1-5:
1 = Wrong/useless, 2 = Partially relevant, 3 = Adequate,
4 = Good and complete, 5 = Excellent

Task: {task_title}
Task description: {task_description}

Response to grade:
{response}

Respond with ONLY a JSON object: {{"score": N, "reason": "brief"}}"""


async def grade_response(
    task_title: str,
    task_description: str,
    response_text: str,
) -> float | None:
    """Grade a response using the cheapest available model.

    Returns a score 1-5, or None if grading fails.
    """
    if not response_text or len(response_text.strip()) < 10:
        return None

    try:
        # Use cheapest tier for grading
        grading_result = await call_model(
            tier="routing",
            messages=[{
                "role": "user",
                "content": GRADING_PROMPT.format(
                    task_title=task_title[:100],
                    task_description=(task_description or "")[:300],
                    response=response_text[:2000],
                ),
            }],
        )

        raw = grading_result.get("content", "").strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        parsed = json.loads(raw.strip())
        score = float(parsed.get("score", 3))
        score = max(1.0, min(5.0, score))  # clamp to 1-5
        logger.info(
            f"Response graded: {score}/5 — "
            f"{parsed.get('reason', 'N/A')[:60]}"
        )
        return score
    except Exception as e:
        logger.debug(f"Response grading failed: {e}")
        return None


# ─── Cost Budget Checking (Phase 4) ─────────────────────────────────────────

async def check_cost_budget() -> dict:
    """Check if daily cost budget is exceeded.

    Returns {"ok": True/False, "reason": str}.
    """
    try:
        from db import check_budget
        return await check_budget("daily")
    except Exception as e:
        logger.debug(f"Budget check failed: {e}")
        return {"ok": True, "reason": f"check failed: {e}"}
