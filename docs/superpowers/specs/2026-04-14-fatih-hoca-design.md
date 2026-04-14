# Fatih Hoca — Design Spec

## Problem

Model selection logic is spread across three components with blurred boundaries:

- **Router** (`src/core/router.py`) — 15-dimension scoring, task profiles, capability mapping, model selection
- **Dispatcher** (`src/core/llm_dispatcher.py`) — swap budget, overhead exclusion, cold-start wait, proactive GPU loading, pinned-model fallback, timeout calculation
- **Registry** (`src/models/model_registry.py`) — model catalog, YAML parsing, GGUF scanning, demotion tracking

All three make model selection decisions. Router scores, but dispatcher second-guesses it (pinned model attempts, overhead exclusion, swap budget gates). Registry defines ModelInfo that both router and dispatcher query independently. Agents fixing unrelated bugs stumble into scoring logic because there's no hard boundary.

## Decision

Extract model selection into a new package: **Fatih Hoca** — the coach who knows every player's strengths and picks the lineup. Router scoring + registry + capabilities + quota planner merge into one package with a hard boundary.

Three roles, cleanly separated:

- **Fatih Hoca** (manager) — knows models, evaluates them, picks the best one for each job
- **Dispatcher** (assignment desk) — asks the manager, loads the model, calls the talking layer, reports failures back
- **Nerd Herd** (eyes and ears) — unified system state, feeds Fatih Hoca with current GPU, model, and cloud capacity data

## Architecture After

```
Orchestrator (task lifecycle, queue)
  → Dispatcher (ask, load, call, retry)
      → Fatih Hoca (select model, swap budget, failure adaptation)
          → Nerd Herd (system state snapshot)
      → DaLLaMa (ensure_model, load/swap)
      → Talking Layer (LLM call execution)
          → DaLLaMa (GPU semaphore, inference tracking)
          → KDV (pre_call, post_call)
          → Doğru mu Samet (quality check)

Nerd Herd (collects from all infrastructure)
  ← DaLLaMa — pushes loaded model, thinking, swap state
  ← KDV — pushes provider utilization, failures, limits
  ← GPU hardware — polls VRAM, temperature
  ← llama-server — polls inference metrics
```

No circular dependencies. Every arrow is one-directional.

## Fatih Hoca Responsibilities

Owns:
- Model catalog (YAML parsing, GGUF scanning, ModelInfo)
- 15-dimension capability scoring
- Task profiles, agent requirement templates, capability-to-task mapping
- Composite scoring formula (weights, ranking adjustments)
- Swap stickiness (loaded model preference)
- Swap budget (max 3 per 5 min, exemptions)
- Specialty matching
- Sibling rebalancing across providers
- Circuit breaker decisions (from consecutive failure counts + rate limits)
- Failure-adaptive re-selection (timeout → faster model, context overflow → larger context)
- Minimum time estimation for selected model
- Quota planning (penalize paid models below threshold)

Does not own:
- Model loading/swapping (DaLLaMa)
- LLM call execution (Talking Layer)
- Cloud capacity tracking (KDV) — consumes via Nerd Herd
- Message preparation (Dispatcher — secret redaction, thinking adaptation)
- Timeout enforcement (Dispatcher)

## Interface

```python
# ── Init (called once at startup, scans GGUFs) ──
models = await fatih_hoca.init(
    models_dir="D:/models",
    catalog_path="data/models.yaml",
)
# returns list[str] — model names
# select() works after this returns

# ── Selection (called per LLM request, ~30 microseconds) ──
pick = fatih_hoca.select(
    task="shopping_advisor",
    agent_type="shopping_advisor",
    difficulty=5,
    needs_function_calling=True,
    needs_vision=False,
    needs_thinking=True,
    estimated_input_tokens=2000,
    estimated_output_tokens=2500,
    min_context_length=0,      # explicit override, 0 = auto from token estimates
    max_cost=0.0,              # 0 = no budget cap
    prefer_speed=True,
    prefer_local=True,
    priority=5,
    failures=[],               # empty on first attempt
)
# returns Pick | None

# ── Informational ──
fatih_hoca.all_models() → list[ModelInfo]
```

### Return Types

```python
@dataclass
class Pick:
    model: ModelInfo
    min_time_seconds: float    # minimum estimated generation time

@dataclass
class Failure:
    model: str                 # litellm_name that failed
    reason: str                # "timeout", "rate_limit", "context_overflow",
                               # "quality_failure", "server_error", "loading"
    latency: float | None      # seconds, if available
```

Score and reasons are logged internally by Fatih Hoca, never returned. Dispatcher doesn't interpret scores.

### Selection With Failures

On retry, dispatcher passes accumulated failures. Fatih Hoca adapts:

- `timeout` → exclude slow models, boost speed weight
- `rate_limit` → avoid that provider, check siblings
- `context_overflow` → require larger context window
- `quality_failure` → escalate difficulty, prefer stronger model
- `server_error` → factor into consecutive failure count for provider
- `loading` → exclude that model

```python
pick = fatih_hoca.select(
    task="shopping_advisor",
    difficulty=5,
    failures=[
        Failure(model="qwen3-30b", reason="timeout", latency=120.0),
        Failure(model="groq/llama-8b", reason="rate_limit"),
    ],
)
# Returns a faster model, avoiding groq
```

No separate `exclude` parameter — exclusions are derived from failures.

### No Ranked List

`select()` returns a single model, not a ranked list. On failure, dispatcher calls `select()` again with the failure appended. Each call is ~30 microseconds (pure arithmetic over ~15 models), and recalculating ensures fresh system state from Nerd Herd — important because swaps can happen between attempts.

### No suggest_preload

Proactive GPU loading is eliminated. Tasks trigger model loading when they run. OVERHEAD calls (grading, classification) go through the same `select → load → call` path as MAIN_WORK. The MAIN_WORK vs OVERHEAD distinction affects timeout floors and thinking, not model selection eligibility.

## OVERHEAD Call Overhaul

Today OVERHEAD calls (grading, classification, self-reflection) are second-class citizens with special-case machinery:

- Hard-gated from triggering swaps — `_prepare_overhead_reqs()` excludes all unloaded local models
- Cold-start wait polling loop when no model is loaded and no cloud exists
- Proactive GPU loading (`ensure_gpu_utilized`) to ensure something is loaded for overhead
- Deferred grading queue that drains opportunistically on model swap events
- Fixed 20s timeout that's too short if a swap is needed, and the swap can't happen anyway

**After**: OVERHEAD calls go through the same `select → load → call` path as MAIN_WORK. The differences are inputs to Fatih Hoca, not separate code paths:

| Concern | Before | After |
|---------|--------|-------|
| Model selection | Excluded unloaded locals, separate candidate loop | Same `select()` call. Fatih Hoca factors in swap cost — strongly prefers loaded model for overhead but *can* swap if nothing else works |
| Swap budget | OVERHEAD cannot spend swap budget at all | Fatih Hoca weighs swap cost higher for overhead (it's cheap work, not worth a 25s swap usually) but allows it when no alternative exists |
| Timeout | Hard 20s / 60s for grading | `min_time_seconds` from Fatih Hoca accounts for model speed. Dispatcher applies a floor but doesn't cap. Slow model grading gets enough time |
| Cold start | Polling loop waiting for model | `select()` returns None if nothing available. Dispatcher fails, task retries later via normal retry pipeline |
| Proactive loading | `ensure_gpu_utilized` preloads for overhead | Eliminated. Grading tasks trigger loads when they run, like any other task |
| Deferred grading | Side-channel drain on swap events | Grading tasks are normal tasks. Orchestrator submits them, dispatcher routes them through Fatih Hoca |
| Thinking | Hardcoded `needs_thinking=False` | Dispatcher still passes `needs_thinking=False` for OVERHEAD — this is a real constraint, not an artificial gate |

The key insight: OVERHEAD's "no swap" rule existed to prevent wasteful swaps for cheap work. Fatih Hoca achieves the same result through scoring — loaded model gets massive stickiness bonus for overhead calls, unloaded models get penalized by swap cost. But when the loaded model literally cannot do the job (wrong capabilities, too slow, no model loaded at all), Fatih Hoca can pick an unloaded one instead of letting tasks rot.

This eliminates `_prepare_overhead_reqs`, `_exclude_unloaded_local`, `_exclude_all_local`, `_should_wait_for_cold_start`, `_wait_for_model_load`, `ensure_gpu_utilized`, and all the grade-draining machinery from dispatcher.

## Nerd Herd Expansion

Nerd Herd becomes the single source of system state. Two new collectors, both push-based:

- **LocalModelStateCollector** — receives updates from DaLLaMa callbacks (`on_ready`)
- **CloudCapacityCollector** — receives updates from KDV callbacks (`on_capacity_change`)

### SystemSnapshot

```python
@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None        # absolute epoch seconds

@dataclass
class RateLimits:
    rpm: RateLimit = field(default_factory=RateLimit)
    tpm: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)

@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimits = field(default_factory=RateLimits)

@dataclass
class CloudProviderState:
    provider: str = ""
    utilization_pct: float = 0.0
    consecutive_failures: int = 0
    last_failure_at: int | None = None   # epoch seconds
    limits: RateLimits = field(default_factory=RateLimits)
    models: dict[str, CloudModelState] = field(default_factory=dict)

@dataclass
class LocalModelState:
    model_name: str | None = None
    thinking_enabled: bool = False
    vision_enabled: bool = False
    measured_tps: float = 0.0
    context_length: int = 0
    is_swapping: bool = False
    kv_cache_ratio: float = 0.0

@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)
```

Fatih Hoca calls `nerd_herd.snapshot()` inside `select()`. One dependency, one call, complete picture.

Key design decisions:
- `vram_available_mb` is a single number — Nerd Herd calculates it (total minus llama-server minus external minus budget policy). Fatih Hoca never sees raw GPU fields.
- No `circuit_breaker_open` flag — Fatih Hoca makes circuit breaker decisions from `consecutive_failures`, `last_failure_at`, and rate limits.
- No `daily_exhausted` flag — derived from `rpd.remaining == 0`.
- No `load_mode` — if VRAM is 0, no local model fits. Nerd Herd's policy is internal.

## Package Structure

```
packages/fatih_hoca/
  pyproject.toml
  src/
    fatih_hoca/
      __init__.py            # exports: init, select, all_models, Pick, Failure,
                             #   ModelInfo, ModelRequirements
      registry.py            # ModelInfo, GGUF scanning, YAML parsing, model catalog
      capabilities.py        # Cap enum, TASK_PROFILES, score_model_for_task(),
                             #   15-dim dot product
      requirements.py        # ModelRequirements, AGENT_REQUIREMENTS,
                             #   CAPABILITY_TO_TASK, QuotaPlanner
      selector.py            # select(), candidate filtering (eligibility hard gates)
      ranking.py             # composite weights, swap stickiness, specialty,
                             #   sibling rebalancing, failure adaptation
      types.py               # Pick, Failure, SwapBudget, shared types
  tests/
    test_registry.py
    test_capabilities.py
    test_requirements.py
    test_selector.py
    test_ranking.py
```

Each file under 200 lines. Dependencies: `nerd_herd` only.

## Dispatcher After Extraction

Dispatcher shrinks to a mechanical loop:

```python
class LLMDispatcher:

    async def request(self, category, task, agent_type, difficulty,
                      messages, tools=None, failures=None, **kwargs):

        pick = fatih_hoca.select(
            task=task,
            agent_type=agent_type,
            difficulty=difficulty,
            needs_thinking=category != CallCategory.OVERHEAD,
            failures=failures or [],
            **kwargs,
        )
        if pick is None:
            raise ModelCallFailed(...)

        if pick.model.is_local:
            await dallama.ensure_model(pick.model.name, ...)

        messages = self._prepare_messages(messages, pick.model)
        timeout = max(pick.min_time_seconds, self._timeout_floor(category))

        result = await talking_layer.call(
            model=pick.model,
            messages=messages,
            tools=tools,
            timeout=timeout,
        )

        if isinstance(result, CallResult):
            return self._to_response_dict(result)

        # Failed — tell Fatih Hoca what happened
        failure = Failure(
            model=pick.model.litellm_name,
            reason=result.category,
            latency=result.latency,
        )
        # Guard against infinite recursion
        all_failures = (failures or []) + [failure]
        if len(all_failures) >= 5:
            raise ModelCallFailed(...)

        return await self.request(
            category, task, agent_type, difficulty,
            messages, tools,
            failures=all_failures,
            **kwargs,
        )
```

### What moves out of dispatcher

| Removed | Destination |
|---------|-------------|
| `SwapBudget` | Fatih Hoca |
| `_select_candidates()` | Fatih Hoca |
| `_prepare_overhead_reqs()` | Fatih Hoca (scoring handles via weights) |
| `_try_pinned_loaded()` | Fatih Hoca (swap stickiness scoring) |
| `_exclude_unloaded_local()` / `_exclude_all_local()` | Fatih Hoca |
| `_should_wait_for_cold_start()` / `_wait_for_model_load()` | Eliminated (select returns None) |
| `ensure_gpu_utilized()` / `_find_best_local_for_batch()` | Eliminated |
| `_has_pending_overhead_needs()` / `_loaded_model_can_grade()` | Eliminated |
| `_find_fastest_general_model()` / `_get_grade_exclusions()` | Eliminated |
| `_compute_timeout()` | Simplified to `_timeout_floor()`, Fatih Hoca provides estimate |
| `on_model_swap()` grade draining | Orchestrator handles directly |

### What stays in dispatcher

- `_prepare_messages()` — secret redaction for cloud, thinking adaptation
- `_timeout_floor()` — OVERHEAD 20s / 60s for grading, MAIN_WORK minimum
- `_to_response_dict()` — convert CallResult to legacy dict
- `ModelCallFailed` — raised when select returns None
- Recursion depth guard on retries

## What Happens to Existing Files

| Current file | Fate |
|---|---|
| `src/core/router.py` | Thin shim re-exporting from fatih_hoca during migration. Deleted when callers updated. |
| `src/models/model_registry.py` | Deleted. Merged into `fatih_hoca/registry.py`. |
| `src/models/capabilities.py` | Deleted. Merged into `fatih_hoca/capabilities.py`. |
| `src/models/quota_planner.py` | Deleted. Merged into `fatih_hoca/requirements.py`. |
| `src/models/local_model_manager.py` | Deleted. Callers import `dallama` directly. |
| `src/models/gpu_monitor.py` | Deleted. Callers import `nerd_herd` directly. |
| `src/models/gpu_scheduler.py` | Deleted. Already dead. |
| `src/models/rate_limiter.py` | Deleted. KDV handles this. |
| `src/models/header_parser.py` | Deleted. KDV handles this. |
| `src/models/models.yaml` | Stays (or moves to `data/`). Path passed to `fatih_hoca.init()`. |

`src/models/` directory potentially goes away entirely.

## Migration Safety

1. **Shim-first**: `router.py` re-exports `ModelRequirements`, `call_model`, `select_model`, etc. from fatih_hoca. All existing callers work unchanged.
2. **Gradual migration**: Move callers from shim imports to direct `fatih_hoca` imports one file at a time.
3. **Test parity**: Fatih Hoca tests must cover every scoring path the current router handles.
4. **Kill switch**: If Fatih Hoca has a bug, revert the shim to call old code path.

## Future: Registry Separation

Registry may split from Fatih Hoca later when it grows (benchmark fetching, auto-discovery, GGUF header parsing). The split would be:

```
packages/fatih_hoca/     → scoring, task profiles, requirements
packages/model_registry/ → ModelInfo, catalog, YAML parsing, benchmarks
```

Not planned now. The code will signal when it's time.

## Open Items

None. All design decisions settled during brainstorming.
