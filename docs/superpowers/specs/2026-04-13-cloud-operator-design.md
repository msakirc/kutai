# Kuleden Dönen Var — Design Spec

Tracks rate limits, quotas, and health across cloud LLM providers and tells KutAI who has capacity.

## What It Is

A dumb pipe for cloud provider health. Manages rate limits (per-model + per-provider), quota thresholds, circuit breakers, and response header parsing. Does not discover models, pick models, or make LLM calls.

Extracted from `src/models/rate_limiter.py`, `src/models/quota_planner.py`, `src/models/header_parser.py`, and the `CircuitBreaker` class in `src/core/router.py`.

## What It Is Not

- Does not discover cloud providers or available models (registry's job)
- Does not select which model to use (router's job)
- Does not make litellm calls (talking layer's job, future)
- Does not manage local models (DaLLaMa's job)

## Interface

### Construction

```python
from kuleden_donen_var import KuledenDonenVar, KuledenConfig

kdv = KuledenDonenVar(
    config=KuledenConfig(...),
    on_capacity_change=callback,  # called whenever capacity state changes
)
```

### `register(model_id, provider, rpm, tpm, provider_aggregate_rpm, provider_aggregate_tpm)`

Host pushes model config at startup. KDV doesn't pull from registry.

### `pre_call(model_id, provider, estimated_tokens) -> PreCallResult`

Called before every cloud LLM call. Returns:
- `allowed: bool` — can this call proceed?
- `wait_seconds: float` — if not allowed, how long until capacity frees up (0 if allowed)
- `daily_exhausted: bool` — daily limit hit, skip this model entirely

Does NOT block/sleep. Caller decides whether to wait or pick another model.

### `post_call(model_id, provider, response_headers, token_count)`

Called after every successful cloud LLM call. Updates:
- Token usage tracking
- Rate limit state from parsed response headers
- Circuit breaker (records success)
- Fires `on_capacity_change` if state changed meaningfully

### `record_failure(model_id, provider, error_type)`

Called on cloud call failure. Error types:
- `rate_limit` (429) — adaptive limit reduction, fires `on_capacity_change`
- `server_error` (5xx) — circuit breaker failure count
- `timeout` — circuit breaker failure count
- `auth` — not tracked (permanent, not transient)

### `status -> dict[str, ProviderStatus]`

Current state of all providers. Read-only property for diagnostics and dispatcher queries.

```python
@dataclass
class ProviderStatus:
    provider: str
    circuit_breaker_open: bool
    utilization_pct: float          # 0-100, worst across RPM/TPM
    rpm_remaining: int | None
    tpm_remaining: int | None
    rpd_remaining: int | None       # daily limit remaining
    reset_in_seconds: float | None  # earliest reset timer
    models: dict[str, ModelStatus]  # per-model breakdown
```

### `on_capacity_change` Callback

```python
def on_capacity_change(event: CapacityEvent) -> None: ...

@dataclass
class CapacityEvent:
    provider: str
    model_id: str | None            # None = provider-level change
    event_type: str                  # "capacity_restored", "limit_hit", "circuit_breaker_tripped", "circuit_breaker_reset", "daily_exhausted"
    snapshot: ProviderStatus        # full current state
```

Fires on:
- Rate limit reset timer expiry (capacity restored)
- 429 received (limit hit)
- Circuit breaker trip/reset
- Daily limit exhausted
- Response headers reveal significant capacity change (>20% swing)

This is the key difference from DaLLaMa. DaLLaMa sends simple "GPU free/busy" events. KDV sends rich capacity snapshots because the dispatcher/quota planner needs utilization %, remaining counts, and reset timers to make look-ahead routing decisions.

## Internal Modules

```
kuleden_donen_var/
  ├── __init__.py          # public API: KuledenDonenVar, config, types
  ├── kdv.py               # main class, composes modules (~100 lines)
  ├── rate_limiter.py       # two-tier rate limiting (per-model + per-provider)
  ├── quota_tracker.py      # utilization tracking, threshold calculation
  ├── circuit_breaker.py    # per-provider failure tracking + cooldown
  ├── header_parser.py      # provider-specific header normalization
  └── config.py             # KuledenConfig, ProviderStatus, CapacityEvent, PreCallResult
```

### Dependency Cuts

Current modules import from `src/`:

| Import | Current Location | Resolution |
|--------|-----------------|------------|
| `src.infra.logging_config.get_logger` | rate_limiter, quota_planner, header_parser | Use stdlib `logging.getLogger(__name__)` |
| `src.infra.db.accelerate_retries` | rate_limiter (lazy), quota_planner (lazy) | Replaced by `on_capacity_change` callback — host wires it to `accelerate_retries` |
| `src.models.model_registry.get_registry` | rate_limiter `_init_from_registry()` | Replaced by `register()` — host pushes model config |
| `src.models.header_parser.RateLimitSnapshot` | rate_limiter | Moves into package |

### Quota Planner Split

The current `quota_planner.py` has two responsibilities:
1. **Tracking** utilization and 429 frequency — moves into KDV as `quota_tracker.py`
2. **Deciding** the expensive threshold based on queue profile look-ahead — stays in KutAI (dispatcher/router)

KDV reports utilization. KutAI decides policy. The `QueueProfile` dataclass and `recalculate()` threshold logic stay in `src/` because they need task queue knowledge.

## What Changes in KutAI

### Router (`src/core/router.py`)
- Delete `CircuitBreaker` class and `_circuit_breakers` dict (~50 lines)
- Replace `_get_circuit_breaker(provider).is_degraded` check with `kdv.status[provider].circuit_breaker_open`
- Replace `_get_circuit_breaker(provider).record_success/failure()` with `kdv.post_call()` / `kdv.record_failure()`
- Replace `rate_limit_manager.has_capacity()` with `kdv.pre_call()`
- Replace `_update_limits_from_response()` with `kdv.post_call(..., response_headers)`
- Delete direct imports of `rate_limiter`, `header_parser`

### Dispatcher (`src/core/llm_dispatcher.py`)
- Subscribe to `on_capacity_change` at startup
- Wire `capacity_restored` events to `accelerate_retries`
- Read `kdv.status` for quota-aware routing decisions (replaces direct `quota_planner` reads)

### Orchestrator (`src/core/orchestrator.py`)
- Feed `QueueProfile` to the quota threshold logic (stays in `src/`, not in KDV)

### Shim
- `src/models/rate_limiter.py` becomes a thin shim: `get_rate_limit_manager()` delegates to KDV
- `src/models/header_parser.py` re-exports from KDV
- `src/models/quota_planner.py` keeps the threshold logic, delegates utilization tracking to KDV

## Package Setup

- Location: `packages/kuleden_donen_var/`
- Layout: src layout (`src/kuleden_donen_var/`)
- Install: editable in `requirements.txt`
- Dependencies: none (stdlib only)
- Logging: stdlib `logging`
- Tests: `packages/kuleden_donen_var/tests/`

## Files

| File | What |
|------|------|
| `packages/kuleden_donen_var/` | Package root |
| `packages/kuleden_donen_var/pyproject.toml` | Package metadata |
| `packages/kuleden_donen_var/src/kuleden_donen_var/` | Source modules |
| `src/models/rate_limiter.py` | Becomes shim |
| `src/models/quota_planner.py` | Keeps threshold logic, delegates tracking |
| `src/models/header_parser.py` | Becomes re-export shim |
| `src/core/router.py` | Loses CircuitBreaker, gains KDV calls |
| `src/core/llm_dispatcher.py` | Subscribes to on_capacity_change |
