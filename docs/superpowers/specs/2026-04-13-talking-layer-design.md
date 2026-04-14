# HaLLederiz Kadir — Design Spec

## Problem

Router (`src/core/router.py`, 1,700 lines) does two unrelated jobs: **model selection** (15-dimension capability scoring) and **LLM call execution** (litellm.acompletion, streaming, retries, error handling, GPU semaphore, rate limiting, metrics, auditing). These are tangled into a single 640-line `call_model()` function. Agents editing router to fix a retry bug can break scoring logic, and vice versa.

Meanwhile, dispatcher (`src/core/llm_dispatcher.py`, 550 lines) sits between orchestrator and router but only manages policy (swap budget, categorization, timeout). The actual call execution bypasses it — dispatcher calls `router.call_model()` which does everything.

The result: three components (orchestrator, dispatcher, router) with blurred boundaries, doing each other's jobs.

## Decision

Extract call execution into a new package: **HaLLederiz Kadir**. This follows the same pattern as DaLLaMa (process lifecycle), KDV (capacity tracking), and Nerd Herd (observability) — focused packages with hard boundaries.

## Architecture After

```
Orchestrator (task lifecycle, queue)
  → Dispatcher (policy: candidate iteration, swap budget, timeout calc)
      → Router (pure scoring, returns ranked candidates)
      → HaLLederiz Kadir (execution: litellm, retry, streaming, parsing, quality, metrics)
          ├── DaLLaMa (local: infer, GPU semaphore, inference tracking)
          ├── KDV (cloud: pre_call, post_call, record_failure)
          ├── Dogru mu Samet (quality: degenerate output detection)
          ├── Nerd Herd (metrics: prometheus, speed tracking)
          └── Yazbunu (logging: structured JSONL)
```

## Responsibility Map

### Router (pure scoring, ~500 lines after split)

- 15-dimension capability scoring
- Task profile matching
- S7 sibling rebalancing across providers
- Rate limit scoring via KDV status
- Local vs cloud preference logic
- Fallback relaxation (lower min_score when nothing matches)
- Model override handling
- `select_model(reqs) → list[ScoredModel]`

**No I/O.** Router never calls litellm, never touches DaLLaMa, never acquires GPU.

### Dispatcher (policy + orchestration, ~700 lines after absorbing candidate loop)

Owns:
- MAIN_WORK vs OVERHEAD categorization
- Swap budget (max 3 per 5 min)
- Timeout calculation (TPS-based adaptive)
- Candidate iteration loop (moved from router)
- `ensure_model()` calls via DaLLaMa (swap/load decisions)
- Capacity look-ahead via KDV (exclude exhausted providers)
- OVERHEAD model exclusion (exclude unloaded local models)
- Cold-start wait logic
- Proactive GPU loading (`ensure_gpu_utilized`)
- On-model-swap callback (wake tasks, drain grades)
- Secret redaction on messages before passing to HaLLederiz Kadir
- Thinking model message adaptation (assistant prefill → user message)

Does not own:
- litellm calls (HaLLederiz Kadir)
- Response parsing (HaLLederiz Kadir)
- Retry logic (HaLLederiz Kadir)
- GPU semaphore (HaLLederiz Kadir via DaLLaMa)
- Metrics/audit/tracing recording (HaLLederiz Kadir)

### HaLLederiz Kadir (execution hub, ~400 lines, new package)

Owns:
- Build `completion_kwargs` from `ModelInfo` (sampling params, api_base, api_key, tools, response_format)
- `litellm.acompletion()` call
- Streaming with partial content accumulation
- Per-model retry loop (2 retries local, 3 cloud)
- `asyncio.wait_for` timeout enforcement
- Response parsing (content, tool_calls, thinking extraction, usage, cost)
- Think-tag stripping
- Error classification (timeout, rate_limit, auth, server_error, loading, etc.)
- Partial content salvage on timeout
- Quality check via Dogru mu Samet
- GPU semaphore acquire/release via DaLLaMa
- Inference tracking (mark_start/mark_end) via DaLLaMa
- Model-swap-during-GPU-wait detection
- KDV `pre_call()` / `post_call()` / `record_failure()` per call
- Speed measurement → feed back to model registry + Nerd Herd
- Prometheus metrics recording
- Audit trail recording
- Trace recording
- Performance logging via Yazbunu

Does not own:
- Model selection (router)
- Candidate iteration across models (dispatcher)
- Swap budget decisions (dispatcher)
- Model loading / ensure_model (dispatcher via DaLLaMa)
- Timeout calculation (dispatcher)
- Message redaction or adaptation (dispatcher)

### Shared Dependencies

| Package | Dispatcher uses | HaLLederiz Kadir uses |
|---------|----------------|-------------------|
| DaLLaMa | `ensure_model()`, `current_model`, swap state queries | `infer()`, GPU semaphore, inference tracking |
| KDV | Capacity queries, provider exclusion lists | `pre_call()`, `post_call()`, `record_failure()` |
| Router | `score(reqs)` → ranked candidates | Nothing |
| Dogru mu Samet | Nothing | Quality check on responses |
| Nerd Herd | Nothing | Metrics recording |
| Yazbunu | Nothing | Structured logging |

All dependencies are direct imports. No registration pattern — HaLLederiz Kadir is a KutAI package that knows its siblings.

## Interface

### Dispatcher → HaLLederiz Kadir

```python
result = await talker.call(
    model=model_info,              # ModelInfo from router scoring — contains litellm_name,
                                   # api_base, max_tokens, sampling_overrides, is_local,
                                   # thinking_model, supports_function_calling, etc.
    messages=messages,             # Already redacted and thinking-adapted by dispatcher
    tools=tools,                   # Tool definitions or None
    timeout=120.0,                 # Computed by dispatcher (TPS-based adaptive)
    task="shopping_advisor",       # For sampling profile lookup and logging
    needs_thinking=True,           # Whether to enable thinking for this call
    estimated_output_tokens=500,   # For max_tokens calculation
)
```

Seven parameters. Dispatcher sends what it wants, HaLLederiz Kadir decides how.

### Return Types

```python
@dataclass
class CallResult:
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    usage: dict                    # prompt_tokens, completion_tokens
    cost: float                    # 0.0 for local
    latency: float                 # seconds
    model: str                     # litellm name used
    model_name: str                # human-readable name
    is_local: bool
    provider: str                  # "local" or provider name
    task: str
    # Note: capability_score is NOT here — HaLLederiz Kadir doesn't know
    # about scoring. Dispatcher attaches it when converting to response dict.

@dataclass
class CallError:
    category: str                  # "timeout", "rate_limit", "auth", "server_error",
                                   # "loading", "gpu_busy", "quality_failure", etc.
    message: str
    retryable: bool                # Hint for dispatcher's candidate loop
    partial_content: str | None    # Salvaged from streaming on timeout
```

### What HaLLederiz Kadir Derives from ModelInfo

The `ModelInfo` object (from router scoring) contains everything HaLLederiz Kadir needs to build the litellm call without importing router:

| ModelInfo field | HaLLederiz Kadir uses it for |
|----------------|-------------------------|
| `litellm_name` | `model=` parameter in litellm call |
| `is_local` | DaLLaMa path vs cloud path, retry count, api_key |
| `api_base` | Endpoint URL for litellm (local models) |
| `max_tokens` | Cap on `max_tokens` parameter |
| `sampling_overrides` | Per-model sampling param overrides |
| `thinking_model` | Think-tag handling, sampling behavior |
| `supports_function_calling` | Tools vs json_mode fallback |
| `supports_json_mode` | Fallback when no function calling |
| `has_vision` | (informational, ensure_model already handled by dispatcher) |
| `location` | "ollama" special-casing |
| `provider` | For KDV calls and result metadata |
| `name` | Human-readable name for logging |
| `tokens_per_second` | Fallback speed estimate |
| `is_free` | Affects rate limit recording behavior |

## Call Flow

### Happy Path (local model)

```
Dispatcher:
  1. router.score(reqs) → [qwen3-30b (local), groq/llama-8b (cloud), ...]
  2. dallama.ensure_model("qwen3-30b", thinking=True, vision=False)
  3. talker.call(model=qwen3_info, messages=redacted_msgs, timeout=120, ...)

HaLLederiz Kadir (internal):
  4. Model is local → DaLLaMa path
  5. dallama.acquire_inference_slot(priority, timeout)
  6. dallama.mark_inference_start()
  7. Verify model didn't change during GPU wait
  8. Build completion_kwargs from ModelInfo (sampling, api_base, api_key, etc.)
  9. litellm.acompletion(**kwargs) with streaming + asyncio.wait_for
  10. Parse response (content, tool_calls, thinking, usage)
  11. Strip think tags if thinking not requested
  12. Quality check via Dogru mu Samet
  13. Measure speed → registry.update_measured_speed() + Nerd Herd
  14. Record audit + trace
  15. dallama.mark_inference_end()
  16. dallama.release_inference_slot()
  17. Return CallResult

Dispatcher:
  18. Success → return to orchestrator
```

### Happy Path (cloud model)

```
HaLLederiz Kadir (internal):
  4. Model is cloud → KDV path
  5. kdv.pre_call(model, provider, estimated_tokens)
  6. If daily_exhausted → return CallError("daily_exhausted")
  7. If rate limited with wait → sleep, then proceed
  8. Build completion_kwargs from ModelInfo
  9. litellm.acompletion(**kwargs) with asyncio.wait_for
  10. Parse response
  11. kdv.post_call(model, provider, headers, token_count)
  12. Calculate cost via litellm.completion_cost()
  13. Quality check, metrics, audit
  14. Return CallResult
```

### Failure + Candidate Fallback

```
Dispatcher:
  1. router.score(reqs) → [qwen3-30b (local), groq/llama-8b (cloud)]
  2. dallama.ensure_model("qwen3-30b", ...) → success
  3. talker.call(model=qwen3_info, ...) → CallError(category="timeout", retryable=True)
  4. Try next candidate: talker.call(model=groq_info, ...) → CallResult
  5. Return result
```

### Timeout with Partial Content

```
HaLLederiz Kadir (internal):
  - Streaming locally, accumulated 800 tokens
  - asyncio.wait_for fires timeout
  - Salvage accumulated content from stream buffer
  - Return CallError(category="timeout", partial_content="The analysis shows...")

Dispatcher:
  - Decides: return partial to agent, or try next candidate
```

## Package Structure

```
packages/hallederiz_kadir/
  pyproject.toml
  src/
    hallederiz_kadir/
      __init__.py                 # exports: call, CallResult, CallError
      caller.py                   # Main call() function (~200 lines)
                                  #   - local vs cloud routing
                                  #   - completion_kwargs building
                                  #   - litellm.acompletion + streaming
                                  #   - asyncio.wait_for timeout
      response.py                 # Response parsing (~80 lines)
                                  #   - content, tool_calls, thinking extraction
                                  #   - think-tag stripping
                                  #   - cost calculation
      retry.py                    # Per-model retry loop (~80 lines)
                                  #   - retry with backoff
                                  #   - error classification
                                  #   - partial content salvage
      errors.py                   # CallError, error categories (~40 lines)
      types.py                    # CallResult, internal config types (~50 lines)
  tests/
    test_caller.py
    test_response.py
    test_retry.py
```

Estimated total: ~400-450 lines of production code.

## What Moves Where

### From router.py to HaLLederiz Kadir

| Code | Current location | Description |
|------|-----------------|-------------|
| `_stream_with_accumulator()` | router.py:43-100 | Streaming + partial accumulation |
| completion_kwargs building | router.py:1027-1172 | Sampling, api_base, api_key, tools, response_format |
| `litellm.acompletion()` call | router.py:1309-1319 | The actual call |
| Response parsing | router.py:1321-1483 | Content, tool_calls, thinking, usage, cost, metrics, audit |
| Think-tag stripping | router.py:1398-1425 | Clean unwanted thinking output |
| `_extract_thinking()` | router.py:1628-1638 | Thinking content extraction |
| Retry loop | router.py:1283-1579 | Per-model retry with error handling |
| Error classification | router.py:1485-1578 | Timeout, rate_limit, auth, server_error, loading |
| `_classify_error_category()` | router.py:1602-1623 | Error string → category |
| `ModelCallFailed` | router.py:19-27 | Exception when all candidates fail |
| GPU semaphore | router.py:1197-1271 | acquire/release, inference tracking, swap detection |
| KDV pre/post | router.py:1174-1195, 1330-1345 | Rate limit checks, usage recording |
| Speed measurement | router.py:1348-1373 | tok/s calculation, registry update |
| Metrics/audit/trace | router.py:1427-1466 | Prometheus, audit trail, tracing |
| litellm config | router.py:32-41 | suppress_debug_info, request_timeout |

### From router.py to dispatcher

| Code | Current location | Description |
|------|-----------------|-------------|
| Candidate iteration loop | router.py:1024-1597 (outer for loop) | Try up to 5 candidates |
| ensure_model logic | router.py:1034-1075 | Vision/thinking reload decisions |
| Fallback relaxation | router.py:999-1017 | Relax min_score when no candidates match |
| `ModelCallFailed` raising | router.py:1590-1597 | All candidates exhausted |
| Secret redaction | router.py:1106-1118 | Redact secrets for cloud messages |
| Thinking message adaptation | router.py:1128-1137 | Convert assistant prefill for thinking models |

### Stays in router.py

| Code | Description |
|------|-------------|
| `select_model()` | 15-dimension scoring, returns ranked candidates |
| `ModelRequirements` | Requirements dataclass |
| `ScoredModel` | Scored candidate dataclass |
| All scoring functions | Capability matching, rate limit scoring, preference logic |
| `select_for_task()` | Simplified selection by task name |
| `check_cost_budget()` | Cost budget checking |

## Router.py After Split

Router drops from ~1,700 to ~500 lines. It becomes a pure scoring module:
- `ModelRequirements` dataclass
- `ScoredModel` dataclass  
- `select_model(reqs) → list[ScoredModel]`
- `select_for_task(task, **kwargs) → list[ScoredModel]`
- All scoring internals (capability matching, rate limit scoring, sibling rebalancing)
- `check_cost_budget()`
- Helper constants (`CAPABILITY_TO_TASK`, `AGENT_REQUIREMENTS`)

No litellm import. No DaLLaMa import. No KDV import. No I/O of any kind.

## Dispatcher After Split

Dispatcher grows from ~550 to ~700 lines. It absorbs the candidate loop and pre-call preparation:
- Everything it currently has (categorization, swap budget, timeout calc, ensure_gpu_utilized)
- Candidate iteration loop (from router's call_model outer loop)
- ensure_model calls before each candidate attempt
- Secret redaction and thinking message adaptation
- `ModelCallFailed` raising when all candidates exhausted
- Fallback relaxation logic

The `request()` method becomes:

```python
async def request(self, category, reqs, messages, tools=None):
    timeout = self._compute_timeout(category, reqs)
    candidates = self._get_candidates(category, reqs)
    messages = self._prepare_messages(messages, candidates[0].model)

    for candidate in candidates:
        model = candidate.model
        if model.is_local:
            await self._ensure_local_ready(model, reqs)

        result = await talker.call(
            model=model,
            messages=messages,
            tools=tools,
            timeout=timeout,
            task=reqs.effective_task,
            needs_thinking=reqs.needs_thinking and model.thinking_model,
            estimated_output_tokens=reqs.estimated_output_tokens,
        )

        if isinstance(result, CallResult):
            return self._to_response_dict(result, candidate)  # attaches capability_score here
        # CallError — try next candidate
        if not result.retryable:
            break

    raise ModelCallFailed(...)
```

## Shim

Original `router.call_model()` becomes a thin shim that calls dispatcher, preserving all existing import paths during migration:

```python
async def call_model(reqs, messages, tools=None, timeout_override=None, 
                     partial_buf=None, on_chunk=None):
    """Legacy shim — routes through dispatcher."""
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        reqs=reqs, messages=messages, tools=tools,
    )
```

## Migration Safety

1. **Shim-first**: `call_model()` becomes a shim on day one. All existing callers work unchanged.
2. **Gradual migration**: Move callers from `call_model()` to `dispatcher.request()` one at a time.
3. **Test parity**: New HaLLederiz Kadir tests must cover every error path the current `call_model()` handles.
4. **Kill switch**: If HaLLederiz Kadir has a bug, revert the shim to call the old code path.

## Name

**HaLLederiz Kadir** — named after the TV character who talked his way through everything. "Hallederiz" (we'll handle it) with "LL" referencing LLM. Package name: `hallederiz_kadir`.

## Open Items

None. All design decisions settled during brainstorming.
