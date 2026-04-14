# HaLLederiz Kadir — Design Brief

## The Question

Router currently does two unrelated jobs: **model selection** (scoring 15-dimension capability vectors) and **LLM call execution** (litellm.acompletion, streaming, retries, timeout calculation, error handling). These should be separated. The open question is WHERE the call execution lives:

**Option A: Refactored Dispatcher.** The existing dispatcher (`src/core/llm_dispatcher.py`) already sits between orchestrator and router. Expand it to own the litellm call. Router shrinks to pure scoring, dispatcher becomes the execution layer.

**Option B: New Layer.** A dedicated component between router and the backends. Dispatcher stays as-is (swap budget, MAIN_WORK vs OVERHEAD categorization). The new layer owns the wire.

## Current Architecture

```
Orchestrator (picks tasks from queue)
  → Dispatcher (MAIN_WORK vs OVERHEAD, swap budget, model-aware scheduling)
    → Router (scores models + makes litellm.acompletion call + error handling)
      ├── DaLLaMa (local: process lifecycle)
      └── KDV (cloud: rate limits, quotas, circuit breakers)
```

## What Router Does Today (1,700 lines)

### Model Selection (~lines 370-950)
- 15-dimension capability scoring (fit, cost, availability, performance, ...)
- Graduated rate limit scoring via KDV status
- S7 sibling rebalancing across providers
- Task profile matching
- Local vs cloud preference logic

### Call Execution (~lines 1000-1600)
- `call_model()` — the main function, ~600 lines
- litellm.acompletion() with streaming support
- Retry loop (configurable max_retries)
- Timeout calculation (currently uses runtime_state.measured_tps, thinking_enabled)
- Response parsing and token extraction
- Error classification (rate_limit, auth, timeout, server_error)
- KDV pre_call/post_call/record_failure wiring
- Local model performance logging (tok/s)

### What Dispatcher Does Today (~550 lines, `src/core/llm_dispatcher.py`)
- Categories: MAIN_WORK (can trigger model swaps) vs OVERHEAD (no swaps, uses loaded model)
- Swap budget: max 3 swaps per 5 min
- `request()` — main entry point, calls `call_model()` from router
- Adaptive timeout calculation from measured TPS + thinking mode
- `ensure_gpu_utilized()` — proactive local model loading (contains model selection logic that should be router's job)
- Model state reads: current_model, runtime_state.measured_tps, thinking_enabled
- Deferred grading via GradeQueue

## What Belongs Where (Decided)

| Concern | Owner | Status |
|---------|-------|--------|
| Task queue + scheduling | Orchestrator | Current (has model affinity leak — future cleanup) |
| MAIN_WORK vs OVERHEAD | Dispatcher | Current, stays |
| Swap budget | Dispatcher | Current, stays |
| Model scoring/selection | Router | Current, stays |
| Rate limits, quotas, circuit breakers | KDV | Done (just extracted) |
| Local process management | DaLLaMa | Done |
| **litellm.acompletion call** | **???** | **This design** |
| **Streaming** | **???** | **This design** |
| **Retry loop** | **???** | **This design** |
| **Timeout calculation** | **???** | **This design** |
| **Error classification** | **???** | **This design** |

## Timeout Calculation Detail

Currently lives in dispatcher (lines 182-193) AND router. Dispatcher calculates adaptive timeout from `runtime_state.measured_tps` and `thinking_enabled`. Router applies it in the litellm call. HaLLederiz Kadir would need access to this — either it calculates timeout itself, or dispatcher passes it in.

## Constraints

- DaLLaMa is a dumb can opener — process lifecycle only
- KDV is a dumb pipe — capacity tracking only
- Router should be pure scoring — no I/O
- Whatever owns the litellm call also owns: streaming, retries, error handling, response parsing
- Must work for BOTH local (litellm → DaLLaMa endpoint) and cloud (litellm → provider API)

## Key Files

| File | Lines | Read For |
|------|-------|----------|
| `src/core/llm_dispatcher.py` | ~550 | Current dispatcher — swap budget, categorization, timeout calc |
| `src/core/router.py` | ~1700 | Model selection + call execution (the thing to split) |
| `src/core/router.py:1000-1600` | ~600 | The call execution code specifically |
| `packages/dallama/src/dallama/dallama.py` | ~96 | DaLLaMa interface pattern (for symmetry reference) |
| `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` | ~215 | KDV interface pattern |
| `docs/architecture-modularization.md` | — | Full architecture context |

## Backend Asymmetry (Key Insight)

DaLLaMa and KDV have fundamentally different interfaces:

```
Local:  DaLLaMa.infer(config) → yields URL → litellm.acompletion(api_base=url)
Cloud:  KDV.pre_call() → check ok → litellm.acompletion(model="groq/llama-8b") → KDV.post_call()
```

DaLLaMa participates in the call — it provides the endpoint URL via a context manager that tracks inflight requests. KDV is purely pre/post bookkeeping — it doesn't participate in the call itself.

HaLLederiz Kadir would unify these two call paths behind one interface:

```
TalkingLayer.call(model, messages, ...)
  ├── local: DaLLaMa.infer() → get URL → litellm(api_base=url) → done
  └── cloud: KDV.pre_call() → litellm(model=name) → KDV.post_call() → done
```

Caller doesn't know or care if it's local or cloud. This is a strong argument for Option B (new layer) rather than Option A (refactored dispatcher) — HaLLederiz Kadir's core value is unifying two different backend protocols.

## What To Decide

1. Is this a refactored dispatcher (Option A) or a new component (Option B)?
2. Does it live in `src/core/` or `packages/`? (Agents get confused when logic stays in src/)
3. Who owns timeout calculation?
4. Who owns retry policy? (Currently hardcoded in router, could be configurable)
5. Should streaming be a separate concern or part of HaLLederiz Kadir?
6. Name?
