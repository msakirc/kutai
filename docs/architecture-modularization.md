# Architecture & Modularization

KutAI decomposes into standalone packages to create hard boundaries. Code in `packages/dallama/` is a dependency, not source — agents can't accidentally edit it while fixing a shopping bug. The goal is blast radius control, not PyPI publishing.

## Why It Exists

With 237 Python files in `src/` and no enforced boundaries, an agent fixing a shopping bug could touch router internals, or an agent investigating a model error could reach into DB code. Nothing told it which modules were owned by shopping vs shared infrastructure. Package extraction makes wrong fixes impossible.

## System Layers

```
Telegram / API
       │
  Orchestrator ─── picks tasks from queue by priority, dependencies
       │
  LLM Dispatcher ─── candidate iteration, swap budget, model selection (via Router)
       │
  Talking Layer ─── litellm call, streaming, retry, quality check, metrics
       │
  ┌────┴────┐
  │         │
DaLLaMa   KDV
(local)   (cloud)
  │
llama-server
```

**Orchestrator** reads the task queue, picks next task. Does NOT select models.

**Dispatcher** picks the best model via Router scoring, iterates candidates, manages swap budget. Calls Talking Layer per candidate.

**DaLLaMa** manages the llama-server process. Start, stop, swap, health, idle unload. Three methods: `infer(config)`, `keep_alive()`, `status`. Does NOT select models, route calls, or manage cloud.

**Router** scores models using 15-dimension capability vectors, selects best candidate. Pure scoring, no I/O.

**Talking Layer** executes LLM calls via litellm — builds completion kwargs, manages streaming, retries, response parsing, quality checks (via Doğru mu Samet), and metrics recording.

**Registry** knows all models (local GGUFs + cloud providers). Builds `ServerConfig` for DaLLaMa from `ModelInfo`.

## Data Flow: Local LLM Call

```
Agent needs LLM call
  → Dispatcher.request(messages, MAIN_WORK)
    → Router scores models, picks "qwen3-30b"
    → Dispatcher: ensure_model via DaLLaMa, prepare messages
    → TalkingLayer.call(model, messages, ...)
       → DaLLaMa.infer() → GPU acquire → endpoint URL
       → litellm.acompletion(api_base=url, ...)
       → Parse response, quality check, metrics
    → Response flows back: TalkingLayer → Dispatcher → Agent
  → DaLLaMa marks inference done, resets idle timer
```

## Extracted Packages

| Package | What It Does | Location | Status | Deps |
|---------|-------------|----------|--------|------|
| **yazbunu** | Zero-dep JSONL logging + web viewer (port 9880) | Separate repo ([github](https://github.com/msakirc/yazbunu)) | Stable v0.2.0 | None |
| **vecihi** | Tiered web scraper: HTTP -> TLS -> Stealth -> Browser | `packages/vecihi/` | Stable, ready for own repo | aiohttp |
| **yasar_usta** | Telegram-controlled process manager with heartbeat watchdog | `packages/yasar_usta/` ([github](https://github.com/msakirc/yasar-usta)) | Active development | aiohttp |
| **dogru_mu_samet** | Heuristic LLM output quality detection (degenerate/repetitive) | `packages/dogru_mu_samet/` | Stable | None |
| **dallama** | Async llama-server process manager | `packages/dallama/` | Stable v0.1.0 | httpx |
| **nerd_herd** | Observability: GPU, VRAM budget, health, inference metrics, Prometheus | `packages/nerd_herd/` | Stable v0.1.0 | yazbunu, pynvml, psutil, prometheus_client, aiohttp |
| **kuleden_donen_var** | Cloud provider capacity tracker: rate limits, quotas, circuit breakers | `packages/kuleden_donen_var/` | Stable v0.1.0 | None |
| **talking_layer** | LLM call execution hub: litellm, streaming, retries, quality | `packages/talking_layer/` | New v0.1.0 | litellm |

All packages: `packages/<name>/`, src layout, editable install via requirements.txt. Original module becomes a thin shim preserving all import paths.

## DaLLaMa Internals

```
DaLLaMa (dallama.py, ~90 lines)
  ├── ServerProcess (server.py)   — build cmd, start/stop, poll /health
  ├── SwapManager (swap.py)       — lock, drain inflight, circuit breaker
  ├── HealthWatchdog (watchdog.py) — crash + hang detection → auto-restart
  ├── IdleUnloader (watchdog.py)   — timer-based unload
  ├── MetricsParser (metrics.py)   — GET /metrics → tps, kv_cache
  └── PlatformHelper (platform.py) — Job Objects, orphan cleanup, OS shutdown
```

**Circuit breaker:** After N consecutive load failures for the same model, refuse for cooldown period. Resets on success or cooldown expiry.

**Inference drain:** Before killing the server for a swap, wait for in-flight requests to finish (up to 30s). After timeout, force-drain by bumping a generation counter so orphaned `mark_inference_end()` calls are ignored.

**VRAM check:** Before loading, call `get_vram_free_mb()` (injected by host from gpu_monitor). Refuse if below threshold. DaLLaMa doesn't own pynvml — the host provides the reading.

**Idle unload:** Configurable timeout. `keep_alive()` resets the timer. If nobody calls DaLLaMa within the window, it unloads.

**`on_ready(model, reason)`:** Single callback for all state changes. Reasons: `model_loaded`, `inference_complete`, `idle_unload`, `load_failed`, `circuit_breaker_active`, `circuit_breaker_reset`, `crash_recovery`. Host wires this to wake sleeping tasks.

### Why Not llama-server Router Mode?

llama-server has router mode (`--models-dir`) with hot swap since late 2025. Three blockers:

| Need | Router Mode | Impact |
|------|-------------|--------|
| Per-model ctx/gpu_layers | Broken ([#20851](https://github.com/ggml-org/llama.cpp/issues/20851)) | Can't swap between different-size models |
| Vision projector on-demand | Blocked arg ([#20855](https://github.com/ggml-org/llama.cpp/discussions/20855)) | Wastes 876MB VRAM on non-vision tasks |
| Dynamic context from VRAM | `POST /models/load` takes no flags | Can't calculate context at swap time |

DaLLaMa's `swap.py` is designed for future migration — swap strategy changes from "kill→restart" to "POST /models/load" without touching callers.

**Per-request thinking:** Works today via `thinking_budget_tokens` if server starts without `--reasoning-budget`. Future optimization to avoid restarts for thinking-mode toggling.

## What Was Removed

**GPU Scheduler** (`gpu_scheduler.py`): Priority queue for GPU inference slots. No longer needed — dispatcher checks `dallama.status.busy` instead of queuing. File still exists for shim backward compat; deleted when dispatcher is refactored.

**local_model_manager.py** (1,193 → 449 lines): Replaced with shim wrapping DaLLaMa. All existing import paths preserved.

## Extraction Decisions

### Don't Extract

| Component | Why |
|-----------|-----|
| LLM Router / Dispatcher | Router: 15-dimension scoring is KutAI-shaped. Dispatcher: swap budget + candidate orchestration is KutAI-shaped. Call execution extracted to talking_layer. |
| ReAct Agent Framework | Crowded space (LangGraph, CrewAI). Heavy refactor, marginal value. |
| Workflow Engine | Coupled to DB, blackboard, tools. Not separable. |
| Skills System | Coupled to DB persistence and learning loop. |
| Web Search Pipeline | Value is in the composition. Vecihi covers scraping tier. Rest is glue. |

### Extract Next

**GPU Monitor → nerd_herd (DONE).** Extracted as Nerd Herd. Unified collector for GPU, inference speed, VRAM budget policy, health. Serves Prometheus `/metrics` on port 9881 for Grafana. Prometheus container removed — Nerd Herd pre-computes rates via ring buffers.

**Cloud Operator → kuleden_donen_var (DONE).** Extracted as Kuleden Dönen Var. Tracks rate limits, quotas, and circuit breakers across cloud LLM providers. Reports capacity changes via callback. Router and dispatcher consume status instead of directly managing rate_limiter/header_parser/circuit_breaker.

**Turkish Shopping Scrapers → own package.** 8,500 LOC, 19 scrapers. Largest blast radius. Now connected to workflows, increasing confusion risk.

## Dispatcher Refactoring (Future)

Three components do overlapping scheduling work today:

| Component | Does | Should Do |
|-----------|------|-----------|
| Orchestrator | Picks tasks + model affinity boost | Pick tasks only |
| Dispatcher | Gates swaps + categorizes | Route local/cloud, own swap budget |
| Router | Scores + calls litellm | Same |

Target:
```
Dispatcher ("what to do")
  ├── DaLLaMa ("local backend")
  └── Cloud Operator ("cloud backend" — rate limits, quotas, fallback)
```

Steps: remove model affinity from orchestrator → replace acquire/release_inference_slot with busy check → delete gpu_scheduler → extract cloud operator (rate_limiter, header_parser, quota_planner).

## Model Metadata Standards

No single industry standard. Three complementary sources:

| Layer | Source | Covers |
|-------|--------|--------|
| Architecture | GGUF headers | Params, layers, context, quant |
| Capabilities | LiteLLM cost map | supports_vision, pricing, context windows |
| Richest REST | OpenRouter `/v1/models` | Identity + arch + pricing + modality |

Future registry extraction should align with LiteLLM's field vocabulary.

## Files

| File | What |
|------|------|
| `packages/dallama/` | DaLLaMa package (6 modules + tests) |
| `packages/nerd_herd/` | Nerd Herd observability package (8 modules + tests) |
| `packages/kuleden_donen_var/` | Kuleden Dönen Var package (6 modules + tests) |
| `src/models/local_model_manager.py` | Shim wrapping DaLLaMa |
| `src/models/gpu_scheduler.py` | Deprecated — used only by shim |
| `src/models/gpu_monitor.py` | GPU state — future extraction to observability |
| `src/models/model_registry.py` | Model catalog + capability vectors |
| `src/models/capabilities.py` | 15-dim scoring, task profiles |
| `src/core/llm_dispatcher.py` | Routing policy, swap budget |
| `src/core/router.py` | Model selection (pure scoring, no I/O) |
| `docs/superpowers/specs/2026-04-12-dallama-design.md` | DaLLaMa design spec |
| `packages/talking_layer/` | Talking Layer package (4 modules + tests) |
| `docs/extraction/extraction_report_v2.md` | Extraction decisions and status |

## What Agents Need to Know

**If you're fixing a model loading/swap error:**
The real logic is in `packages/dallama/src/dallama/swap.py` (circuit breaker, drain, VRAM check) and `server.py` (cmd building, health polling). The shim in `src/models/local_model_manager.py` just translates types. Don't edit the shim for swap bugs — fix DaLLaMa.

**If you're fixing a routing/dispatch error:**
Stay in `src/core/llm_dispatcher.py` and `src/core/router.py`. DaLLaMa doesn't route. Don't touch DaLLaMa for routing bugs.

**If you're fixing an agent/task error:**
Stay in `src/agents/` and `src/core/orchestrator.py`. The model layer is a black box — call `Dispatcher.request()`, get a response.

**If you're fixing a GPU/metrics/monitoring error:**
The real logic is in `packages/nerd_herd/src/nerd_herd/`. The shims in `src/models/gpu_monitor.py` and `src/infra/load_manager.py` just delegate. Don't edit the shims for GPU bugs — fix Nerd Herd.

**If you're fixing a cloud rate limit/quota/circuit breaker error:**
The real logic is in `packages/kuleden_donen_var/src/kuleden_donen_var/`. The shims in `src/models/rate_limiter.py` and `src/models/header_parser.py` just delegate. Don't edit the shims for cloud capacity bugs — fix Kuleden Dönen Var.

**If you're fixing an LLM call error (timeout, retry, streaming, response parsing):**
The real logic is in `packages/talking_layer/src/talking_layer/`. The caller in dispatcher just iterates candidates. Don't touch dispatcher or router for call execution bugs — fix the talking layer.

**If you're adding a new package:**
Follow the convention: `packages/<name>/`, src layout, pyproject.toml, editable install. Original module becomes a shim. Preserve all import paths.
