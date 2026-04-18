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
  LLM Dispatcher ─── ask, load, call, retry
       │
  ┌────┼──────────────┐
  │    │              │
Fatih Hoca    Talking Layer    DaLLaMa
(model        (litellm call,   (local model
 selection)    streaming,       process mgmt)
  │            retry, quality)     │
Nerd Herd         │           llama-server
(system       ┌───┴───┐
 state)       │       │
           DaLLaMa   KDV
           (GPU sem)  (cloud capacity)
```

**Orchestrator** reads the task queue, picks next task. Does NOT select models.

**Dispatcher** asks Fatih Hoca for a model, loads it via DaLLaMa, calls the Talking Layer, reports failures back for re-selection. Owns message preparation (secret redaction, thinking adaptation) and timeout floors. Does NOT score or select models.

**Fatih Hoca** (planned extraction) is the model manager — knows every model's strengths, picks the best one for each job. Owns model catalog, 15-dimension capability scoring, swap budget, failure-adaptive re-selection. Queries Nerd Herd for system state. See `docs/superpowers/specs/2026-04-14-fatih-hoca-design.md`.

**DaLLaMa** manages the llama-server process. Start, stop, swap, health, idle unload. Three methods: `infer(config)`, `keep_alive()`, `status`. Does NOT select models, route calls, or manage cloud.

**Talking Layer** executes LLM calls via litellm — builds completion kwargs, manages streaming, retries, response parsing, quality checks (via Doğru mu Samet), and metrics recording.

**Nerd Herd** is the unified system state provider — GPU, VRAM, loaded model state (from DaLLaMa), cloud provider capacity (from KDV), inference metrics. Single `snapshot()` call for consumers.

**KDV** tracks cloud provider rate limits, quotas, and failures. Pushes state changes to Nerd Herd.

## Data Flow: Local LLM Call

```
Agent needs LLM call
  → Dispatcher.request(category, task, messages, ...)
    → Fatih Hoca.select(task, difficulty, failures, ...)
       → Nerd Herd.snapshot() → system state
       → Score models, pick best → Pick(model, min_time)
    → DaLLaMa.ensure_model(model) → swap if needed
    → Dispatcher: prepare messages, compute timeout
    → TalkingLayer.call(model, messages, timeout, ...)
       → DaLLaMa.infer() → GPU acquire → endpoint URL
       → litellm.acompletion(api_base=url, ...)
       → Parse response, quality check, metrics
    → Response flows back: Talking Layer → Dispatcher → Agent
    → On failure: Dispatcher adds Failure, calls select() again
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
| **hallederiz_kadir** | LLM call execution hub: litellm, streaming, retries, quality | `packages/hallederiz_kadir/` | New v0.1.0 | litellm |
| **fatih_hoca** | Model manager: scoring, selection, swap budget, failure adaptation | `packages/fatih_hoca/` | Stable v0.1.0 | nerd_herd |

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

**GPU Scheduler** (`gpu_scheduler.py`): Priority queue for GPU inference slots. Dead code — scheduled for deletion.

**local_model_manager.py** (1,193 → 449 lines): Shim wrapping DaLLaMa. Scheduled for deletion when Fatih Hoca extraction removes all callers.

**Router dead code** (removed 2026-04-14): `RateLimiter` class (replaced by KDV), `refresh_perf_cache()` (never called), perf cache globals (never populated). Dispatcher unified `_route_main_work` / `_route_overhead` into single candidate loop, removed dead `partial_buf`/`on_chunk` params.

**Fatih Hoca extraction** (2026-04-14): `model_registry.py` (2050→39 lines), `capabilities.py` (492→13 lines), `model_profiles.py` (1557→10 lines), `quota_planner.py` (229→2 lines) all became thin shims re-exporting from `packages/fatih_hoca/`. `router.py` (856→488 lines) kept `select_model()`, `call_model()`, `get_kdv()` but re-exports `ModelRequirements`, `ScoredModel`, `AGENT_REQUIREMENTS` from fatih_hoca.

## Extraction Decisions

### Don't Extract

| Component | Why |
|-----------|-----|
| LLM Dispatcher | Coordinator that wires Fatih Hoca + DaLLaMa + Talking Layer. Its value is in the wiring, not reusable logic. |
| ReAct Agent Framework | Crowded space (LangGraph, CrewAI). Heavy refactor, marginal value. |
| Workflow Engine | Coupled to DB, blackboard, tools. Not separable. |
| Skills System | Coupled to DB persistence and learning loop. |
| Web Search Pipeline | Value is in the composition. Vecihi covers scraping tier. Rest is glue. |

### Extract Next

**GPU Monitor → nerd_herd (DONE).** Extracted as Nerd Herd. Unified collector for GPU, inference speed, VRAM budget policy, health. Serves Prometheus `/metrics` on port 9881 for Grafana. Prometheus container removed — Nerd Herd pre-computes rates via ring buffers.

**Cloud Operator → kuleden_donen_var (DONE).** Extracted as Kuleden Dönen Var. Tracks rate limits, quotas, and circuit breakers across cloud LLM providers. Pushes state changes to Nerd Herd.

**Model Manager → fatih_hoca (DONE).** Router + registry + capabilities + quota planner merged into Fatih Hoca. 7 modules, 253 tests. Original files are thin shims preserving import paths. Dispatcher simplification (converting request() to use fatih_hoca.select()) is planned as a follow-up. See `docs/superpowers/specs/2026-04-14-fatih-hoca-design.md`.

**Nerd Herd expansion (DONE, part of Fatih Hoca).** Added `SystemSnapshot`, `LocalModelState`, `CloudProviderState` types. `snapshot()` method on NerdHerd. Push-based state from DaLLaMa (`push_local_state`) and KDV (`push_cloud_state`).

**Turkish Shopping Scrapers → own package.** 8,500 LOC, 19 scrapers. Largest blast radius. Now connected to workflows, increasing confusion risk.

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
| `packages/hallederiz_kadir/` | Talking Layer package (4 modules + tests) |
| `packages/fatih_hoca/` | Fatih Hoca model manager (7 modules, 152 tests) |
| `src/models/local_model_manager.py` | Shim wrapping DaLLaMa — pushes state to Nerd Herd |
| `src/models/gpu_scheduler.py` | Deprecated — still imported by local_model_manager |
| `src/models/gpu_monitor.py` | Shim wrapping Nerd Herd — still imported by local_model_manager |
| `src/models/model_registry.py` | Shim re-exporting from `fatih_hoca.registry` |
| `src/models/capabilities.py` | Shim re-exporting from `fatih_hoca.capabilities` |
| `src/models/quota_planner.py` | Shim re-exporting from `fatih_hoca.requirements` |
| `src/models/model_profiles.py` | Shim re-exporting from `fatih_hoca.profiles` |
| `src/core/llm_dispatcher.py` | Ask-load-call-retry loop (dispatcher simplification pending) |
| `src/core/router.py` | Shim — `ModelRequirements`/`ScoredModel` from fatih_hoca, keeps `select_model()`/`call_model()` |
| `docs/superpowers/specs/2026-04-12-dallama-design.md` | DaLLaMa design spec |
| `docs/superpowers/specs/2026-04-14-fatih-hoca-design.md` | Fatih Hoca design spec |
| `docs/extraction/extraction_report_v2.md` | Extraction decisions and status |

## What Agents Need to Know

**If you're fixing a model loading/swap error:**
The real logic is in `packages/dallama/src/dallama/swap.py` (circuit breaker, drain, VRAM check) and `server.py` (cmd building, health polling). The shim in `src/models/local_model_manager.py` just translates types. Don't edit the shim for swap bugs — fix DaLLaMa.

**If you're fixing a model selection/scoring error:**
Stay in `packages/fatih_hoca/` (or `src/core/router.py` + `src/models/` pre-extraction). DaLLaMa doesn't route. Dispatcher doesn't score. Don't touch them for scoring bugs.

**If you're fixing a dispatch/routing error:**
Stay in `src/core/llm_dispatcher.py`. Dispatcher is a thin loop — ask Fatih Hoca, load via DaLLaMa, call Talking Layer. Don't touch scoring for dispatch bugs.

**If you're fixing an agent/task error:**
Stay in `src/agents/` and `src/core/orchestrator.py`. The model layer is a black box — call `Dispatcher.request()`, get a response.

**If you're fixing a GPU/metrics/monitoring error:**
The real logic is in `packages/nerd_herd/src/nerd_herd/`. Don't edit shims for GPU bugs — fix Nerd Herd.

**If you're fixing a cloud rate limit/quota/capacity error:**
The real logic is in `packages/kuleden_donen_var/src/kuleden_donen_var/`. Don't edit shims for cloud capacity bugs — fix Kuleden Dönen Var.

**If you're fixing an LLM call error (timeout, retry, streaming, response parsing):**
The real logic is in `packages/hallederiz_kadir/src/hallederiz_kadir/`. Dispatcher just calls it. Don't touch dispatcher or scoring for call execution bugs — fix the Talking Layer.

**If you're adding a new package:**
Follow the convention: `packages/<name>/`, src layout, pyproject.toml, editable install.

---

## Phase 1 In-Tree Refactor (2026-04-17)

Orchestrator untangled in-tree without package extraction. New in-tree modules:

- `src/core/decisions.py` — `Allow`, `Block`, `Cancel`, `GateDecision`, `Dispatch`, `NotifyUser` types (Phase 2b targets)
- `src/core/task_context.py` — centralized `parse_context` / `set_context` helpers
- `src/core/task_gates.py` — async `run_gates(task, task_ctx, approval_fn) -> GateDecision`
- `src/core/result_router.py` — pure `route_result(task, agent_result) -> list[Action]` (dataclasses + state machine). **Not wired into `process_task` yet** — existing branches have 30-80 lines of per-status guard logic that doesn't fit a simple handler dispatch. Phase 2b handles this.
- `src/core/watchdog.py` — `check_stuck_tasks` + `check_resources` (natural 2-function split, not 4)
- `src/core/mechanical/` — in-tree home for non-LLM executors (Phase 2a target): `workspace_snapshot.py` live; `git_commit.py` dormant (call site disconnected per user direction)
- `src/app/scheduled_jobs.py` — proactive/cron-triggered jobs (todo reminders, API discovery, daily digest, price watches)

**Line counts:**

- `src/core/orchestrator.py`: 3,865 → 2,800 (−1,065, −28%)
- `process_task`: 1,143 → 945 (−198, via `_prepare` extraction; `_dispatch` / `_record` deferred)
- `watchdog`: 519 → ~10 line delegator

**What shipped:**
- Task 1: Decision vocabulary
- Task 2: Centralized task context parsing (4+ call sites → one module)
- Task 3: `assess_risk` made async (no more event-loop blocking)
- Task 4: Gate logic extracted (`run_gates` returns `GateDecision`; orchestrator still owns Telegram I/O via injected `approval_fn`)
- Task 5: `result_router` module and tests; wire-up deferred to Phase 2b
- Task 6: `mechanical/` directory with `workspace_snapshot`; `_auto_commit` dormant pending i2p refactor
- Task 7: Proactive jobs moved to `ScheduledJobs` class (todo reminders, API discovery, digest, price watches)
- Task 8 (scoped): `_prepare` stage extracted; `_dispatch` / `_record` stay inline pending Phase 2b
- Task 9: `watchdog` split into focused async module functions

**Deferred to Phase 2:**
- `result_router` wire-up in `process_task` (needs handler signature normalization + guard-code triage)
- `process_task` reduction from 945 → ~25 lines (depends on router wire-up)
- `mechanical/` extraction to `packages/mechanical_dispatcher/` (Phase 2a)
- Gates / context / watchdog extraction to `packages/gorev_ustasi/` (Phase 2b)
- `_handle_*` → Decision emissions (inverts Telegram coupling fully)

---

## Phase 2 — Plan A (2026-04-18): in-tree untangle of `process_task`

Finished the two items Phase 1 deferred: `result_router` wire-up (D1) and
the `process_task` split (D2).  Plan stayed in-tree; no new packages.

**Line-count impact:**
- `process_task`: 945 → 27 lines (target was < 50)
- `src/core/orchestrator.py`: 2,800 → 2,548
- `src/core/result_router.py`: 97 → 109 (`raw: dict` added to every Action
  so handlers keep their `(task, result)` signatures — Option 1)
- `src/core/result_guards.py`: **new** (353 lines)

**What shipped (Tasks 1–7 of `docs/superpowers/plans/2026-04-18-orchestrator-plan-a-in-tree.md`):**
- `_handle_exhausted` / `_handle_failed` extracted verbatim from the inline
  `if/elif` status chain
- Every `result_router.Action` type gained a `raw: dict` field so the router
  can be wired through without touching any existing `_handle_*` signatures
- New `src/core/result_guards.py` module: typed `GuardHandled` + async
  guards for workflow post-hook, clarification suppression, subtask blocking,
  pipeline artifacts, ungraded post-hook (the pre-handler logic that used
  to sit inline inside `process_task`)
- New `Orchestrator._dispatch_action` isinstance-dispatches router Actions
  to their matching `_handle_*` method
- New `Orchestrator._run_guards_for` runs the guards that apply to each
  Action type; returns True when a guard fully consumed the task
- `_dispatch` owns agent/pipeline invocation + timeout wrapper + partial
  result recovery; returns `None` if the timeout handler consumed the task
- `_record` owns status routing: `ungraded`/`pending` inline + everything
  else via `route_result` → `_run_guards_for` → `_dispatch_action`
- Outer exception handlers moved to `_handle_availability_failure` and
  `_handle_unexpected_failure`
- `process_task` is now a 27-line orchestrator chaining
  `_prepare` → `_dispatch` → `_record`

**Still deferred to Phase 2b:**
- Moving `_handle_*` out of `Orchestrator` into a package
- Moving `result_guards` into a package
- Replacing the `ungraded` inline branch with a router Action type
- Inverting the `self.telegram.*` coupling inside the guards
