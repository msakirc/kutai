# KutAI Orchestrator X-Ray: Model Routing, Concurrency & Resource Management

> Architecture reference document. All 11 problems and 10 solutions are **implemented**.
> Living document — update as system evolves.

---

## Current Architecture (As-Is)

### Request Flow: Task Creation to Inference

1. **Task arrives** via Telegram or scheduled cron → inserted into SQLite `tasks` table (`status='pending'`)
2. **Main loop** (`orchestrator.py`) runs every ~3s:
   - `get_ready_tasks(limit=8)` — fetches pending tasks, filters by dependency completion, ordered by `priority DESC, created_at ASC`
   - Partitions: only `assistant` agent_type is "cloud-safe"; at most 1 "local" task runs concurrently; rest deferred
3. **`process_task()`**: Claims task → classifies (LLM call) → dispatches to agent
4. **Agent `execute()`**: Builds `ModelRequirements` → enters ReAct loop → each iteration calls `call_model()`
5. **`call_model()`**: Runs `select_model()` (14-dimension scoring) → iterates top 5 candidates → for local: acquires GPU slot + ensures model loaded → for cloud: checks rate limits → executes via litellm
6. **Grading**: After agent completes, `grade_response()` calls `call_model()` with a different model

### call_model() Call Sites (8 total, 6 inline/swap-capable)

| # | File:Line | Purpose | Can Swap? | Inline? |
|---|-----------|---------|-----------|---------|
| 1 | `base.py:1092` | ReAct iteration | YES | No (main work) |
| 2 | `base.py:1836` | Single-shot agents | YES | No |
| 3 | `task_classifier.py:167` | Classification (d=3, speed) | YES | YES |
| 4 | `router.py:1242` | Grading (d=3, excludes generator) | YES | YES |
| 5 | `shopping/_llm.py:55` | Shopping LLM | YES | Inside agent tools |
| 6 | `orchestrator.py:1622` | Subtask classification (up to 8) | YES | YES |
| 7 | `base.py:1342` | Self-reflection | YES | YES |
| 8 | `base.py:1640` | Sub-agent (ask_agent) | YES | YES |

### Scoring Pipeline (select_model)

**5 dimensions** weighted by difficulty tier:
- Capability fit (14-dimension dot product, 0-100)
- Cost efficiency (local=95, free cloud=85, paid tiered)
- Availability (loaded=100, unloaded=35-75, cloud by rate limit headroom)
- Performance history (success_rate * avg_grade, needs 3+ calls)
- Speed (measured TPS for local, provider tier for cloud)

**3-layer scoring (S8)**: Layer 1 (eligibility hard filters incl. coding mismatch), Layer 2 (capability gate), Layer 3 (3 multiplier groups: thinking 1.20x, specialty 1.15x, stickiness 1.40x/1.10x/0.75x)

**Weight profiles by difficulty**:
- d<=3: cost=35, cap=20, avail=20, speed=15, perf=10
- d<=5: cap=30, cost=20, avail=20, perf=15, speed=15
- d<=7: cap=35, speed=20, cost=15, avail=15, perf=15
- d>=8: cap=45, perf=20, speed=20, avail=10, cost=5

### Rate Limiting

- Two-tier: per-model RPM/TPM + per-provider aggregate
- Adaptive reduction on 429: -20% each hit, min 50% of original
- Restoration: +10% after 10 minutes without 429
- Header-based updates from cloud responses
- `QuotaPlanner` has `set_max_upcoming_difficulty()` but it is **never called** (dead code)

### GPU Management

- Single-slot priority queue (`GPUScheduler`)
- Model swap: drain in-flight (30s) → stop server → start new → wait healthy (30-120s)
- Idle unload after 10 minutes
- Thinking state tracked but intentionally NOT enforced at runtime (avoids swap storms)

---

## Problems Identified

### P1: Scattered call_model — No Single Decision Maker
8 separate call sites independently invoke `call_model()`, each constructing own `ModelRequirements`. Each can trigger a swap. No central coordinator knows what's loaded, what's coming, swap count, cloud quota remaining.

### P2: Overhead Calls Treated Same as Main Work
Classification, grading, self-reflection, sub-agent calls all go through same swap-capable path. Single task lifecycle = 3 LLM calls minimum (classify+execute+grade). 67% overhead for single-iteration tasks.

### P3: Grading Forces Swap or Cloud
Grading must exclude generating model. With 1 GPU slot: swap (25s+x2 round-trip), cloud (costs budget), or skip. No mechanism to defer grading until natural model switch.

### P4: No Cloud Quota Reservation
`set_max_upcoming_difficulty()` never called. Easy classifiers burn through cloud quota needed by hard coding tasks later. Peek-ahead doesn't exist.

### P5: GPU Idle When Nothing Loaded
`prefer_local` net multiplier (0.75 x 1.15 = 0.86) loses to cloud. Even without prefer_local, if tasks exist that local can handle, GPU should be loaded — local is free.

### P6: Task Ordering Model-Blind
`get_ready_tasks()` orders by priority only. Doesn't batch compatible tasks per loaded model.

### P7: Scoring Complexity
5 dimensions x 14 capabilities x 4 tiers x 8+ multipliers. Hard to reason about, harder to debug.

### P8: Loaded Model Runtime State Not Tracked
No tracking of actual thinking state, context size allocated, GPU layers offloaded. Scorer uses static ModelInfo.

### P9: Multi-Model Provider Selection Underutilized
Per-model rate limits exist but selection doesn't spread load across provider's models to maximize throughput.

### P10: Fixed Timeouts Don't Match Reality
120s GPU acquire, 60s cloud, 30s drain — don't account for slow inference, deep queues, measured model speed.

### P11: Missing shopping_advisor Task Profile
`CAPABILITY_TO_TASK` maps "shopping" -> "shopping_advisor" but no such profile in TASK_PROFILES.

---

## Solutions (All Implemented)

### S1: Centralized LLM Dispatcher (`src/core/llm_dispatcher.py`)

**Replace ALL call_model() sites with `dispatcher.request()`**

Two call categories:
- **MAIN_WORK**: Agent execution (ReAct, single-shot, shopping, sub-agents). CAN trigger swaps.
- **OVERHEAD**: Classifier, grader, self-reflection, subtask classification. CANNOT trigger swaps — uses loaded model or cloud.

The dispatcher is the ONLY component that can:
- Trigger a model swap (via ensure_model)
- Acquire a GPU slot (via GPUScheduler)
- Decide between local and cloud
- Check and reserve cloud quota

| Category | Can swap? | GPU queue | Cloud fallback |
|----------|-----------|-----------|----------------|
| MAIN_WORK | YES | Full priority queue | Yes, unless local_only |
| OVERHEAD | NEVER | Skip if loaded available | Yes, error on total failure |

No "fail silently" — every failure propagates with context. Calling code decides retry/skip/escalate.

### S2: Deferred Grading Queue

New task state: `executed_pending_grade` between execution and grading.

**Rules:**
1. If loaded model != generator AND capable: grade immediately (free)
2. Otherwise: push to GradeQueue

**GradeQueue drains when:**
- Model swap happens for MAIN_WORK → drain grades old model can handle before swap, then grades new model can handle after
- Cloud quota has headroom → batch-grade via cheapest cloud
- Queue exceeds 20 pending → force drain via cloud
- No MAIN_WORK tasks remain → self-grade as last resort

**Urgent tasks (priority >= 8):** Skip deferred queue, grade immediately via cloud.

Tasks in `executed_pending_grade` are "done" for dependency purposes — downstream tasks can start. Grading updates quality metadata retroactively.

### S3: Cloud Quota Management ✅

**3a. Forward-looking queue scan** (every main loop cycle):
Builds a `QueueProfile` dataclass from upcoming tasks analyzing: max difficulty, vision count, tool count, thinking count, hard_tasks count, cloud_only count. Passed to `QuotaPlanner.set_queue_profile()`.

**3b. Graduated availability scoring:**
Replaced binary `has_capacity()` gate with continuous headroom curve: `avail_score = max(5, 95 - effective_util * 0.90)`. Uses `max(model_util, provider_util)` for effective utilization. Daily limit exhaustion remains a hard gate (`avail_score=0`). Added `RateLimitManager.is_daily_exhausted()`.

**3c. Reserve quota for hard upcoming tasks:**
`QueueProfile.cloud_only_count >= 3` → threshold ≥ 6. `needs_thinking_count >= 2` with moderate util → threshold ≥ 6. Max difficulty ≥ 8 → threshold ≥ max_diff - 1.

### S4: Proactive GPU Loading

When GPU is idle and queue has ANY task a local model can handle (regardless of local_only/prefer_local):
- Scan queue for best local model match
- Load proactively before tasks run
- Local is free; don't waste GPU on principle

Idle unload still applies (10 min no tasks). Proactive loader means GPU ramps up immediately when work appears.

### S5: Model-Aware Task Ordering

After get_ready_tasks(), reorder by model affinity:
- Boost tasks matching loaded model by up to +0.9 priority
- Never override a 2+ priority gap
- `quick_capability_check()` considers: agent type, difficulty, tool usage, vision, thinking needs vs loaded model
- Boost is per-model (what's loaded), reducing swaps by batching compatible work

### S6: Track Loaded Model Runtime State

New `ModelRuntimeState` tracked by LocalModelManager:
- thinking_enabled (actual server state)
- context_length (actual --ctx-size)
- gpu_layers (actual --n-gpu-layers)
- measured_tps (rolling average from recent inferences)

Scorer uses runtime state for loaded model:
- Thinking-heavy task + thinking disabled: reduced stickiness (1.10x instead of 1.40x)
- Context insufficient: hard reject
- Measured TPS: speed scoring and timeout calculation

### S7: Multi-Model Provider Load Balancing

When a model from a provider has high utilization (>70%) and a sibling has low (<30%), slightly penalize the busy model's availability score. Nudges selection toward underutilized models.

### S8: Scoring Reorganization

**Layer 1: Eligibility (pass/fail)** — context, capabilities, constraints, budget
**Layer 2: Capability Gate (threshold)** — reject if below effective_min_score
**Layer 3: Ranking (4 dimensions)** — Capability+Track, Cost, Availability, Speed

**3 post-composite multipliers only:**
1. Loaded stickiness: 1.40x (1.10x if runtime mismatch)
2. Specialty match/mismatch: 1.15x / 0.50x
3. Unloaded penalty: 0.75x

Everything else folds into dimensions.

### S9: Adaptive Timeouts ✅

Based on measured TPS (`ModelRuntimeState.measured_tps`) and estimated tokens:
- **OVERHEAD**: hard cap 20s (via `LLMDispatcher._compute_timeout()`)
- **MAIN_WORK**: `output_tokens / measured_tps * 2.0`, clamped [20, 300]s; difficulty heuristic fallback
- **GPU acquire**: `max(30, min(180, est_gen * 3.0 + 15))` using measured TPS; difficulty heuristic when no TPS; priority ≥ 10 → 30s (fail fast to cloud)

### S10: Swap Budget

Max 3 swaps per 5 minutes. Exemptions: local_only (no choice), priority>=9 (urgent). When exhausted: use loaded model (suboptimal) or cloud.

---

## Implementation Order (All Complete)

| Phase | Solution | Commit | Tests |
|-------|----------|--------|-------|
| 1 | S1: LLM Dispatcher + S10: Swap Budget + S2: Deferred Grading | `072a156` | 27 |
| 2 | S3c: QuotaPlanner queue scan | `d16afca` | — |
| 3 | S4: Proactive GPU loading | `dd46a03` | — |
| 4 | S5: Model-aware task ordering + shopping_advisor profile | `6908aa2` | 15 |
| 5 | S6: Runtime state tracking | `017da5c` | 17 |
| 6 | S7: Provider sibling rebalancing | `5deb446` | 8 |
| 7 | S8: 3-layer scoring architecture | `574d443` | 12 |
| 8 | S9: Adaptive timeouts (LLM call + GPU acquire) | `58f3ecc` | 11 |
| 9 | S3a: Queue profile + S3b: Graduated availability | (pending commit) | 15 |

---

## Key Metrics to Track

- **Swaps per hour**: Target < 6 (currently unbounded)
- **Overhead LLM calls per task**: Target < 1.5 (currently ~2.0)
- **GPU idle time with pending tasks**: Target 0%
- **Cloud quota utilization**: Target even spread across providers
- **Grade queue depth**: Monitor, alert if > 30
- **Task completion latency p50/p95**: Track improvement over time
