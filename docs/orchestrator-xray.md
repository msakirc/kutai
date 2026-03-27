# KutAI Orchestrator X-Ray: Model Routing, Concurrency & Resource Management

> Future architecture document. Describes current state problems and planned solutions.
> Living document — update as implementation progresses.

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

**8+ post-composite multipliers**: thinking 1.20x, specialty match 1.15x, coding mismatch 0.50x, prefer_local 1.15x, loaded stickiness 1.40x, unloaded 0.75x

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

## Planned Solutions

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

### S3: Cloud Quota Management

**3a. Forward-looking queue scan** (every main loop cycle):
```
upcoming = get_ready_tasks(limit=30)
queue_profile = analyze(upcoming)  # difficulty, vision, tools, thinking, priorities
quota_planner.update_queue_profile(queue_profile)
quota_planner.recalculate()
```

**3b. Per-model utilization in selection:**
Replace binary has_capacity() with headroom-weighted availability score. Naturally spreads load across provider's models.

**3c. Reserve quota for hard upcoming tasks:**
When queue contains difficulty>=7 tasks, raise expensive_threshold so easy overhead doesn't consume paid cloud.

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

### S9: Adaptive Timeouts

Based on measured TPS and estimated tokens:
- OVERHEAD: hard cap 20s
- MAIN_WORK: generation_time * 2.0, clamped 20-300s
- GPU acquire: estimated_queue_wait + model_load + generation + buffer

### S10: Swap Budget

Max 3 swaps per 5 minutes. Exemptions: local_only (no choice), priority>=9 (urgent). When exhausted: use loaded model (suboptimal) or cloud.

---

## Implementation Order

| Phase | Solution | Effort | Dependencies |
|-------|----------|--------|-------------|
| 1 | S1: LLM Dispatcher | Large | None — foundation |
| 2 | S3c: QuotaPlanner queue scan | Small | S1 |
| 3 | S4: Proactive GPU loading | Medium | S1 |
| 4 | S2: Deferred grading queue | Medium | S1 |
| 5 | S5: Model-aware task ordering | Medium | S1 |
| 6 | S6: Runtime state tracking | Small | S1 |
| 7 | S3a+S3b: Full quota management | Medium | S1, S6 |
| 8 | S7: Provider load balancing | Small | S3b |
| 9 | S8: Scoring reorganization | Large | All above stable |
| 10 | S9+S10: Timeouts + swap budget | Medium | S6 |
| 11 | Add shopping_advisor task profile | Tiny | None |

---

## Key Metrics to Track

- **Swaps per hour**: Target < 6 (currently unbounded)
- **Overhead LLM calls per task**: Target < 1.5 (currently ~2.0)
- **GPU idle time with pending tasks**: Target 0%
- **Cloud quota utilization**: Target even spread across providers
- **Grade queue depth**: Monitor, alert if > 30
- **Task completion latency p50/p95**: Track improvement over time
