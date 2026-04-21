# Pool Pressure — Shared Primitive + Beckman Admission — Design Spec

**Date:** 2026-04-21
**Status:** Approved design, awaiting implementation plan
**Owners:** `packages/kuleden_donen_var/` (kdv), `packages/nerd_herd/`, `packages/fatih_hoca/`, `packages/general_beckman/`, `src/core/llm_dispatcher.py`

## 1. Motivation

Phase 2d introduced a continuous pool-pressure signal inside Fatih Hoca's scarcity layer: `remaining_frac × exp(-reset_in/24h)` with depletion, abundance, and queue arms. This signal drives Hoca's per-call model routing with proven equilibrium behaviour.

Two gaps remain:

1. **t=0 burn risk for cloud pools.** When multiple tasks dispatch in the same tick, each sees the same stale `remaining` count before any call lands on the wire. With 30 daily Claude credits, a parallel burst of 6 can drain 20% in seconds. KDV's confirmed-usage counter only updates on response headers — mid-flight calls are invisible.

2. **Beckman's admission decision lacks cloud awareness.** Current `next_task()` gates dispatch behind a single `system_busy` bit from nerd_herd. It cannot distinguish "cloud about to reset, burn through it" from "cloud near depletion, hold back." This leaves quota wasted at reset boundaries (`use it or lose it`) and risks bursting through tight pools.

The two problems share a primitive: **pool pressure as a signed signal in [-1, +1] per cloud (provider, model), accounting for confirmed usage, reset horizon, and in-flight dispatches.** Hoca already derives this for per-call routing. Beckman needs the same primitive for admission decisions. Making it a shared, snapshot-exposed field eliminates duplication and closes both gaps.

This spec describes:

- A new `in_flight` counter in KDV that tracks cloud calls between dispatch and response.
- A `pool_pressure` field derived from KDV state, exposed via `nerd_herd.snapshot()`.
- A Beckman admission algorithm using `snapshot.pressure_for(model)` and task-specific urgency.
- A precondition refactor making Hoca's `select()` pure, enabling Beckman to call it freely without telemetry or swap-budget side effects.

## 2. Scope

**In scope:**

- KDV tracks in-flight cloud calls via `begin_call` / `end_call` API. In-flight counter decays via TTL safety net.
- Pool pressure computed lazily from KDV push-state during `snapshot()` build. Exposed via `snapshot.pressure_for(model)` accessor.
- Beckman extended: `next_task()` scans top-K ready tasks by urgency, asks Hoca for a Pick per candidate, admits based on `pressure_for(pick.model)` vs urgency-derived threshold.
- Queue profile (2 fields: `hard_tasks_count`, `total_ready_count`) pushed from Beckman to nerd_herd on queue-change events. Feeds Hoca's per-call scarcity.
- Dispatcher's iteration loop wraps each cloud-bound call with `begin_call` / `end_call`.
- Hoca purity refactor (precondition): move swap-budget counter to nerd_herd, move `model_pick_log` write to dispatcher.

**Out of scope:**

- Priority scheduling rework. Beckman's `pick_ready_task` ordering logic stays as-is; urgency composition only adjusts admission threshold.
- Local-pool saturation smoothing beyond the existing `LocalModelState`-derived pressure (idle/busy/swapping).
- Cloud provider autoscaling, multi-region failover, or provider cost optimization.
- Simulator reshape. Phase 2d scenarios must keep passing; Beckman-level scenarios are added incrementally.

## 3. Architecture

### 3.1 Layers and responsibilities

```
┌──────────────────┐
│ Beckman          │  owns queue, decides "dispatch yes/no" per ready task
│                  │  (reads snapshot.pressure_for(pick.model), applies threshold)
└─────┬────────────┘
      │ push queue_profile on queue change
      │ call hoca.select() per candidate
      ▼
┌──────────────────┐
│ Fatih Hoca       │  owns model selection (PURE)
│ (stateless)      │  reads snapshot; returns Pick; no side effects
└─────┬────────────┘
      │ reads snapshot
      ▼
┌──────────────────┐
│ nerd_herd        │  aggregates pushed state + computes derived views
│ (push-based)     │  pool_pressure derived lazily; queue_profile stored from Beckman push
└─────┬────────────┘
      │ receives push events
      ▲
      │
┌──────────────────┐
│ KDV              │  tracks cloud quota (remaining, limit, reset_at) + in_flight
│                  │  pushes CloudProviderState to nerd_herd on call boundaries
└──────────────────┘

┌──────────────────┐
│ Dispatcher       │  iteration loop: select → load or begin_call → call → end_call
│ (dumb pipe)      │  writes model_pick_log post-dispatch; records swap to nerd_herd
└──────────────────┘
```

**Single-owner principles:**

- Queue state: Beckman. Queue_profile pushed to nerd_herd for read access by Hoca.
- Model/pool decisions: Hoca. Consulted by dispatcher per iteration and by Beckman per admission probe.
- Cloud rate-limit state (including in-flight): KDV. Pushed to nerd_herd.
- Local/GPU state: nerd_herd's existing collectors + DaLLaMa push.
- Swap-budget counter: nerd_herd (monitoring). Hoca reads it during scoring. Dispatcher writes to it post-swap.
- Pick telemetry (`model_pick_log`): dispatcher writes after actual dispatch completes.

### 3.2 Push-based integration with nerd_herd (no new tick)

Nerd_herd today is a push-aggregator: KDV pushes cloud state on each provider response; DaLLaMa pushes local state on each swap; GPU collector has its own internal TTL cache. There is no background tick. `NerdHerd.snapshot()` builds fresh on every call from the most recent pushed state.

This spec preserves that pattern:

- **KDV push points (extended):** same as today, plus `in_flight` field in the pushed `CloudProviderState.limits.rpd.in_flight`.
- **Beckman push points (new):** Beckman pushes queue_profile on queue-change events: `enqueue`, `on_task_finished`, `sweep`. Receiver: `nerd_herd.push_queue_profile(profile)`.
- **Pool pressure:** NOT pushed. Computed lazily inside `NerdHerd.snapshot()` from pushed cloud state.

No new coroutine, no new tick. All ingestion remains event-driven push.

### 3.3 In-flight tracking in KDV

**New API surface on kdv:**

```python
@dataclass
class InFlightHandle:
    provider: str
    model: str
    started_at: float
    ttl_s: float
    token: uuid.UUID

def begin_call(provider: str, model: str, ttl_s: float = 180.0) -> InFlightHandle
def end_call(handle: InFlightHandle, result: dict | Exception | None = None) -> None
```

**State:**

KDV maintains a per-(provider, model) in-flight list:
```python
_in_flight: dict[tuple[str, str], list[InFlightHandle]] = {}
```

On `begin_call`:
- Prune expired handles (lazy cleanup).
- Append new handle.
- Push updated `CloudProviderState` (with new in-flight count) to nerd_herd.

On `end_call`:
- Remove matching handle by token.
- Apply header-derived updates to confirmed remaining/reset (existing KDV logic).
- Push updated state to nerd_herd.

**Effective remaining (consumed by pool_pressure):**

```python
effective_remaining = max(0, limit - confirmed_used - in_flight_count)
remaining_frac = effective_remaining / limit
```

**TTL semantics:**

- Default TTL: 180 seconds. Env override: `KDV_INFLIGHT_TTL_S`.
- Happy path: `end_call` fires from dispatcher's try/finally well before TTL expiry. TTL never triggers.
- Failure path (dispatcher crash mid-call, process SIGKILL, power loss): TTL-expired handle is pruned on next `begin_call` or snapshot build. Prevents permanent in-flight leaks.
- TTL is long enough to exceed any legitimate cloud call (Claude with thinking on hard tasks peaks around 90s; 180s gives 2x margin).

**Cleanup:** lazy. Pruning runs on each `begin_call`, and during `snapshot()` build when pool pressure is computed. No background task.

### 3.4 Pool pressure primitive

Computed inside `nerd_herd.snapshot()`, cached per snapshot instance (lazy-on-first-read).

```python
@dataclass
class PoolPressure:
    value: float                # signed, [-1, +1]
    depletion: float            # [-1, 0] arm
    abundance: float            # [0, +1] arm
    time_weight: float          # exp(-reset_in/86400); 0 for per_call pools
    in_flight_count: int        # informational

def compute_pool_pressure(
    remaining: int,
    limit: int,
    reset_at: int | None,
    in_flight_count: int,
) -> PoolPressure:
    ...
```

**Arms** (unchanged from Phase 2d, relocated):

- **Depletion** (arm 1): low `remaining_frac` → strongly negative. Activates below 15% remaining.
- **Abundance** (arm 2): high `remaining_frac` + imminent reset → strongly positive. `exp(-reset_in/86400)` continuous weighting.
- Queue-aware arm from Phase 2d (`_per_call_scarcity` arm 3) stays inside Fatih Hoca's scarcity layer. It consumes `queue_profile` from snapshot. It is NOT part of the shared primitive, because Beckman does not need it (Beckman already reasons about queue directly).

Per-pool burn-rate cap: unchanged; abundance arm saturates at +1, so no runaway positive signal from a flush pool with slow reset.

### 3.5 Queue profile

```python
@dataclass
class QueueProfile:
    hard_tasks_count: int      # ready tasks with difficulty >= 7
    total_ready_count: int     # ready tasks (excluding blocked)
    # Other fields (oldest_age_s, blocked_count, priority_histogram) deferred
    # until a consumer actually needs them.
```

**Push sites (Beckman → nerd_herd):**

- After `enqueue(spec)`: new ready task may shift counts.
- After `on_task_finished(task_id, result)`: completion may unblock tasks, shifting ready count.
- After `sweep_queue(...)`: DLQ / stuck-task moves may shift counts.

**Consumer:** Fatih Hoca's `scarcity._per_call_scarcity` queue arm reads `snapshot.queue_profile`. Beckman does NOT consume this from snapshot — it reads its own queue directly when computing urgency (freshest possible).

### 3.6 snapshot.pressure_for accessor

```python
def pressure_for(self, model: ModelInfo) -> float:
    if model.is_local:
        return self._local_pressure()    # derived from self.local (idle/busy)
    prov = self.cloud.get(model.provider)
    if prov is None:
        return 0.0                        # no data → neutral
    m = prov.models.get(model.name)
    if m is None:
        return self._provider_level_pressure(prov)   # fallback (anthropic case)
    if m.pool_pressure is None:
        m.pool_pressure = compute_pool_pressure(
            remaining=m.limits.rpd.remaining,
            limit=m.limits.rpd.limit,
            reset_at=m.limits.rpd.reset_at,
            in_flight_count=m.limits.rpd.in_flight,
        )
    return m.pool_pressure.value
```

**Caching semantics:**

- Snapshot instances are constructed fresh per call; cache lives on the returned `SystemSnapshot` dataclass.
- First `pressure_for(model)` computes and stores. Subsequent reads hit cache.
- No async contention (single-threaded asyncio).

## 4. Precondition refactor — Hoca purity

Hoca's `select()` today has two side effects that would pollute Beckman's admission probes:

1. `SwapBudget.record_swap()` at `selector.py:195` — commits swap budget consumption during selection.
2. Fire-and-forget `INSERT INTO model_pick_log` at `selector.py:244-299` — writes telemetry during selection.

**Target state: `select()` is pure.** No mutations. Callable from Beckman, dispatcher, and simulator identically.

### 4.1 Swap budget → nerd_herd (monitor)

- Move `SwapBudget` state into nerd_herd (e.g., `nerd_herd.swap_budget`).
- Nerd_herd exposes read API: `recent_swap_count(local_only, priority) -> int` (consumed by Hoca scoring).
- Nerd_herd exposes write API: `record_swap(model_name)` (called by dispatcher after successful swap).
- **Decision authority stays in Hoca.** Hoca reads the counter and decides whether to score a swap-requiring pick as feasible. Nerd_herd is a dumb monitor.
- **Swap execution stays in dispatcher.** Dispatcher already triggers swaps via `ensure_local_model`. After a successful swap, dispatcher calls `nerd_herd.record_swap(model_name)`.
- If Hoca scored assuming a swap was viable but dispatcher fails to execute (e.g., DaLLaMa error), the counter simply doesn't increment. No corruption.

### 4.2 model_pick_log → dispatcher

- Remove `_log_pick` call (and DB INSERT) from `selector.py`.
- Dispatcher writes the log row after its iteration completes (success or failure), using the `Pick` it received from Hoca plus the actual outcome.
- Benefit: log records **what was actually dispatched**, not what was theoretically picked. Admission rejections from Beckman no longer pollute the distribution.

### 4.3 Audit `select_for_simulation`

`selector.py:481` has a separate pure variant for simulation. After purity refactor, consolidate: `select()` and `select_for_simulation()` should share the same code path (or `select_for_simulation` deletes in favour of `select()`).

## 5. Beckman admission algorithm

### 5.1 Flow

```python
async def next_task() -> Task | None:
    # Beckman's own in-flight task count (queue bookkeeping).
    # Distinct from kdv's in-flight *cloud call* counter.
    dispatched_count = await count_currently_dispatched_tasks()
    if dispatched_count >= BECKMAN_HARD_CAP:
        return None

    snapshot = nerd_herd.snapshot()

    for task in pick_ready_top_k(k=BECKMAN_TOP_K):
        pick = fatih_hoca.select(
            task=task.profile,
            agent_type=task.agent_type,
            difficulty=task.difficulty,
            ...,
        )
        if pick is None:
            continue   # no eligible model for this task; try next candidate

        pressure = snapshot.pressure_for(pick.model)
        urgency = _compute_urgency(task)

        if pressure >= _threshold(urgency):
            task.preselected_pick = pick
            return task
        # Rejection here does NOT permit early break: the next candidate
        # may receive a Pick for a different model whose pool pressure
        # passes its (higher) threshold. Always continue through K.

    return None
```

### 5.2 Urgency composition

```python
def _compute_urgency(task: Task) -> float:
    priority_term = task.priority / 10.0                          # [0.1, 1.0]
    age_term = min(1.0, task.age_seconds / 86400) * 0.05          # [0, 0.05]
    unblock_count = task.downstream_unblocks_count                # weak tiebreaker
    blocker_term = min(1.0, unblock_count / 5.0) * 0.05           # [0, 0.05]
    return max(0.0, min(1.0, priority_term + age_term + blocker_term))
```

Scale parameters (`AGE_SCALE = 86400`, blocker cap `5`, term weights `0.05`) are simulator-tunable. Starting values chosen so age + blocker combined max +0.10 on top of priority, cannot flip a lower-priority task above a higher-priority one.

**Implementation follow-up:** `task.age_seconds` (derived from `created_at`) and `task.downstream_unblocks_count` (derived from queue dependency graph) may not exist as ready-made fields on Beckman's Task type today. Implementation must add helpers to compute them — preferably as `@property` on Task, reading from existing columns, not new persisted state.

### 5.3 Threshold function

```python
def _threshold(urgency: float) -> float:
    return max(-1.0, min(1.0, 0.5 - urgency))
```

Pure linear. No floor. Intercept `0.5` means an idle task (urgency ≈ 0) needs clearly abundant pool pressure (≥ +0.5) to admit. Max-urgency task (≈ 1.0) accepts mild depletion (≥ -0.5). Simulator calibrates intercept + slope.

**Deadlock mitigation:** no hard floor. If the only eligible pool is at -0.6 and the task is mission-critical, Beckman still admits (threshold_at_urgency_1 = -0.5, task urgency must be > 1 to reach -0.6, caps at 1.0). Hoca's own scoring further disfavours depleted pools — second layer of defence.

### 5.4 Preselected pick hand-off

Beckman attaches the Hoca `Pick` onto the returned Task (new field `Task.preselected_pick`). Dispatcher's iteration 1 uses this pick directly — no redundant Hoca call. Iteration 2+ invokes `fatih_hoca.select()` fresh (per existing retry/swap behaviour).

No staleness guard needed: dispatcher runs synchronously off `next_task()`. No async backpressure between Beckman's return and dispatcher's first iteration.

### 5.5 K (top-K scan)

`K = 5` provisional. Rationale: during admission, Beckman probes up to 5 candidates per call. Orchestrator pump loops `next_task()` until None, so multi-task draining happens naturally across successive calls. K>5 adds overhead; K<3 risks missing dispatchable tasks behind stuck leaders.

Tuneable via `BECKMAN_TOP_K`. Final value set from measured `hoca.select()` latency: if average < 30 ms, K=5 or higher; if > 100 ms, drop to 3.

## 6. Dispatcher iteration loop

### 6.1 Updated shape

```python
async def request(category, task, ..., preselected_pick=None):
    failures = []
    for iteration in range(MAX_ITERATIONS):
        pick = preselected_pick if iteration == 0 and preselected_pick else fatih_hoca.select(
            task=task, ..., failures=failures,
        )
        preselected_pick = None   # only valid for iteration 0

        if pick is None:
            raise ModelCallFailed(...)

        model = pick.model

        if model.is_local:
            await ensure_local_model(model, ...)      # DaLLaMa path
            if model_was_swapped:
                await nerd_herd.record_swap(model.name)
            call_fn = lambda: hallederiz.call(model, ...)
        else:
            handle = kdv.begin_call(model.provider, model.name)
            async def call_fn():
                try:
                    return await hallederiz.call(model, ...)
                finally:
                    kdv.end_call(handle, result=<response or None>)

        try:
            result = await call_fn()
            write_model_pick_log(pick, result, success=True)
            return result
        except Exception as e:
            write_model_pick_log(pick, error=e, success=False)
            failures.append(Failure.from_exception(e, model))
            # continue to next iteration with accumulated failures
    raise ModelCallFailed(...)
```

### 6.2 Guarantees

- Cloud calls: `begin_call` before wire transmission; `end_call` in try/finally. Zero leak window.
- Local calls: DaLLaMa load pre-hook; no kdv involvement (local pool pressure derives from `LocalModelState`, not kdv).
- Swap recording: fires only if DaLLaMa actually swapped, only inside dispatcher.
- Pick log: writes after each iteration ends (success or final failure), capturing actual outcome.

## 7. Failure cases

| Case | Handling |
|---|---|
| Dispatcher process dies mid-call | `end_call` never fires; TTL (180 s) prunes the handle lazily on next `begin_call` or snapshot build. In-flight counter self-heals. |
| Beckman admission granted but Hoca select fails at dispatcher iteration 0 | Happens only if state shifted drastically between Beckman's probe and dispatcher's iteration. Dispatcher treats as a normal select-None and raises `ModelCallFailed` — task re-queues. Rare. |
| Cloud pool drains to zero mid-tick (burn at t=0 still possible?) | First `begin_call` bumps in-flight; subsequent admissions see reduced `effective_remaining` → pressure drops → threshold check may reject. If multiple admissions happen between two snapshot reads, minor overshoot possible; bounded by tick-to-admission rate and HARD_CAP. |
| Hoca returns Pick for model whose pool has no KDV state (new provider) | `pressure_for()` returns 0.0 (neutral). Threshold check passes for mid-urgency tasks. Acceptable default. |
| Beckman push to nerd_herd fails (e.g., transient error) | Push-based pattern already tolerates this (KDV and DaLLaMa push best-effort). Queue profile simply stays stale in snapshot until next successful push. |
| Stale queue_profile in snapshot misleads Hoca's per-call scarcity | Hoca reads snapshot; staleness is bounded by push frequency. Worst case: old profile shows more hard tasks than reality → Hoca conserves more on easy tasks. Self-correcting within one push cycle. |

## 8. Testing strategy

### 8.1 Phase 2d scenario regression

Existing simulator (`packages/fatih_hoca/tests/sim/run_scenarios.py` and `run_swap_storm_check.py`) MUST pass unchanged. This spec is an additive refactor — Hoca's per-call scoring behaviour is preserved. Any divergence signals regression in the purity refactor.

### 8.2 New unit tests

- `compute_pool_pressure`: each arm in isolation; boundary cases (remaining=0, reset_at=None, full quota).
- `begin_call` / `end_call`: happy path, TTL expiry, double-end (idempotent), parallel calls same provider.
- `pressure_for`: cached on repeat reads, falls back to provider-level on missing per-model.
- Beckman urgency composition: each term in isolation, clamping at [0, 1].
- Beckman threshold: pure function, returns [-1, +1].
- Hoca purity: `select()` does not mutate pick_log, does not call `SwapBudget.record_swap`. Verify with mock persistence layer.

### 8.3 Beckman admission integration

Extension of Phase 2d simulator:
- Add `Beckman` harness that drives `next_task()` against a stateful queue + mocked nerd_herd snapshot.
- Scenarios:
  1. Cloud near reset, hot queue → admission rate grows, quota drains smoothly.
  2. Cloud depleted, cold queue → admission drops to zero, no burn.
  3. Mixed: one abundant pool + one depleted → admitted tasks dispatch via Hoca routing (sim only verifies admission count, not routing correctness — that is Phase 2d's domain).
  4. i2p-shaped burst: 180 correlated tasks with dependencies. Measure admission rate across wave phases (onset, peak, tail).
  5. Starvation recovery: stuck task ages past 24h; age_bump nudges admission threshold without breaking priority ordering.
- Metrics: admission rate vs pool pressure, admission-count vs HARD_CAP, task age histograms.

### 8.4 Dispatcher integration

- In-flight counter visible in `snapshot().cloud[p].models[m].limits.rpd.in_flight` during an active mock call.
- Counter decrements to zero after call completes.
- TTL prune test: manually skip `end_call`, wait TTL + 1s, trigger `begin_call` — expired handle gone.

## 9. Tuning knobs (initial + how to calibrate)

| Knob | Initial | Calibration |
|---|---|---|
| `KDV_INFLIGHT_TTL_S` | 180 | Longest legitimate cloud call observed + 2x margin. |
| `BECKMAN_TOP_K` | 5 | Measured `hoca.select()` latency; target total admission overhead < 50 ms/tick. |
| `BECKMAN_HARD_CAP` | TBD (existing env?) | Existing concurrency limit from current Beckman config; unchanged unless sim shows otherwise. |
| `AGE_SCALE` | 86400 (24 h) | Aligns with i2p runtime expectations (hours to days). |
| Urgency term weights | `priority=1.0, age=0.05, blocker=0.05` | Simulator shows no inversions under realistic priority distributions. |
| Threshold intercept/slope | `0.5 / 1.0` | Simulator scenarios 1–5 above. Watch for starvation and burn patterns. |

## 10. Migration

### 10.1 Precondition phase (must ship first)

1. Add `nerd_herd.swap_budget` module; move `SwapBudget` class there; expose read + write API.
2. Update `Selector` to read swap counter from `nerd_herd`, stop writing (`record_swap()` moves out).
3. Move `_log_pick` DB write from `Selector` to dispatcher post-iteration.
4. Audit + consolidate `select_for_simulation` with `select()`.
5. Full targeted tests + Phase 2d scenario regression green.

### 10.2 In-flight + pool pressure phase

6. Add `in_flight` field to `RateLimit`; KDV pushes it.
7. Implement `begin_call` / `end_call` API on kdv with TTL prune.
8. Wire into dispatcher's iteration loop (cloud branch).
9. Implement `compute_pool_pressure` in nerd_herd; attach to `CloudModelState.pool_pressure`.
10. Implement `snapshot.pressure_for(model)` accessor + lazy cache.
11. Fatih Hoca scarcity layer: replace direct KDV-state reads with `snapshot.pressure_for()`. Queue arm (`_per_call_scarcity` arm 3) stays in Hoca.
12. Phase 2d scenario regression must still pass.

### 10.3 Queue profile + Beckman admission phase

13. Add `QueueProfile` dataclass to `nerd_herd.types`.
14. `NerdHerd.push_queue_profile(profile)` receiver.
15. Beckman push sites: `enqueue`, `on_task_finished`, `sweep_queue`.
16. Hoca consumes `snapshot.queue_profile` in its existing queue arm (was `queue_state` param → now snapshot-sourced).
17. Beckman `next_task()` rewrite: top-K loop, urgency composition, threshold check, preselected_pick attached.
18. Dispatcher honours `Task.preselected_pick` on iteration 0.
19. Beckman simulator scenarios green; Phase 2d regression green.

### 10.4 Cleanup

20. Remove obsolete `system_busy` single-bit admission check in Beckman (if still present post-step 17).
21. Update `docs/architecture-modularization.md` with the new data flow.

## 11. Non-goals and deferred items

- Shared "dispatch simulator" infrastructure joining Phase 2d and Beckman scenarios into one. Each simulator stays its own harness; integration comes from running both per PR.
- Adaptive TTL tuning from observed call latency distribution. Ship with fixed 180 s.
- Priority reshape or new priority tiers. Urgency uses existing priority scale unchanged.
- Per-task feedback loop from Hoca's routing history into Beckman's admission prediction.
- Historical continuity of `model_pick_log` semantics (pre-refactor rows are selection records; post-refactor rows are dispatch records). Counterfactual CLI will need awareness of the cutover date.

## 12. Open questions (resolved during implementation)

- Initial `BECKMAN_HARD_CAP` confirmation: does an existing concurrency env var cover this?
- Whether Hoca's nerd_herd injection in production uses the in-process `NerdHerd` object (always-fresh `snapshot()`) or the `NerdHerdClient` (cached, requires periodic refresh). If the latter, refresh cadence becomes a configurable parameter and is covered by the existing `src/app/run.py:496,518` callers.
- Exact shape of `Task.preselected_pick` field: dataclass field vs context-dict key. Lean dataclass field for type safety.
