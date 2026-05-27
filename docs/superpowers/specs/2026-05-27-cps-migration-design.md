# Design — Continuation-Passing migration: delete `await_inline`

**Date:** 2026-05-27
**Status:** approved (brainstorm), SP1 ready for planning
**Owner:** founder + agent

---

## Problem

`await_inline` (general_beckman) is a synchronous "enqueue a child task and block on an
in-memory future for its result" primitive. It was intended as a telegram-edge convenience
but drifted into a general in-task control-flow primitive used at ~23 sites. When called
from **inside a dispatched, cap-counted task** (grading, code_review, summarize, …), the
parent **holds a lane concurrency slot while blocking up to `INLINE_TIMEOUT=600s`** waiting
for a child that needs a slot from the **same capped lane** (`ONESHOT_CONCURRENCY=4`). With
several such parents blocked at once, the cap fills with blocked parents, their children
can't be admitted, every parent times out → DLQ. Confirmed on 2026-05-27 (mission 77): at
4/4 in-flight, 3 slots were grader parents blocked on `await_inline` (cerebras), each
released exactly ~600s after admit. Secondary fragilities: the waiter future is **in-memory
(lost on every orchestrator restart)** and the 600s block is opaque.

## Goal

**Delete `await_inline` / `resolve_inline` / `_inline_waiters` / `INLINE_TIMEOUT` entirely.**
Replace the synchronous blocking model with **durable continuation-passing (CPS)**: a task
enqueues its child and *returns* (no blocking, no held slot); when the child reaches a
terminal state, a named resume-handler runs with the saved parent state. State lives in the
DB → survives restarts.

Non-goal: the coulson ReAct loop. It calls `dispatcher.execute()` directly per iteration
(coulson *is* the worker) and is **not** an `await_inline` caller — out of scope.

## Substrate decision (approved)

Approach **B — durable framework**, realised by **extending the existing `enqueue` API**
rather than adding a new function. `enqueue` already carries `on_complete` (a named handler
wired into `on_task_finished`); today it is fire-and-forget (`asyncio.create_task`,
no state, errors swallowed). We **upgrade `on_complete` to be durable + state-carrying** and
add an opt-in failure handler. No `enqueue_then`.

YAGNI cuts (approved): **no join/fan-in** (no current caller needs parallel fan-in; every
continuation is 1:1 with one child) and **no periodic orphan sweep** (restart-reconcile + a
TTL cover it).

## Decomposition (each sub-project: own spec → plan → implement)

- **SP1 — Foundation (this spec).** Durable continuation substrate. **Zero call-site
  changes.** `await_inline` stays fully intact and coexists.
- **SP2 — Edge group.** Migrate telegram + jobs (`task_classifier`, `interview`, `meetings`,
  `faq_regen`, `investor_bullets`). Lowest risk; validates the substrate in production.
- **SP3 — In-task / deadlock set.** Migrate `grading`, `code_review`, `hooks` summarize, the
  `dispatcher.request` shim. **This is what removes the DLQ deadlock.**
- **SP4 — Tools + mechanicals.** `vision` tool, mr_roboto LLM executors (`reviews_*`,
  `incident_draft_update`, `press_kit_assemble`, `demo_storyboard`, `crisis_draft_holding`),
  `yalayut/discovery/synthesize`.
- **SP5 — Delete the primitive.** Remove `await_inline`/`resolve_inline`/`_inline_waiters`/
  `INLINE_TIMEOUT`; fix conftest/tests; add a guard test that the primitive is gone.

No stopgap (founder decision): DLQ bleeding continues until SP3; missions may be paused
meanwhile via Telegram lifecycle.

---

## SP1 — Durable continuation substrate (detailed)

### Data model

One dedicated table (1:1 — keyed by the child it waits on; two-table was overkill once join
was cut, and columns-on-`tasks` would be schema debt on an already-hot, wide table):

```
continuations(
  child_task_id   INTEGER PRIMARY KEY,   -- the task whose terminal state fires this
  resume_name     TEXT NOT NULL,         -- registered success handler
  on_error_name   TEXT,                  -- registered failure handler (opt-in, nullable)
  state_json      TEXT NOT NULL,         -- serialized parent state (default '{}')
  status          TEXT NOT NULL,         -- 'pending' | 'fired'
  created_at      TEXT NOT NULL          -- strftime('%Y-%m-%d %H:%M:%S') (NOT isoformat)
)
```

### API (extends existing `enqueue`)

```
enqueue(spec, *, on_complete=<name>, on_error=<name>|None,
        cont_state=<dict>|None, lane=..., parent_id=..., ...) -> child_task_id
```
- When `on_complete` is set, `enqueue` returns the child `task_id` immediately (never blocks).
- Writes the `continuations` row **in the same DB transaction as / before** the child
  `add_task`, so a fast child always finds its row (writing the child first would race the
  fire-check — the missed-fire bug class).
- `register_resume(name, handler)` — handler signature
  `async (task_id: int, result: dict, state: dict)`. `task_id` is kept (the existing
  registry passes it and handlers use it); `state` is the new third arg, defaulting to `{}`
  when no `cont_state` was given. The 2 existing `on_complete` handlers (`analytics_digest`,
  `classify_signals`) gain the `state` param (unused → `{}`).

### Fire / error / restart semantics

- **Fire (success):** in `on_task_finished`, when a task reaches terminal **success**, look up
  `continuations WHERE child_task_id = ? AND status='pending'`; if found, load `state_json`,
  invoke `resume_name(task_id, result, state)`, mark `status='fired'`. Idempotent via the
  status guard.
- **Failure — default (≈90% of callers, zero extra code):** a terminal **failed** child with
  **no** `on_error` fires nothing. The child already went through normal task retry/DLQ; a
  source left stuck `ungraded` is caught by the existing ungraded sweep. (A grade that graded
  *as fail* is a **success** result carrying `{passed: false}` — only infra failures hit this
  path.)
- **Failure — opt-in observe (≈10%):** a terminal failed child **with** `on_error` set invokes
  `on_error_name(task_id, failed_result, state)`, then marks `fired`. Lets the parent fall
  back / mark the source / etc.
- **Restart:** the table is durable. Startup recovery scans `continuations(status='pending')`
  whose `child_task_id` is already terminal (completed/failed/DLQ) and processes them — closes
  the gap where a child completed while the orchestrator was down.
- **TTL safety net:** a `pending` continuation older than `CONTINUATION_TTL` (default 1h) whose
  child is not terminal is failed (fires `on_error` if set, else logged + marked `fired`).
  Replaces the opaque 600s block with a bounded, observable path. No periodic bespoke sweep —
  folded into the existing startup/periodic recovery pass.

### Scope guard

SP1 adds the table, the `enqueue` params, the `on_task_finished` fire-check (replacing the
detached `create_task(dispatch_on_complete)`), `register_resume`, restart-reconcile, TTL, and
updates the 2 existing `on_complete` callers. It changes **no** `await_inline` call site;
`await_inline` and the new durable `on_complete` coexist until SP5.

### Module boundaries

- `general_beckman/continuations.py` — registry (`register_resume`, lookup) + the durable
  fire logic (`fire_for_task(task_id, result)`), pure and unit-testable.
- DB schema/migration for `continuations` (in `src/infra/db.py` init path).
- `general_beckman.enqueue` — new params + the atomic write.
- `on_task_finished` — replace the fire-and-forget block with a call into
  `continuations.fire_for_task`.
- Startup recovery — reconcile + TTL pass.

### Testing (host-path, DB-isolated, `timeout` prefix)

- success fires `resume(state, result)` with the exact state + result;
- terminal-failed child with `on_error` fires `on_error(state, failed_result)`;
- terminal-failed child without `on_error` is a no-op (no fire);
- double `on_task_finished` for the same child fires exactly once (idempotency);
- restart-reconcile: child went terminal while the row stayed `pending` → fires on the
  reconcile pass;
- TTL: a `pending` row past TTL with a non-terminal child is failed/expired;
- the 2 pre-existing `on_complete` callers still fire correctly under the durable path.

### Acceptance

- New `continuations` table created on init.
- `enqueue(spec, on_complete=…, cont_state=…)` returns a `child_task_id`, never blocks, and
  the resume handler runs (with state) when the child completes — proven by a host-path test
  driving the real `enqueue` → `on_task_finished` path against a temp DB.
- All SP1 tests green; no `await_inline` call site touched; existing beckman suite green.
