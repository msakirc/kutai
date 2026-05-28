# Design — Continuation-Passing migration: delete `await_inline`

**Date:** 2026-05-27 (rev. 2 — SP1 hardened against `add_task` atomicity/dedup reality + concurrency)
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

- **SP1 — Foundation (this spec).** Durable continuation substrate. **Zero *production*
  call-site changes** — but SP1 includes a **throwaway spike** that migrates one SP3 call site
  (grading) against the substrate to prove the handler shape is sufficient (see Scope guard).
  `await_inline` stays fully intact and coexists.
- **SP2 — Edge group.** Migrate telegram + jobs (`task_classifier`, `interview`, `meetings`,
  `faq_regen`, `investor_bullets`). Lowest risk; validates the substrate in production.
- **SP3 — In-task / deadlock set.** Migrate `grading`, `code_review`, `hooks` summarize, the
  `dispatcher.request` shim. **This is what removes the DLQ deadlock.**
  ⚠️ **`dispatcher.request` is the riskiest single migration, not a footnote.** Its contract
  is *"returns the LLM result to inline code"* — CPS cannot return a value to a synchronous
  caller, so migrating it inverts the contract for every caller (callers must split into
  enqueue-then-resume). Per [[feedback_singular_dispatcher_caller]] only Beckman calls
  `request` directly, which bounds the blast radius — but SP3 must scope `request`'s callers
  explicitly before touching it, and it may warrant its own sub-spec.
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
CREATE INDEX idx_continuations_pending ON continuations(status);
```
The `status` index serves the reconcile + TTL passes, which scan
`WHERE status='pending'` (the PK only helps the per-fire point lookup).

`child_task_id` is the PK, so **exactly one** continuation can exist per child.
That assumption is load-bearing and is the reason `add_task` dedup must be
handled explicitly (see *Enqueue atomicity & dedup* below) — two enqueues that
dedup to the same child id would otherwise collide on this PK.

### API (extends existing `enqueue`)

```
enqueue(spec, *, on_complete=<name>, on_error=<name>|None,
        cont_state=<dict>|None, lane=..., parent_id=..., ...) -> child_task_id
```
- When `on_complete` is set, `enqueue` returns the child `task_id` immediately (never blocks).
- `register_resume(name, handler)` — handler signature
  `async (task_id: int, result: dict, state: dict)`. `task_id` is the **child** id (the
  existing registry passes it and handlers use it); `state` is the new third arg, defaulting
  to `{}` when no `cont_state` was given. The 2 existing `on_complete` handlers
  (`analytics_digest`, `classify_signals`) gain the `state` param (unused → `{}`).
  **Handlers that need to resume the *parent's* work (SP3: grading, code_review) must carry
  `parent_id` inside `cont_state` and drive their follow-up via `enqueue` /
  `on_task_finished` themselves** — the substrate hands back `(child_id, child_result, state)`
  and nothing else. Validated by the SP3 spike (see Decomposition).

### Enqueue atomicity & dedup (corrected — `add_task` is the constraint)

The original "write the `continuations` row in the same transaction as / before
`add_task`" is **infeasible** and is replaced as follows:

- `add_task` (`src/infra/db.py`) runs on an **isolated aux connection** (`connect_aux`)
  with its own `BEGIN IMMEDIATE`/`COMMIT` — deliberately NOT the shared singleton (sharing
  triggers "cannot rollback - no transaction is active"). It **commits the child before
  returning its id**. So there is no outer transaction to join, and the PK (`child_task_id`)
  doesn't exist until after commit. You cannot write the row "before" or "in the same
  transaction as" `add_task` from `enqueue`.
- **Fix:** thread `on_complete` / `on_error` / `cont_state` **into `add_task`** so the
  `continuations` INSERT happens inside `add_task`'s own `BEGIN IMMEDIATE` region, committed
  atomically with the child row. The child therefore never exists (claimable) without its
  continuation row — closing the missed-fire window at the only place it can be closed.
- **Dedup is a first-class case** (the original spec ignored it). `add_task` dedups on
  `task_hash` and can return: (a) an **existing** id — possibly already running or already
  **terminal**; (b) **None** (dedup-skip / rollback); (c) collapse **two** parents' children
  into **one** id. Each breaks naive 1:1 keying — and note `await_inline` already mis-handles
  (c) today (`_inline_waiters[id]` silently overwrites). SP1 resolves it by **disabling
  dedup for continuation children**: when `on_complete`/`on_error` is set, `add_task` skips
  the dedup probe and always inserts a fresh child (continuation children are call-scoped,
  never legitimately shared). `enqueue` asserts a real child id came back; a `None` return
  with a continuation set is a hard error, not a silent orphan.
- **Coexistence guard:** `await_inline=True` together with `on_complete`/`on_error` on the
  same `enqueue` is rejected (both would fire). They are mutually exclusive until SP5 deletes
  `await_inline`.
- `next_task_spec` (the existing fire-and-forget chain) is untouched by SP1 and coexists; it
  is not part of the durable substrate and migrates with its callers (if any) later.

### Fire / error / restart semantics

- **Claim-then-fire (idempotency, ordering matters):** in `on_task_finished`, on a terminal
  child, **first** atomically claim the row —
  `UPDATE continuations SET status='fired' WHERE child_task_id=? AND status='pending'` — and
  check `rowcount==1`. Only the winner of that CAS proceeds to invoke the handler. The
  original "invoke, *then* mark fired" leaves a double-fire window: `on_task_finished` can be
  re-entered for the same child (retry / re-grade paths exist) and two callers both read
  `pending` before either marks `fired`. In a project whose whole purpose is fixing a
  concurrency deadlock, this ordering is not optional.
- **Detached invocation:** after winning the CAS, dispatch the handler **detached**
  (`asyncio.create_task`), as the current `on_complete` path already does (`__init__.py:881`).
  A synchronous `await` of the resume would let a slow handler stall the terminal pipeline and
  the pump. CAS is synchronous (cheap, ordering-critical); the handler body is detached.
- **Status mapping (3 terminal states, not 2):** the fire reads the **raw agent result
  status** (before `route_result`/`rewrite_actions`, which can flip it):
  - `completed` → **success** path: invoke `resume_name`. A grade that graded *as fail* is a
    `completed` result carrying `{passed: false}` — it fires the resume, not `on_error`.
  - `failed` (infra failure: timeout, retries exhausted) → **failure** path.
  - `needs_clarification` (agent worked, needs human) → **leave the row `pending`, do NOT
    claim, do NOT fire.** It is *not* terminal: the task pauses for human input and later
    resumes to a real terminal (`completed`/`failed`), which fires the continuation then.
    Claiming on `needs_clarification` would permanently suppress the eventual fire. (Without
    this third case the child would wrongly fire the success resume on a non-terminal state.)
- **Failure — default (≈90% of callers, zero extra code):** a `failed` child with **no**
  `on_error` fires nothing. The child already went through normal task retry/DLQ; a source
  left stuck `ungraded` is caught by the existing ungraded sweep.
- **Failure — opt-in observe (≈10%):** a `failed` child **with** `on_error` invokes
  `on_error_name(task_id, failed_result, state)` (same claim-then-detach discipline). Lets the
  parent fall back / mark the source / etc.
- **Restart:** the table is durable. Startup recovery scans `continuations(status='pending')`
  whose `child_task_id` is already terminal (completed/failed/DLQ) and processes them — closes
  the gap where a child completed while the orchestrator was down. **The child result is
  reconstructed from the persisted `tasks` row (`tasks.result` + artifacts), not the in-memory
  `result` dict (which is gone after restart).** SP1 must therefore verify that the fields a
  resume handler needs (e.g. a reviewer's structured verdict) are actually persisted on the
  child task before completion — if a handler depends on a field that lives only in the
  in-memory envelope, the durability claim is hollow. A reconcile test asserts the
  reconstructed result carries the verdict. **Handler-registration gotcha:** the registry is
  in-memory and is repopulated by importing the handler modules. `classify_signals` registers
  only inside its `run()`, so after a restart its handler is absent until that mechanical runs
  again — reconcile would find no handler and drop the continuation. SP1 adds a
  `register_startup_handlers()` (imports the known continuation-bearing modules so their
  `register()` side-effects fire) that runs **before** the reconcile pass. Terminal in the
  `tasks` table = status ∈ {`completed`, `failed`} (DLQ writes `failed` then quarantines).
- **TTL safety net (alive-aware):** a `pending` continuation older than `CONTINUATION_TTL`
  (default 1h) is expired **only when the child is neither terminal nor still alive** — alive =
  has a recent heartbeat / appears in `src.core.in_flight`. A legitimately long-running child
  (a 20-iteration coder/planner can exceed 1h) must NOT be abandoned; expiring it would fire a
  premature `on_error` while the child later completes and finds `status='fired'` (silent
  drop) — the exact premature-abandon flaw of the old 600s timeout. On genuine expiry: fire
  `on_error` if set, else log + mark `fired`. No periodic bespoke sweep — folded into the
  existing startup/periodic recovery pass.

### Scope guard

SP1 adds the table + index, the `enqueue` params, the continuation-aware path **inside
`add_task`** (atomic INSERT + dedup-skip for continuation children), the `on_task_finished`
claim-then-fire check (replacing the detached `create_task(dispatch_on_complete)`),
`register_resume`, restart-reconcile, alive-aware TTL, and updates the 2 existing
`on_complete` callers. It changes **no** `await_inline` *production* call site; `await_inline`
and the new durable `on_complete` coexist until SP5.

**Spike (throwaway, gates SP1 sign-off):** migrate the **grading** call site to the new
substrate on a scratch branch — grader enqueues its reviewer child with
`on_complete=resume_grade` + `cont_state={parent_id, …}`, returns, and the resume handler
produces the verdict and re-enters routing. Prove end-to-end (verdict reaches
`_apply_posthook_verdict` per [[feedback_verify_verdict_roundtrip]]) that the
`(child_id, child_result, state)` handler shape is sufficient. If it isn't, fix the substrate
*now*. The spike is then discarded — real grading migration ships in SP3.

### Module boundaries

- `general_beckman/continuations.py` — registry (`register_resume`, lookup) + durable fire
  logic: `claim_for_fire(child_task_id) -> bool` (the CAS) and `fire_for_task(task_id, result,
  raw_status)`, pure and unit-testable.
- DB schema/migration for `continuations` (in `src/infra/db.py` init path) + the
  continuation-aware INSERT/dedup-skip branch in `add_task`.
- `general_beckman.enqueue` — new params; rejects `await_inline` + `on_complete` together;
  asserts a non-None child id when a continuation is set.
- `on_task_finished` — replace the fire-and-forget block with claim-then-detach into
  `continuations.fire_for_task`, branching on raw status (completed / failed / needs_clarification).
- Startup recovery — reconcile (terminal-while-down, result reconstructed from `tasks` row) +
  alive-aware TTL pass.

### Testing (host-path, DB-isolated, `timeout` prefix)

- success (`completed`) fires `resume(child_id, result, state)` with exact state + result;
- `completed` carrying `{passed: false}` fires **resume**, not `on_error`;
- `needs_clarification` child fires **neither** (terminal no-fire, row marked `fired`);
- `failed` child with `on_error` fires `on_error(child_id, failed_result, state)`;
- `failed` child without `on_error` is a no-op;
- **double `on_task_finished` for the same child fires exactly once** (CAS / claim-then-fire);
- **dedup:** a continuation child is never deduped — two `enqueue(on_complete=…)` with
  identical specs yield two distinct child ids + two rows (no PK collision, no lost handler);
- `enqueue(await_inline=True, on_complete=…)` raises (mutual-exclusion guard);
- restart-reconcile: child went terminal while the row stayed `pending` → fires on reconcile,
  **with the result reconstructed from `tasks.result`** (assert the verdict survives);
- TTL: a `pending` row past TTL whose child is **dead** (not terminal, no heartbeat) is
  expired; a `pending` row past TTL whose child is **still alive** is **left pending** (no
  premature abandon);
- the 2 pre-existing `on_complete` callers (`analytics_digest`, `classify_signals`) still fire
  under the durable path with `state={}`.

### Acceptance

- New `continuations` table + `idx_continuations_pending` created on init.
- `enqueue(spec, on_complete=…, cont_state=…)` returns a fresh `child_task_id`, never blocks,
  never deduped, and the resume handler runs (with state) when the child completes — proven by
  a host-path test driving the real `enqueue` → `add_task` → `on_task_finished` path against a
  temp DB.
- Claim-then-fire proven idempotent under a double terminal call.
- Restart-reconcile proven to fire with a result reconstructed from the persisted task row.
- **SP3 grading spike green** (verdict round-trips) — substrate shape confirmed before sign-off.
- All SP1 tests green; no `await_inline` production call site touched; existing beckman suite green.
