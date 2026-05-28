# SP3b Kickoff — `dispatcher.request` contract-inversion set

**For:** a session designing/building **SP3b** of the CPS migration, **running in parallel with SP3**.
**Date:** 2026-05-29
**Author:** SP3-design session.
**Runs alongside:** SP3 (`docs/superpowers/specs/2026-05-29-cps-sp3-design.md`) — see *Parallel-safety boundary* below before touching any shared file.

---

## Mission

Migrate the **`dispatcher.request` shim** (`src/core/llm_dispatcher.py:273`, inventory site #10)
off `await_inline=True`. This is the **riskiest single migration** in the whole CPS project and
was deliberately carved out of SP3 into its own sub-project because it cannot be migrated the
same way as grading/code_review/summarize.

SP3 kills the DLQ deadlock for the **post-hook** call paths. SP3b kills the same deadlock
mechanism for the **`dispatcher.request` fan-out** — but the design problem is different: it's a
*contract inversion*, not just a call-site relocation.

## Why this is its own sub-project (the core problem)

`request()` is a **deprecation alias** whose contract is *"return the LLM result to a synchronous
inline caller"*:

```python
async def request(self, category, ...) -> dict:
    spec = _request_kwargs_to_spec(...)
    async with _hb.keepalive():
        result = await general_beckman.enqueue(spec, await_inline=True)   # <-- the await_inline
    return _task_result_to_request_response(result)                       # <-- returns a value
```

CPS (continuation-passing) **cannot return a value to a synchronous caller** — it enqueues and
fires a handler later. So unlike SP3's post-hook sites (where the caller already returns and a
resume handler re-enters routing), `request`'s callers consume the result **inline to continue
their own work**. Migrating `request` to CPS would invert the contract for **every** caller —
each would have to split into enqueue-then-resume, and several can't (they're mid-loop /
mid-step / mid-pipeline).

So `await_inline=True` here is doing two jobs at once: (1) it deadlocks when the caller is a
cap-counted task, and (2) it's the only thing that lets `request` keep its return-a-value
contract. SP3b must solve (1) **without** breaking (2).

## Read order (before anything)

1. `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (rev2 umbrella) — the SP3 bullet
   flags `request` as ⚠️ riskiest-and-not-a-footnote, and the **Non-goal** section is
   load-bearing for SP3b: *"the coulson ReAct loop calls `dispatcher.execute()` directly per
   iteration (coulson IS the worker) and is NOT an `await_inline` caller — out of scope."*
   That direct-execute path is the model your leading hypothesis should study (see below).
2. `docs/superpowers/specs/2026-05-27-cps-migration-call-site-inventory.md` — site #10 row.
3. `docs/superpowers/specs/2026-05-29-cps-sp3-design.md` — the SP3 spec. Read it to know exactly
   what SP3 owns so you don't collide (Parallel-safety boundary repeats the essentials).
4. Memory `[Singular dispatcher caller](feedback_singular_dispatcher_caller.md)` — the policy is
   *"only Beckman calls `LLMDispatcher.request`; every other LLM caller goes through
   `beckman.enqueue`"* with a **known-violators list**. The callers below ARE those violators.
   SP3b has to decide, per caller, whether the right fix is "route `request` differently" or
   "migrate this violator onto `beckman.enqueue`."
5. Substrate invariants — the SP3 kickoff (`docs/handoff/2026-05-28-sp3-kickoff.md`,
   *"Substrate invariants SP3 MUST honor"*) is the single source of truth. SP2 confirmed SP2
   moved none of it. If your design ends up needing a continuation at all, honor that list.
6. Live code: `src/core/llm_dispatcher.py` — `request()` (`:222`), `_request_kwargs_to_spec`
   (`:59`), `_do_dispatch()` (`:292`), `_task_result_to_request_response`. Understand how
   `_do_dispatch` (the actual dispatch the orchestrator pump calls for raw_dispatch tasks)
   relates to `request` and to whatever `coulson` calls directly.

## Caller inventory (re-confirm each — `feedback_audit_call_sites`; line numbers as of 2026-05-29)

Real (non-test) callers of `.request(`:

| Caller | File:line | Caller context | Deadlock-prone? | Contract-locked? |
|--------|-----------|----------------|-----------------|------------------|
| coulson reflection | `packages/coulson/src/coulson/reflection.py:335` | **mid-ReAct** self-reflection inside an agent task | **YES** (cap-counted) | **YES** (value continues the loop) |
| coulson single_shot | `packages/coulson/src/coulson/single_shot.py:46` | one-shot LLM helper | likely (verify caller) | YES |
| constrained_emit | `src/workflows/engine/constrained_emit.py:147` | post-execution structured-emit pass on a step | likely (verify) | YES |
| shopping pipeline_v2 | `src/workflows/shopping/pipeline_v2.py:363, 487` | shopping workflow steps | likely (verify) | YES |
| shopping labels | `src/workflows/shopping/labels.py:22` | label generation | verify | YES |
| shopping intelligence | `src/shopping/intelligence/_llm.py:42` | review synthesis | verify | YES |
| mr_roboto critic_gate | `packages/mr_roboto/src/mr_roboto/critic_gate.py:220` | **mechanical** pre-hook (cap-EXEMPT) | **no** | YES |

Not a live caller (ignore): `mr_roboto/executors/alert_triage.py:146` is a comment
(*"T4+ wires a real dispatcher.request()"*) — no call yet. `llm_dispatcher.py:206` is internal
to the module (verify it isn't the public path).

**Key split:** `critic_gate` is mechanical → cap-exempt → **no deadlock**, but still
contract-locked. The coulson/constrained_emit/shopping callers are mid-task → both
deadlock-prone and contract-locked. Different fixes may apply to each group.

## Candidate approaches to explore in brainstorm (do NOT pre-decide — this is the brainstorm's job)

1. **Bounded direct-dispatch path (leading hypothesis).** Route `request()` through the same
   `dispatcher._do_dispatch()` / `dispatcher.execute()` path that coulson already uses directly,
   **bypassing `enqueue`/the lane/`await_inline` entirely**. This preserves the return-a-value
   contract (no inversion) AND removes the held-slot deadlock (no lane admission to block on).
   The umbrella spec's non-goal already legitimizes this path for coulson — SP3b would extend it
   to the other `request` callers. **Open questions for the brainstorm:** does bypassing
   `enqueue` lose cap-counting / in-flight accounting / telemetry that something relies on? Does
   it double-count or under-count against `ONESHOT_CONCURRENCY`? Is a direct OVERHEAD call
   acceptable un-capped, or does it need its own soft limit? This is the crux.
2. **Per-caller CPS split.** Migrate each caller to enqueue-then-resume. Almost certainly
   rejected for the mid-loop callers (coulson reflection can't suspend the ReAct loop without
   touching coulson, which the umbrella spec rules out of scope), but may fit an edge-ish caller.
3. **Migrate violators onto `beckman.enqueue` (per the singular-caller policy).** Where a caller
   genuinely tolerates async delivery, move it to the sanctioned `enqueue` path and drop its
   `request` use. Combine with (1) for the rest.
4. **Hybrid (most likely outcome).** Mechanical/edge callers (`critic_gate`) keep a synchronous
   path; mid-task callers get the bounded direct-dispatch path. Decide the matrix in the spec.

The end state SP5 needs: **`request` no longer calls `enqueue(await_inline=True)`** (by any of
the above), so SP5 can delete `await_inline`/`resolve_inline`/`_inline_waiters`/`INLINE_TIMEOUT`.

## Parallel-safety boundary (SP3 is running concurrently — READ THIS)

**SP3 owns (do not edit):**
- `src/core/grading.py`, `src/core/code_review.py`, `src/workflows/engine/hooks.py`
- `src/agents/grader.py`, `src/agents/code_reviewer.py`, `src/agents/artifact_summarizer.py` (deleted by SP3)
- `packages/general_beckman/src/general_beckman/apply.py` — the post-hook **spawn** sites
  (`_apply_request_posthook`, `_posthook_agent_and_payload`, the grade-pass summary loop) and the
  verdict-apply functions
- new `packages/general_beckman/src/general_beckman/posthook_continuations.py`

**SP3b owns:**
- `src/core/llm_dispatcher.py` — `request()`, `_request_kwargs_to_spec`, dispatch routing
- the caller files: `coulson/{reflection,single_shot}.py`, `constrained_emit.py`,
  `shopping/{pipeline_v2,labels}.py`, `shopping/intelligence/_llm.py`, `mr_roboto/critic_gate.py`

**Shared-file watch (the only real collision points):**
- **`src/core/llm_dispatcher.py`** — SP3 *imports* `_task_result_to_request_response` from here
  (in grading/code_review/hooks) but does **not** modify it. SP3b must **keep
  `_task_result_to_request_response`'s signature/behavior stable**, or coordinate. If SP3b
  changes only `request()` internals + adds a direct path, the collision is nil.
- **`_HANDLER_MODULES`** (`general_beckman/continuations.py`) — SP3 appends `posthook_*` entries.
  SP3b only touches this if its design needs a continuation module (the leading direct-dispatch
  hypothesis does **not**). If both touch it, resolve take-both (same pattern as the
  SP1.1×SP2 merge — small list file, low risk).

If SP3b sticks to `llm_dispatcher.py` + the caller files and leaves
`_task_result_to_request_response` alone, **the two sub-projects do not overlap** and merge
cleanly in either order. Coordinate the merge: whoever lands second rebases/merges `--no-ff`,
re-runs both test sets on the merged tree, then the founder restarts via Telegram.

## Workflow

1. **Brainstorm first** (`superpowers:brainstorming`). The contract-inversion question and the
   cap-counting trade-off of the direct-dispatch path are genuine design forks — don't skip.
2. **Write the SP3b spec** under `docs/superpowers/specs/YYYY-MM-DD-cps-sp3b-design.md`. Cite the
   umbrella rev2 as parent and this kickoff. Decide the per-caller matrix explicitly.
3. **Write the SP3b plan** (`superpowers:writing-plans`). Use the SP1 plan
   (`docs/superpowers/plans/2026-05-27-cps-sp1-continuations.md`) as the format template.
4. **Execute via subagent-driven development** in a fresh worktree
   (`superpowers:using-git-worktrees` → `EnterWorktree`).

## Working environment basics

- Windows host, Python 3.10, venv at `.venv/`. Use the venv python by absolute path inside
  worktrees: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`.
- Always prefix pytest with `timeout` (e.g. `timeout 120 ...`) — project rule; zombie pytest
  holds SQLite locks and crash-loops live KutAI.
- Worktree-root `conftest.py` resolves `packages/*/src` to the worktree. Run pytest from the
  worktree root.
- Live KutAI runs from the main checkout; worktrees protect it. Founder restarts via Telegram
  (**never** `taskkill`).
- Git: `rtk git ...` (token-optimizing passthrough). Project pushes to `main` directly, but
  branch first, merge `--no-ff`, verify tests on the merged result.

## What "done" looks like

- `dispatcher.request` no longer routes through `enqueue(await_inline=True)` for any caller; a
  cap-counted task calling `request` (e.g. coulson reflection mid-ReAct) no longer holds a lane
  slot blocking on a child.
- The return-a-value contract is preserved for every caller that needs it (no caller broke).
- The per-caller decision matrix is documented in the spec; any caller moved to `beckman.enqueue`
  is noted against `feedback_singular_dispatcher_caller`.
- Tests pin: no deadlock under concurrent `request` callers; each caller still gets its result;
  the keepalive heartbeat behavior (the reason `request` wraps in `_hb.keepalive()`) is preserved
  or made unnecessary.
- SP5 unblocked: with SP3 + SP3b done, the only remaining `await_inline` users are SP4 (tools +
  mechanicals) and the two SP5-deferred carve-outs (#2 task_classifier, #6 investor_bullets).

## Known traps

- **Don't touch the coulson ReAct loop.** The umbrella spec rules it out of scope; coulson
  already calls `dispatcher.execute()` directly. SP3b changes how `request` dispatches, not how
  coulson iterates.
- **The `_hb.keepalive()` wrapper exists for a reason** (`llm_dispatcher.py:259` comment): the
  parent's no-progress watchdog would otherwise kill the runner during a 60s+ child call. Any new
  path must keep the parent's heartbeat fresh or the watchdog regression returns (11+ wedged ❌
  pings in 5 min, observed 2026-05-04 when the alias first landed).
- **Cap-counting is the crux of the direct-dispatch hypothesis.** Bypassing `enqueue` removes the
  call from `ONESHOT_CONCURRENCY` accounting. Decide deliberately whether that's correct (these
  are mostly OVERHEAD calls) or whether the direct path needs its own bound.
- **Pre-existing dispatcher test failures** flagged in the SP2 handoff
  (`tests/core/test_dispatcher_in_flight.py`, `test_dispatcher_records_swap.py`) — verify whether
  they're already fixed by a later SP1.1 commit before assuming SP3b broke them.

---

Hand this file to the parallel session. The umbrella spec, the SP3 spec, and the call-site
inventory are the required reading; the memory loads automatically.
