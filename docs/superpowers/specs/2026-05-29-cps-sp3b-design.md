# Design — SP3b: dumb-pipe dispatcher + hüsam, the non-agentic worker

**Date:** 2026-05-29
**Status:** approved (brainstorm), ready for planning
**Parent:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (CPS umbrella, rev 2)
**Kickoff:** `docs/handoff/2026-05-29-sp3b-kickoff.md`
**Sibling (lands first):** `docs/superpowers/specs/2026-05-29-cps-sp3-design.md` (SP3 — post-hook CPS)
**Provisional names:** module/package `hüsam` (full: *kumandan hüsamettin*) — founder may rename; single public entry `run(task)` (founder wanted `pass`, but `pass` is a reserved Python keyword → use `run`/`command`/`handle`).

---

## 1. Why this exists (corrected from the kickoff)

The kickoff framed SP3b as "migrate the `dispatcher.request` shim and its 7 callers off `await_inline`." Brainstorming corrected the scope **upward** to the real architectural goal the founder is after. Two binding goals:

- **Goal 1 — the dispatcher is a dumb pipe.** It must bear no logic. Its only job: connect HaLLederiz Kadir (call), DaLLaMa (load), Kuleden Dönen Var (capacity) for **one** LLM call. All intelligence — model selection, retry, prompt assembly, result mapping — lives in the **workers**. The dispatcher needs **one** interface, not the current `request`/`dispatch`/`_do_dispatch`/`execute` spread.
- **Goal 2 — workers are pure and never touch each other.** coulson (ReAct) and hüsam (non-agentic) each prepare a call and hand it to the dumb pipe, per Beckman's admission. **No coulson↔hüsam edge.** Anything one worker would need from the other becomes a **Beckman task** instead (admit again — that is the sanctioned channel, not laziness).

The deadlock the kickoff cares about (`request → enqueue(await_inline=True)` double-admits, parent holds a lane slot blocking on a child that needs the same lane) is a **symptom** of the dispatcher carrying admission logic it should not. Fixing Goal 1 + Goal 2 removes it by construction for every in-scope caller.

### Invariant this preserves (verified, do not break)

`beckman.enqueue` is the **sole admission gate**. Every path that reaches the dispatcher is downstream of `beckman.next_task()` → `orchestrator._dispatch(task)`:

```
orchestrator._dispatch (orchestrator.py:173, fed by beckman.next_task)
  ├── mechanical?         → mr_roboto
  ├── raw_dispatch/direct → (today) dispatcher.dispatch()   ← becomes hüsam.run
  ├── shopping_pipeline_v2→ ShoppingPipelineV2.run          ← out of scope (see §6)
  └── else                → coulson.execute(task)
```

There are **no shortcuts to the dispatcher**: the dispatcher is reached only by a worker that Beckman already admitted. SP3b keeps it that way — it does **not** introduce un-admitted dispatcher access. (It also removes a *current* shortcut: `request → enqueue(await_inline)` creates a **second** admitted task for one call — the double-admit flagged in `feedback_singular_dispatcher_caller`. Routing those calls through the worker that already holds the admission deletes the second admission.)

---

## 2. End-state architecture

```
Beckman            admission + lifecycle (enqueue = the only gate)            [unchanged]
workflow_engine    task DEFINITIONS (i2p, shopping json plans)               [unchanged, out of scope]
      │ pump dispatches each admitted task to a worker, by agent/runner
      ├── coulson    ReAct worker. Owns agent-profile execution + per-iter select. Pure.
      ├── hüsam      non-agentic worker. Owns raw profile-less single calls. Pure.
      └── mr_roboto  mechanical                                              [unchanged]
                 │ worker brings an already-selected pick
            dispatcher   DUMB PIPE — one interface: execute(pick, messages, …)
                         = load (DaLLaMa) + call (HK) + meter (in-flight / KDV) + map.
                         NO select, NO retry, NO spec unpacking.
                 │
            fatih_hoca.select()        hallederiz_kadir.call()
```

**Dependency directions (the thing Goal 2 protects):**
- `coulson → {beckman, fatih_hoca, dispatcher}` — never `hüsam`.
- `hüsam → {fatih_hoca, dispatcher}` — never `coulson`.
- Nothing imports `hüsam` except the orchestrator pump (which already routes to every worker).
- No worker↔worker import in either direction.

---

## 3. The dispatcher becomes a dumb pipe (Goal 1)

**Keep:** `execute(pick, messages, …)` — the one-attempt primitive coulson already calls (`react.py:596`). It loads the pick (DaLLaMa), calls (HK), meters the transport (`begin_call`/`end_call`, `record_pick`), and **returns the raw `CallResult`/`CallError`** — it does **not** map to the legacy dict and does **not** select. The worker maps the result (coulson already does this via `dispatch_helpers.result_to_response_dict`; hüsam gets its own).

**Delete (logic that belongs in workers):**
- `dispatch(spec)` — the raw_dispatch worker entry. Its job moves to `hüsam.run(task)`.
- `_do_dispatch(...)` — select + execute + error-shaping + pool-empty forensics. This **is hüsam's core**; its body moves into hüsam.
- `_request_kwargs_to_spec(...)` — only used by `request`; goes when `request` does (see §7, after shopping).

**Keep but re-home consideration:** `_task_result_to_request_response` — consumed by **SP3** (grading), **`posthook_handlers/brand_voice_lint.py:363/431`**, and **`src/tools/vision.py:45/100`** (SP4). Keep its signature/behavior stable. SP3 lands first; if SP3b relocates it, update all three imports on the merged tree, do not change its shape.

**Support helpers stay** (pipe mechanics): `_ensure_local_model`, `_prepare_messages`, `_record_pick`, `_estimate_prompt_tokens`, `get_loaded_model_*`. **`_result_to_dict` moves to hüsam** (it is worker-side result mapping; coulson already has its own equivalent).

After this, `execute()` has no `fatih_hoca.select` in its path → the dispatcher carries no selection logic. One interface.

---

## 4. hüsam — the non-agentic worker (new package)

**Role:** the worker the pump dispatches for **raw, profile-less single LLM-call tasks** — grader, summarizer, hooks-summarize (SP3's callers), and the new reflection/constrained_emit post-hooks (§5). It is **generic**: the *call definition* (messages, schema, exclusions) lives in the **task spec**; hüsam never builds an agent prompt and never parses the ReAct action DSL.

**Single public interface:** `run(task: dict) -> dict`. Only the orchestrator pump calls it. There is **no** second `call(messages)` face — every consumer reaches hüsam as a Beckman task, so one entry suffices (this answers the "do we need two interfaces" question: no).

**Body** (moved from `dispatcher._do_dispatch`):
1. Unpack `task.context.llm_call` → category, messages, hints.
2. Pick: use `task.preselected_pick` (Beckman attached it at admission) else `fatih_hoca.select(...)`.
3. Handle `SelectionFailure` / empty pool (budget pause + admission forensics — port from `_do_dispatch`).
4. `await get_dispatcher().execute(pick, messages, …)`.
5. Map `CallResult → dict` / raise `ModelCallFailed` on `CallError`.

**Imports:** `fatih_hoca`, `src.core.llm_dispatcher` (execute), `src.core.in_flight`, forensics. **Never `coulson`.**

**Location:** `packages/hüsam/` peer to `packages/coulson/` (or ASCII-safe `packages/husam/` — decide at impl; Windows path + Python package name must be import-safe, so the package dir is almost certainly `husam`, with "hüsam"/"kumandan hüsamettin" as the display name in docs/logs).

---

## 5. coulson becomes react-only; reflection + constrained_emit become post-hooks (Goal 2)

coulson is the **react worker**. It must not make non-react LLM calls inline, and must not call hüsam. The two inline post-passes leave coulson and become **Beckman post-hook tasks** routed to hüsam:

- **`self_reflect`** (`reflection.py`, sole caller `react.py:1018`) → new post-hook kind, e.g. `self_reflect`. Auto-wired on agents with `enable_self_reflection`. Severity `warning` (advisory). Verdict can **rewrite** the source result (`verdict:"fix"` → `corrected_result`).
- **`constrained_emit`** (`constrained_emit.maybe_apply`, sole caller `coulson/__init__.py:105`) → new post-hook kind, e.g. `constrained_emit`. Auto-wired on steps declaring a constrainable `artifact_schema`. Verdict **rewrites** the source result (draft → schema-conforming JSON). Finally lands the long-promised move (`coulson/__init__.py:22`: *"Phase A.12 will move constrained_emit out to workflow_engine post-hooks"*).

Both register a `PostHookSpec` row in `POST_HOOK_REGISTRY` (`posthooks.py`) and are spawned as **raw_dispatch LLM children** via `apply._enqueue_posthook_llm_child` → executed by hüsam → verdict applied by `_apply_posthook_verdict` (apply.py:1190/3779). hüsam stays generic — it receives **pre-built messages** in the task spec.

**Prompt-building relocation:** the reflection prompt (`REFLECTION_BLOCKS`, `STACK_BLOCKS`, `build_reflection_prompt`) and the emit schema logic move **down to the post-hook layer, next to `grading.py`'s `GRADING_PROMPT`** — the scheduler that builds the post-hook task spec owns the prompt. coulson keeps only what react itself needs.

### 5a. Two NEW apply-layer capabilities SP3b must build (verified absent)

Verification (`_apply_posthook_verdict`, apply.py:3779) confirmed **no existing post-hook rewrites the source task's `result`** — all 40 registry kinds either gate (pass/fail → retry) or surface (founder_action). reflection and constrained_emit are **result-rewriting**, so SP3b adds:

1. **Result-rewrite verdict path.** A post-hook verdict that **replaces `tasks.result`** of the source task (then continues the chain), via `update_task(result=…)` inside `_apply_posthook_verdict`. New verdict shape (e.g. `PostHookVerdict(action="rewrite", new_result=…)`). Idempotent under the SP1 claim-then-fire substrate.
2. **Ordered post-hook chain.** Today `determine_posthooks` returns `["grade", …extras]` (grade first) and the apply layer spawns them. reflection/emit **must complete and rewrite the result BEFORE grade spawns** (grade scores the rewritten result). SP3b adds explicit ordering so the chain runs **emit → reflect → grade**, each gating the next via the SP1/SP3 continuation substrate — not a parallel fan-out. (Both rewrites must land before the grade child is enqueued.)

These two are the real engineering core of SP3b (beyond standing up hüsam) — they have no precedent in the existing 40-kind registry. Build on **merged SP3**'s `posthook_continuations` / `_apply_posthook_verdict` shapes (STEP-0, §9).

### single_shot — vestigial, shopping-only, no action needed

Investigation (2026-05-29, `feedback_audit_call_sites`): `execution_pattern="single_shot"` is set by **exactly one profile — `shopping_clarifier.py:21`.** `PlannerAgent` and `SignalClassifierAgent` have **no override → `react_loop`** (the docstrings in `single_shot.py` and `base.py:75` claiming "planner, classifier" are **stale**). react already subsumes single_shot's behavior (`max_iterations` can be 1; `decompose → needs_subtasks` with alt-key fallback at `react.py:940-959`; the `is_workflow_step` guard; `final_answer`).

Therefore **single_shot is not migrated by SP3b.** Its only consumer is shopping (out of scope, §6); its coulson dependencies (`build_system_prompt`, `parse_action`) **stay in coulson**, exercised by react + the dormant single_shot branch. Moving single_shot to hüsam would drag coulson's ~1100-line context engine + action DSL into hüsam — the exact coupling Goal 2 forbids. So **there is no shared "low layer" to extract; hüsam imports nothing from coulson.** When shopping migrates to real workflow steps (separate effort), `shopping_clarifier` becomes a `max_iterations=1` react agent or a plain step, and `single_shot.py` is deleted then (react absorbs it).

---

## 6. Out of scope: shopping

Shopping is a **workflow definition** (`shopping_v2.json` + workflow_engine), not a lifecycle actor and not a sibling to coulson/hüsam. `ShoppingPipelineV2` being a bespoke inline-LLM worker (`pipeline_v2.py:363/487`, `labels.py:22`, `intelligence/_llm.py:42`, all via `request()`) is the wrong shape — but correcting it is shopping's own migration to proper step definitions (each LLM step → an admitted task → routed to hüsam by the engine). **That is not SP3b.**

Consequence: shopping's `request()` callers (and `single_shot.run`, which only shopping reaches) keep using `request()`. SP3b therefore does **not** fully delete `request()` (see §7). Shopping's residual deadlock risk persists until shopping migrates — an explicit, accepted deferral, not a half-done SP3b.

---

## 7. `request()` and `await_inline` disposition

- **`request()`** loses its in-scope callers (reflection, constrained_emit, critic_gate). It is **retained as an explicitly-marked, shopping-only deprecated shim** (callers: `single_shot.run` + the 3 shopping pipeline sites). Mark it clearly; it dies when shopping migrates — **before** SP5 deletes `await_inline`.
- **`await_inline`** is **not** deleted here. After SP3 + SP3b its remaining users are: shopping (via `request`, deferred), SP4 (tools + mechanicals), and the SP5 carve-outs (#2 `task_classifier`, #6 `investor_bullets`). SP5 still owns the primitive's deletion; shopping migration is a prerequisite to it.

---

## 8. critic_gate — split (mechanical must not touch the dispatcher)

`critic_gate.py:220` is a **mechanical** pre-hook that calls `LLMDispatcher().request(...)` (note: it instantiates a fresh dispatcher, not the singleton). Per `feedback_no_direct_dispatcher`, a mechanical reaching the dispatcher is itself the violation — even though its task is admitted (cap-exempt) and it does not deadlock.

**Fix:** split into (a) an **admitted agent step** that produces the critic verdict (runs via hüsam as a normal task), and (b) a **mechanical confirm gate** that reads the persisted verdict and decides pass/block with **no LLM call**. This removes the mechanical→dispatcher edge entirely.

---

## 9. Parallel-safety & sequencing

SP3 is "almost done" in its worktree and will land on `main` **before** SP3b finishes. SP3b therefore **builds on top of merged SP3**, not in parallel — the earlier collision concern is resolved by ordering.

**STEP-0 (first task of the plan):** branch from **merged** `main` (not the SP3 worktree — stale-worktree hazard, cf. z9). Reconcile against SP3's actual shapes: `posthook_continuations` (does it chain? §5), `_task_result_to_request_response` (keep stable, §3), any `_HANDLER_MODULES` entries.

Layering composes cleanly: SP3 is **caller-side** (post-hook *spawns* use CPS, no `await_inline`); SP3b is **worker-side** (the pump dispatches those spawned tasks through `hüsam.run` instead of inline `dispatcher.dispatch()`). SP3's continuations see a `TaskResult` either way.

---

## 10. Caller migration matrix

| Caller | Today | After SP3b |
|---|---|---|
| pump raw_dispatch (grader, summarizer, hooks — SP3) | `dispatcher.dispatch()` | `hüsam.run(task)` |
| `self_reflect` (coulson, react.py:1018) | inline `request()` | Beckman post-hook task → `hüsam.run` |
| `constrained_emit.maybe_apply` (coulson/__init__.py:105) | inline `request()` | Beckman post-hook task → `hüsam.run` |
| `critic_gate` (mechanical) | `request()` | split: admitted agent step (→hüsam) + mechanical confirm gate |
| `single_shot.run` | `request()` | **untouched** (shopping-only, keeps `request()`; §5/§6) |
| shopping pipeline_v2 / labels / intelligence | `request()` | **untouched** (out of scope; §6) |
| coulson react per-iter | `dispatcher.execute()` | unchanged (already the dumb-pipe path) |

---

## 11. Tests pin

- **No deadlock:** N concurrent post-hook tasks (reflect/emit/grade) under saturated `ONESHOT_CONCURRENCY` complete — no parent blocks on a same-lane child (the original mission, now satisfied because nothing blocks inline).
- **Result fidelity:** grader/summarizer round-trip through `hüsam.run`; SP3's CPS continuations still receive a correct `TaskResult`.
- **Result-rewrite verdict:** an emit/reflect post-hook with a `rewrite` verdict replaces `tasks.result`; idempotent under a double terminal call (no double-rewrite).
- **Ordered chain:** emit → reflect → grade — the rewrites land before the grade child is enqueued; grade scores the rewritten result, not the draft. A task with only `grade` (no emit/reflect) behaves exactly as today.
- **Reflection/emit as post-hooks:** a coder task whose draft needs emit → schema-conforming result before grade; a react result flagged `fix` by reflection applies the correction before grade.
- **hüsam purity:** import test asserts `hüsam` imports nothing from `coulson`; dispatcher `execute()` path imports no `fatih_hoca.select`.
- **No un-admitted dispatcher access:** dispatcher reached only via an admitted worker (assertion/test that `hüsam.run` runs under a task context).
- **Heartbeat:** a 60s+ child call keeps the parent/worker heartbeat fresh (no watchdog regression — the 2026-05-04 reason for the `_hb.keepalive()` wrap).
- **critic_gate:** split path produces the same pass/block decision; the mechanical gate makes no LLM call.
- **Pre-existing dispatcher tests** (`test_dispatcher_in_flight.py`, `test_dispatcher_records_swap.py`, flagged in SP2 handoff): verify status on merged SP3 before assuming SP3b broke them.

---

## 12. What "done" looks like

- Dispatcher = dumb pipe: one interface (`execute`), no `select`/retry/spec logic; `dispatch`/`_do_dispatch` deleted.
- `hüsam` exists as a peer worker package with a single `run(task)` entry; pump routes raw-dispatch + reflection/emit post-hooks to it; it imports nothing from coulson.
- coulson makes no inline non-react LLM call; reflection + constrained_emit are post-hook tasks; their prompts live in the post-hook layer.
- apply layer gains a **result-rewrite verdict path** and an **ordered emit→reflect→grade chain** (the two capabilities verified absent); both registered as new `POST_HOOK_REGISTRY` kinds.
- `critic_gate` split; no mechanical→dispatcher edge.
- No coulson↔hüsam import in either direction; `beckman.enqueue` remains the sole admission gate; nothing reaches the dispatcher un-admitted.
- `request()` survives only as a marked shopping-only shim; full `request`/`await_inline` deletion remains SP5 (gated on shopping migration).
- Built on merged SP3; both test suites green on the merged tree.

---

## 13. Known traps

- **Don't touch the coulson ReAct iteration loop.** SP3b changes coulson's *post-passes* (removes inline reflect/emit) and routing, not how react iterates.
- **`_hb.keepalive()` must survive.** Any new hüsam path keeps the worker heartbeat fresh during long child calls or the no-progress watchdog kills the runner (11+ wedged ❌ pings in 5 min, observed 2026-05-04).
- **`_task_result_to_request_response` is shared with SP3.** Keep it stable; relocate only with SP3's import updated on the merged tree.
- **Post-hook ordering** (emit→reflect→grade) may require extending SP3's `posthook_continuations` to ordered chains. Verify at STEP-0.
- **Cap-counting:** post-hook reflect/emit are now their own admitted tasks (more task churn, more lane contention than the old inline calls) — this is the deliberate, accepted price of purity; ensure admission/urgency tuning still lets them through under load.
- **`brand_voice_lint` is a sibling deadlock site, out of scope.** Its handler (`_run_llm_tone_pass`, brand_voice_lint.py:411) makes a **nested `await_inline` LLM call** — the same pattern, inside a post-hook handler. Not SP3b's (it's a Z7 handler; likely SP4 or a separate sweep) — but do **not** model reflection/emit on it. Use the spawned-child path (`_enqueue_posthook_llm_child`), never a nested `await_inline`.
- **Result-rewrite must be idempotent.** The rewrite verdict runs under SP1's claim-then-fire substrate; a double terminal call must not double-rewrite. Reuse the existing CAS guard.
- **Naming is provisional** (`hüsam`, `run`) — founder may change; keep the name out of load-bearing identifiers where cheap, or accept a rename pass.
