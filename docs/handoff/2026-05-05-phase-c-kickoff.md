# Handoff — Phase C kickoff (C.1 + C.2a shipped)

Anchor: 2026-05-05. User: sakircimen@gmail.com.
Continues `docs/handoff/2026-05-04-runtime-phase-a-complete.md` and
`docs/handoff/2026-05-04-record-model-call-audit.md`.

## What shipped

| SHA | What |
|---|---|
| `37d73b2` | C.1 — extract `_execute_attempt` from `_do_dispatch` body. In_flight bracket + local-load + hallederiz call + record_pick lives in primitive. `_do_dispatch` becomes selection + retry-recurse around the primitive. Loading failure is now `CallError(category="loading", retryable=False)` which the existing non-retryable branch maps back to `ModelCallFailed(error_category="loading")` — outward contract preserved. |
| `2953476` | C.2a — rename `_execute_attempt` → public `execute`. No external callers yet. |

Tests after each commit: same 3 pre-existing failures only
(`tests/core/test_dispatcher_in_flight.py` x2,
`tests/core/test_dispatcher_records_swap.py`). Pre-existed on `dc49695`,
verified via git stash.

## What's next — C.2b (the substantial one)

Goal: coulson.react calls `dispatcher.execute(pick, messages, ...)` per
iter with locally-managed `fatih_hoca.select()` + failures list. Bypass
`dispatcher.request → beckman.enqueue` per-iter sub-tasking. Single
retry surface inside coulson.

### Where the call lives today

`packages/coulson/src/coulson/react.py:421-466`:

```python
from src.core.llm_dispatcher import get_dispatcher, CallCategory
response = await get_dispatcher().request(
    CallCategory.MAIN_WORK,
    task=reqs.effective_task or reqs.primary_capability,
    agent_type=reqs.agent_type,
    difficulty=reqs.difficulty,
    messages=messages,
    tools=litellm_tools,
    needs_thinking=reqs.needs_thinking,
    needs_function_calling=reqs.needs_function_calling,
    needs_vision=reqs.needs_vision,
    local_only=reqs.local_only,
    prefer_speed=reqs.prefer_speed,
    prefer_quality=reqs.prefer_quality,
    prefer_local=reqs.prefer_local,
    estimated_input_tokens=reqs.estimated_input_tokens,
    estimated_output_tokens=reqs.estimated_output_tokens,
    min_context=reqs.effective_context_needed,
    priority=reqs.priority,
    exclude_models=reqs.exclude_models or [],
    remaining_budget=max(0.0, _remaining),
    preselected_pick=task.get("preselected_pick") if iteration == 0 else None,
    task_obj=task,
    iteration_n=iteration,
)
```

`dispatcher.request` builds a Beckman spec, calls `beckman.enqueue(spec,
await_inline=True)`, which admits a sub-task that the orchestrator pump
later routes to `dispatcher.dispatch(spec)` → `_do_dispatch`. That admission
path runs Hoca.select inside Beckman's pre-dispatch step (preselected_pick
is the result), reserve_task, and pool_pressure gate.

### What C.2b must replace

1. **Per-iter Pick selection.** Today: Beckman admission picks via Hoca
   for iter 0; for iter N>0, `_do_dispatch`'s retry-recurse re-selects.
   New: coulson handles both — uses `task.preselected_pick` for iter 0
   if present, else calls `fatih_hoca.select(...)`. Bumps urgency on
   retries (mirror lines 358-366 of dispatcher).

2. **Kwarg massaging that today happens inside `_do_dispatch`** (lines
   307-344):
   - `needs_thinking` defaulting (overhead = False, else True)
   - `needs_function_calling` from tools presence
   - `min_context` floor: `int((est_in + est_out) * 1.3) + 512`
   - `response_format` extraction + `needs_json_mode` injection if set
   - `task_obj`, `iteration_n` plumbed to hallederiz for B-table rollup

   These are inputs to either `fatih_hoca.select` (selection-shaping)
   or `dispatcher.execute` (call-shaping). Coulson will need both
   paths.

3. **Failures plumbing.** Today: `_do_dispatch` accumulates Failures
   across retry-recurses, passes to `fatih_hoca.select(failures=...)`.
   New: coulson maintains a per-task `failures: list[Failure]` list.
   Appends on each `CallError`. Passes to next select. Reset between
   outer iters? **Decide:** spec says "iter N>0 OR transport failure"
   triggers re-select — implies failures persist across outer iters.
   Verify against today's `_do_dispatch` semantics (recursion is per
   outer call, so today failures DON'T persist across react iters —
   each react iter starts fresh). C.2b should match: failures list is
   per-iter, recreated each outer iter. Transport-retry budget within
   one outer iter ~3 attempts (spec says fixed; expose later if
   needed).

4. **Surface mapping.** `dispatcher.execute` returns `CallResult |
   CallError`. Coulson must map CallResult → response dict (the
   coulson loop reads `content`, `model`, `cost`, `latency`, `tool_calls`,
   `thinking`, `usage`, `partial_content`). Use `_result_to_dict`
   logic — promote it out of dispatcher into a small helper (or
   inline the 12-line mapping in coulson).

5. **Lose Beckman per-iter sub-tasking.** That's the *point* of C — per
   ReAct iter no longer admits a sub-task. Single retry surface. The
   `record_model_call` double-emission (audit doc) goes away because
   only one path emits per attempt.

### What stays the same

- `dispatcher.request` keeps its current shape for non-react callers
  (graders, structured_emit, classifier, raw_dispatch). They still
  enqueue via Beckman. C.2b only touches react.
- Beckman admission still picks iter-0 model and attaches
  `preselected_pick` to the task — react reads it as before.
- `record_model_call` and `record_cost` calls in react.py:474-489 stay
  put. The dispatcher-side `record_pick` (different table —
  `model_pick_log`) also stays put inside `dispatcher.execute`.

### Risks / hotspots

- **`heartbeat.keepalive`** wraps the `dispatcher.request` call today
  (lines 241-243 of dispatcher) for parent-task no-progress watchdog.
  After C.2b that wrapper needs to land inside coulson.react around
  the new call site — otherwise the parent goes 300s without a bump
  during slow swaps and gets killed. Production triage 2026-05-04
  showed this was real.
- **Forensics on pool-empty mid-task** (dispatcher lines 386-413): the
  `record_admission_violation` write currently fires inside
  `_do_dispatch` when Hoca returns None on retry. After C.2b that path
  moves to coulson — preserve the forensics write.
- **`fatih_hoca.select` kwarg surface.** Big. The current call (lines
  367-376 of dispatcher) takes `**kwargs` covering urgency,
  prefer_speed/quality/local, local_only, exclude_models,
  remaining_budget, etc. Build a small `pick_for_iter(reqs, task,
  failures, iteration)` helper inside coulson that owns this mapping.

### Suggested step order for C.2b

1. Add `pick_for_iter(reqs, task, failures, iteration_n) -> Pick`
   helper inside `coulson/dispatch_helpers.py`. Mirror today's
   selection logic from `_do_dispatch` lines 307-376.
2. Add `_to_response_dict(result)` helper (the existing
   `_result_to_dict` body).
3. Rewrite the `try: response = ...` block in `react.py` to:
   - `pick = pick_for_iter(reqs, task, failures, iteration)`
   - inner transport-retry loop (max 3 attempts):
     - `result = await dispatcher.execute(pick=pick, messages=messages, ...)`
     - if `CallResult`: convert to dict, break out
     - if `CallError` retryable: append Failure, `pick =
       pick_for_iter(reqs, task, failures, iteration)` (re-select),
       continue
     - if `CallError` not retryable: raise `ModelCallFailed`
4. Wrap the new call in `heartbeat.keepalive()`.
5. Run trace replay (still pending from Phase A) — capture 5
   production ReAct tasks, run new path, diff TaskResult.

### After C.2b

C.3 (delete `_do_dispatch` retry recursion + Hoca call): becomes mostly
trivial — the retry block is unused once react bypasses it. Keep the
selection + execute wrapping for `dispatcher.dispatch` callers (raw
overhead via Beckman). Maybe the right shape is to make `dispatch`
call `pick_for_iter` (shared helper) + `execute` directly, deleting
`_do_dispatch` entirely.

C.5 verification:

```bash
grep -nE "fatih_hoca\.select|hoca\.select" src/core/llm_dispatcher.py
# Should return only the call inside the shared pick helper, NOT in
# any retry-recurse position.

grep -rE "track_model_call_metrics" src/infra/db.py
# Per audit doc fix: remove the call from db.record_model_call so
# hallederiz_kadir.caller is the single in-memory metric emitter.
```

Counter inflation should drop ~50% on active models.

## Quick-resume

```bash
git log --oneline -5     # 2953476 → 37d73b2 → dc49695 ...
git diff dc49695..HEAD -- src/core/llm_dispatcher.py | wc -l
```

Read this handoff + the audit doc + spec/plan, then start at "Suggested
step order for C.2b" above.
