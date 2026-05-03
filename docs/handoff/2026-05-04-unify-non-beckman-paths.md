# Handoff — Unifying Non-Beckman Dispatcher Paths

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries.
Code/commits/security: write normal.

Anchor: 2026-05-04

---

## The Architectural Question

> "I am more worried for even a path exists without beckman's admission"
> — user, 2026-05-04

Beckman is the task admission layer. It pulls from queue, runs eligibility,
calls fatih_hoca.select(), reserves an in_flight slot via `reserve_task`,
plumbs `est_tokens` so pool_pressure can back-pressure subsequent
admissions. Tracks `worker_attempts`, runs the retry ladder, decides
DLQ-vs-defer.

But ~10 of 16 dispatcher.request() callers fire **outside** the Beckman
pump. They go through KDV.pre_call (admission gate at call time) and
KDV.record_attempt (TPM reservation at call time), but bypass everything
Beckman provides for queue/budget/retry coordination.

---

## Inventory (16 dispatcher.request() sites)

### Category A — Beckman task path (1 site)
Inside an admitted task, Beckman has reserved the slot.

- `src/agents/base.py:2498` — ReAct main loop (MAIN_WORK)

### Category B — Within-task overhead (6 sites)
Task is admitted, but these overhead/grader/hook calls are SEPARATE LLM
invocations within the parent task's lifetime. `task_id` IS set via
heartbeat contextvar → in_flight slot is task-keyed (good for retry
recursion accounting). But Beckman's reserve_task fired for the PARENT
call only — these inherit that est_tokens, possibly an under-estimate
when overhead calls are large (constrained_emit on 30k draft).

- `src/agents/base.py:3782` — constrained_emit / structured_emit (OVERHEAD)
- `src/agents/base.py:3870` — alt-prompt retry (MAIN_WORK)
- `src/agents/base.py:3977` — self-reflection (OVERHEAD)
- `src/core/grading.py:305` — reviewer grader (OVERHEAD)
- `src/workflows/engine/hooks.py:46` — post-execute hooks (e.g. summary)
- `src/tools/vision.py:29` — vision tool from agent (MAIN_WORK)

### Category C — Standalone, no task context (9 sites)
`task_id` IS None → uuid-keyed in_flight entry. No Beckman queue
membership, no worker_attempts, no retry ladder beyond dispatcher's
internal max_recursion=5. Pool_pressure now sees their est_tokens
(thanks to commit `5f7f905`) but they fire whenever the calling
code path runs — no admission rate-limiting beyond KDV.pre_call.

- `src/app/telegram_bot.py:4145, 4570` — Telegram conversational replies
- `src/core/task_classifier.py:258` — pre-task classifier
  (decides what to create — can't be a task itself)
- `src/shopping/intelligence/_llm.py:42` — shopping helper
- `src/workflows/shopping/labels.py:22` — labeler
- `src/workflows/shopping/pipeline_v2.py:363, 487` — shopping pipeline
- (the dispatcher docstring example at `llm_dispatcher.py:66` doesn't count)

---

## What's Already Done (commit `5f7f905`)

`in_flight.begin_call` accepts `est_tokens: int = 0`. Dispatcher computes
from kwargs.estimated_input + estimated_output (same formula caller.py
uses for KDV's TPM reservation), passes through. Standalone calls now
contribute projected token cost to pool_pressure's in_flight overlay.
Task slots take `max(prior_reserve, passed)` so Beckman's value isn't
downgraded.

This closes the **TPM accounting gap** for non-Beckman paths but does
NOT address the bigger architectural question.

---

## The Real Question

Should some Category C paths be **wrapped as proper Beckman tasks**?

### Pro wrapping (full Beckman admission)
- Single source of truth for "what's running on the system right now"
- Queue back-pressure works uniformly — Telegram floods don't bypass
  the queue gate
- worker_attempts counting works uniformly — graders that crash N
  times in a row visible in DLQ rather than silently retrying
- Retry ladder + backoff applies uniformly
- Forensics (`admission_violations`, `model_pick_log`) capture
  consistent context

### Pro keeping standalone
- Telegram chat replies need low latency — task overhead would add
  hundreds of ms
- task_classifier MUST be standalone (decides what task to create)
- Shopping pipeline already has its own coordination layer
- Workflow hooks fire from inside Beckman dispatch — wrapping them as
  new Beckman tasks creates re-entrance risk
- Many Category C paths are short-lived single calls — task overhead
  is real cost for no benefit

### My read of the trade-off

| Path | Wrap as task? | Reason |
|------|---------------|--------|
| `task_classifier` | NO | Pre-task, can't be a task |
| `telegram_bot.py:4145, 4570` chat replies | MAYBE | Low-latency need; consider lightweight admission gate that's NOT full task lifecycle |
| `shopping/intelligence/_llm.py` | YES (probably) | Background work, not user-facing; should respect queue back-pressure |
| `workflows/shopping/labels.py` | YES | Labeling is task-like work |
| `workflows/shopping/pipeline_v2.py:363, 487` | YES | Pipeline stages are task-like |
| `tools/vision.py` | NO | Inside agent task already; Category B not C |
| Workflow hooks (post-execute) | NO | Inside Beckman dispatch already; Category B-ish |
| Grader / constrained_emit / reflection | NO | Within-task overhead; Category B |

So the real candidates are: `shopping/intelligence/_llm.py`,
`workflows/shopping/labels.py`, `workflows/shopping/pipeline_v2.py`,
and possibly Telegram chat replies under heavy load.

### Alternative: lightweight admission tier

Instead of full task wrapping, introduce a **lightweight admission gate**
for high-volume non-Beckman paths:

```python
# new in src/core/admission.py
async def acquire_call_slot(
    *, category: str, agent_type: str, est_tokens: int,
    timeout: float = 30.0,
) -> AcquiredSlot | None:
    """Async backpressure gate for non-Beckman dispatcher entries.

    Polls fatih_hoca.select() until urgency-threshold is met OR timeout.
    Returns context manager that releases on exit. NOT full task
    lifecycle — no DB row, no worker_attempts, no DLQ. Just queue-aware
    admission timing.
    """
```

Cheaper than task wrapping, gives queue-aware back-pressure to standalone
paths. Can ramp up its complexity later.

---

## Recommended Next Steps

1. **Audit each Category C site for actual call volume** in production.
   Use `model_pick_log` (no `task_id` filter) — get hits/min per source.
   Without volume data, "is wrapping worth it?" is unanswerable.

2. **Decide per-site policy** based on volume + latency requirements:
   - High volume + latency tolerant → wrap as task or lightweight gate
   - High volume + latency sensitive → lightweight gate only
   - Low volume → leave as-is, rely on KDV.pre_call

3. **For shopping paths specifically**: shopping pipeline is the most
   obvious wrap candidate. Consider creating Beckman tasks per shopping
   stage with `agent_type="shopping_advisor"` etc. — would also surface
   shopping work in `/dlq`, `/ps`, etc.

4. **Workflow hooks (post-execute)** are NOT actually Category C — they
   run inside Beckman dispatch already. The hook's call inherits parent
   task_id via contextvar. Verify this is working (grep for
   `current_task_id.get()` in hooks code path).

---

## Files involved

- `src/core/in_flight.py` — begin_call now takes est_tokens (5f7f905)
- `src/core/llm_dispatcher.py:268-291` — passes est_tokens (5f7f905)
- `src/core/heartbeat.py` — current_task_id contextvar
- `packages/general_beckman/src/general_beckman/__init__.py:372-382` —
  reserve_task call site (admission)
- `src/core/orchestrator.py` — `_hb.current_task_id.set(...)` site
- All 16 dispatcher.request() callers listed above

---

## Pitfalls

- **Don't wrap `task_classifier`** — it decides what task to create.
  Recursion deadlock.
- **Don't wrap Telegram chat replies blindly** — user-facing latency
  matters. Lightweight gate at most.
- **Don't break workflow hooks** — they fire from inside Beckman
  dispatch; making them their own tasks could deadlock the parent.
- **Beckman's worker_attempts is per-task** — wrapping a recurring
  shopping pipeline as a task means "one shopping task, N attempts"
  not "N shopping tasks each with M attempts." Match the unit
  semantically.
- **Don't introduce a SECOND admission queue** — that's the path to
  the divergent admission gate problem we just fixed (selector vs
  KDV). One queue, one source of truth.

---

## Open questions for next session

1. Is the lightweight admission tier worth building, or just wrap
   high-volume sites as full tasks?
2. Telegram chat replies: actual latency budget? Can we afford ~100ms
   admission overhead?
3. Shopping pipeline: who owns the architectural call — wrap as tasks
   vs. keep its own coordination?
4. Should there be a new "non-task" category in the in_flight registry
   that's distinct from "task" but tracked uniformly with the same
   est_tokens / worker_attempts plumbing?

---

## DON'T

- Don't revert `5f7f905` — the est_tokens plumbing is correct
  regardless of which architectural direction wins for the bigger
  question.
- Don't introduce a second admission gate alongside Beckman —
  unify, don't duplicate.
- Don't `pytest` without `timeout` prefix.
