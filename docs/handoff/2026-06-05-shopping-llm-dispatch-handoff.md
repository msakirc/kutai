# Handoff — how shopping should do LLM dispatches (SP5 unblock)

**For:** the parallel shopping-iteration session, blocked on "how to proceed with LLM dispatches."
**From:** the SP4b session (reviews CPS + the founder's mechanical-only ruling).
**Date:** 2026-06-05.
**Read alongside:** `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md` (the carrier reasoning), memories [[no-direct-dispatcher-from-mechanical]], [[feedback_singular_dispatcher_caller]], [[project_cps_migration_20260527]].

---

## The one rule (non-negotiable)

**No code calls `LLMDispatcher.request()` / the dispatcher / husam directly. Every LLM hop is an admitted Beckman task** — `beckman.enqueue(...)`. The dispatcher is Beckman's private worker. The shopping `request()` shim is the LAST direct caller; migrating it off is exactly what lets **SP5 delete the primitive**. So your job = route every shopping LLM call through `enqueue`.

## How to pick the CARRIER — by job-shape, one decision tree

This is the reusable part of the SP4b design. Two carriers, chosen by the job's natural grain:

```
Is the LLM hop part of a multi-step / fan-out pipeline?
  ├─ YES → WORKFLOW STEP. Make each agent hop an `agent:<type>` step in the
  │        workflow JSON; the engine sequences via depends_on + passes artifacts.
  │        (mechanical steps stay `agent:mechanical`.)
  └─ NO (single producer→consumer hop, fired imperatively):
        Is a human/caller WAITING on the result in the same turn?
          ├─ YES (interactive) → INTERACTIVE CPS / husam-inline.
          │     Resume back into the conversation (SP2 telegram/interview family),
          │     OR husam.run inline (SP4a shape-a) for one synchronous call.
          └─ NO (fire-and-surface-later) → CPS CONTINUATION → mechanical sink.
                enqueue(spec, on_complete="...", on_error="...", cont_state={...});
                a registered sink persists/surfaces. (This is the reviews SP4b shape.)
```

**The axis that decides single-hop cases: is the consumer waiting?** Waiting → interactive carrier. Not waiting → mechanical sink. Get this wrong and you either block the pump on a human (bad) or make a waiting user stare at nothing (bad).

## Applied to shopping (do NOT blindly copy reviews)

| Shopping path | Shape | Carrier | Notes |
|---------------|-------|---------|-------|
| Complex query (3-task mission: researcher→analyst→advisor) | multi-step | **workflow steps** | It's ALREADY a mission. Workflow JSONs exist (`shopping_v2`, `product_research_v2`, `combo_research`). Just make each agent hop a real admitted `agent:` step. Identical to SP4b's workflow carrier — no new pattern. |
| Simple query (single agent, user typed "find me X" and is WAITING in Telegram) | single hop, **interactive** | **interactive CPS resume** (SP2 telegram family) OR **husam-inline** | ⚠️ Do NOT use the reviews "fire → surface via a later founder_action" sink — the user is blocked on the answer in THIS turn. Resume into the conversation, or a single synchronous husam.run if acceptable. |

**Why the warning:** reviews verbs surface their output as a *later* founder_action card (async-tolerant — the founder isn't blocked). Shopping simple-query is the opposite: synchronous, user-facing. Same principle, different carrier — that's the whole point of the decision tree.

## Concrete CPS mechanics (if you do use continuations)

- `enqueue(spec, *, lane=LANE_ONESHOT, on_complete="name", on_error="name", cont_state={...})` → returns task_id (NO `await_inline`). `await_inline` is mutually exclusive with on_complete/on_error.
- Handler signature: `async def h(child_task_id: int, result: dict, state: dict) -> None`. Register via `register_resume("name", h)` inside a module-level `register_continuations()`; **add the module to `general_beckman.continuations._HANDLER_MODULES`** or the handler is absent after restart (silent correctness bug — rows stay pending).
- Read LLM text with a dual-shape decoder (normal `result['result']['content']` vs restart-reconcile top-level `result['content']`). Copy `_extract_content` from `posthook_continuations.py` / `mr_roboto/executors/reviews_continuations.py`.
- **lane = "oneshot" ONLY.** Never `overhead` (phantom lane the pump never selects → orphaned). `ongoing` may also be unpumped.
- raw_dispatch overhead spec shape: `context.llm_call = {raw_dispatch:True, call_category:"overhead", agent_type, difficulty, messages, failures:[], estimated_*_tokens}`.

## Where prompts live (founder ruling)

The LLM and its prompt **leave** any mechanical module. Workflow-carried → prompt in the step JSON (built by the expander). CPS/imperative → a thin producer module OUTSIDE the mechanical package (see `src/reviews/producers.py` for the reviews example — it holds the prompts + builds the spec + enqueues). Shopping prompts go in a shopping producer/agent-step home, never inside a mechanical verb.

## Scope caveat

Shopping is a 25.9k-LOC sprawl with its own pipeline (`src/shopping/`, ShoppingPipeline). It gets its **own brainstorm + spec** — the carrier rule transfers, the specific sinks do NOT. Don't reuse reviews' founder_action sink for a synchronous product answer.

## Live status (shared substrate — both sessions depend on it)

- Substrate verified healthy 2026-06-05: all tasks `lane=oneshot`; 156 post-hook children (constrained_emit/self_reflect/grade/critic) all completed, zero orphaned. The pump/continuation path works.
- `husam` is now `pip install -e`'d. A `/restart` is still needed to pick it up before anything husam-inline runs live.
