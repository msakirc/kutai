# Decision handoff — `vision.py` `husam.run` direct call (ruling-#1)

**Date:** 2026-06-11
**Type:** founder decision (architectural). Carries forward the residual from the SP5 await_inline closure (`docs/handoff/2026-06-11-sp5-await-inline-DONE.md`).
**Status:** UNRESOLVED — needs a one-line founder ruling. No code change made (a prior founder finding called this "non-compliant"; reversing that is the founder's call, not mine).

---

## The thing

`src/tools/vision.py::analyze_image` (line 83) calls `husam.run(spec)` **directly** to do a vision LLM call, instead of `beckman.enqueue()`. A prior founder finding (shopping SP5, memory `project_shopping_sp5_group_findings_20260608`) flagged this as ruling-#1 non-compliant ("all LLM beckman-admitted, no shortcuts").

## Why it isn't a simple fix — and why the calculus changed today

`analyze_image` is a **mid-ReAct tool**: an agent (already a beckman-admitted task, executing inside `husam.run`) calls `analyze_image` as a tool and needs the analysis string returned **synchronously** to continue its ReAct loop. Two hard constraints:

1. **No synchronous beckman path exists anymore.** SP5 (this session) deleted `await_inline` — the only mechanism that let a caller block on a beckman task and get its result. `beckman.enqueue` is now fire-and-continue (`-> int | None`) with `on_complete`/`on_error` continuations only. A tool cannot use a continuation — it has nowhere to "resume" to; it must return a value inline.
2. **Admitting the nested call would recreate the lane-deadlock.** A nested LLM call admitted while the parent task still holds its lane slot is exactly the deadlock `await_inline` was deleted to prevent (parent holds slot → nested child can't admit → both stall). Calling `husam.run` directly — bypassing admission — *avoids* that deadlock.

`husam.run` is NOT a shortcut around model intelligence: it does `fatih_hoca.select` + `hallederiz_kadir` execute + result mapping (the same executor the orchestrator pump uses at `orchestrator.py:212`). What the nested call skips is only beckman's *admission* layer: lane-budget accounting, quota look-ahead, and lifecycle failure-tracking. For a rare, bounded vision call that is acceptable.

## Reachability

Live. `analyze_image` is a registered optional tool (`src/tools/__init__.py:188`), in the allowed-tools of `visual_reviewer` and others (`src/security/permissions.py:55,66`). Any agent with the tool can invoke it mid-loop.

## Options

**(a) Sanction `husam.run` for nested tool-internal LLM calls. — RECOMMENDED, ~zero cost.**
Declare that a mid-ReAct tool needing one inline LLM result uses `husam.run` directly; this is the deliberate pattern, not a violation, because (1) there is no sync-beckman alternative post-await_inline and (2) admission would deadlock. Action: update the comment in `vision.py` (lines 79-82) to state this rationale; optionally add a one-line note to ruling-#1 docs that "nested, inline, single-shot tool LLM calls use `husam.run` (un-admitted by design — admission would recreate the lane-deadlock)." No behavior change. Closes the residual.

**(b) Redesign vision as an async agent step. — BIG, not justified for one tool.**
Turn `analyze_image` from a synchronous tool into a beckman-admitted `visual_reviewer` task, and make the ReAct loop suspend/resume around a pending tool (CPS-for-tools). This breaks the tool contract (tools are synchronous) across the whole agent model — a large project, only worth it if a class of mid-ReAct nested LLM calls emerges (not just vision).

## Recommendation

**(a).** The "non-compliant" label was correct under the old world (where `await_inline` offered a sync-beckman path); SP5 removed that path and made `husam.run`-direct the only deadlock-free way to serve a synchronous nested tool call. Reframe it as sanctioned, update the comment, done.

**To close:** founder replies "(a)" → I update the `vision.py` comment + ruling-#1 note and close. Founder replies "(b)" → open a `vision-async-step` project (spec + plan; touches the tool/agent contract).
