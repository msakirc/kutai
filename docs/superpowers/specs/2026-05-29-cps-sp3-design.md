# Design — CPS SP3: in-task deadlock set (grading / code_review / summarize)

**Date:** 2026-05-29
**Status:** approved (brainstorm); ready for planning
**Owner:** founder + agent
**Parent specs:**
- `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (rev2 umbrella)
- `docs/superpowers/specs/2026-05-27-cps-migration-call-site-inventory.md` (sites #7–#10)
**Companion handoffs:**
- `docs/handoff/2026-05-28-sp3-kickoff.md`
- `docs/handoff/2026-05-29-cps-sp2-shipped.md`

---

## Problem

`await_inline=True` is used by three **post-hook agent** call paths that run **inside a
dispatched, cap-counted, non-mechanical task** — the exact triad that deadlocks
(`ONESHOT_CONCURRENCY=4` lane). Confirmed root cause (mission 77, 2026-05-27): a grader
parent holds a lane slot while blocking up to `INLINE_TIMEOUT=600s` on a reviewer child that
needs a slot from the same capped lane. Several such parents fill the lane, their children
can't be admitted, every parent times out → DLQ.

The three sites all share one shape after the G-grounding refactor:

```
source task completes
  → _apply_request_posthook parks source 'ungraded', enqueues a post-hook AGENT task
    (grader / code_reviewer / artifact_summarizer — cap-counted, non-mechanical)
    → agent.execute() calls grade_task / code_review_task / _llm_summarize
      → enqueues a raw_dispatch reviewer/summarizer LLM CHILD via await_inline=True
        ← agent task BLOCKS holding its slot = DEADLOCK
    → agent returns a `posthook_verdict` dict
  → _apply_posthook_verdict applies it to the source (pass→complete+advance / fail→retry-feedback / DLQ at cap)
```

Live sites (re-confirmed 2026-05-29, line numbers match inventory):
- **#7 grading** — `src/core/grading.py:373` (`grade_task`), agent `src/agents/grader.py`.
- **#8 code_review** — `src/core/code_review.py:179` (`code_review_task`), agent `CodeReviewerAgent`.
- **#9 summarize** — `src/workflows/engine/hooks.py:84` (`_llm_summarize`), agent
  `src/agents/artifact_summarizer.py`. **Note:** the *inline* `_llm_summarize` caller in
  `hooks.py` is gone — `post_execute_workflow_step` now stores a fast non-LLM
  `_structural_summary` and the LLM upgrade is scheduled as a post-hook (`apply.py:4211`
  grade-pass branch). So `_llm_summarize` is reached **only** through the summarizer agent.

#10 `dispatcher.request` (`src/core/llm_dispatcher.py:273`) is **out of scope** — see SP3b.

## Goal

Remove the DLQ deadlock by migrating the three inner `await_inline` reviewer/summarizer child
enqueues to the durable continuation substrate (SP1+SP1.1+SP2, frozen). After SP3 no grading,
code-review, or summarize work holds a lane slot while blocking on a child.

## Decisions (locked in brainstorm)

1. **Migration shape = B (collapse the agent layer).** The post-hook enqueues the
   `raw_dispatch` reviewer/summarizer child **directly** with `on_complete`/`cont_state`; the
   `grader` / `code_reviewer` / `artifact_summarizer` agent classes are **deleted**. Rejected
   shape A (keep agent as a thin launcher) — it leaves a vestigial instant-complete task per
   grade and forces the apply layer to tolerate verdict-less grader completions. B removes the
   cap-counted parent outright, halves task rows per grade/review/summary, and the thin agent
   adapters carry no logic worth preserving (real work is in the core fns).
2. **dispatcher.request → SP3b** (own spec + brainstorm). 7+ contract-locked callers
   (coulson `single_shot`/`reflection`, shopping `pipeline_v2`×2/`labels`/`intelligence`,
   mr_roboto `critic_gate`, `constrained_emit`), several mid-ReAct/mid-step expecting an inline
   return value. CPS inverts that contract; the likely answer is a bounded direct-dispatch
   path, not blanket CPS. Not part of SP3.
3. **Grading 2-attempt retry = continuation-chaining.** The inline
   `for attempt in (0, 1)` loop re-rolls the grader model on a **parse-fail of a `completed`
   call** (thinking-model reasoning leak → unparseable verdict). Beckman/Fatih task-retry fires
   only on `status == "failed"`, so it structurally cannot cover the completed-but-unparseable
   case — this loop is grading-specific (not coulson). Under CPS the loop becomes a chain: the
   grade resume handler re-enqueues a second reviewer child (failed grader added to exclusions)
   when attempt 0 fails to parse; a second parse-fail auto-fails the grade. Rejected
   "Beckman-native" (make the child self-fail on unparseable verdict) — it would leak
   grading-format knowledge into the generic dispatch/quality gate (`dogru_mu_samet`/HK only
   knows degenerate detection); chaining keeps grading semantics in `grading.py` (correct
   layer) and is lower risk.

## Architecture

**After SP3:**

```
source task completes
  → _apply_request_posthook parks source 'ungraded', enqueues the raw_dispatch
    reviewer/summarizer CHILD DIRECTLY:
       enqueue(spec, on_complete='posthook.<kind>.resume',
                     on_error='posthook.<kind>.resume_err',
                     cont_state={source_task_id, kind, attempt, exclusions, mission_id})
    → returns immediately; NO lane slot held; NO agent task row
  child reaches terminal state
    → posthook.<kind>.resume(child_id, result, state):
         parse child output → build PostHookVerdict → _apply_posthook_verdict(child_task, verdict)
```

**Phase-2 apply is reused verbatim.** `_apply_posthook_verdict` (`apply.py:3740`) and its
delegates (`_apply_code_review_verdict` `:2254`; grade fail `:3761` / grade pass `:4211`;
summary `:4275`) are unchanged. The resume handler's only job is to reconstruct the same
`PostHookVerdict` the deleted agents used to return and feed it to `_apply_posthook_verdict` —
exactly the "re-enter routing" the SP1 grading spike proved sufficient.

## Changes

### 1. Pure spec-builders (extract; keep `parse_*` functions)
- `grading.py::build_grading_spec(source, exclusions) -> spec` — lift the `GRADING_SYSTEM` +
  `GRADING_PROMPT.format(...)` message/spec construction (incl. the 30000-char response cap and
  the early degenerate/trivial auto-fail checks, which return a verdict without enqueueing).
- `code_review.py::build_code_review_spec(source, exclusions) -> spec` — lift `CODE_REVIEW_*`
  construction (incl. `produces`, degenerate auto-fail).
- `hooks.py::build_summary_spec(text, artifact_name) -> spec` — lift the summarizer
  message/spec (16k input cap, `prefer_speed`/`prefer_local`).

`grade_task` / `code_review_task` / `_llm_summarize` are removed (their parse halves move to the
resume handlers; their early auto-fail/degenerate checks move to the spawn helper, which can
short-circuit to an immediate verdict via `_apply_posthook_verdict` without a child).

### 2. New continuation module
`packages/general_beckman/src/general_beckman/posthook_continuations.py`, with a
`register_continuations()` that registers six handlers and is added to `_HANDLER_MODULES`:
- `posthook.grade.resume` / `posthook.grade.resume_err`
- `posthook.code_review.resume` / `posthook.code_review.resume_err`
- `posthook.summary.resume` / `posthook.summary.resume_err`

### 3. Spawn refactor (apply.py)
New helper `_enqueue_posthook_llm_child(kind, source, source_ctx)`:
- builds the raw_dispatch spec via the matching `build_*_spec`,
- handles the immediate-verdict short-circuit (trivial/empty/degenerate source → apply auto-fail
  verdict now, no child),
- otherwise `enqueue(spec, parent_id=source_id, on_complete=..., on_error=..., cont_state=...)`.

Call it from **both** LLM-kind spawn sites:
- `_apply_request_posthook` (`:1189` add_task) for `kind in {grade, code_review, summary:*}`.
- the grade-pass summary loop (`:4217` add_task, `agent_type="artifact_summarizer"`).

Mechanical post-hook kinds (`verify_artifacts`, `grounding`, `test_run`, `imports_check`,
`pattern_lint`, and the Z3/Z4 reviewer kinds that already run config-only/mechanical) are
**untouched** — cap-exempt, no deadlock, not in SP3.

### 4. Delete the agent layer
- Files: `src/agents/grader.py`, `src/agents/code_reviewer.py`, `src/agents/artifact_summarizer.py`.
- Refs: agent-registry/classifier entries for `grader`/`code_reviewer`/`artifact_summarizer`;
  `_OVERHEAD_POSTHOOK_AGENTS` membership (`apply.py:1145`); `_posthook_kind` mapping;
  `_send_step_progress` SP1.1-I6 exclusions (those agent types vanish — the raw_dispatch child's
  `agent_type="reviewer"`/`"summarizer"` exclusion stays); the schema-validation producer-skip
  for those agent types (`hooks.py:1465`); the grader-DLQ-cascade (`apply.py:832`) → relocate
  into `posthook.grade.resume_err`.

## Resume handler logic

State (JSON, set at spawn): `{source_task_id, kind, attempt, exclusions, mission_id}`.
`result` is the child's normalized result dict (the raw reviewer text rides in `result`, decoded
in-handler per the I5 no-magical-unwrap policy).

- **`posthook.grade.resume(child_id, result, state)`**
  - `verdict = parse_grade_response(extract_content(result))`.
  - parse OK → `PostHookVerdict(kind="grade", source_task_id, passed=verdict.passed, raw=verdict-dict)` → `_apply_posthook_verdict`.
  - parse-fail & `state.attempt == 0` → re-enqueue reviewer child #2 with
    `exclusions + [failed_grader_model]`, `cont_state.attempt = 1`, same handlers. Return.
  - parse-fail & `state.attempt == 1` → `PostHookVerdict(kind="grade", passed=False, raw="auto-fail: grader_incapable after 2 attempts")` → apply.
- **`posthook.grade.resume_err(child_id, failed_result, state)`** — reviewer child terminally
  failed (infra). Build `PostHookVerdict(kind="grade", passed=False, raw="auto-fail: grader call failed")` → apply. Carries the relocated grader-DLQ-cascade semantics (source → permanently failed when the grade path can produce no verdict and the source is at cap).
- **`posthook.code_review.resume`** — `parse_code_review_response` →
  `PostHookVerdict(kind="code_review", passed, raw=issues)` → apply. No chaining (single-shot
  today). **`.resume_err`** → `passed=False` → apply (drives source retry-with-feedback).
- **`posthook.summary.resume`** — extract content; `passed = bool(summary) and len(summary) >= 50`
  and not degenerate; `PostHookVerdict(kind=f"summary:{artifact_name}", passed, raw={"summary", "artifact_name"})` → apply (stores `{name}_summary` to blackboard, drains pending, completes source if last). **`.resume_err`** → `passed=False` → apply (structural summary already stored by `post_execute`; just drain pending).

## Substrate invariants honored

- Fire on DB `tasks.status` post-apply — substrate already gates this; resume gets the agent
  pre-post-hook snapshot. SP3 adds no fire/route logic.
- `cont_state` JSON-serializable (ints / strings / list of model-name strings).
- `await_inline` XOR `on_complete`/`on_error` — SP3 uses only `on_complete`/`on_error`.
- `needs_clarification` — N/A: raw_dispatch overhead reviewer/summarizer children never emit it.
- Every new continuation-bearing module (`posthook_continuations`) added to `_HANDLER_MODULES`.
- Reconcile reconstructs from `tasks.result` top-level; the reviewer envelope (`{"content": ...}`)
  is decoded inside the handler. The fields the resume needs (the reviewer's raw text) live in
  `tasks.result` — verified by a reconcile test.

## Testing (host-path, DB-isolated, `timeout` prefix)

- **Happy:** grade pass → source completes (+ summary children spawned); grade fail → source
  re-pends with grader feedback + exclusion; code_review pass/fail; summary stored to blackboard.
- **Chaining:** grade attempt-0 parse-fail → a 2nd reviewer child enqueued with the failed grader
  in `exclude_models` and `cont_state.attempt == 1`; attempt-1 parse-fail → auto-fail verdict.
- **C1 regression (keystone):** reviewer child `failed → retried → completed` fires the resume
  exactly once on the final completed status (no silent drop, no premature on_error).
- **on_error:** reviewer child terminal-failed → auto-fail verdict applied; grader-path DLQ
  cascade fires from `resume_err` when source at cap.
- **CAS idempotency:** double `on_task_finished` for the same child fires the resume once.
- **Deadlock closure:** `_apply_request_posthook` for `kind="grade"` enqueues a raw_dispatch
  child and **returns without enqueuing any `grader` agent task** (assert no agent task row, child
  carries a continuation row).
- **Deletion guards:** `grader`/`code_reviewer`/`artifact_summarizer` agent classes are gone and
  unreferenced by the classifier/registry; `await_inline=True` removed from `grading.py`,
  `code_review.py`, `hooks.py`.
- **Apply reuse:** `_apply_posthook_verdict` called from the resume handler produces identical
  source-state transitions to the pre-SP3 agent-return path (verdict round-trips per
  `feedback_verify_verdict_roundtrip`).

## Risks

- **`_apply_posthook_verdict(task, a)` first arg.** Today `task` = the grader agent task; the
  resume handler will pass the reviewer **child** task. Confirm `task` is used only for
  logging/attribution and that `source` is re-fetched from `a.source_task_id` (it is, at `:3746`).
  If `task` is read for anything load-bearing, pass a compatible dict.
- **Source `ungraded` window** between spawn and resume — identical to the await_inline window
  today; restart is covered by reconcile + alive-aware TTL.
- **Two summary spawn sites** (`:1211` and `:4217`) must both route through the new helper or one
  path silently keeps the agent task.

## Acceptance

- No `await_inline=True` in `grading.py`, `code_review.py`, `hooks.py`.
- Grading/code-review/summarize post-hooks enqueue a raw_dispatch child + continuation and return
  without holding a lane slot; a grader parent no longer occupies a slot for 600s blocking on its
  reviewer child (DLQ deadlock closed).
- All resume handlers re-enter `_apply_posthook_verdict` with verdicts equivalent to the deleted
  agents'; source transitions unchanged (pass→complete, fail→retry/DLQ).
- The three agent classes deleted; `posthook_continuations` in `_HANDLER_MODULES`.
- Full SP3 test set green; existing beckman + grading + code-review + hooks suites green.
- SP4 (tools + mechanicals) and SP5 (delete primitive) unblocked; SP5 carve-outs (#2
  task_classifier, #6 investor_bullets) unaffected.

## Out of scope

- **SP3b** — `dispatcher.request` shim (#10) and its contract-locked callers.
- **SP4** — `vision` tool, mr_roboto LLM executors, `yalayut/discovery/synthesize`, the two
  `posthook_handlers/` LLM gates (`brand_voice_lint`, `copy_compliance_review`).
- **SP5** — delete `await_inline`/`resolve_inline`/`_inline_waiters`/`INLINE_TIMEOUT`.
- The substrate itself (`continuations.py` fire logic, `db.py`, `enqueue`) — frozen.
