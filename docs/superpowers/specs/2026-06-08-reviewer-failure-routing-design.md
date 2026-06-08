# Reviewer-Failure Routing — Design

**Date:** 2026-06-08
**Status:** design (approved dimensions, pending spec review)
**Origin:** live DLQ — task #289752, step 3.11 requirements_review:
`requirements_review_result.status: value 'fail' not in allowed set ['pass']`.

## Problem

i2p reviewer steps (3.11 requirements_review, 4.16 architecture_review, 6.6
project_plan_review, …) are LLM steps that judge an upstream artifact and emit
a `*_review_result` artifact with a `status` verdict. Their instructions
explicitly say **`REJECT (status=fail)`**, but two things are broken:

1. **Schema drift.** The `status` `equals` enum on these steps lists only
   pass-class values (`['pass']`, `['pass','approved']`). The deterministic
   schema gate (shipped 2026-06-05) now hard-checks enums, so a reviewer
   *correctly* rejecting a bad artifact emits `status='fail'` → gate rejects
   the verdict as malformed → the reviewer task DLQs with a misleading
   "schema validation" error. The reviewer **succeeded** (it found a real
   problem) but its valid verdict cannot persist.

2. **No routing.** Reviewer steps carry **no `checks` and no `post_hooks`**.
   Even if `fail` persisted, nothing acts on it — the reviewer task would just
   complete and its lone dependent (e.g. 4.1) would proceed to build on the
   failed artifact. The verdict is decorative.

Fixing only (1) is a band-aid: it converts a DLQ-halt into a **silent
pass-through**, which is worse for quality (the mission builds architecture on
requirements that genuinely failed review).

## Why auto-routing the fix is not viable

The natural fix — "reviewer fail re-pends the producer to fix it" — does not map
onto the existing blocker rail. `_apply_simple_blocker_verdict`
(general_beckman/apply.py:4872) re-pends the **check's own source step**. A
reviewer is a *separate* step from the producer it reviews, so attaching a check
to the reviewer re-pends the reviewer (re-runs the review) — the same wrong loop
as the DLQ. The rail cannot target a named upstream step.

And the target can't be inferred. Audit of all 11 reviewer steps:

| reviewer | distinct producers reviewed |
|----------|-----------------------------|
| 1.7, 14.2 | 1 |
| 4.16, 7.16, 10.5 | 2 |
| 11.5 | 3 |
| 12.5 | 4 |
| 0.6, 6.6 | 5 |
| 1.13 | 6 |
| 3.11 | several (requirements_spec, prd, 4× falsification results) |

**Most reviewers review multiple producers.** Single-producer inference works
for only 2 of 11. "Re-pend all producers" over-fires (6.6 fail → re-pend 5
steps, most of which were fine). And *which* producer is at fault lives in the
reviewer's free-form `issues` text, not in structure — so deterministic
auto-routing to the right producer is impossible without re-architecting every
reviewer to emit per-issue machine-routable target pointers.

## Design — the founder is the router

A reviewer `fail` carries human-readable issues spanning several producers. The
founder reads the issues and knows which producer to fix. So routing is a
**founder-halt**, not an auto-loop. This sidesteps the multi-producer problem
entirely, reuses existing human-in-loop infra, and fits the product's
founder-controlled ethos.

### Flow

```
reviewer LLM step  ──emits──▶  *_review_result {status, issues, ...}
        │
        ▼  (post-step check, mechanical)
  verify_review_verdict
        │
   status pass-class? ──yes──▶ complete; dependents proceed
        │ no (fail)
        ▼
  FOUNDER-HALT (waiting_human)
   Telegram card: "<step> review FAILED" + issues, with buttons:
     · 🔁 Regenerate: <producer A> | <producer B> | …   (the reviewed producers)
     · ✅ Accept anyway (override)
        │
   ┌────┴───────────────┐
   ▼                    ▼
 regenerate           accept-anyway
 re-pend chosen       mark verdict overridden,
 producer with        complete reviewer,
 issues as feedback;  dependents proceed;
 producer re-runs →   audit_log the override
 reviewer re-reviews
```

### Components

1. **Schema reconcile (the #1 fix, generalised).** For every reviewer step whose
   instruction can emit a reject verdict, add the reject value to the `status`
   `equals` enum so the verdict persists. Scope = reviewers whose instruction
   declares `status=fail` (3.11, 4.16, 6.6 confirmed; audit the rest). Leave
   pass-only reviewers (7.16 sprint_0_review, 12.5 legal_review, 14.2
   launch_checklist_review — no `fail` in instruction) untouched. Verdict
   classes: **pass-class** = `{pass, approved}` (and `needs_minor_fixes` where
   declared — advisory, proceeds); **reject-class** = `{fail}` (halts).

2. **`verify_review_verdict` mechanical verifier** (new, in mr_roboto). Reads the
   review_result `status`. pass-class → `completed`. reject-class → a new
   verdict kind that triggers the founder-halt (NOT the producer-re-pend rail).
   Payload carries the reviewed-producer list (derived from the reviewer's
   `input_artifacts` → producer index at expand time) so the halt can render
   regenerate buttons.

3. **Founder-halt apply path** (general_beckman). On the reject verdict: set the
   reviewer task `waiting_human`, send the Telegram card (issues + producer
   regenerate buttons + accept-anyway), block dependents until resolved. Reuses
   the clarify/artifact-confirm keyboard + callback infrastructure
   (send_artifact_confirm_keyboard / the `sc:`/`rpc:` callback pattern).

4. **Callbacks** (telegram_bot):
   - **Regenerate <producer>** → re-pend that producer step with the reviewer's
     issues as retry feedback (reuse `regenerate_step_id` / regen rail, memory
     4af6e21c). Producer re-runs; the reviewer re-reviews on the re-pend
     cascade. *(Plan must confirm the re-pend cascade re-runs the reviewer after
     the producer.)*
   - **Accept anyway** → stamp the review_result with an `overridden_by_founder`
     marker, complete the reviewer, let dependents proceed, write an
     `audit_log` row.

### Explicitly out of scope (YAGNI)

- Auto-routing / per-issue target pointers / reviewer output schema redesign.
- Edit-in-place and abort-mission halt actions (founder chose regenerate +
  accept-anyway only).
- Retry/loop budget — there is no auto-loop; the founder paces retries.

## Distinguishing reviewer-fail from reviewer-task-failure

The verifier must separate **reviewer SUCCESS with a fail verdict** (well-formed
result, status=fail → founder-halt) from **reviewer TASK failure** (model error,
no parseable result → normal DLQ). Only the former enters the halt path.

## Testing

- Schema invariant (already drafted): every reviewer fixture's emitted verdict
  is in the step's `status` enum — the gate must permit what the reviewer emits.
  RED on 3.11/4.16/6.6 pre-fix, GREEN after enum reconcile.
- `verify_review_verdict`: pass-class → completed; reject-class → halt verdict
  with producer list; malformed/empty → task-failure path (not halt).
- Producer-list derivation: reviewer input_artifacts → correct producer step ids.
- Apply path: reject verdict sets waiting_human + blocks dependents; accept-anyway
  completes + audit row; regenerate re-pends the chosen producer with feedback.
- Callback handlers: button parse → correct re-pend / override.

## Interim / live DLQ

The live mission stays halted at 3.11 — which is the **desired** end state
(don't proceed on failed requirements). Shipping the schema reconcile first
turns the misleading "schema validation" DLQ into the clean founder-halt once
the subsystem lands. The schema reconcile is the first plan task and is already
implemented + green (enum edits on 3.11/4.16/6.6 + invariant test).

## Key files

- `src/workflows/i2p/i2p_v3.json` — reviewer `status` enums + `checks` wiring.
- `packages/mr_roboto/src/mr_roboto/` — new `verify_review_verdict`.
- `packages/general_beckman/src/general_beckman/apply.py` — reject-verdict halt path.
- `src/app/telegram_bot.py` — halt card + regenerate/accept callbacks.
- `src/workflows/engine/` — producer-index derivation for the verifier payload.
