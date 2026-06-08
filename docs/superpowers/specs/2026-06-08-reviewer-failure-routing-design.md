# Reviewer-Failure Routing — Design

**Date:** 2026-06-08
**Status:** design (approved dimensions, pending spec review)
**Origin:** live DLQ — task #289752, step 3.11 requirements_review:
`requirements_review_result.status: value 'fail' not in allowed set ['pass']`.

## Problem

i2p reviewer steps (3.11 requirements_review, 4.16 architecture_review, 6.6
project_plan_review, …) are LLM steps that judge upstream artifacts and emit a
`*_review_result` with a `status` verdict. Their instructions explicitly say
**`REJECT (status=fail)`**, but two things are broken:

1. **Schema drift.** The `status` `equals` enum lists only pass-class values
   (`['pass']`, `['pass','approved']`). The deterministic schema gate (shipped
   2026-06-05) now hard-checks enums, so a reviewer *correctly* rejecting a bad
   artifact emits `status='fail'` → gate rejects the verdict as malformed → the
   reviewer task DLQs with a misleading "schema validation" error. The reviewer
   **succeeded** (found a real problem) but its verdict cannot persist.

2. **No routing.** Reviewer steps carry **no `checks` and no `post_hooks`**.
   Even if `fail` persisted, nothing acts on it — the reviewer task completes and
   its dependents proceed to build on the failed artifact. The verdict is
   decorative.

Fixing only (1) is a band-aid: it turns a DLQ-halt into a **silent
pass-through**, worse for quality.

## Principle: the system auto-recovers; the founder is a last resort

This is an autonomous agent. A reviewer `fail` must drive the AI to **find the
problematic producer(s) and retry them**, automatically. The founder is escalated
to **only** when auto-recovery is exhausted or a fault genuinely cannot be
localised. Founder-halt is the safety net, not the mechanism.

## Why the existing blocker rail does not fit

`_apply_simple_blocker_verdict` (general_beckman/apply.py:4872) re-pends the
check's **own source step**. A reviewer is a *separate* step from the producers
it reviews, so a check on the reviewer would re-pend the reviewer (re-run the
review) — the same wrong loop as the DLQ. The rail cannot target a named
upstream step, and the target is not structurally inferable: most reviewers
review **multiple** producers.

| reviewer | distinct producers reviewed |
|----------|-----------------------------|
| 1.7, 14.2 | 1 |
| 4.16, 7.16, 10.5 | 2 |
| 11.5 | 3 |
| 12.5 | 4 |
| 0.6, 6.6 | 5 |
| 1.13 | 6 |
| 3.11 | several (requirements_spec, prd, 4× falsification results) |

Single-producer inference works for 2 of 11; "re-pend all" over-fires (6.6 →
5 steps). The fault attribution lives in the reviewer's issues — so the router
must read the issues, not the graph.

## Design — hybrid auto-router, founder-halt as fallback

### Reviewer output becomes routable

Reviewer `issues` change from free-form to structured:

```
issues: [ { target_artifact: <artifact name | null>,
            severity: "blocker" | "major" | "minor",
            problem: <one line> } , … ]
status: pass | fail            (+ approved where that reviewer uses it)
```

`target_artifact` is the artifact the issue is about — the reviewer knows this
and names it when it can. It MAY be null (issue is systemic / the reviewer is
unsure). `severity` lets minor issues pass without blocking (advisory).

### Routing (on `status=fail`)

```
fail issues
   │
   ├─ each issue with target_artifact that maps to a producer  ──┐
   │     (artifact → producer index, built from output_artifacts) │
   │                                                              ▼
   ├─ issues with null/unmappable target_artifact ──▶ ROUTER LLM ─┤  group by producer
   │     (reads issue + the reviewed-artifact→producer list,      │
   │      assigns each to the most likely producer, or "unknown") │
   │                                                              ▼
   ▼                                              re-pend each implicated producer
 nothing mappable at all ───────────────────────▶  with its issues as retry feedback
        │                                            (reuses _stamp_retry_feedback →
        ▼                                             worker_attempts + model escalation)
   FOUNDER-HALT                                              │
                                                             ▼
                                              producers re-run → reviewer re-reviews
```

- **Tag path (deterministic):** issue.target_artifact → producer via the
  artifact→producer index (the reviewer's `input_artifacts` already name the
  reviewed artifacts; the index maps artifact → the step whose
  `output_artifacts` produced it).
- **LLM fallback:** untagged/unmappable issues go to a router LLM (OVERHEAD
  call) that assigns each to a producer from the reviewed-producer set, or
  emits `unknown`.
- **Re-pend:** each implicated producer is re-pended with the relevant issues
  as feedback, through the existing retry rail (`_stamp_retry_feedback` →
  per-producer `worker_attempts` climb to 3 + model escalation). After the
  producers fix, the reviewer re-reviews on the re-pend cascade.
  *(Plan must confirm the re-pend cascade re-runs the reviewer after its
  producers — the reviewer depends on them, so a dependency-aware re-pend
  should; verify.)*

### Termination — the existing retry cap, no new budget

A reviewer `fail` is a quality failure, so it rides the **existing** retry rail
rather than a bespoke budget. The review-fix re-pend MUST target the producer's
**existing task row** (not a fresh attempt=0 task), so `worker_attempts`
increments (`_stamp_retry_feedback`, apply.py:515) and the normal
`max_worker_attempts` cap + model escalation bound the loop automatically:

- Same producer re-blamed each round → its `worker_attempts` climbs → it DLQs
  terminally at its cap (existing rail). No reviewer-specific round budget.
- The reviewer re-runs only when a producer it depends on is re-pended-and-
  completes, so total reviewer re-runs ≤ Σ producer attempts — also bounded.

**Founder-halt triggers (last resort only):** (a) the router returns `unknown`
for all blocker issues (no localisable target); (b) every implicated producer
has exhausted its normal attempts (terminal DLQ). There is no separate round
counter to tune.

### Founder-halt (fallback UX)

When escalated, surface to the founder via Telegram (reuse the
clarify/artifact-confirm keyboard + callback infra) with the issues and two
actions:

- **🔁 Regenerate <producer>** — buttons for the reviewed producers; founder
  picks one to re-run with the issues as feedback; reviewer re-reviews.
- **✅ Accept anyway (override)** — founder overrules; stamp the review_result
  `overridden_by_founder`, complete the reviewer, dependents proceed, write an
  `audit_log` row.

(Edit-in-place and abort-mission were considered and dropped — YAGNI.)

### Distinguishing reviewer-fail from reviewer-task-failure

The verifier separates **reviewer SUCCESS with a fail verdict** (well-formed
result, status=fail → routing) from **reviewer TASK failure** (model error / no
parseable result → normal DLQ). Only the former enters the routing path.

## Components

1. **Schema reconcile — all 11 reviewers.** Convert `issues` to the structured
   shape above on every reviewer (0.6, 1.7, 1.13, 3.11, 4.16, 6.6, 7.16, 10.5,
   11.5, 12.5, 14.2) so the subsystem is uniform. Add `fail` to the `status`
   `equals` enum for every reviewer whose instruction can reject — audit each;
   3.11/4.16/6.6 confirmed. Reviewers with no reject path today (e.g.
   7.16/12.5/14.2) still get structured issues + the verify check (it completes
   on pass), so adding a reject path later needs no rewiring. pass-class =
   `{pass, approved}` (+ `needs_minor_fixes` advisory); reject-class = `{fail}`.
2. **Reviewer instruction update — all 11.** Each reviewer must emit
   `target_artifact` + `severity` per issue (name the artifact each issue is
   about; null only when systemic).
3. **`route_review_failure`** (new, general_beckman/coulson). The hybrid router:
   tag-map → LLM fallback → group by producer → re-pend each producer's
   **existing task row** with feedback (so `worker_attempts` carries forward);
   escalate to founder-halt only on the two triggers above.
4. **`verify_review_verdict`** check on each reviewer step: pass-class →
   complete; reject-class → hand to `route_review_failure`.
5. **Producer index** derivation (workflow engine): artifact → producing step.
6. **Founder-halt path + callbacks** (general_beckman + telegram_bot): the
   fallback card + regenerate/accept-anyway handlers.

## Out of scope (YAGNI)

- Per-issue auto-fix without re-running the producer (we re-pend producers, not
  patch artifacts directly).
- Edit-in-place / abort-mission founder actions.
- Cross-mission learning from review failures (separate Z9 concern).

## Testing

- Schema invariant (drafted): every reviewer fixture's emitted verdict is in the
  step's `status` enum. RED on 3.11/4.16/6.6 pre-fix, GREEN after reconcile.
- Structured issues: reviewer fixtures emit `[{target_artifact, severity,
  problem}]`; schema validates.
- Tag routing: tagged issue → correct producer via the index (deterministic, no
  LLM).
- LLM fallback: untagged issue → router assigns a producer or `unknown` (stub
  the LLM).
- Termination: re-pend targets the producer's existing row → `worker_attempts`
  increments (not reset); all-`unknown` → founder-halt; producer terminal DLQ →
  founder-halt.
- Re-pend: implicated producer re-pended with the right feedback; reviewer
  re-reviews after.
- Founder-halt callbacks: regenerate <producer> re-pends correctly;
  accept-anyway overrides + audit row.

## Interim / live DLQ

The live mission stays halted at 3.11 (the desired end state — don't proceed on
failed requirements). The schema reconcile is the first plan task and is already
implemented + green (enum edits on 3.11/4.16/6.6 + invariant test, uncommitted);
it turns the misleading "schema validation" DLQ into the routed path once the
subsystem lands.

## Key files

- `src/workflows/i2p/i2p_v3.json` — reviewer `status` enums, structured `issues`,
  `checks` wiring, per-issue instruction.
- `packages/mr_roboto/src/mr_roboto/` — `verify_review_verdict`.
- `packages/general_beckman/src/general_beckman/apply.py` — `route_review_failure`,
  round budget, founder-halt escalation.
- `packages/coulson/src/coulson/` — router LLM prompt (OVERHEAD).
- `src/workflows/engine/` — artifact→producer index.
- `src/app/telegram_bot.py` — founder-halt card + regenerate/accept callbacks.
