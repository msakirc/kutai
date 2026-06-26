# Handoff — DLQ-noise fixes shipped + the verify-step→post-hook refactor (scoped, not started)

**Date:** 2026-06-01
**Predecessor:** `docs/handoff/2026-05-31-mission-debug-continuation.md`
**HEAD after this session:** `88735dec` (+ the two i2p commits `9050a899`, `3ce122bf` — confirm with `git log --oneline -8`).
**Live build:** founder restarted ~3 times during the session. The grade-availability fix (`bfdffea8`) was live mid-session; the rest need the NEXT restart to load (they were committed after the last restart).

---

## 1. What shipped this session (6 commits, all green, all TDD)

| Commit | Fix | Tests |
|--------|-----|-------|
| `bfdffea8` | **grade auto-fail → availability backoff.** `apply._grade_verdict_is_availability` (auto-fail SHAPE `{raw:"auto-fail: grader call failed (...)"}` + availability marker, shape-guarded so a grader insight mentioning "daily"/"quota" never false-positives) → routes through `_retry_or_dlq(category="availability")` instead of hardcoded quality DLQ. | `test_grade_autofail_availability_backoff.py` (5) |
| `7958475e` | **sweep overcap honors `effective_max_attempts`.** Sweep section 8 (`sweep.py:398`) compared `worker_attempts >= raw max_worker_attempts` in SQL — force-DLQ'd availability tasks waiting at 6/6. The "3-site unification" (`e4f9a1c2`) wired decide_retry + admission but MISSED this SQL (only unit-tested the pure fn). Now applies effective cap in the Python loop. | `test_sweep_overcap_honors_effective_cap.py` (2, real-DB) |
| `eb8187da` | **rewrite Rule 0c: mechanical check-fail → terminal.** A validator that RAN its check and returned `{ok:False}` was kept as `Failed` → self-retried 5× to "Worker attempts exceeded" DLQ. Now converts the validator's own action `Failed→Complete` (terminal) while the producer verdict stays byte-identical. | `test_rewrite_mechanical_checkfail_terminal.py` (3) |
| `88735dec` | **orchestrator carries verdict result on failed path.** `_mech_action_to_result` (`src/core/orchestrator.py:72`) DROPPED `action.result` for `status=failed`, so `eb8187da` NEVER FIRED on the real path (its unit test passed only because it synthesised the result the orchestrator strips — verify-roundtrip trap on my own fix). Now carries `json.dumps(action.result)`; empty default `{}` has no verdict keys so executor-error path stays retryable. **`eb8187da` + `88735dec` are a PAIR — neither works alone.** | `test_mech_failed_result_chain.py` (3, real chain) |
| `9050a899` | **i2p `direct_competitors_list` min_items 1→3.** #225586: researcher returned 1 competitor; min_items=1 let the cheap gate pass it, only the LLM grader caught it (COMPLETE:NO) then DLQ'd. Workflow already assumes ≥3 (step 1.4 `length(competitors) >= 3`). | `test_i2p_competitor_floor.py` (3) |
| `3ce122bf` | **i2p `research_review_result.verdict` enum.** #225600: schema `equals:["pass"]` rejected the instructed `needs_minor_fixes` → EVERY non-pass review DLQ'd on schema (reviewer couldn't report problems). Now `["pass","fail","needs_minor_fixes"]`. | `test_i2p_review_verdict_enum.py` (3) |

Regression: 216 beckman + 56 failed-path-consumer tests pass. `tests/` and `packages/general_beckman/tests/` CANNOT be run in one pytest invocation (conftest name collision) — run them separately.

---

## 2. The mission_79 triage (the 4 recurring DLQs) — fully root-caused

mission_79 is **poisoned** (predecessor handoff said so). The founder bulk-`/dlq retry`'d the 4 rows; they re-failed. Root causes:

- **#225600** `research_quality_review` — two bugs, both fixed: sweep force-DLQ at 6/6 (`7958475e`) + verdict-enum schema (`3ce122bf`).
- **#225586** `direct_competitor_identification` — researcher found only 1 competitor (genuine thin output); schema floor raised (`9050a899`). The grader was RIGHT.
- **#225576** `interview_script_shape_check` — see §3 (the dead-end). Producer 225575 (analyst) emitted **5KB of prose narration** instead of a structured script → genuine producer-quality. The validator-terminal pair (`eb8187da`+`88735dec`) stops the *validator* DLQ noise but does NOT fix the analyst.
- **#227677** `prior_art_min_coverage` — producer 225583 is **dead** (`failed` 3/6); it emitted bare-string `attempted_solutions`. This IS a post-hook (`source_task_id` set), so it DLQ'd the producer correctly (Z1 blocker semantics — see §3). Reviving it needs the producer re-run.

**The mission needs a FRESH run to validate the fixes** — retrying poisoned #79 hits dead producers + cloud exhaustion (was rate-limited at 17:01). Don't keep whacking #79.

---

## 3. THE BIG OPEN ITEM — verify steps that dead-end (the founder's question)

**Founder's question:** "if the validator doesn't validate the actual task, why doesn't it fail the actual task?" — correct instinct.

**Two shapes of output-validation exist:**
1. **Post-hook** (`source_task_id` + `posthook_kind` in ctx): on fail, routes a `PostHookVerdict` back to the producer → producer fails/re-pends. THIS is the right mechanism. (e.g. `prior_art_min_coverage` on step 1.0.)
2. **Standalone `.verify` workflow STEP** (`is_workflow_step=True`, only `depends_on` the producer, NO `source_task_id`): on fail it fails ITSELF → DLQ → blocks the phase, while the **producer stays `completed` and is never re-run.** Dead-end. (e.g. `0.0c.verify` interview — #225576.)

**Scale of the dead-end:** `i2p_v3.json` has **~39 standalone `.verify` steps**. Only **`verify_falsification_present` (4 steps)** is also registered as a post-hook (correct; standalone step is redundant belt-and-suspenders — see comment `posthooks.py:451`). The other **~31 steps across 19 verb-kinds have NO post-hook sibling → all dead-ends.** The 19 orphan verbs:

```
verify_adr_shape (8 steps), verify_cost_curve_present (6), verify_html_prototype_shape (2),
verify_screen_plan_shape (2), vendor_call (2), verify_interview_script_shape, verify_charter_shape,
verify_reverse_pitch_shape, verify_non_goals_shape, verify_competitive_positioning_shape,
verify_premortem_shape, verify_design_tokens_shape, verify_surfaces_shape, verify_user_flow_shape,
verify_screen_inventory_shape, verify_shared_shell_shape, verify_screen_consistency, verify_adr_register
(1 each)
```
Audit command (paste into a python -c against i2p_v3.json): registered verbs = `re.findall(r'verb="([a-z_]+)"', posthooks.py)`; standalone verify steps = mechanical steps whose id ends `.verify`/name ends `_check`. Diff them.

**THE FIX = convert each dead-end verify STEP to a post-hook on its PRODUCER.** The mechanism is already built. Per-verb recipe (5 sites), mirror `verify_falsification_present`:
1. `packages/general_beckman/src/general_beckman/posthooks.py` — add a `PostHookSpec(kind, verb, default_severity=...)`.
2. `apply.py::_posthook_agent_and_payload` (~line 2146+) — add a payload-builder branch. **BESPOKE per verb** — each reads different producer inputs (interview needs `script_paths` from `produces` + min/max_questions; adr_shape needs the ADR artifact; cost_curve needs its inputs).
3. `apply.py` verdict routing — pick the handler (see below).
4. `apply.py::_posthook_title` (~line 3940) — add a title branch.
5. `i2p_v3.json` — add `post_hooks:[verb]` on the producer step; **delete the standalone `.verify` step**; rewire any step whose `depends_on` named the deleted `.verify` to name the producer instead. (For interview: only `0.0c.request` depends on `0.0c.verify` → repoint to `0.0c`.)

**CRITICAL design choice per verb — two post-hook fail-semantics:**
- `_apply_z1_mechanical_verdict` (`apply.py:3798`, kinds in `_Z1_BLOCKER_KINDS` `apply.py:3781`): blocker fail → **DLQ the producer, NO retry** (docstring: deterministic check, retry re-emits same artifact, founder must intervene).
- `_apply_simple_blocker_verdict` (`apply.py:3957`, kinds in the tuple at `apply.py:4564`): fail → **re-pend producer WITH feedback + retry to cap.**

The founder wants *fail-and-re-run-with-feedback* → use **`_apply_simple_blocker_verdict`** for shape checks a producer can plausibly fix on retry (interview: "emit structured Questions, not prose"). Use the Z1 DLQ path only where retry genuinely can't help. **This is a judgment call per verb — the refactor is design-laden, not rote.**

Note: `_apply_simple_blocker_verdict`'s feedback reads `raw.get("findings")`; the verify verbs return `{ok, problems/question_problems}` — map `problems→findings` (in the payload or a small adapter) or the producer's retry feedback will be generic.

**Why it must be SEQUENTIAL, not parallel subagents:** all 19 verbs edit the SAME three files (`apply.py`, `posthooks.py`, `i2p_v3.json`). Parallel worktrees would collide ([[feedback_canonical_first_for_tier3plus]]). One verb → test → next.

**Suggested batching:** phase-0/1 blockers first (5 verbs: interview_script, charter, reverse_pitch, non_goals, competitive_positioning), validate against a fresh mission, then the rest. `verify_falsification_present` is the worked reference at all 5 sites.

---

## 4. Lessons (don't repeat my mistakes)

- **Verify the whole round-trip, not a synthetic shape.** `eb8187da` was inert for an entire restart because I unit-tested rewrite with a hand-built `raw` carrying `result`, but the orchestrator strips `result` on the failed path. Always test through the REAL producer of the data ([[feedback_verify_verdict_roundtrip]]). `test_mech_failed_result_chain.py` now runs the real `_mech_action_to_result→route_result→rewrite` chain.
- **Live vs stale is the trap.** I repeatedly claimed fixes were/weren't live and was wrong (restart loads `.py` from disk, not git; a bulk-retry re-runs rows; `mode=ro` DB reads can lag WAL). Check `started_at`/`completed_at` timestamps against restart time before claiming anything.
- **DLQ notices on restart re-surface the backlog** (dlq-blocks-phases) — not necessarily fresh failures. Confirm with timestamps.
- The "3-site unification" memory claimed sweep was covered; it wasn't (only the pure fn was tested). **A registry/shape test passing ≠ the wiring is live.**

---

## 5. Environment / gotchas (carried, still true)

- Live DB: `C:\Users\sakir\ai\kutai\kutai.db` (read-only probe: `scripts/_probe_task.py <id>`; or `sqlite3 'file:...?mode=ro'`).
- venv python: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`. pytest with `-p no:warnings` + a timeout, ALWAYS.
- `tests/` and `packages/*/tests/` collide on conftest — run in SEPARATE pytest invocations.
- Each DB-integration test calls `init_db()` which loads the embedding model (~19s) — suites with several are slow (~50-90s), not hung. Don't kill them; the heavy long-running python procs are KutAI/llama.
- Restart KutAI via Telegram, never taskkill llama-server.

---

## 6. Open / deferred (besides §3)

- **Latent siblings of `3ce122bf`** (flagged, NOT fixed): `research_review_result.issues=[]` trips the empty-placeholder check (a clean pass with zero issues would DLQ); `go_no_go` step 1.14 `recommendation` enum `["Go","go"]` cannot express No-Go. The other 11 `equals` fields are intentional pass/approved GATES — do NOT loosen.
- **Producer-quality residue** (not code-fixable): weak local models narrate instead of emitting artifacts (analyst 0.0c), emit bare strings (prior_art). The §3 refactor routes these to the right task with feedback, but a model that can't produce the shape will still ride retries → DLQ on the PRODUCER. That's correct behavior, not a bug.
- Predecessor's still-open items (grade-reject availability §1 was THIS session's `bfdffea8`; `accelerate_retries`/`capacity_restored` e2e still unverified).
- The uncommitted VRAM-aware ctx-floor stopgap + the load-mode redesign (`2026-05-31-load-mode-redesign-ideas.md`) are untouched, still pending the founder's "revert in redesign" call.

---

## 7. Suggested first move next session

1. `git log --oneline -8` to confirm the 6 commits landed; restart KutAI if not already (loads `7958475e`/`eb8187da`/`88735dec`/`9050a899`/`3ce122bf`).
2. Read `verify_falsification_present` at all 5 sites (grep it in `apply.py`, `posthooks.py`, `i2p_v3.json`) — that's the reference implementation.
3. Convert `verify_interview_script_shape` (smallest rewire: 1 dependent) through `_apply_simple_blocker_verdict`, TDD'd with a real-chain test, commit. That's the template for the other 18.
4. Then batch the phase-0/1 blockers; validate with a FRESH mission, not poisoned #79.
