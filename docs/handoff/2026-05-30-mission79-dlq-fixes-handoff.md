# Handoff — mission_79 DLQ-storm fixes + open items

**Date:** 2026-05-30 (evening)
**Session:** SP4-kickoff turned into a live-debugging run against the test mission (mission_79, "HabitFlow"). 13 fix/test commits on `main`. **KutAI has NOT been restarted with any of them** — restart via Telegram to load.
**Branch:** `main` (committed directly per repo convention). HEAD = `90698212`.

---

## TL;DR

A test mission (`mission_79`) was run, hit a cascade of DLQs across multiple restarts. Each DLQ was triaged to root cause and fixed. Two themes:

1. **Artifact plumbing** — structured-emit envelopes, workspace paths, stale files, malformed shapes were tripping validators even though content was fine.
2. **The founder principle** — *"if capacity is not available, tasks should not be admitted / should WAIT, not DLQ."* Several availability (cloud daily-exhausted) failures were fast-DLQ'ing instead of backing off. Fixed across classification + all 3 cap-enforcement sites.

mission_79 itself is now a heavily-poisoned zombie (many hand-recovered tasks, stale artifacts). **Recommend a FRESH mission for the next e2e validation**, not another recovery of #79.

---

## Commits this session (oldest→newest, all `main`, all 2026-05-30)

| Hash | What | Root cause |
|------|------|-----------|
| `f216ca49` | coulson: self_critique verdict no longer clobbers graded artifact | react loop took the `{"verdict":"clean"}` self-critique reply as the task result, overwriting the real artifact → grade failed → DLQ |
| `3e6f86c0` | workflows: unwrap structured_emit CallResult envelope before schema validation | array/object artifacts persisted as raw `{"content":"<stringified>","model":...}` envelope; `_unwrap_envelope` didn't peel it → validators counted ~0 items / missing fields |
| `9a8d09aa` | mr_roboto: resolve workspace-relative report_path in prior_art_min_coverage | validator used `report_path` verbatim; cwd=repo root ≠ WORKSPACE_DIR → "no report payload"; + str `search_summary` AttributeError |
| `c68bb2ca` + `b0c0a8f2` + `95017486` | workflows: overwrite stale junk produces artifact (+ 2 fixture fixes) | "only fill missing" persist guard locked in a truncated final_answer envelope at `.intake/interview_script.md` forever |
| `f81d33bd` | mr_roboto: tolerate non-dict attempted_solutions | model emitted `attempted_solutions: ["Habitica: ...", ...]` (bare strings) → `sol.get()` crashed |
| `f2bf9271` | beckman: availability failures classified by error-text sniff | `_apply_failed` fell back to stale `quality` category → fast-DLQ; `_classify_availability_text` overrides |
| `a01c4137` + `e00c27fe` | beckman: transient categories ride full backoff ladder (+ test fix) | `max_worker_attempts=6` bottomed the ladder at ~9min; transient now rides to 24h (past quota reset) |
| `ebed13fb` | telegram: Regenerate button resolves step id from task context | step id read from ephemeral `_pending_action` (clobbered) → "Regenerating step `?`" + re-emitted same artifact |
| `301546d4` | beckman: surface auto-fail grade message instead of "grader verdict unavailable" | auto-fail verdict `{"passed":False,"raw":"auto-fail: grader call failed (...)"}` — `_grader_verdict_text` never read `raw` |
| `90698212` | beckman: admission cap-guard + sweep honor the transient retry ladder | `#225600` cat=availability still DLQ'd "6/6" — admission/sweep used raw cap, ignoring decide_retry's ladder extension; unified via `effective_max_attempts()` |

Memory files written (in `…/memory/`, indexed in MEMORY.md):
`project_self_critique_clobber_20260530`, `project_schema_validation_dlq_20260530`, `project_mission79_cluster3_20260530`, `project_regen_button_fix_20260530`, `project_availability_dlq_fix_20260530`, `project_grader_verdict_autofail_20260530`, `project_transient_cap_three_sites_20260530`.

---

## ⛔ OPEN / not done — read before continuing

### 1. Grade-reject branch ignores availability (PART 2 of grader-autofail) — HIGHEST
`packages/general_beckman/src/general_beckman/apply.py::_apply_posthook_verdict_locked` grade-reject branch (~4348–4470) hardcodes `category="quality"` and `_dlq_write(category="quality")` at `attempts>=max`. So an **availability-caused GRADE-CHILD failure** (grade child can't get a model → auto-fail) still gets quality fast-DLQ there — the worker-side fixes (`f2bf9271`/`a01c4137`/`90698212`) do NOT cover this branch.
**Fix shape:** at top of the grade-reject branch, if `_classify_availability_text(error_str)` → availability, re-enqueue the grade child with backoff (or set source pending w/ availability category + next_retry_at) WITHOUT counting a quality attempt. **DELICATE** — branch runs under `_source_verdict_guard` (per-source lock) + manages `_pending_posthooks`/`grade_excluded_models`/`_bonus_count`. Needs careful TDD. (301546d4 only made the *reason text* honest; the *routing* is still quality.)

### 2. `accelerate_retries` / `capacity_restored` early-wake — VERIFY then maybe build
retry.py comments claim deferred tasks wake early when capacity returns. The wiring EXISTS (`kdv.py` fires `capacity_restored` → `router.py` → `schedule_accelerate_retries` → `db.accelerate_retries`). But NOT verified end-to-end this session. Without it, an availability task waits the FULL ladder step (up to 24h) even if capacity frees sooner. Verify it actually wakes deferred rows; if not, that's the highest-leverage availability improvement.

### 3. Genuine model-quality failures remain (NOT bugs)
Several mission_79 DLQs are real: writer outputs under-spec (18 grade children on #225586 all `COMPLETE: NO`), compliance_overlay `required_documents` emitted as a string not a list. These ride retries correctly once capacity returns; they're a model-routing/prompt-quality matter, not code. Don't chase them as bugs.

### 4. interview_script regen is dormant on a VALID artifact (design Q)
Fix C (`c68bb2ca`) only overwrites a JUNK existing file. Regenerating a *valid* artifact may reproduce the same content because the preserve guard (intake #73) keeps it. Real force-regen = the `/regen` primitive (`regen_artifact`, versions the file). **Design question for founder:** should the confirm-card ♻️ delete the existing produces file first so the writer always re-emits?

### 5. Pre-existing test failures (NOT introduced this session — confirmed via baseline)
- `tests/test_grading.py::TestApplyGradeResultPass::test_pass_with_rich_verdict` + `test_pass_empty_verdict_uses_mechanical_fallback` — `add_skill` not called in `apply_grade_result` skill-capture path. Fail at baseline `ebed13fb`. Unrelated; not investigated.
- `packages/mr_roboto/tests/test_reversibility_registry.py::test_every_dispatcher_action_is_in_registry` — `publish_preview_pages` missing from `VERB_REVERSIBILITY`. Fails at baseline `b4d9c05a`. One-line fix available, out of scope.

---

## SP4 status (the original reason for the session)

**Not started.** The SP4 kickoff (`docs/handoff/2026-05-30-sp4-kickoff.md`) prerequisite was "validate the SP3b post-hook substrate end-to-end with one real graded mission." Doing that surfaced all the bugs above. Substrate findings:
- Post-hooks DO dispatch and run (self_critique, grade, grounding children all complete) — the kickoff's fear that they "never dispatched" was wrong.
- The break was the ReAct self_critique loop (f216ca49) + artifact plumbing, not CPS.
- husam is installed. The grade-child availability handling (open item #1) is the remaining substrate gap before SP4.

**Before SP4:** restart, run ONE FRESH graded mission (not mission_79), confirm a coder/writer step with `enable_self_reflection` + `artifact_schema` completes clean through emit→reflect→grade with the new fixes. Then resume the SP4 plan.

---

## Environment notes / gotchas hit this session
- Live DB: `C:\Users\sakir\ai\kutai\kutai.db` (NOT `./data/kutai.db`). `tasks` keyed by integer `id` (no `task_id` column). Probe read-only: `scripts/_probe_task.py <id>`.
- WORKSPACE_DIR = repo checkout `…kutay\workspace` (NOT `ai/kutai/workspace`).
- The session's tool channel was flaky (frequent blank tool returns + a few cascade-cancelled parallel batches). Mitigation that worked: **write output to a file, then Read it** — avoids the rtk/console truncation + blank-return issue. Run pytest with `-p no:warnings` and grep the summary.
- venv python: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`.
- A pre-existing stash (`z8 stray work — pre-T3-merge`) is present — NOT from this session, left untouched.

---

## Suggested next steps (in order)
1. Restart KutAI via Telegram → load all 13 commits.
2. Run a FRESH graded mission (not #79). Watch for: clean grade pass, availability tasks backing off (not DLQ'ing), regen button working.
3. Tackle open item #1 (grade-reject availability routing) — the last availability gap.
4. Verify open item #2 (`accelerate_retries` wakes deferred rows).
5. Resume SP4.
