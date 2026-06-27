# Handoff — Mission-90 DLQ debugging (continuation, 2026-06-27)

**Date:** 2026-06-27
**HEAD:** `fcc8cbf8`. Origin is 13 commits behind — **nothing pushed this session.**
**Status:** Three m90 DLQs fixed (5 commits), TDD'd, two Opus review passes, all green. **Restart-gated** (editable installs → `/restart` loads them). The DLQ'd task rows are frozen `failed` — they must be re-reset AFTER restart to clear.
**DB inspect (read-only):** `sqlite3.connect(r"file:C:\Users\sakir\ai\kutai\kutai.db?mode=ro", uri=True, timeout=3)` + `PRAGMA busy_timeout=3000`. Workspace artifacts under `workspace/mission_90/`.

## The through-line (whole saga)
Every m90 DLQ = a VALID / near-valid artifact false-rejected by a gate that is mismatched to the artifact's form OR mis-handles a failure. Fixes target the gates, never weaken a real check. Two of my own first-pass diagnoses this saga were WRONG (shallow log reads); an Opus sub corrected them — **trust disk artifacts + rejection ledgers over log lines.**

## This session's fixes (all local `main`, NOT pushed)
| Commit | Fix | DLQ |
|---|---|---|
| `bcfa90b1` | `verify_falsification_present` → `_PRODUCER_QUALITY_Z1_BLOCKERS` (producer re-pend, not single-shot DLQ at wa=1); wiring surfaces JSON parse error | 567413 [3.1] |
| `be5757bb` | (review) actionable feedback moved to `res["problems"]` — `res["error"]` was clobbered by `rewrite.py:269` `{**res,"error":Action.error}` | 567413 |
| `88c30d42` | `validate_artifact_schema(produces_markdown=…)` skips the object/array prose field-NAME text-fallback on `.md` produces; defers to `verify_*_shape` | 567452 [5.0c] |
| `df0106c8` | grade gate runs the step's verify check INLINE; PASS auto-skips the confab-prone LLM grade (kills confab→degenerate-DLQ loop) | 567449 [5.0a] |
| `fcc8cbf8` | (review) grade-skip authority = `_is_grade_authoritative_check` registry (incl `verify_adr_register`), not a `_shape` suffix; + markdown-step invariant test | 567449 / 4.14 |

Parallel session also committed `ece7bdc8` (reviewer Tier-1 grounding) + `283e2350` (verify reviewer verdicts) + `8e96c4a8` (clarify/telegram) — not mine, interleaved, intact.

## Immediate next steps
1. **`/restart`** via Telegram (loads all 5 + the parallel-session work).
2. **Re-reset the three DLQ'd tasks** (reset SQL below) → they should advance:
   - **567413 [3.1]** — model's JSON had a corrupt seam; it now re-pends with "re-emit valid JSON" feedback instead of single-shot DLQ. Expect it to regenerate valid JSON and pass.
   - **567449 [5.0a]** — the on-disk `design_tokens.json` is already shape-valid (`verify_design_tokens_shape` → ok); the inline-verify short-circuit should auto-PASS the grade. Expect immediate completion.
   - **567452 [5.0c]** — ⚠️ **see caveat below.** The schema-gate false-reject leg is fixed, but the OPERATIVE root is narration-clobber.
3. **Push** everything once live-verified (origin 13 behind).

## ⚠️ 567452 caveat — narration-clobber is the real root, NOT yet confirmed fixed
The on-disk `workspace/mission_90/.flow/user_flow.md` (3999c) is the analyst's **"## Analysis / Summary / Findings" narration**, with the real flow doc (proper `surfaces:` frontmatter + mermaid) buried inside a ```yaml fence. So both gates legitimately complained:
- The grade object text-fallback searched the literal `mermaid_per_surface` (Fix `88c30d42` removes this false-reject leg — good), AND
- `verify_user_flow_shape` fails on missing top-level `surfaces:` frontmatter (correctly — the clean doc is fenced inside narration).

Fix `88c30d42` stops the schema-gate false-reject, but on a fresh run the analyst must **write a CLEAN `user_flow.md`** (frontmatter at top, not wrapped in an Analysis report). That is the narration-clobber class — targeted by `6feaeb33` (materialize/writer, committed earlier). After restart, **re-run 5.0c and inspect the disk file**: if the top is still `## Analysis` narration with a fenced doc, the writer/analyst-prompt clobber is NOT solved for this step and needs a separate fix (analyst emits the artifact body directly, not a narration wrapper; or the produces-markdown materialize pulls the fenced block). Do NOT relax `verify_user_flow_shape` to accept the narration — that would be a band-aid (the frontmatter check is the one thing catching the clobber).

## KNOWN TRADEOFF (separate handoff)
Fix `df0106c8` auto-PASSes the grade (skips the LLM grade) on a shape-verify PASS for **24 steps** → drops the LLM topicality/relevance check (a shape-valid but off-topic artifact auto-passes). Deliberate, sub-validated. Full analysis + the "advisory-COMPLETE" refinement option: **`docs/handoff/2026-06-27-grade-skip-llm-relevance-tradeoff-handoff.md`**.

## Reset SQL (re-pend a task; clears the stale-reqs checkpoint)
```sql
UPDATE tasks SET status='pending', task_state=NULL, result=NULL, error=NULL,
  worker_attempts=0, grade_attempts=0, next_retry_at=NULL, exhaustion_reason=NULL,
  retry_reason=NULL, sleep_state=NULL WHERE id=<TID> AND status IN ('failed','waiting_human');
```
(rw connect: `sqlite3.connect(db, timeout=10)` + `PRAGMA busy_timeout=8000`; clearing `task_state` is REQUIRED — react restores stale `reqs` from the checkpoint otherwise, bypassing the requirements-layer fixes.)

## Tests / how to verify the fixes (per-file — mixing two package test dirs in one pytest invocation triggers a conftest collision)
```
.venv/Scripts/python.exe -m pytest -q packages/general_beckman/tests/test_falsification_repend.py packages/general_beckman/tests/test_grade_verify_authority.py
.venv/Scripts/python.exe -m pytest -q packages/mr_roboto/tests/test_falsification_feedback.py
.venv/Scripts/python.exe -m pytest -q tests/test_schema_gate_markdown_produces.py
```
All green (beckman 18 / mr_roboto 13 / root 12 in the consolidated run).

## Known pre-existing failure (NOT mine, NOT a regression)
`tests/test_i2p_v3.py::TestV3WorkflowLoading::test_v3_all_steps_have_artifact_schema` fails: step `[13.demo_storyboard_draft]` (reviewer) has no `artifact_schema`. Pre-exists at session-start HEAD (proven via stash). A data gap in `i2p_v3.json` — either add a schema or exempt reviewer steps in the test. Unrelated to the DLQ work.

## Housekeeping
Working tree has scratch junk NOT to commit: `.claire/`, `_*.txt`, `bash.exe.stackdump`, `temp*.txt`, `count_sections.ps1`, `search_sp6.py`, plus auto-generated `egg-info/SOURCES.txt` churn and `.claude/settings.local.json`. Gitignore or delete; do NOT sweep into `main`.

## Memory
`project_m90_three_gate_fixes_20260627` (this session), `project_grade_gate_canonical_pull_20260625`, `project_narration_clobber_selfcritique_prosescope_20260625`.
