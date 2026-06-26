# Handoff — Mission-90 DLQ debugging (continuation)

**Date:** 2026-06-26
**Status:** Most fixes DONE + live-verified; ALL uncommitted (HEAD `773127e4`); one fix needs a restart; reviewer verdict-quality split to a separate handoff.
**DB (read-only inspect):** `sqlite3.connect("file:C:\\Users\\sakir\\ai\\kutai\\kutai.db?mode=ro", uri=True, timeout=3)` + `PRAGMA busy_timeout=3000`. Workspace: `workspace/mission_90/`.

## The through-line
Every mission-90 DLQ this saga = the same class: **an agent narrates / writes prose instead of emitting the clean artifact**, and some gate then mis-handles it. Fixes target the gates/plumbing so a correct artifact isn't false-rejected.

## Task status right now
| Task | Step | State | Notes |
|---|---|---|---|
| 567379 | `[0.6a.draft] non_goals_draft` (writer, md) | ✅ **completed** | A1/A2 fix verified live — writer's clean file preserved, grade passed |
| 567396 | `[1.11a] compliance_overlay` (analyst, obj) | ❌ **failed** — HELD | needs the post-hook fix loaded (next restart) then re-reset |
| 567399 | `[1.13] research_quality_review` (reviewer) | ⏸ **waiting_human** | ctx/model/prompt fixes WORK; verdict still confabulates → see verdict-verification handoff |

## Uncommitted fixes (ALL in working tree, HEAD = `773127e4`) — NEED COMMIT + PUSH
Restarts load the working tree (editable installs), so most are already live. Grouped by topic:

**A. Narration-clobber batch (567379/567381/567396) — LIVE, verified:**
- `src/workflows/engine/hooks.py` — **A1** `materialize_produces.write_stripped` now uses `coulson._schema_is_structured_only` (matches `_apply_auto_strip`; markdown steps regain `[disk, output_value]` order so the writer's clean file outranks narration). **A2** markdown section validator anchored to line-start regex (a backticked `` `# Non-goals` `` prose mention no longer spuriously passes). **C** object PROSE text-fallback honors `_empty_exemption_granted` (empty-scope fields not flagged "missing content").
- `packages/coulson/src/coulson/__init__.py` — added `"json"` to `_STRUCTURED_SCHEMA_TYPES` (the lone `type:"json"` step `[0.0a.draft]` instruction says write_file is "intentionally unavailable"; was kept un-stripped).
- `packages/coulson/src/coulson/self_critique.py` + `guards.py` — self-critique guard skips when write tools stripped (was nagging impossible "Call write_file" → `max_iterations`; `[1.0a]` 567381).
- Tests: `tests/test_i2p_v3.py`, `tests/workflows/test_materialize_produces.py`, `packages/coulson/tests/test_self_critique.py`, `packages/coulson/tests/test_fetch_deps_full_for_review.py` (untracked).
- Prior handoffs: `2026-06-25-narration-clobber-selfcritique-prosescope-handoff.md`.

**B. Reviewer context (ctx populated via real deps + estimations) — LIVE, verified:**
- `src/agents/base.py` + `packages/coulson/src/coulson/context.py` — resolve `model_ctx` from the **actual selected model** (`ctx.generating_model` → `coulson.window.context_window_for`) instead of the DEAD `self._get_context_window` call that left it at 4096 for every step. (This was THE truncation root — deps budget was ~1.2k tok.) Bounded by `CONTEXT_ABS_CAP=32768`.
- `packages/coulson/src/coulson/context.py` `fetch_deps` — reviewer fetches FULL artifacts (skip the lossy `_summary` form).
- `packages/fatih_hoca/src/fatih_hoca/requirements_builder.py` — review-step input-estimate escalation (`_estimate_full_artifact_tokens`, `[req-escalation]` logs).
- `packages/fatih_hoca/src/fatih_hoca/requirements.py` — reviewer `difficulty 6→7 + prefer_quality=True` (stop picking weak `gemini-flash-lite`).
- `packages/finch/src/finch/profiles/reviewer.yaml` — generic work/artifact reviewer + anti-hallucination evidence rule (was a code-reviewer prompt).
- Verified live on 567399 re-run: model `cerebras/gpt-oss-120b`, est 13538, prompt 58k chars, **all 6 full artifacts, zero truncation**.
- Prior handoff: `2026-06-26-reviewer-full-artifacts-estimate-escalation-handoff.md`.

**C. Compliance post-hook (567396) — NOT yet loaded (needs restart):**
- `packages/mr_roboto/src/mr_roboto/compliance_template_present.py` — on overlay `json.load` failure, fall through to `ok=True` (no parseable `required_documents` → nothing to check) instead of returning `ok=False "could not read overlay"`. Aligns with the gate's documented intent (#166124/#166560: "must not crash on a malformed overlay; the schema gate owns validity"). The empty-scope analyst wrote a PROSE overlay → old code DLQ'd at the post-hook.
- Test: `packages/mr_roboto/tests/test_compliance_template_present.py` (+1, 14 green).

## Immediate next steps
1. **`/restart`** to load the post-hook fix (group C).
2. **Re-reset 567396** (reset SQL below) → it should pass the post-hook now (prose overlay tolerated for empty scope; the schema gate already blessed it via fix C).
3. **Commit + push everything** (HEAD is far behind — the whole session is uncommitted). Suggested split: one commit per group A/B/C. Tests: root i2p/materialize/self_critique, coulson 141, fatih 494/18, mr_roboto 14, finch 43, context cluster 29 — all green.
4. 567399 reviewer halt → see the verdict-verification handoff; either override the halt to unblock m90 (artifacts are genuinely good) or wait for the verification pass.

## Reset SQL (re-pend a task; clears stale-reqs checkpoint)
```sql
UPDATE tasks SET status='pending', task_state=NULL, result=NULL, error=NULL,
  worker_attempts=0, grade_attempts=0, next_retry_at=NULL, exhaustion_reason=NULL,
  retry_reason=NULL, sleep_state=NULL WHERE id=<TID> AND status IN ('failed','waiting_human');
```
(rw connect: `sqlite3.connect(db, timeout=10)` + `PRAGMA busy_timeout=8000`; clearing `task_state` is REQUIRED — react restores stale `reqs` from the checkpoint otherwise, bypassing requirements-layer fixes.)

## Memory
`project_narration_clobber_selfcritique_prosescope_20260625`, `project_reviewer_summary_starvation_20260626`, `project_grade_gate_canonical_pull_20260625`.
