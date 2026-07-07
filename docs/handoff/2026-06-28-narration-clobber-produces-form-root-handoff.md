# Handoff — Narration-clobber TRUE root fixed (produces-form vs schema-type), 2026-06-28

> **UPDATE (2026-07, sub-validated full set — HEAD `197f7130`, origin ahead 7, push HELD).**
> Live re-runs showed `734b1c96`+`6871f9aa` fix the narration but not completion. An adversarial sub-investigator confirmed the mechanism and corrected the plan. **Full validated set now committed (`197f7130`):**
> 1. `schema→markdown` for the AUTHORING steps 5.0c/5.0d/6.5z (i2p_v3.json) + `tools_hint:[write_file]` + an explicit frontmatter/```mermaid template. **4.14 stays array** — it RETURNS ADR JSON and the engine rebuilds `register.md` mechanically. Guard test `test_v3_md_produces_steps_use_markdown_schema` (allowlist 4.14).
> 2. `reset_workflow_step` now clears `_rejection_ledger`/`_schema_error`/`_prev_output` (stale `out_hash` was instant-DLQ'ing regens as "degenerate repeat").
> 3. `produces_markdown` plumbed through the GRADE-path schema gate (`schema_gate.py`+`apply.py`) — producer gate had it, grade gate didn't.
> 4. The `6871f9aa` invariant refined: it does NOT restore write_file for a structured-return `.md` step (4.14 pattern).
>
> **Correction:** `schema→markdown` only relocates the reject to `verify_*_shape`; the remaining live blocker on 5.0c is analyst NON-COMPLIANCE (erratic `gemma-4-26b` from a fleet-exhausted pool writing markdown that lacks `surfaces:` frontmatter / unfenced mermaid). Proven earlier: given write_file the analyst DID write a clean compliant doc. Completion now needs the markdown schema (no JSON confusion) + a capable model.
> **Validate on a FRESH mission** (avoids 567452's stale-ledger + gets capable-model runs), then push (ahead 7). Regression: coulson+mr 63 / beckman 37 / root 125, zero fail.
> Full detail: memory `project_narration_clobber_produces_form_root_20260628` (UPDATE section).

**HEAD:** `734b1c96` (local main). Origin **2 behind** — `734b1c96` (this fix) + `3dd5f54d` (parallel session's advisory-COMPLETE). **Nothing pushed for this fix** (restart-gated + hold per user).
**Status:** The last un-closed root of the narration-clobber class is fixed at the SOURCE. TDD + regression green. Restart-gated (editable install → `/restart` loads it).

## What was wrong (the real root, after dozens of band-aid sessions)
Write-tool stripping (`coulson._apply_auto_strip` + `hooks.py materialize_produces.write_stripped`) decided "result IS the artifact, strip write_file" off the schema **`type`** (`_schema_is_structured_only`). But a step's **`.md` produces extension** is the authoritative artifact-form signal, and it legitimately diverges from schema type: an analyst step producing a `.md` doc while carrying an **object/array** schema (to validate structured frontmatter like `surfaces`/`mermaid_per_surface`) was mis-classed structured → `write_file` stripped → the narration-prone analyst was forced down the `final_answer` path → wrapped the doc in `## Analysis/Summary/Findings` → `materialize_produces` wrote that narration to disk → `verify_*_shape` failed on the missing top-level frontmatter → identical re-emit → **degenerate-repeat DLQ**.

Every prior fix keyed off schema `type` and so missed this: `#524995` (markdown/string SCHEMAS keep write tools), `6feaeb33` (writer disk file outranks narration — but a write-stripped step has NO disk file), `df0106c8`/`88c30d42` (downstream grade-skip / schema-gate defenses).

## The fix (`734b1c96`)
New shared predicate `coulson._write_tools_redundant(schema, produces)`: strip write tools **only** when the schema is structured-only **AND** no produces path ends in `.md`. A `.md` produces keeps write tools regardless of schema type → the analyst authors a CLEAN file → the disk-first materializer picks it → narration is harmless. Used by BOTH `_apply_auto_strip` and `materialize_produces.write_stripped` so they never drift.

Regression guards intact: `.json` produces + object schema (intake_todo_draft) still strips; object schema with NO produces (reviewer verdict) still strips.

**Closes the class for all 4 affected analyst steps** (i2p_v3 scan `.md` produces ∩ structured schema = exactly these): `5.0c` user_flow, `5.0d` screen_inventory/shared_shell, `4.14` register, `6.5z` premortem.

## Immediate next steps (yours)
1. **`/restart`** via Telegram (loads `734b1c96` + parallel session's `3dd5f54d`).
2. Re-reset **only** `567452 [5.0c]` (reset SQL below). The other 3 affected steps are still `pending` (`567441` 4.14, `567453` 5.0d, `567467` 6.5z) → they run fresh with the fix and never clobber.
3. **Verify:** after 5.0c runs, inspect the disk `user_flow.md` — the FIRST line must be `---` (frontmatter), NOT `## Analysis`. (Artifact is in the blackboard / workspace, not `C:\Users\sakir\ai\kutai\workspace` — that path was empty; produces land under the configured WORKSPACE_DIR.)
4. **Push** `734b1c96` once 5.0c verified clean. (Pushing also carries the parallel session's `3dd5f54d` — confirm that's intended/verified before pushing, or coordinate.)

## Separate m90 failure — `567426` [reviewer], NOT this class, NOT a code bug
All 5 capable models were in `failed_models` (transient rate-limit) → it fell to `gpt-oss-20b:free`, which degenerated: **96% of its 6000-char output was literal `!`**. Schema reject is CORRECT. The good review sitting in `tasks.result` is a STALE earlier attempt stored as `str(dict)`. **Fix = operational re-reset** (below) when the fleet is healthy so a capable model does the review. If it re-degenerates because `failed_models` persists in context, clear that too (ping me — needs a context-JSON edit, not in the reset SQL).

## Reset SQL (re-pend; clears the stale-reqs checkpoint)
```sql
UPDATE tasks SET status='pending', task_state=NULL, result=NULL, error=NULL,
  worker_attempts=0, grade_attempts=0, next_retry_at=NULL, exhaustion_reason=NULL,
  retry_reason=NULL, sleep_state=NULL WHERE id IN (567452, 567426) AND status IN ('failed','waiting_human');
```
(rw connect: `sqlite3.connect(db, timeout=10)` + `PRAGMA busy_timeout=8000`. Clearing `task_state` is REQUIRED — react restores stale `reqs` from the checkpoint otherwise.)

## Tests (per-file — mixing package test dirs in one invocation → conftest collision)
```
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider packages/coulson/tests/test_auto_strip_schema_scope.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/workflows/test_materialize_produces.py
```
Regression run this session (all green): coulson 57 · root hooks/dialect/grounding/produces 109 · materialize 12 · beckman grade 9 · schema-gate 7.

## ⚠️ Environment hazard
Dozens of **zombie pytest** processes (`-p no:cacheprovider` + `--deselect …`) + `dbg.py` from a **parallel live session** hold the SQLite lock → a full-suite `packages/coulson/tests` run DEADLOCKED. Workaround used: killed only my own hung run (PIDs 144408/113364), ran targeted PURE-PYTHON files foreground with `timeout` (no DB lock → no deadlock). Do NOT mass-kill — the parallel session is active (its commit `3dd5f54d` landed mid-session). Consider `token-optimizer:health` to clean safely once that session is done.
