# Handoff — requirement-conservation gate + m90 DLQs (2026-07-22, evening)

## TL;DR
Shipped a **mechanical requirement-conservation gate** (root fix for assembly steps silently
compressing requirement lists) + 2 live-reload fixes it needed + an ADR-materialize fix + 3
adversarial-review fixes. All 5 commits **pushed to origin/main**, all TDD'd. The gate is
**fully validated end-to-end live** (it caught a real drop and self-healed). m90 advanced from
the requirements phase to phase 13+ (613 tasks completed). Two DLQs were triaged: one root-fixed
(ADR double-escape, restart-gated), one re-pended with the weak model excluded (per-screen plans).

## Commits (all on origin/main; parallel Yaşar-Usta session interleaves the log)
| SHA | What |
|---|---|
| `97e15519` | **The gate**: `verify_requirement_conservation` (mr_roboto pure checker + dispatch + reversibility), beckman `POST_HOOK_REGISTRY` reg, `checks` on i2p 3.9a/3.10a/3.10b + hardened instructions. |
| `7982e2de` | **coulson**: add `"checks"` to `_CTX_FIELDS` so `checks` live-reload onto in-flight tasks (they froze at expansion → gate was inert on m90). |
| `7883789d` | **coulson**: substitute `{mission_id}` in the live-reloaded step (the live-reload assigned raw checks → `mission_{mission_id}/…` paths didn't resolve → **vacuous pass** masked the gap). |
| `007d00c1` | **hooks**: `_unwrap_envelope` recovers a whole-string **double-escaped JSON** artifact (`{\n \"adr_id\"…}`) at materialize — fixes m90 4.1 ADR DLQ + any consumer of a corrupt-on-disk json artifact. |
| `f697f219` | **3 adversarial-review fixes** (below). |

## Gate mechanics (how it works — don't re-learn)
- Deterministic post-hook, **presence=mechanical / quality=LLM** (sibling of `verify_falsification_present`).
- Pure checker `packages/mr_roboto/src/mr_roboto/verify_requirement_conservation.py`: for each
  `{label, source_text, id_pattern}`, `missing = source_ids − produced_ids` (subset check; extras OK).
  `empty` (no ids in any source) = **vacuous pass** (safe direction — never a false re-pend).
- Declared on a producer via the `checks` field (payload = `produced_paths` + `sources[]`); a drop
  routes through `apply._CHECK_KINDS → _apply_simple_blocker_verdict` → **re-pends the producer** with
  feedback naming the dropped ids (self-heals; NOT a founder halt), bounded by `max_worker_attempts`.
- mr_roboto dispatch reads produced+source artifacts off the mission workspace via `_resolve_path_list`.
- **LIVE PROOF**: post-hook 864428 returned `ok=False, missing=[all 15 FR]` → re-pended 3.10b → 864433
  `ok=True checked=15` → spec 15 FRs. The gate armed → ran non-vacuously → caught a real drop →
  self-healed. No halt.

## ⭐ DO THIS FIRST (restart-gated — the mr_roboto/hooks/coulson CODE needs a fresh orch)
`007d00c1` (ADR fix) + `f697f219` (mr_roboto observability) are CODE. The i2p_v3.json edits reload
live. After `/restart`:

1. **Re-pend m90 4.1 ADR (task 567427)** — the materialize fix now decodes the double-escaped JSON,
   so a re-run writes a clean `.adr/architecture_pattern_decision.json` and `verify_adr_shape` passes.
   Clear the degenerate ledger:
   ```sql
   UPDATE tasks SET status='pending', worker_attempts=0, grade_attempts=0, result=NULL, error=NULL,
     task_state=NULL, sleep_state=NULL,
     context=json_remove(context,'$._rejection_ledger','$._schema_error',
       '$._schema_error_for_attempt','$._prev_output')
   WHERE id=567427;
   -- also resolve its dead_letter row:
   UPDATE dead_letter_tasks SET resolved_at=datetime('now'), resolution='materialize double-escape fixed (007d00c1)'
   WHERE task_id=567427 AND resolved_at IS NULL;
   ```
   Watch: 4.1 should complete; the whole phase-4 architecture chain unblocks.

2. **Re-pend m90 3.10b (task 567425)** — the review fix (`f697f219`) added US+BR conservation. The
   current `requirements_spec.md` is **missing BR-002 and BR-006** (the FR-only gate missed them).
   A re-pend under the new gate will catch the BR drop and self-heal (validates the review fix live):
   ```sql
   UPDATE tasks SET status='pending', worker_attempts=0, result=NULL, error=NULL, task_state=NULL,
     sleep_state=NULL, context=json_remove(context,'$._rejection_ledger','$._schema_error',
       '$._schema_error_for_attempt','$._prev_output')
   WHERE id=567425;
   ```
   Expect a conservation post-hook with `ok=False missing=[BR-002,BR-006]` → re-pend → converge.
   NOTE: 3.11 (review) is `completed` and phase-4+ already consumed the old spec; re-running 3.10b
   fixes the artifact but does NOT re-propagate to already-completed downstream (demo mission — fine).

3. **Check m90 5.20a (task 567454)** — re-pended THIS session with `gemma-4-31b` excluded (it fumbled
   5× `write_file` calls with empty args despite inputs being on the blackboard). If it completed on a
   stronger model → good. If it re-DLQ'd, it's a **model-selection** problem (Fatih Hoca gave a medium
   analyst too weak for multi-file `.screens/*.md` writing) — see residuals.

## Adversarial review (Opus sub-agent) — 3 majors FIXED (`f697f219`), minors DEFERRED
Fixed:
1. **3.10b conserved FR only** → verified the final m90 spec dropped **BR-002/BR-006** undetected.
   Added US+BR rules + instruction mandate.
2. **3.9a/3.10a `depends_on` race**: they conserve `user_stories_refined`(3.8)/`business_rules`(3.7)
   but only depended on 3.1-3.4 → the post-hook could fire before 3.7/3.8 materialize → vacuous pass.
   Added 3.7, 3.8 to both `depends_on`. (Dispatch gates on `depends_on`, NOT `input_artifacts` —
   confirmed `runner.py:227` / `expander.py:498`.)
3. **Silent vacuous pass**: dispatch now sets `res["wiring_suspect"]=True` + logs a WARNING when
   declared paths resolve to zero ids (the exact foot-gun that hid the `{mission_id}` bug).

Deferred (minor/nit — NOT blocking; good first tasks next session):
- **coulson `_CTX_FIELDS` None-skip**: a step that *removes* a check won't clear stale `checks` from
  in-flight tasks (`if _live_val is None: continue`). Fine for the ADD case; latent for pulling a bad
  gate mid-mission.
- **Malformed `id_pattern`** silently disables a rule (`re.error` → `[]` → vacuous). Same observability
  hole; consider folding into the `wiring_suspect` signal.
- **Test patch fragility**: `test_step_refresh_checks.py` patches `src.infra.db.update_task` but the code
  calls `general_beckman.update_task` (works only because beckman re-delegates to dabidabi at call time).

The review's **verified-correct** list (registration/routing, subset semantics, sort, no cross-match,
paths, `{mission_id}` substitution safety, unfiltered-checks safety, no circular import) is solid — trust it.

## Open DLQs (m90) — status
- **567427** (4.1 architecture ADR): ROOT-diagnosed + fixed at materialize (`007d00c1`). Restart-gated
  re-pend (step 1 above). The corrupt `.adr/architecture_pattern_decision.json` on disk is still the old
  double-escaped file until 4.1 re-runs.
- **567454** (5.20a per-screen plans): re-pended with weak model excluded (live). Model-selection root.

## Residuals (deferred, not blocking)
1. **m90 3.1 lacks 3 Medium/Low PRD MVP features** (Evolving Challenges #3, Personalized Habit Bundles #9,
   Progressive Disclosure UI #10) — the requirements review PASSED without them this run (LLM variance),
   so they're not currently blocking. If a future review flags them, re-pend 3.1 (attempt budget: check;
   was 6/8) to add distinct FRs, then cascade 3.8→3.9a→3.10a→3.10b→3.11.
2. **2 non-mechanizable reviewer findings** (high-risk-vagueness validation methods; Rule-B non-goals-
   contradiction grounded against the wrong artifact) — LLM-judgment territory; verdict-verification
   handled them without halting. See `project_reviewer_falsification_presence_mechanical_20260722` residuals.

## Deeper root hunts for next session (with smart help)
- **The double-escape ROOT**: `007d00c1` recovers it at materialize (defense that matches the codebase's
  existing fence-strip / repr defenses), but WHERE the architect result got double-`json.dumps`'d in the
  first place is unfound. The escaped form is `json.dumps(pretty_json)` — a pretty artifact escaped once
  too many. Trace the object-schema producer → `constrained_emit` (rewrite.py) → result-store path.
  `traceability_matrix` (object, written `.md`) is clean; only the `.json`-written artifact corrupted —
  so suspect the JSON write/serialize seam specifically. This is the "canonical seam" cluster
  (`project_grade_gate_canonical_pull`, `project_m90_verdict_repr`).
- **5.20a / weak-model tool fumbling**: `gemma-4-31b` emitted `write_file({})` 5×. Is this a fleet-quality
  issue (Fatih Hoca picking too-weak a model for a multi-file-write medium step) or a prompt/tool-schema
  issue? Consider bumping 5.20a difficulty or an eligibility floor for file-writing steps.

## Key gotchas
- **Instruction reloads live; CODE does not.** i2p_v3.json = `(path,mtime)` via
  `_refresh_workflow_step_config`. mr_roboto/hooks/coulson need `/restart`.
- **`checks` now live-reload** (this session) — but the running orch must be newer than `7883789d` for
  the substitution to be correct; a mid-version orch live-reloads checks with unsubstituted `{mission_id}`
  → vacuous pass. After any i2p `checks` edit, ensure the orch is at ≥`7883789d`.
- **Dispatch gates on `depends_on` (task-ids), NOT `input_artifacts`** — a conservation source must be a
  `depends_on` producer or the check can race ahead of it.
- **Vacuous pass is silent by design** but now flagged (`wiring_suspect` + WARNING). Grep logs for
  `conservation vacuous pass` after any checks change.
- ⚠️ Zombie-pytest deadlock risk with the live session — run targeted + `timeout`, kill only own hung procs.

## Memory
`project_requirement_conservation_gate_20260722` (the gate + 3 live-reload gaps + full validation).
`project_reviewer_falsification_presence_mechanical_20260722` (the earlier halt fix, validated).

## Files touched this session
| File | Change |
|---|---|
| `packages/mr_roboto/src/mr_roboto/verify_requirement_conservation.py` | NEW pure checker |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | dispatch branch + `wiring_suspect` observability + export |
| `packages/mr_roboto/src/mr_roboto/reversibility.py` | `verify_requirement_conservation: full` |
| `packages/general_beckman/src/general_beckman/posthooks.py` | registry entry |
| `packages/coulson/src/coulson/__init__.py` | `checks` in `_CTX_FIELDS` + `{mission_id}` substitution |
| `src/workflows/engine/hooks.py` | `_unwrap_envelope` double-escaped-JSON recovery |
| `src/workflows/i2p/i2p_v3.json` | conservation `checks` on 3.9a/3.10a/3.10b; hardened instrs; 3.9a/3.10a `depends_on` +3.7/3.8; 3.10b FR+US+BR |
| tests | `test_verify_requirement_conservation.py` (11), `test_step_refresh_checks.py` (3), `test_unwrap_envelope.py` (+4) |
