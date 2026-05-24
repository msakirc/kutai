Legacy Removal — Phase 6 Final Handoff (2026-05-25)
=====================================================

## What was removed (phases 1-5)

### Deleted workflow steps (16 total)

Phase 0 deletions:
- 0.2  problem_statement_extraction
- 0.4  scope_ambiguity_detection
- 0.5  human_clarification_request

Phase 5 deletions (all 13 design-spec steps):
- 5.1  information_architecture
- 5.2  navigation_design
- 5.3  user_flows
- 5.4  layout_and_wireframes
- 5.5  forms_and_states
- 5.6  wireframe_review
- 5.7  brand_and_design_tokens
- 5.8  component_specs
- 5.9  screen_specifications
- 5.10 interaction_responsive_accessibility_specs
- 5.11 design_system_handoff
- 5.11b design_handoff_document
Plus: design_review (was also referenced as 5.10 in reviewer_regression)

### legacy_pre_* gates stripped

All skip_when gates of the form `mission.legacy_pre_* == '1'` were removed
from i2p_v3.json. Steps that had these gates now run unconditionally.

Gates removed (by feature name):
- legacy_pre_compliance   (0.4a, 1.11a)
- legacy_pre_premortem    (6.5z)
- legacy_pre_spec_alive   (7.0z, 8.0z, 9.0z, 10.0z, 11.0z, 12.0z)
- legacy_pre_inheritance  (6.7z)
- legacy_pre_prior_art    (1.0)
- legacy_pre_github_init  (6.7)
- legacy_pre_adr          (4.1, 4.2, 4.2a, 4.4, 4.6, 4.8, 4.9, 4.10, 4.14)
- legacy_pre_falsification (3.1.verify, 3.2.verify, 3.3.verify, 3.7.verify)
- legacy_pre_non_goals    (0.6a)
- legacy_pre_p7           (verify_schema_version function parameter — kept for
                           backward compat with in-flight missions, not a gate)

Total: 9 distinct legacy_pre_* DB column names dropped. Migration in
`src/infra/db.py` via `_LEGACY_DROP_COLS` list applies the ALTER TABLE DROP
on next init_db() call.

### Engine branches removed

- `mission.<col> skip_when` branch in workflow engine loader/runner
- Legacy rescue branches for the deleted design-spec phase

## Phase 6 — test cleanup (this session)

### Files modified

| File | Action |
|------|--------|
| tests/i2p/test_t5a_steps.py | Updated legacy_pre_compliance assertions to assert gate is GONE; fixed 0.4a depends_on assertion (0.4 deleted, now depends on 0.1) |
| tests/i2p/test_t5b_steps.py | Updated legacy_pre_premortem + legacy_pre_spec_alive assertions to assert gate is GONE; fixed WAVE_PHASES: 9.0z now depends on 8.spike.git_commit |
| tests/i2p/test_t6a_steps.py | Removed test_step_0_5_has_surface_prior_mission_hints_post_hook (step 0.5 deleted); updated legacy_pre_inheritance assertion |
| tests/i2p/test_t6b_steps.py | Updated legacy_pre_prior_art assertion to assert gate is GONE |
| tests/i2p/test_t6c_steps.py | Renamed test + updated legacy_pre_github_init assertion to assert gate is GONE |
| tests/i2p/test_adr_shape.py | Renamed test_i2p_v3_adr_steps_have_skip_when_legacy -> test_i2p_v3_adr_steps_have_no_legacy_gate; asserts no legacy_pre_ in skip_when |
| tests/i2p/test_falsification.py | Updated falsification verify sibling skip_when assertion to assert gate is GONE |
| tests/i2p/test_non_goals.py | Updated 0.6a skip_when assertion to assert gate is GONE |
| tests/i2p/reviewer_regression/test_reviewer_regression.py | Removed 5.10 from REVIEWER_STEP_IDS; added 5.10 to deleted_step_ids filter in _discover_fixtures(); updated minimum fixture count 10->8 |
| tests/workflows/test_no_legacy_residue.py | REWROTE schema guard test to be fully synchronous (source-parsing); eliminated Windows asyncio teardown hang |

### Files left untouched (correct as-is)

- tests/test_clarify_action_schema_skip.py — references "0.5" as a dummy
  step ID string in mock context; the clarify logic checks triggers_clarification
  flag, not the step ID. Test is still valid.
- tests/test_workflow_expander.py — uses "0.2" as a generic example ID.
- tests/test_workflow_runner.py — uses "0.2" as a generic example ID.
- tests/test_z3_t2c_integration_review.py — uses "5.1" as a dummy mock step.
- tests/i2p/test_falsification.py tests for legacy_pre_falsification as a
  function PARAMETER (not a skip_when gate) — that parameter is kept for
  backward-compat with in-flight missions and remains valid.
- tests/i2p/reviewer_regression/test_reviewer_regression.py — legacy_pre_p7
  function parameter tests remain valid (same reason).

### Tests run and results

All run without DB (safe):
- tests/workflows/test_no_legacy_residue.py          2 passed
- tests/workflows/test_i2p_v3_dep_integrity.py       3 passed (primary gate)
- tests/i2p/test_t5a_steps.py                        7 passed
- tests/i2p/test_t5b_steps.py                        8 passed
- tests/i2p/test_t6a_steps.py                        3 passed
- tests/i2p/test_t6b_steps.py                        6 passed
- tests/i2p/test_t6c_steps.py                        5 passed
- tests/i2p/test_adr_shape.py                       (part of 73 passed)
- tests/i2p/test_falsification.py                   (part of 73 passed)
- tests/i2p/test_non_goals.py                       (part of 73 passed)
- tests/i2p/reviewer_regression/test_reviewer_regression.py  26 passed

### Guard-test approach

The test_fresh_missions_schema_has_no_legacy_columns test was REWRITTEN to
be fully synchronous: it reads db.py source text and searches the CREATE TABLE
missions statement for any `legacy_pre_` column names. This eliminates the
Windows asyncio + aiosqlite teardown hang. Both tests in the file pass and
exit cleanly (0.08s).

## Pre-existing failures (NOT caused by this work)

tests/test_workflow_loader.py has pre-existing failures from stale v2-shape
assertions that do not match the current i2p_v3.json:
- Asserts version "2.0" (current is different)
- Asserts 31-step template (template count has changed)
- Asserts 7 conditional_groups (structure changed)

These failures predate the legacy-removal branch and are unrelated to this work.
Do NOT fix them as part of this cleanup.

## Open semantic-validation debt

Phase 2 rewired phase-7/11/13 frontend+docs consumers from the deleted
design-spec artifacts to paraflow html_prototype outputs. The structural
dependency integrity gate (test_i2p_v3_dep_integrity.py) passes. However,
SEMANTIC correctness — whether the implementers actually produce good UI
components when given html_prototypes instead of wireframe/design-spec
artifacts — is UNVALIDATED. This requires a real i2p mission reaching
phase 7+ to confirm.

## Commit SHAs (branch: chore/legacy-removal)

Run `git log --oneline -5` on the branch to get the latest 5 commits.
Phase 6 adds 2 commits on top of phases 1-5:
- test(legacy): make schema guard fully synchronous to avoid Windows asyncio hang
- test(legacy): update/remove tests asserting removed legacy steps + gates

## FOUNDER ACTIONS REQUIRED

(a) DATABASE MIGRATION — apply column-drop migration:
    1. Back up the live DB FIRST:
       copy C:/Users/sakir/ai/kutai/kutai.db C:/Users/sakir/ai/kutai/kutai.db.backup-pre-legacy-removal
    2. Send /restart via Telegram (or /stop then restart manually).
       init_db() will run _LEGACY_DROP_COLS ALTER TABLE DROP statements.
    3. Confirm the bot comes back up in logs.

(b) FULL TEST SUITE — run after /stop (DB-touching tests cannot run while
    orchestrator holds the SQLite write lock):

    DB-touching files (DO NOT run with orchestrator live):
    - tests/test_i2p_v3.py           (opens DB via get_db fixture)
    - tests/i2p/test_intake_todo.py  (DB-backed workflow trigger test)
    - tests/integration/             (full integration, needs DB)
    - Any test importing init_db, get_db, or aiosqlite directly

    After /stop:
    timeout 120 .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/integration -p no:cacheprovider

(c) E2E MISSION VALIDATION — run a fresh i2p mission to validate:
    - Phase 0: steps 0.1 -> 0.0a.draft -> ... no longer wait for 0.2/0.4/0.5
    - Phase 5: confirm the phase is now entirely skipped (all 13 design steps gone)
    - Phase 7+: confirm frontend implementers produce valid output from
      html_prototypes (the semantic-validation debt noted above)
    - Review the DLQ for any unexpected failures from newly unconditional steps
