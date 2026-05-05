# Handoff — i2p_v3 grounding & verdict gates (2026-05-05)

## Frame

Session investigated mission 57 (still in progress, paused). Found
3,542 "completed" tasks producing essentially zero verifiable code:
56 fabricated features off a 1-entry backlog, 17× `code_review:
verdict=fail` followed by fake staging deploys, "10 tests passed" with
no test files. Existing grading + validation + dependency-cascade
machinery was correct; the gaps were upstream (lenient schemas,
unbounded loop) and downstream (no link between review verdicts and
the build step they reviewed). Seven commits land the fixes.

## Diagnosis (grouped)

| Group | Status |
|-------|--------|
| A — design phases 0–6 (prose) | fine; no work |
| B — phase 7 deploy fabrication | folds into NEEDS-REAL-TOOLS |
| C — `[8.0]` backlog generator collapsed 16 → 1 | **closed** |
| D — feature loop invented 56 features | **closed** |
| E — feature-template build steps narrate, no file check | **closed** for 9 build/test steps |
| F — review verdicts ignored downstream | **closed** for 10 of 17 reviews + per-feature `code_review` post-hook |
| G — agent-side grounding (tool-only finish) | **deferred** to user's runtime rewrite |

User explicitly deferred G (already redesigning base agent + tool
calling layer). Goal of this session: build a mechanical safety net
under unreliable coder output without touching the agent layer.

## What shipped

7 commits on `main`, all pushed. ~3500 LOC, 137 new tests, no
regressions in pre-existing suites.

| # | SHA prefix | Title |
|---|-----------|-------|
| 1 | 41a6d59 | feat(salako): verify_artifacts + run_cmd verbs, git_commit require_diff |
| 2 | be60643 | feat(workflows): tighten implementation_backlog schema; pattern, unique_by, min_items_from |
| 3 | (D)     | feat(workflows): bound feature-template expansion (cap, dedup, halt-on-empty) |
| 4 | 8c7de28 | feat(posthook): verify_artifacts kind end-to-end |
| 5 | f93b2b1 | feat(posthook): code_review kind end-to-end |
| 6 | 6c1f9c9 | feat(i2p_v3): wire produces + post_hooks on feature_implementation_template |
| 7 | 5655f8b | feat(i2p_v3): wire string equals gates on 10 review steps |

(commits 8c7de28 and f93b2b1 carry the verify_artifacts + code_review
post-hook plumbing; 6c1f9c9 wires it on the 9 template build steps;
5655f8b adds the dialect string `equals` rule and uses it on top-level
reviews.)

## Architectural changes

### Salako (mechanical executor) verbs
New entries in `packages/salako/src/salako/`:

- `verify_artifacts.py` — given `paths` declared on the source step,
  resolve each under the mission workspace (rejecting absolute /
  traversal), check existence + `min_bytes`, optional compile/parse
  for known extensions (`.py` py_compile, `.json` json.load,
  `.yaml/.yml` yaml.safe_load). Returns `{verified, missing, failed,
  all_ok, sha256s}`.
- `run_cmd.py` — argv-list-only subprocess (no shell=True), cwd
  jailed under mission workspace, timeout with kill, stdout/stderr
  tail capped at 8 KB.
- `git_commit.py` — added `require_diff` flag; when set and the diff
  is empty, returns `failed` so silent no-op commits surface as
  hard failures. Backwards compatible (default off).

Routed via `salako.run({payload: {action: "verify_artifacts", ...}})`
and friends.

### Workflow engine — schema dialect

`src/workflows/engine/schema_dialect.py`:

- String rule now accepts `pattern: "<regex>"` (translates to JSON
  Schema `pattern`) and `equals: "<v>" | ["<v>", ...]` (alias
  `one_of`). The `equals` constraint is **post-emit only** — never
  translates to JSON-schema `enum`. Same anti-fabrication rationale
  as the existing boolean `equals: true` for verification flags
  (forcing the token at decode time made small models fabricate the
  success value).
- Array rule now accepts `unique_by: "<field>" | "."` — reports first
  duplicate with both indices, runs before per-item validation.
- New async `resolve_dynamic_constraints(schema, mission_id)` in
  `hooks.py` walks the schema, replaces
  `min_items_from: {artifact, path?, floor}` with the literal item
  count of the upstream artifact (drilled via dot-path). Failure
  degrades to floor. Wired into `validate_artifact_schema`.

### Workflow engine — feature loop bound

`src/workflows/engine/hooks.py:_trigger_template_expansion`:

- New config `MAX_FEATURES_PER_MISSION` (default 30, env override)
- Halt with WARNING (was DEBUG) when backlog empty / non-list /
  unparseable
- Dedup by `feature_id`, keep first occurrence
- Skip entries with missing/blank id
- Cap to `MAX_FEATURES_PER_MISSION` with warning

### Beckman — post-hook plumbing extension

`packages/general_beckman/src/general_beckman/`:

- `posthooks.determine_posthooks` reads `ctx.post_hooks` and combines
  with default `["grade"]`. Filtered through `_KNOWN_EXTRA_KINDS =
  {verify_artifacts, code_review}`. `_NO_POSTHOOKS_AGENT_TYPES`
  extended with `code_reviewer` (its verdict IS the gate, no
  judge-of-judge).
- `apply._posthook_agent_and_payload` + `_posthook_title` extended
  with `verify_artifacts` (mechanical) and `code_review`
  (code_reviewer agent) branches.
- New `_apply_verify_artifacts_verdict` and `_apply_code_review_verdict`:
  pass → drop kind from pending, complete source when empty; fail →
  retry source with feedback in `_schema_error` / `_prev_output`,
  honor worker attempt cap + bonus-progress budget. Neither bumps
  `failed_models` (the failures are agent-behaviour / review-judgment,
  not model-quality verdicts on the coder).
- `_posthook_dlq_cascade` gate widened from
  `agent_type ∈ {grader, artifact_summarizer}` to
  `ctx.source_task_id + ctx.posthook_kind` so any post-hook task
  cascades to source. New cascade branches for `verify_artifacts`
  and `code_review` mark source `failed` so depends_on blocks
  downstream.
- `rewrite.py`:
  - Rule 0 widened to translate `code_reviewer` Complete →
    PostHookVerdict (the agent emits `posthook_verdict` field same as
    grader).
  - Rule 0b (new): mechanical Complete with posthook ctx →
    synthesise PostHookVerdict (parses JSON-stringified salako result;
    `passed = result.all_ok`).
  - Rule 0c (new): mechanical Failed with posthook ctx → synthesise
    fail verdict.
  - `is_bookkeeping` widened so post-hook tasks (mechanical / reviewer
    / code_reviewer) don't recurse into MissionAdvance / RequestPostHook.

### New agent: code_reviewer

- `src/core/code_review.py` — `code_review_task(source)` mirrors
  `grade_task` but with a code-review-flavoured prompt (correctness,
  security, error handling, completeness against `produces`). Auto-
  fails on trivial/empty/degenerate output before round-tripping a
  reviewer model. `parse_code_review_response` extracts VERDICT
  (PASS/FAIL with bare-keyword fallback) plus bullet issues bounded
  by the `ISSUES:` header.
- `src/agents/code_reviewer.py` — thin BaseAgent wrapper, registered
  in `AGENT_REGISTRY` as `"code_reviewer"`.

### Workflow engine — expander field propagation

`src/workflows/engine/expander.py`:

- Top-level steps: propagate `produces` (list of paths) and
  `post_hooks` (list of kinds) into task ctx so Beckman's
  `determine_posthooks` reads them at dispatch time.
- Template steps: same, with `{feature_id}` (and any other params)
  interpolated per-feature so each instance gets concrete paths.
  `expand_template` uses Python str.format with the params dict;
  unmatched placeholders pass through untouched (validator catches).

## i2p_v3.json wiring

### `[8.0]` implementation_backlog_initialization (commit be60643)

- `min_items: 5`
- `min_items_from: {artifact: "mvp_scope", path: "mvp_feature_list", floor: 5}`
- `unique_by: "feature_id"`
- `feature_id` field rule: `pattern: "^F-\\d{2,3}$"`
- `feature_name` field rule: `min_length: 3`
- Instruction strengthened: "MUST cover EVERY feature from
  mvp_scope.mvp_feature_list — do not collapse, deduplicate, or drop"

### feature_implementation_template build steps (commit 6c1f9c9)

9 steps gained `produces` (with `{feature_id}` placeholder) +
`post_hooks` + a "REQUIRED OUTPUT PATH" sentence appended to the
instruction:

| Step                  | produces                                   | post_hooks                          |
|-----------------------|--------------------------------------------|-------------------------------------|
| database_migration    | migrations/versions/{fid}_initial.py       | verify_artifacts                    |
| backend_service       | backend/app/{models,services}/{fid}*.py    | verify_artifacts + code_review      |
| backend_endpoints     | backend/app/routes/{fid}.py                | verify_artifacts + code_review      |
| backend_tests         | backend/tests/test_{fid}.py                | verify_artifacts                    |
| frontend_state        | frontend/src/{types,api,state}/{fid}.ts    | verify_artifacts + code_review      |
| frontend_components   | frontend/src/components/{fid}/index.tsx    | verify_artifacts + code_review      |
| frontend_pages        | frontend/src/app/{fid}/page.tsx            | verify_artifacts + code_review      |
| frontend_tests        | frontend/src/__tests__/{fid}.test.tsx      | verify_artifacts                    |
| e2e_tests             | e2e/{fid}.spec.ts                          | verify_artifacts                    |

Untouched intentionally: `spec_review`, `implementation_plan` (pure
design, no produces); `code_review` template step (left as
decoration — the new post-hook is the actual gate); `staging_deploy`,
`staging_validation`, `quality_checks` (NEEDS-REAL-TOOLS).

### Top-level review verdict gates (commit 5655f8b)

10 of 17 reviews now declare `equals` on their verdict/status/
recommendation field:

| Step                          | Field          | Allowed       |
|-------------------------------|----------------|---------------|
| 1.13 research_quality_review  | verdict        | {pass}        |
| 1.14 go_no_go_assessment      | recommendation | {Go, go}      |
| 3.11 requirements_review      | status         | {pass}        |
| 4.16 architecture_review      | status         | {pass, approved} |
| 5.5  wireframe_review         | status         | {pass, approved} |
| 5.10 design_review            | status         | {pass, approved} |
| 6.6  project_plan_review      | status         | {pass, approved} |
| 7.16 sprint_0_review          | status         | {pass, approved} |
| 12.5 legal_review             | status         | {pass, approved} |
| 14.2 launch_checklist_review  | status         | {pass, approved} |

Each schema converted from legacy `required_fields: [...]` to
canonical `fields: {...}` so the equals constraint can attach to a
single field while others stay presence-only.

Not wired (7), with reason:

| Step                              | Reason |
|-----------------------------------|--------|
| 0.6 idea_brief                    | markdown artifact, no verdict field |
| 3.8 acceptance_criteria_audit     | outputs refined stories, not a verdict |
| 4.1 architecture_pattern_selection | decision (selected_pattern), not pass/fail |
| 4.2 tech_stack_research_and_selection | same — selection, not pass/fail |
| 10.1 owasp_audit                  | emits critical_findings list; gate would be array max_items=0 (false-positive prone). Defer to a "no-criticals" rule. |
| 10.5 encryption_and_logging_review | same shape as 10.1 |
| 11.5 documentation_review_and_fixes | fix step, not a gate |
| 13.14 launch_go_no_go             | `approved` field — type ambiguous (string vs boolean). Defer pending instruction clarification. |

## Test coverage

| Suite | New tests |
|-------|----------:|
| salako verbs | 36 |
| schema dialect (pattern, unique_by, equals, min_items_from) | 31 |
| feature loop bound | 9 |
| verify_artifacts post-hook | 18 |
| code_review post-hook | 18 |
| feature template wiring | 5 |
| review verdict gates | 20 |
| **Total** | **137** |

Pre-existing suites (`test_beckman_*`, `test_concurrency_and_cascade`,
`test_feature_template_idempotency`, `test_expander_double_dot`,
`test_reviewer_no_grade`, `test_schema_dialect`, salako legacy)
continue to pass: 100+ pre-existing tests intact.

Two-pass run required (conftest collision between `tests/` and
`packages/general_beckman/tests/`):
```
pytest tests/ ...                     # 190 pass
pytest packages/general_beckman/tests/ packages/salako/tests/...  # 75 pass
```

## How mission 58 differs from mission 57

If a mission with the same shape ran today:

1. **Backlog gate** — `[8.0]` produces a 1-entry backlog → schema
   rejects (`min_items_from` says ≥16). Source retries with feedback
   "implementation_backlog has 1 items, need >= 16". After retries
   exhausted the step DLQs. depends_on cascade halts phase 8 entirely.
   Mission stops honestly, not at 3,542 fake tasks.
2. **Loop bound** — even if a 56-entry backlog slipped through, the
   expander caps to 30, dedups, halts on empty. No invented fids.
3. **Per-feature build** — each build step declares `produces`. Source
   completes → verify_artifacts post-hook fires → if files missing
   under the mission workspace, source retries with explicit feedback
   "Files you said you wrote are missing or empty: missing=[...]".
   After cap, feature DLQs. Sibling features keep going. depends_on
   cascade blocks staging_deploy for the failed feature.
4. **Per-feature review** — code_review post-hook fires after build.
   FAIL verdict → source retries with the reviewer's bullet issues
   list as feedback. Mission 57's pattern (verdict=fail then fake
   deploy) cannot reproduce.
5. **Top-level reviews** — 10 of 17 reject fail values via schema
   equals. Grader retry path → DLQ → depends_on blocks downstream.
   `architecture_review: status=rejected` actually halts subsequent
   phases instead of being decoration.

Without G (agent grounding), the practical outcome on first runs is
expected to be many DLQs because the coder still narrates instead of
calling write_file. That's the honest ceiling — and it's measurable.
When G lands, build success rate jumps; everything wired here keeps
applying.

## Still open

### NEEDS-REAL-TOOLS (separate workstream)
Steps that fundamentally need MCP / API tools the agent doesn't have:

- 7.13 staging_environment
- 9.4  e2e_test_suite (could become real `salako.run_pytest`)
- 13.1 production_infrastructure
- 13.3 monitoring_setup
- 13.11 social_preview_test
- template feat.13 staging_deploy
- template feat.14 staging_validation

Until Vercel/Railway/Supabase deploy adapters land (or whichever stack
applies), these will continue to fabricate. User's framing: do not
"DEFERRED" stub them — that's still LLM-replacement fabrication. Either
real tools or removed from autonomous flow.

### Phase 7 scaffold (14 top-level)
Same `produces` + `post_hooks` wiring as feature template — repetitive
JSON edit, no engine changes:

- 7.2 linting_and_formatting
- 7.3 backend_scaffold
- 7.4 database_setup
- 7.5 frontend_scaffold
- 7.6 test_infrastructure
- 7.7 docker_setup
- 7.8 ci_cd_pipeline
- 7.9 design_system_implementation
- 7.10 primitive_components
- 7.11 composite_components
- (others as relevant)

### 7 unwired reviews
Three need different gate kinds:

- `owasp_audit` (10.1) and `encryption_and_logging_review` (10.5):
  emit `critical_findings` / `findings` arrays. Gate could be
  `max_items: 0` on the criticals list, but false-positive prone.
  Better: a "no-blocker" rule that filters by severity field. Defer.
- `launch_go_no_go` (13.14): `approved` field type ambiguous. Read the
  instruction or examine real outputs to confirm string vs boolean,
  then wire `equals: true` (boolean) or `equals: ["yes", "approved"]`
  (string).

Four don't fit the pattern (markdown / decision / fix step) — leave
alone.

### G (deferred — user's workstream)
Agent-side tool-only finish enforcement. When that lands, build steps
will actually call write_file instead of narrating. Retry-pass-rate
goes from "many before any feature builds" to "first attempt usually
builds". The plumbing here doesn't change.

## Pointers

- Audit script + classifier: re-run by reading `i2p_v3.json` and
  bucketing per the categories in this doc; manual review needed for
  edge cases (the heuristic mis-flags some research/audit steps as
  verdict-gates).
- Mission 57 row remains paused. Its 3,634 top-level tasks (3,542
  completed, 4 failed, 22+ pending) are stale relative to the new
  schema. Either resume (will likely DLQ at first new schema
  rejection) or cancel and start a fresh mission to validate the new
  flow.
- Schema dialect changes are backwards compatible — legacy
  `required_fields: [...]` still works via `_normalize_rule`. New
  rules opt in.
- Feature-loop expander warnings are at WARN level — visible in
  guard.jsonl / orchestrator.jsonl when fired.

## Quick verify

```
# Run all touched suites in two passes (conftest collision).
pytest tests/test_i2p_v3_review_gates.py \
       tests/test_i2p_v3_template_wiring.py \
       tests/test_verify_artifacts_posthook.py \
       tests/test_code_review_posthook.py \
       tests/test_template_expansion_bounds.py \
       tests/test_resolve_dynamic_constraints.py \
       tests/test_schema_dialect_pattern_unique.py \
       tests/test_schema_dialect.py \
       tests/test_beckman_rewrite.py \
       tests/test_beckman_posthooks.py \
       tests/test_beckman_on_task_finished.py \
       tests/test_reviewer_no_grade.py \
       tests/test_feature_template_idempotency.py \
       tests/test_concurrency_and_cascade.py \
       tests/test_expander_double_dot.py
# 190 passed

pytest packages/general_beckman/tests/ \
       packages/salako/tests/test_verify_artifacts.py \
       packages/salako/tests/test_run_cmd.py \
       packages/salako/tests/test_run.py \
       packages/salako/tests/test_git_commit.py
# 75 passed
```
