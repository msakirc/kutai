# Z2 v2 — Build foundation (corrected audit + alt shapes + wiring + migration)

**Supersedes** v1 surface ([02-build-foundation.md](02-build-foundation.md)) per the
"every zone needs v2+" feedback. Inherits frame + open questions; corrects
stale claims; expands each guardrail with alt shapes, wiring details, and
phased migration; converts 6 unbatched guardrails into a 5-tier plan
modelled on the Z1 completion pattern.

## Stale claims in v1 (fix here)

| v1 claim | Reality | Source |
|---|---|---|
| "tools_hint is advisory today; should be binding" | Already binding. `_apply_tools_hint()` **overrides** `profile.allowed_tools`; filter happens before agent sees tools. | `packages/coulson/src/coulson/__init__.py:134-176` |
| "mr_roboto verbs: workspace_snapshot, git_commit, verify_artifacts, run_cmd, check_grounding, clarify, notify_user" (implying that's the full list) | 30+ verbs incl. `run_pytest`, `http_check`, `verify_schema_version`, `check_against_non_goals`, plus shape/consistency checkers. | `packages/mr_roboto/src/mr_roboto/__init__.py:78-119` |
| "G grounding auto-wire shipped" (no detail) | Idempotent prepend in expander when `produces` non-empty; runs before `verify_artifacts`. | `src/workflows/engine/expander.py:173-181` |
| "Mechanical post-hook framework: verify_artifacts + code_review + grounding" (read as: framework supports more) | Framework's KIND set is exactly 3 — frozenset literal. Adding any new kind requires touching this set + the dispatch table. | `packages/general_beckman/src/general_beckman/posthooks.py:39-77` |
| "diff-shaped tools exist but agent defaults to write_file" | Tools exist + are advertised. Defaulting is a **prompt-discipline** issue, not a routing one — `tools_hint` already gives us the lever. | `src/tools/__init__.py:1027-1080`; `packages/coulson/src/coulson/__init__.py:134-176` |
| "feature_implementation_template is recipe scaffold" | One structural template wired via `template_id` + `template_params`. No stack-aware content scaffolds — recipes (auth/upload/search) don't exist. | `src/workflows/i2p/i2p_v3.json:29`; `src/workflows/engine/hooks.py` |
| "stack-aware prompt fragments — new directory `src/agents/prompts/stacks/`" | Plug-point already exists: `REFLECTION_BLOCKS` dict in coulson keys prompts by agent role. Extend with `STACK_BLOCKS` keyed by `tech_stack_detected` — same pattern, no new dir. | `packages/coulson/src/coulson/reflection.py:27-59` |

**Net consequence**: real gap is narrower than v1 suggested. The
foundation (verbs, hint binding, auto-wire pattern, reflection plug-point)
is built — what's missing is **registering new kinds + wiring them on
file-pattern triggers**, plus the memory/recipe substrate.

## Six guardrails — alt shapes + pick + wire

Pattern for each: a) name b) trigger c) alt shapes considered d) **PICK**
with rationale e) wiring sketch f) severity policy.

### 1. `imports_check`
- **Trigger:** step produces `*.py` or `*.ts`/`*.tsx`.
- **Alt shapes:**
  - **A.** `ast.walk` + diff vs `pyproject.toml`/`requirements.txt` (Python-only, zero deps).
  - **B.** `pip-missing-reqs` / `pip-extra-reqs` shellout (Python; uses pkg metadata; covers transitive).
  - **C.** Tree-sitter parse + lockfile diff (multi-lang, heavy).
- **PICK:** language-dispatched, mirroring `src/tools/linting.py` shape: Python→ast (A) for cheapness; TS/JS→tree-sitter (C) since lockfile parsing already costs us a node hop. B is appealing but adds a runtime dependency for a check that ast covers.
- **Wire:** new verb `check_imports` in `packages/mr_roboto/`; add `"imports_check"` to `POST_HOOK_KINDS` frozenset; expander rule: `produces` matches `*.py|*.ts|*.tsx` and project manifest exists → auto-wire.
- **Severity:** missing dep → blocker. Unused dep → warning (no `blockers` entry).

### 2. `migration_apply`
- **Trigger:** step produces `migrations/*.py` or `alembic/versions/*.py` or `*.sql`.
- **Alt shapes:**
  - **A.** Ephemeral sqlite + apply migration → discard (fastest; mismatches PG semantics).
  - **B.** `testcontainers` Postgres per check (real fidelity; ~5–15s startup overhead).
  - **C.** Alembic offline mode → render SQL → parse for syntax (catches syntax, misses FK ordering + extensions).
- **PICK:** stack-aware dispatch by `tech_stack_detected`. SQLite stack→A; Postgres stack→B; unknown→C as cheapest signal. The cost section in v1 ("each gate ~30-60s") was hand-waved — quantify here: B is the only gate above that band; gate it behind opt-in until profiled.
- **Wire:** verb `apply_migration(stack, migration_path, workspace)`; add `"migration_apply"` kind; expander rule: produces matches migration glob.
- **Severity:** any apply error → blocker. Slow apply (>30s) → warning.

### 3. `test_run`
- **Trigger:** step produces `tests/**` or `test_*.py` or `*.test.ts`.
- **Reality check:** `run_pytest` verb already exists in `packages/mr_roboto/`. Wiring is what's missing.
- **Alt shapes:** keep `run_pytest`; add `run_jest` / `run_vitest` / `run_playwright` as thin `run_cmd` wrappers. The shellout-vs-import-and-call decision: shellout. Pytest in-process leaks state; subprocess is cleaner.
- **Wire:** add `"test_run"` kind; dispatch table picks the right runner from stack. Expander rule: produces matches test glob.
- **Severity:** any red → blocker. Slow suite (>2min) → warning + nudge to split.

### 4. `type_sync`
- **Trigger:** step produces a route file OR `openapi.json` OR a TS type file under `types/api`.
- **Alt shapes:**
  - **A.** FastAPI `app.openapi()` → write `openapi.json` → diff vs committed (catches backend drift).
  - **B.** `openapi-typescript` regen → diff vs committed `types/api.ts` (catches frontend drift).
  - **C.** Combined: route change → regen both → fail if either diffs without an accompanying produce.
- **PICK:** A+B as separate kinds (`openapi_sync`, `typescript_sync`) sharing a generic verb `regen_and_diff(generator_cmd, target_path)`. C as a composite over the two.
- **Coupling to Phase 4:** stack picker must constrain to openapi-emitting frameworks (FastAPI / NestJS / Rails-with-rswag). Add this as a stack-selection rule in Phase 4 acceptance, not in this guardrail's runtime.
- **Wire:** verb `regen_and_diff`; two kinds; expander rule: route file → backend kind; openapi.json → frontend kind.
- **Severity:** diff present → blocker.

### 5. `design_system_check`
- **Trigger:** step produces `*.tsx` / `*.jsx`.
- **Alt shapes:**
  - **A.** Regex/AST scan: raw `#[0-9a-f]{3,8}`, inline `style=`, non-allowed component imports.
  - **B.** Stylelint plugin (JS-only; ecosystem-heavy).
  - **C.** Semgrep rule pack (multi-lang; one engine).
- **PICK:** C. Semgrep also covers `pattern_lint` → single engine, two rule packs (`design-system.yml`, `forbidden.yml`). Saves install + config drift.
- **Wire:** verb `run_semgrep(rule_pack_path, target_path)`; kinds `design_system_check`, `pattern_lint` both dispatch to it with different packs.
- **Severity:** rule metadata carries severity (ERROR/WARNING/INFO); map ERROR→blocker.

### 6. `pattern_lint`
- Same verb as #5 with `forbidden.yml`. Rule pack seeded with: no `console.log`, no `time.sleep` in tests, no `assert True`, no `# TODO` in graded commits, no `eval(`, no `exec(`. Founder can add per project.

## Tooling discipline — narrower than v1

v1 frames this as a routing fix; reality is a **hint-generation** fix.
`tools_hint` is binding once set, but i2p_v3.json sets it per step
statically. What's needed:

- **Expander pass `_apply_hint_from_targets`**: when `produces` (or new
  `targets`) lists files that already exist in the workspace → strip
  `write_file` from `tools_hint`; keep `patch_file`, `edit_file`,
  `apply_diff`.
- Lives next to grounding auto-wire (`src/workflows/engine/expander.py:173-181`); same idempotent style.
- Founder-override slot: `force_write: true` in step config bypasses.

## Stack-aware prompts — extend reflection, don't fork a directory

- New dict `STACK_BLOCKS` in `packages/coulson/src/coulson/reflection.py`
  keyed by stack slot (`fastapi`, `nextjs`, `expo`, `django`, `rails`).
- `build_reflection_prompt` already concatenates per-role blocks; add a
  parallel concat for matching stack blocks. Resolution: read
  `tech_stack_detected` artifact (already declared at
  `src/workflows/i2p/i2p_v3.json:444`) at dispatch time via fatih_hoca's
  task-context bundler.
- v1's "src/agents/prompts/stacks/ directory" is fine as a *content*
  storage convention (markdown files loaded by name), but the *plug
  point* is reflection.py — don't re-invent the injection path.

## Cross-mission memory — schema + populator + injector

### Schema (v2 additions to v1 sketch)

```
mission_lessons
  id            INTEGER PK
  stack         TEXT       e.g. "fastapi+postgres+nextjs"
  domain        TEXT       e.g. "auth", "file_upload"
  pattern       TEXT       human label
  fix           TEXT       what to do
  severity      TEXT       blocker | warning | info
  occurrences   INTEGER    bumped on dedup hit
  dedup_key     TEXT UNIQUE  hash(stack || '\n' || domain || '\n' || normalized_pattern)
  source_kind   TEXT       dlq_pattern | posthook_fail | manual
  source_ref    TEXT       json: { table, row_id }
  suppressed    INTEGER    0/1, founder mute switch
  created_at    TIMESTAMP
  last_seen_at  TIMESTAMP
```

`dedup_key` is the v2 lever: upserts replace duplicate inserts. Source
linkage keeps every lesson back-traceable (matches the audit-extension
work in the t1c milestone).

### Populators

- **DLQ → lessons:** existing `src/infra/dlq_analyst.py` already detects cross-task patterns. Add a single sink: when analyst yields a pattern with N≥3 occurrences, upsert into `mission_lessons` with `source_kind='dlq_pattern'`.
- **Post-hook fail → lessons:** in `packages/general_beckman/src/general_beckman/apply.py::_apply_grounding_verdict` (and siblings for other kinds) — when verdict is blocker and the task is on its final attempt, upsert a lesson row.

### Injector

- New mr_roboto verb `inject_lessons(stack, domain) -> list[Lesson]`.
- Fires at mission start (Z0/P0 boundary) via a posthook on the
  preflight phase: `post_hooks: ["inject_lessons"]`.
- Result lands in mission context bucket `lessons_top_n` (default 5,
  ranked by occurrences × recency).
- Coulson reads bucket on prompt build, renders as "watch out for"
  block; suppressed rows excluded.

## Recipe library v1 — sharpened

### Schema sharpening over v1

- **Recipe = template + posthook bundle**, not just code scaffold.
  `recipe.yaml` declares `post_hooks: [imports_check, migration_apply,
  test_run, pattern_lint:<recipe_pack>]`.
- **Versioning by filename suffix**: `recipes/auth/v1/`, `v2/`. Mission
  pins version at start in `recipe_pin_log(mission_id, recipe_id,
  version)`. Upgrades only at phase boundaries (closes v1 open question).
- **Compat matrix in recipe.yaml**:
  ```yaml
  name: auth
  requires:
    tech_stack: ["fastapi+postgres+nextjs", "fastapi+sqlite+nextjs"]
  conflicts_with: ["custom_session_v1"]
  ```
- **`pick_recipe` verb** signature: `pick_recipe(feature_decl, stack) -> {recipe_id, version, fit_score} | None`. Fit < 0.7 → founder confirm.
- **Sequencing**: ship `auth/` first (most missions block). Then `audit_log/` (recipe consumers want one), then `pagination/`, `search/`, `file_upload/`.

## Tier batches (Z1-style)

### T1 — Foundation framework (no new guardrails yet, unblocks all)
- **T1A**: extend `POST_HOOK_KINDS` frozenset + dispatch table to a registry pattern (kind → verb → severity rules). Single source of truth in `packages/general_beckman/src/general_beckman/posthooks.py`.
- **T1B**: expander auto-wire — one rule per kind keyed on file-pattern matches against `produces`. Idempotent prepend pattern (matches grounding's auto-wire).
- **T1C**: expander pass `_apply_hint_from_targets` (write_file → patch_file when target exists).

### T2 — Easy guardrails (existing verbs, wire only)
- **T2A**: `test_run` (verb `run_pytest` exists; add `run_jest`/`run_vitest` thin wrappers).
- **T2B**: `imports_check` (new verb `check_imports`, ast for Python + tree-sitter for TS).
- **T2C**: `pattern_lint` (new verb `run_semgrep` + `forbidden.yml` rule pack).

### T3 — Heavy guardrails (subprocess / sandbox)
- **T3A**: `migration_apply` (sqlite path first; testcontainers Postgres behind opt-in flag).
- **T3B**: `type_sync` (verb `regen_and_diff`; two kinds: `openapi_sync`, `typescript_sync`).
- **T3C**: `design_system_check` (shares `run_semgrep`; adds `design-system.yml`).

### T4 — Memory + stack-aware
- **T4A**: `mission_lessons` table migration + dedup upsert helper in `src/infra/db.py`.
- **T4B**: DLQ→lessons populator + post-hook-fail populator hooks.
- **T4C**: `STACK_BLOCKS` in `coulson/reflection.py` + `inject_lessons` verb + mission-start post-hook.

### T5 — Recipe library
- **T5A**: `recipe.yaml` schema + `recipes/` layout + `pick_recipe` verb + `recipe_pin_log` table.
- **T5B**: ship `auth/v1/` end-to-end as template (specs+templates+tests+migrations+lessons).
- **T5C**: planner gains `pick_recipe` step kind; expander instantiates recipe with stack-aware substitution.

## Migration / back-compat notes

- **i2p_v3.json**: no breaking change. New kinds and auto-wire are
  additive. Steps explicitly listing `post_hooks: ["verify_artifacts"]`
  still run as before — the auto-wire only *prepends* missing kinds.
- **Existing missions**: in-flight missions keep old post-hook set
  (pinned at mission start, same pattern as recipe version pinning).
- **Severity gates**: blockers in new kinds only fire after T4 lessons
  are queryable — until then, ship at warning severity so first batch of
  failures populates the lesson table without halting missions.
- **Cost budget**: T3 gates each cost 10–60s. Add a `gate_budget_ms`
  field in step config (default 60_000); skip with warning if budget
  blown. Profile after T2 ships.

## Open questions — resolved or sharpened

| v1 open Q | v2 answer |
|---|---|
| Recipe versioning? | Filename suffix `v1`/`v2`; mission pins at start (`recipe_pin_log` table); upgrade only at phase boundaries. |
| Memory pruning policy? | TTL = 180d default; `occurrences ≥ 5` immune to TTL; founder `suppressed=1` rows persist but never injected. |
| Stack-aware prompt fragments — how dynamic? | V1: markdown files keyed by stack slot, loaded by name into `STACK_BLOCKS` dict. V2 parametric only if rule explosion observed. |
| Tool-routing layer — heuristic or LLM? | Heuristic only; hint is generated by expander at static analysis time from `produces` file existence. LLM arbitration deferred until we see a heuristic miss class. |
| Mechanical-gate runtime cost? | Per-kind budget (`gate_budget_ms`) + parallel-where-safe (semgrep + ast + test concurrently; migration_apply serial). Profile after T2. |
| Custom-rules engine? | Semgrep, single engine for `pattern_lint` + `design_system_check` (two rule packs). |

## Dependencies (unchanged from v1)

- **Inbound:** [01-pre-code.md](01-pre-code.md) — recipe pick needs spec quality + ADRs.
- **Outbound:** [03-build-review-density.md](03-build-review-density.md), [04-build-visual-review.md](04-build-visual-review.md).
- **Cost hooks:** [10-cross-cutting.md](10-cross-cutting.md).

## Agent task brief — v2 sequencing

1. Read this v2 doc, then v1 for context delta.
2. Tiered batches map to Z1-style parallel dispatch:
   - T1 → one agent (foundation is tightly coupled).
   - T2 → three agents in parallel (independent kinds).
   - T3 → three agents in parallel (independent kinds; T3A may share work with T4A migration).
   - T4 → two agents (A+B as one, C separate).
   - T5 → sequential (A → B → C); B is the template-defining ship.
3. Each task quotes file:line refs; updates `## Updates` log on land.

## Updates

- 2026-05-10 — v2 created. Corrects 7 stale v1 claims, expands 6 guardrails with alt shapes + picks, adds dedup/source-link to memory schema, sharpens recipe library to template+posthook bundle, drops "new prompts/ directory" in favor of REFLECTION_BLOCKS extension, batches work into 5 tiers matching Z1 completion cadence.
- 2026-05-12 — **Z2 CLOSED**. All 6 tiers shipped: T1 (registry + auto-wire + hint-from-targets) → T2 (test_run + imports_check + pattern_lint) → T3 (migration_apply + type_sync + design_system_check) → T4 (mission_lessons schema + DLQ/posthook populators + STACK_BLOCKS + inject_lessons) → T5 (recipe.yaml schema + auth/v1 end-to-end + instantiate_recipe + i2p step 8.0a recipe_pick_all) → T6 (audit_log/pagination/search/file_upload recipe skeletons + semgrep blocker-severity promotion + recipe_picks consumer + emit_dlq_lessons cron + severity-ramp tracker). 33 z2-tagged commits. 309/312 z2-tier tests passing (3 skipped, 0 failed). HEAD `c965680`. Tag `z2-complete-2026-05-12`.
