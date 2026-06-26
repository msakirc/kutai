# Legacy Removal — collapse i2p_v3 to the single canonical (paraflow) path

**Date:** 2026-05-25
**Status:** Design — approved, pending spec review
**Scope decision:** "C: Full + engine rescues" (founder, 2026-05-24)

---

## 1. Problem

Every i2p zone (Z1–Z10) that added a new phase/step also added a `legacy_pre_<feature>`
mission column so that **pre-existing missions** (created before that feature shipped)
would *skip* the new step. New missions get the column default `0`; the Z1 migrations
backfilled every existing mission to `1`.

The pattern was applied uniformly via `skip_when` expressions instead of deleting the
old steps. The Z1 Tier 3 handoff states the motive explicitly:

> *Legacy 5.1-5.11b preserved via skip_when in T3C (avoids cascade into 11+ depends_on
> rewrites). Sweep auto-rescue handles for new missions.*
> — `docs/handoff/2026-05-10-z1-tier3-shipped.md:98`

So the codebase now carries two parallel realities in one 170-step workflow:
the **canonical paraflow path** (what every new mission actually runs) and a **legacy
shadow** (steps + columns + engine back-compat that only ever served the now-stale
missions). This is dead weight, and it actively misfires: production 2026-05-24 mission 75
step `0.2` (`skip_when "mission.legacy_pre_charter != '1'"`) logged "skipped via skip_when"
six times yet still schema-validated the skip sentinel, DLQ'd `quality`, and spammed
Telegram — only the beckman sweep retro-skipped it after the damage.

**Ground truth (live DB, 2026-05-24):** all 19 `legacy_pre_*` columns are `0` for every
mission; exactly one non-terminal mission exists (mission 75, paused). **No real legacy
missions exist to strand.** The legacy branch is pure debt.

## 2. Goal

Collapse i2p_v3 to the single canonical paraflow path. Remove every code path whose
*only* purpose is serving pre-existing missions: the legacy-only steps, the
`legacy_pre_*` gates/columns/migrations, and the engine back-compat rescues for
pre-migration DB rows.

**Success =** a new mission runs an identical sequence to today, the workflow JSON has no
`legacy_pre_*` reference, the missions table has no `legacy_pre_*` column, the engine has
no pre-migration rescue branches, and the dep-integrity + workflow-loader test suites pass.

## 3. Terminology — "legacy" is overloaded

| Term | Meaning | In scope? |
|------|---------|-----------|
| **legacy missions** | pre-existing missions; the `legacy_pre_*` flags | **YES — remove** |
| **legacy rescue paths** | engine back-compat for old DB-row shapes | **YES (scope C) — remove** |
| **legacy schema dialect** | `required_fields`/`item_fields` form in `schema_dialect.py` | **NO — live, used by current steps** |

## 4. Polarity reference (verified against live data + JSON)

Column default `0` (new missions); existing rows backfilled to `1` (legacy).

- `skip_when "X == '1'"` → skips for legacy, **runs for new** → **canonical step → KEEP** (drop the skip_when, becomes unconditional).
- `skip_when "X != '1'"` → skips for new → **legacy-only → DELETE**.
- `skip_when "X == '0'"` → skips for new → **legacy-only → DELETE**.

(The Explore agent's first pass inverted these labels; corrected here and verified:
mission 75 with `legacy_pre_charter=0` *skipped* `0.2`, which is `!= '1'`.)

## 5. Deletion set — 16 legacy-only steps

**Charter chain** (replaced by paraflow charter `0.0z`→`0.1`; `!= '1'` on `legacy_pre_charter`):
- `0.2` problem_statement_extraction → `problem_statement`
- `0.4` scope_ambiguity_detection → `open_questions_list`
- `0.5` human_clarification_request → `clarification_request`

**Design-spec suite** (replaced by paraflow per-screen-plan + HTML prototype `5.20`/`5.30`;
`== '0'` on `legacy_pre_per_screen_plans`):
- `5.1` information_architecture, `5.2` navigation_design, `5.3` user_flows,
  `5.4a` layout_and_wireframes, `5.4b` forms_and_states, `5.5` wireframe_review,
  `5.6` brand_and_design_tokens, `5.7` component_specs, `5.8` screen_specifications,
  `5.9` interaction_responsive_accessibility_specs, `5.10` design_review,
  `5.11a` design_system_handoff, `5.11b` design_handoff_document

These are the **only** `!= '1'` and `== '0'` steps in the workflow. Everything else gated
on `legacy_pre_*` is `== '1'` (canonical).

## 6. Skip_when stripping — ~67 canonical steps

Every step whose `skip_when` is `"mission.legacy_pre_<x> == '1'"` keeps the step and
**removes the `skip_when` key** (it becomes unconditional — identical behavior for new
missions). Flags affected: `legacy_pre_charter, _idea_dedup, _compliance, _non_goals,
_prior_art, _competitive_positioning, _falsification, _adr (×24), _design_tokens,
_user_flow, _per_screen_plans (the ==1 / 5.20-5.30 branch), _html_oids, _preview_url,
_premortem, _github_init, _inheritance, _spec_alive`.

**Preserve** `mission.skip_real_vendor_checks == '1'` — genuine runtime feature flag, not legacy.

## 7. Dependency rewiring (the deferred "11+ depends_on rewrites")

The real work. For each deleted artifact, every remaining step that lists it in
`depends_on` / `input_artifacts` must be rewired to the canonical equivalent or have the
dead reference dropped.

Known anchors (full per-artifact map produced in the plan):
- `problem_statement`, `clarification_request`, `clarification_answers`, `open_questions_list`
  → only real downstream consumer is `0.6` idea_brief, whose instruction already falls
  back to `product_charter` when these are absent. Trim its `input_artifacts` + simplify
  its instruction to charter-only.
- The 5.1-5.11b design artifacts (`information_architecture`, `design_handoff`,
  `screen_specifications`, `component_spec_*`, `brand_identity`, `accessibility_spec`, …)
  → phase 6+ consumers must be rewired to the paraflow outputs (`per_screen_plans_chunk_*`,
  `html_prototypes_chunk_*`). The paraflow path itself (`5.20`/`5.30`) roots at
  `5.0a`/`5.0c`/`5.0d` (canonical 5.0x steps), **not** at 5.1-5.11b — so the canonical
  path is self-contained; only cross-phase document consumers need attention.
- `design_tokens` has two producers (deleted `5.6` + canonical `5.0a design_tokens_generation`).
  Canonical consumers must read from `5.0a`.

**Gate:** `tests/workflows/test_i2p_v3_dep_integrity.py` (every `input_artifacts` /
`depends_on` resolves to a surviving producer) + `tests/test_workflow_loader.py` must pass
after each phase. These tests are the regression harness for the rewiring.

## 8. DB schema + migrations

Drop all legacy columns from `missions`: the 19 `legacy_pre_*` present in the live DB
(`_charter, _adr, _falsification, _non_goals, _competitive_positioning, _per_screen_plans,
_html_oids, _preview_url, _premortem, _spec_alive, _compliance, _critic_gate, _github_init,
_idea_dedup, _inheritance, _prior_art, _design_tokens, _user_flow, _p7`) plus
`interview_skip_reason` (confirm no consumer first).

- Remove the `CREATE`/`ALTER`/backfill blocks for these in `src/infra/db.py`.
- Remove `scripts/z1_migrate_legacy_charter_flag.py` and the legacy-flag portions of any
  `scripts/z1_tier*_patch_*.py`.
- Add one forward migration that drops the columns (SQLite ≥3.35 `ALTER TABLE … DROP
  COLUMN`, applied per-column with existence guard; fall back to table-rebuild if the
  runtime SQLite is older — verify version at migration time).
- **Backup the DB before the drop.** Requires the orchestrator stopped (founder `/stop`),
  not a live-WAL operation.

## 9. Engine rescue removals (scope C)

Remove pure pre-migration back-compat:
- `src/workflows/engine/hooks.py` — `mission.<column>` skip branch (`should_skip_workflow_step`,
  ~1068-1096); the legacy JSON-lookup fallback for tasks lacking `skip_when_expr` (~1011-1048).
- `src/core/orchestrator.py` — executor legacy-shape rescue tiers (~249-315).
- `src/workflows/engine/expander.py` — executor legacy-shape rescue (~468-481).

**Keep:** the general `skip_when_expr` artifact evaluation (shopping_v2 uses it,
e.g. `gate_result.gate.kind != 'chosen'`) and the beckman sweep retro-skip backstop
(`packages/general_beckman/.../sweep.py`) — both serve live non-legacy flows.

## 10. The uncommitted post-hook skip guard

The `hooks.py` `_post_execute_workflow_step_impl` early-return-on-`_skipped` guard +
`tests/test_skip_when_schema_skip.py` written during the 2026-05-24 debug session:
**revert both.** The legacy kill removes its i2p trigger (no `skip_when` left in i2p), and
shopping_v2's skip steps retain the sweep backstop that already handled mission 75.
Keeping it would entangle a shopping-correctness change into a legacy-removal diff.

## 11. Approach — phased by flag-family (#2)

Each phase is independently mergeable and gated by the dep-integrity + loader tests.

1. **Charter chain.** Revert the uncommitted guard+test. Delete `0.2`/`0.4`/`0.5`, rewire
   `0.6`, strip `legacy_pre_charter` skip_when from canonical steps. Proves the dep-rewire
   harness on the smallest case.
2. **per_screen_plans.** Delete `5.1-5.11b`, rewire phase 6+ document consumers to paraflow
   outputs, strip the `== '1'` per_screen_plans skip_when from `5.20`/`5.30`.
3. **Bulk skip_when stripping.** Remove all remaining `legacy_pre_* == '1'` skip_when keys
   (adr/falsification/compliance/non_goals/prior_art/competitive/premortem/spec_alive/etc.).
4. **DB columns + migrations.** Drop columns, remove db.py blocks + z1 scripts, add the
   forward drop-migration. (Founder `/stop` + backup.)
5. **Engine rescues.** Remove the scope-C back-compat branches.

## 12. Testing

- Per phase: `tests/workflows/test_i2p_v3_dep_integrity.py`, `tests/test_workflow_loader.py`,
  and the relevant `tests/i2p/test_*.py` (deleting/rewriting the legacy-flag assertions).
- Engine phase: targeted `should_skip`/expander/orchestrator tests (DB-isolated; the live
  orchestrator deadlocks DB-touching suites — run with the orchestrator stopped or against
  an isolated DB, always with a timeout).
- Update/delete the 12 test files that assert `legacy_pre_*` behavior or steps 0.2/0.4/0.5
  (inventory in §5 of the surface map): `test_skip_when_schema_skip.py` (delete),
  `test_clarify_action_schema_skip.py` (drop 0.2/0.5 refs), `tests/i2p/test_t5a/t5b/t6a/t6b/t6c`,
  `test_adr_shape`, `test_falsification`, `test_non_goals`, `reviewer_regression`,
  `integration/test_workflow_pipeline.py`.
- A new "no-legacy" guard test: assert the workflow JSON contains zero `legacy_pre_`
  substrings and the missions schema has zero `legacy_pre_` columns — locks the cleanup in.

## 13. Risks & mitigations

- **Orphaned cross-phase consumers** (main risk). Mitigated by the dep-integrity test as a
  hard gate after every deletion; rewire per-artifact, never bulk-delete then fix.
- **Column drop on a 60-column live table.** Backup first; orchestrator stopped; per-column
  existence-guarded migration; verify SQLite ≥3.35 or table-rebuild path.
- **Docs reference the legacy flags** (7 handoff/evolution docs). Leave historical handoffs
  as-is (they're dated records); only update `docs/architecture-modularization.md` and any
  live reference doc if it describes current behavior.
- **Hidden non-i2p consumer of a legacy column.** Grep all `.py` for each column name before
  its drop; the `mission.<col>` engine branch is the only known reader and is removed in
  phase 5.

## 14. Rollback

Each phase is a separate commit/branch. Workflow + engine changes revert via git. The DB
column drop is the only non-trivial rollback — the pre-drop backup is the restore path;
the dropped columns held only `0`/`1` flags reconstructable by re-running the (then-removed)
backfill if ever needed, but no rollback path needs them since no legacy missions exist.

## 15. Out of scope

- The `required_fields`/`item_fields` schema dialect (live).
- Shopping_v2 skip_when behavior (separate workflow; keeps the sweep backstop).
- Any phase_5 redesign beyond removing the superseded 5.1-5.11b branch.
- The genuine `skip_real_vendor_checks` flag.
