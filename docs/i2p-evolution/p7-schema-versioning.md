# P7 — Spec Versioning + Reviewer Regression Fixtures

**Status:** Shipped (Tier 0, branch `z1-p7-spec-versioning`).
**Charter:** every later Z1 proposal MUST update fixtures alongside any
artifact-shape bump. No "schema bump only" PRs.

---

## What landed

### 1. `_schema_version` on every phase 0-6 artifact

Every artifact declared in a phase 0-6 step's `artifact_schema` now carries
`_schema_version: "1"`. The pair `(artifact_name, _schema_version)` is the
addressable unit — N4 in `01-pre-code-plan-v3.md` rules out a global schema
version because cross-artifact joins (e.g. an ADR consuming evidence_refs)
must remain unambiguous when one side bumps and the other does not.

The injection was performed by
`scripts/p7_inject_schema_version.py` — idempotent, in-place, preserves
formatting. Re-running on a future v2 bump only adds new fields; existing
ones are skipped.

### 2. `verify_schema_version` mechanical action

New executor at `packages/mr_roboto/src/mr_roboto/verify_schema_version.py`,
wired into `mr_roboto.run` via the `verify_schema_version` action. Pure
function (no I/O). Inputs:

- `artifacts`: dict `{name: value}` or list of `{name, value}`.
  `value` may be a parsed JSON object, a JSON string, or markdown
  containing a fenced ```json``` block.
- `expected_versions`: dict `{name: version_string}`. The expander reads
  these out of `step["artifact_schema"][name]["_schema_version"]`.
- `legacy_pre_p7`: when truthy (sourced from `missions.legacy_pre_p7`),
  *missing* `_schema_version` fields are tolerated. Mismatches are
  always failures — a v9 artifact in a v1 contract is a hard fail
  regardless of mission age.

Output mirrors `verify_artifacts`/`check_grounding`:
`{ok, checked, missing, mismatched, legacy_pre_p7}`. Failure surfaces
the missing/mismatched names in the action error so the source task's
retry-with-feedback path tells the LLM exactly which artifact to fix.

### 3. Auto-wiring decision

**Decision: do not auto-wire as a default post-hook.** Cite
`src/workflows/engine/expander.py:178-181`, where `grounding` auto-attaches
on any step with `produces`. Adding a parallel auto-attach for any step
with `artifact_schema` would spawn a verify_schema_version task per phase
0-6 step (84 extra mechanical tasks per mission), most of which test the
same property — the expander already knows the version, and the writing
agent is the one that emits the field.

Instead, P7 ships the action as **callable infrastructure** that:

- The reviewer regression test suite invokes directly to lock the contract.
- Future Z1 proposals (P3, P1, P4...) can opt-in for steps where they
  bump `_schema_version` to "2" and want the upgrade verified at runtime.
  Add `"verify_schema_version"` to the step's `post_hooks` list and
  feed it the artifacts via the same `source_ctx` plumbing used by
  `grounding`/`verify_artifacts`.

If a future zone wants the auto-attach, the change is one block in
`expander.py` (mirror of the `grounding` block) plus a verdict applier in
`packages/general_beckman/src/general_beckman/apply.py` — model after
`_apply_grounding_verdict`.

### 4. `missions.legacy_pre_p7` migration

Column added to `missions` (default 0) in two places, both idempotent:

- `src/infra/db.py` — runs at startup; ALTER + UPDATE inside a try/except.
  Existing missions backfill to 1 in the ALTER-succeeded branch only.
- `scripts/p7_migrate_legacy_flag.py` — standalone, ops-driven; reads
  `KUTAI_DB` env or default path; safe to re-run.

New missions inserted after the migration default to 0. Pass the value
through `payload["legacy_pre_p7"]` when invoking the verifier so legacy
runs do not regress on the missing-field check.

### 5. Reviewer regression suite

`tests/i2p/reviewer_regression/` locks the structural contract for the
five reviewer steps:

| Step  | Reviewer                  | Output artifact            |
|-------|---------------------------|----------------------------|
| 1.13  | research_quality_review   | research_review_result     |
| 3.11  | requirements_review       | requirements_review_result |
| 4.16  | architecture_review       | architecture_review_result |
| 5.10  | design_review             | design_review_result       |
| 6.6   | project_plan_review       | project_plan_review_result |

Each step has `good.json` + `bad.json` under
`fixtures/v<version>/<step_id>/` (10 fixtures total, 17 tests including
mechanical-action smoke tests). The runner uses a deterministic
`_StubLLM` — **no live LLM calls**.

Per-fixture invariants:

1. Workflow declares `_schema_version` on the output artifact.
2. Stub response parses through `coulson.parsing.parse_action`.
3. Verdict body carries the declared `_schema_version`.
4. `good` fixtures' verdict matches the workflow's `equals` enum;
   `bad` fixtures must NOT.
5. `mr_roboto.run({action: verify_schema_version, ...})` accepts the
   bundle.

Runtime: ~2-3s. Invoked as `pytest tests/i2p/reviewer_regression/`.

---

## Schema-source-of-truth note for future zones

Today, artifact schemas live in **`src/workflows/i2p/i2p_v3.json`** under
each step's `artifact_schema` key. There is no separate registry. The
reviewer prompt's structural expectations are encoded twice:

- Once declaratively in `artifact_schema` (consumed by structural
  validators in `packages/coulson` and constrained-decoding emit passes).
- Once in the reviewer step's `instruction` text (the prose tells the
  LLM to emit `verdict` / `status` plus an `issues` list).

For Z1 going forward, treat the JSON declaration as authoritative. When
bumping a schema version:

1. Update the artifact's `_schema_version` in `artifact_schema`.
2. Update the reviewer step's `instruction` to cite the new version
   ("applies to user_personas v2+").
3. Add a `fixtures/v2/<step_id>/` directory with the new shape.
4. Keep `fixtures/v1/` until every consuming step has migrated — both
   suites run in CI; old fixtures pin the old contract while it still
   has live consumers.

---

## CI gating

The regression suite is collected by the standard `pytest` command (no
opt-in marker, no skip). `pytest.ini` defines no exclusions for
`tests/i2p/`. To wire as a hard gate, run the suite via:

    timeout 60 pytest tests/i2p/reviewer_regression/

This is what every P-proposal merge is expected to keep green.

---

## Known follow-ups (escalated to founder)

1. **No live mission validation.** P7 lands without a mission run that
   actually exercises the new column on a fresh mission. Recommend
   running one z0-preflight mission post-merge and checking
   `SELECT id, legacy_pre_p7 FROM missions ORDER BY id DESC LIMIT 5`.
2. **Fixtures are synthetic.** They lock structural shape, not LLM
   quality. The first real mission whose 1.13 reviewer rejects on a
   ground-truth research report will tell us whether the rubric and the
   schema actually agree.
3. **Auto-wiring decision is opinionated.** If you'd rather have every
   step's verify_schema_version run unconditionally, the change is the
   3 lines suggested above + a verdict applier.
