# Phase 2 Plan — ADDENDUM (Artifact contract reconciliation)

> Read this **with** `2026-05-16-yalayut-phase2-intersect.md`. The base plan
> was written against an `Artifact` shape Phase 1 did NOT ship. This addendum
> prepends **Task 0** and gives a binding rename map. Everything else in the
> base plan stands verbatim.

## The mismatch

Phase 1 `query()` returns `yalayut.contracts.Artifact` with fields:
`artifact_id, name, name_original, artifact_type, kind, vet_tier, score,
exposure_class, applies_to, mechanizable, body_excerpt, payload`.

The base plan's code + `FakeArtifact` assume `id, vector_sim, body, source,
owner, env_status, intent_keywords, inputs_schema`. The plan uses
`getattr(art, ..., default)` everywhere, so against real Phase 1 it would NOT
crash — it would silently produce `confidence == 0.0` for every artifact and
bind nothing. Tests pass (they use `FakeArtifact`). Silent dead path. Fixed
here.

## Task 0 — Enrich Phase 1 `Artifact` (do this FIRST, before base-plan Task 1)

**Files:**
- Modify: `packages/yalayut/src/yalayut/contracts.py`
- Modify: `packages/yalayut/src/yalayut/index.py` (`_row_to_indexrow`)
- Modify: `packages/yalayut/src/yalayut/_query_engine.py` (`_to_artifact`, inline IndexRow build)
- Test: `packages/yalayut/tests/test_artifact_enrichment.py` (new)

**Steps:**

1. Write a failing test `packages/yalayut/tests/test_artifact_enrichment.py`
   asserting an `Artifact` returned by `query_db` carries `source`, `owner`,
   `env_status`, `intent_keywords` (list), `inputs_schema` (dict). Seed an
   in-memory `yalayut_index` row + a temp seed manifest yaml; assert the
   manifest's `intent_keywords`/`inputs_schema` land on the Artifact. Run —
   expect FAIL.

2. `contracts.py` — extend the `Artifact` dataclass (keep existing field
   ORDER; add new fields with defaults AFTER `payload` so positional callers
   in `_to_artifact` still work, or convert `_to_artifact` to all-keyword):

   ```python
   @dataclass
   class Artifact:
       """A ranked query() result."""
       artifact_id: int
       name: str
       name_original: str | None
       artifact_type: str
       kind: str | None
       vet_tier: int
       score: float
       exposure_class: str | None
       applies_to: str | None
       mechanizable: bool
       body_excerpt: str | None
       payload: dict[str, Any] = field(default_factory=dict)
       # --- Phase 2 enrichment ---
       source: str = ""
       owner: str | None = None
       env_status: str = "ready"
       intent_keywords: list[str] = field(default_factory=list)
       inputs_schema: dict[str, Any] = field(default_factory=dict)
   ```

   Also add `env_status: str = "ready"` to the `IndexRow` dataclass.

3. `index.py::_row_to_indexrow` — add `env_status=r["env_status"]`.

4. `_query_engine.py`:
   - The inline `IndexRow(...)` build inside `query_db` MUST also pass
     `env_status=r["env_status"]` (or, cleaner: import and reuse
     `index._row_to_indexrow`).
   - `_to_artifact(row, score)` — populate the new fields:
     - `source = row.source`, `owner = row.owner`, `env_status = row.env_status`
     - `intent_keywords` / `inputs_schema` — load the manifest at
       `row.manifest_path` if present. Use
       `yalayut.manifest.parse_manifest_yaml(open(path, encoding="utf-8").read())`
       and read `.intent_keywords` / `.inputs_schema` off the resulting
       `Manifest`. Wrap in try/except → on any failure default to `[]` / `{}`
       (a missing manifest must not break the hot read path). Make
       `_to_artifact` `async` if a manifest read is needed, OR keep it sync
       with a blocking file read (top_k≤12, tiny yaml — sync read is fine and
       simpler; pick sync).

5. Run the test — expect PASS. Run `timeout 60 pytest packages/yalayut/tests/`
   — expect the existing Phase 1 suite still PASSES (no regression).

6. Commit: `feat(yalayut): enrich Artifact with source/owner/env_status/intent_keywords/inputs_schema`

## Binding rename map — apply throughout base-plan Tasks 1–12

In **every** base-plan code block and test, substitute:

| Base plan writes | Use instead |
|---|---|
| `art.id` / `artifact.id` / `getattr(art, "id", ...)` | `artifact_id` |
| `art.vector_sim` / `getattr(artifact, "vector_sim", ...)` | `score` |
| `art.body` / `getattr(art, "body", ...)` | `body_excerpt` |
| `id=` kwarg in `FakeArtifact` / `_Art` constructors and test calls | `artifact_id=` |
| `vector_sim=` kwarg | `score=` |
| `body=` kwarg | `body_excerpt=` |

`source`, `owner`, `env_status`, `intent_keywords`, `inputs_schema` are now
**real `Artifact` attributes** (Task 0) — base-plan code reading them via
`getattr` works unchanged.

**Concrete touch points** (not exhaustive — apply the map everywhere):
- Task 2 `conftest.py::FakeArtifact.__init__` — rename params `id`→`artifact_id`,
  `vector_sim`→`score`, `body`→`body_excerpt`; assign to matching attr names.
- Task 2 `scoring.py` — `getattr(artifact, "vector_sim", 0.0)` → `"score"`.
- Task 2/3/4/7 tests — every `fake_artifact(id=..., vector_sim=..., body=...)`
  call → renamed kwargs.
- Task 4 `binding.py` — `getattr(artifact, "id", None)` → `"artifact_id"` (2 sites).
- Task 7 `flash.py` — `getattr(art, "id", None)` → `"artifact_id"` (~4 sites:
  `_slot_key`, `preempt_app`, applications dict, conflict-loser dict);
  `getattr(art, "body", "")` → `"body_excerpt"`.
- Task 12 `_Art` stub class — `self.id`→`self.artifact_id`,
  `self.vector_sim`→`self.score`, `self.body`→`self.body_excerpt`; keep
  `source`/`owner`/`env_status`/`intent_keywords`/`inputs_schema` as-is.

The `task["skills"]` envelope key stays `artifact_id` (base plan already uses
`artifact_id` there — only the *Artifact attribute reads* change).

## Untouched

Base-plan Tasks 5, 6, 8, 9, 10, 11 operate on plain dicts / non-Artifact code
— zero change. Task 1 scaffold — zero change. Exposure/budget/telemetry logic,
flow, thresholds, commit messages — all verbatim from the base plan.
