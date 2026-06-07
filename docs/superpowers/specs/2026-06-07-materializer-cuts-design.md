# Materializer cuts — `_schema_version` stamping + multi-produces hardening

**Date:** 2026-06-07
**Status:** Design approved, spec for implementation
**Follows:** `2026-06-05-deterministic-materializer-design.md` (§7 deferred non-goals)

---

## 0. Context

`src/workflows/engine/hooks.py::materialize_produces` is the **sole writer** of a
workflow step's declared `produces` paths (shipped `4a15fbec`, 2026-06-05). Per
file it gathers candidates (agent's on-disk write + the LLM `output_value`),
picks the schema-best via `coulson.grounding.select_canonical`, stamps
`mission_id` front-matter idempotently, and writes the canonical path last.

Two follow-ups were explicitly cut at ship time (design §7):

1. **`_schema_version` stamping.**
2. **Multifile `result→N-artifact` mapping.**

Investigation reshaped both:

- **Version source already exists.** `_schema_version` is authored **per
  artifact-key inside the step's `artifact_schema`** — 84 of 239 i2p_v3 steps
  carry it, values `{"1","2"}`. It is genuinely **per-key**: step `3.6` has
  `platform_requirements`=2, `accessibility_requirements`=1, `i18n_requirements`=2.
  No expander threading is needed; the materializer only needs to *read* it.
- **Cut #2 has no `result→N` consumer.** All 13 multi-produces steps are
  *agent-writes-each* (ADR `<decision>.json` + `register.md`; `5.0d`
  `screen_inventory.md` + `shared_shell.md`). No step returns one bundle to fan
  out. So a splitter is YAGNI. The real gap is a **latent cross-contamination
  bug**: `output_value` (the step's single result) is currently a candidate for
  *every* produces entry, so `register.md`'s candidates are
  `[register_disk, decision_json]` — if the agent's register markdown fails the
  schema while the decision JSON passes it, `select_canonical` returns the
  decision JSON and **`register.md` is overwritten with decision content**.
  Severity is currently gated by `validate_artifact_schema` looseness (no
  phase-4 mission on disk to observe it), but the bug is real.

`verify_adr_shape` reads `_schema_version` *from the artifact* (`adr.get
("_schema_version")`, lenient at "1"/missing). Today that relies on the LLM
authoring the field. Deterministic stamping makes it reliable — the same
"don't trust the LLM for metadata" philosophy as the materializer itself.

---

## 1. Goals / non-goals

**Goals**
- Deterministically stamp `_schema_version` into each materialized artifact when
  its schema entry declares one (Cut #1).
- Eliminate `output_value` cross-contamination across multi-produces files;
  validate/version each file against its own schema key (Cut #2 hardening).

**Non-goals**
- No `result→N` splitter (no consumer).
- No expander changes (version already lives in the schema JSON).
- No change to `validate_artifact_schema` looseness (§6 #2, separate spec).
- No change to the in-memory schema-gate's use of the returned `canonical_out`.

---

## 2. Shared primitive — `_schema_entry_for_path`

One place owns the produces-path ↔ schema-key rule; both cuts call it.

```
_schema_entry_for_path(produces_path: str, schema: dict) -> tuple[str, dict] | None
```

- `schema` is the step's `artifact_schema`, shape `{artifact_name: {type, _schema_version?, ...}}`.
- Match by **basename stem**: strip directory and extension from `produces_path`
  (the `mission_{mission_id}/...` template prefix falls away with the dirname),
  compare to each top-level key.
- Return `(key, entry)` on a unique stem match; `None` otherwise (no key, or
  the file has no schema entry — e.g. `register.md`).
- Pure; no I/O.

Worked cases (from i2p_v3):
- `mission_X/.adr/architecture_pattern_decision.json` → `("architecture_pattern_decision", {…, _schema_version:"1"})`
- `mission_X/.adr/register.md` → `None`
- `mission_X/.flow/screen_inventory.md` → `("screen_inventory", {…})`
- `mission_X/reverse_pitch.md` → `("reverse_pitch", {…, _schema_version:"1"})`

---

## 3. Cut #1 — deterministic `_schema_version` stamp

### 3.1 `stamp_front_matter` gains an optional version (`coulson/grounding.py`)

```
stamp_front_matter(content, mission_id, kind, schema_version: str | None = None) -> str
```

- `kind == "md"`: existing `mission_id` front-matter logic, plus — when
  `schema_version` is not None — ensure a `_schema_version: <v>` line in the
  same front-matter block. Idempotent: no-op if a `_schema_version` line is
  already present (regardless of value — never double-stamp; the authored value
  wins, matching the `mission_id` idempotency contract). Never creates a second
  `---` block.
- `kind == "json"`: existing top-level `mission_id` injection, plus — when
  `schema_version` is not None and the parsed object lacks `_schema_version` —
  add top-level `_schema_version`. No-op if present or the content does not
  parse (best-effort; never corrupt a file).
- `schema_version is None` → behaviour identical to today.

### 3.2 Wiring in `materialize_produces`

For each materialized produces entry, before stamping:
```
match = _schema_entry_for_path(entry, schema)
version = match[1].get("_schema_version") if match else None
chosen = stamp_front_matter(chosen, int(mission_id), kind, schema_version=version)
```
No matching key or no version → `version=None` → only `mission_id` stamped
(e.g. `register.md`). Never invent a version.

---

## 4. Cut #2 — harden multi-produces (no splitter)

In `materialize_produces`, branch on the count of declared `.md`/`.json`
produces entries (`single` already computed):

- **Single (`single == True`):** unchanged —
  `select_canonical([disk, output_value], _schema_ok)`. `canonical_out` becomes
  the chosen content (existing behaviour).
- **Multi (`single == False`):** each file is materialized from **its disk
  content only**. `output_value` is **not** a candidate. Concretely, the
  candidate list per file is `[disk]` (not `[disk, output_value]`), so a file's
  own agent-written content is kept and stamped; nothing from another file's
  result can overwrite it. `canonical_out` stays `output_value` (unchanged
  return for multi).
  - Because each multi file has a **single** candidate (`[disk]`),
    `select_canonical` always returns that disk content (a lone candidate is
    kept whether or not it passes `schema_ok`). So the file is always
    preserved as-written; `schema_ok` is functionally moot in the multi branch.
    Implementation simplification: the multi branch may skip `select_canonical`
    entirely and use the disk content directly (still unwrapping a narration
    fence via `unwrap_fenced_artifact` for parity with single, then stamp).

Disk-missing edge: if a multi-produces file has no disk content (agent failed to
write it), the disk candidate is `None` → `continue` (no file written, no
crash) — same fail-soft contract as today.

---

## 5. Files touched

| File | Change |
|------|--------|
| `packages/coulson/src/coulson/grounding.py` | `stamp_front_matter` gains `schema_version` param; add `_schema_entry_for_path` (pure) |
| `src/workflows/engine/hooks.py` | `materialize_produces`: per-file version lookup + stamp; multi-produces disk-only candidate list + per-file schema_ok |

No expander, gate, or i2p JSON changes.

---

## 6. Testing

**`_schema_entry_for_path` (pure):** stem match (`.adr/x.json`→`x`), miss
(`register.md`→None), `mission_{mission_id}/` template prefix stripped, multi-key
schema unique match.

**`stamp_front_matter` version:**
- md: adds `_schema_version` to existing front-matter; prepends a block with both
  `mission_id` + `_schema_version` when absent; idempotent (authored value kept).
- json: adds top-level `_schema_version`; no-op if present; no-op on unparseable.
- `schema_version=None` ⇒ byte-identical to pre-change output (regression lock).

**`materialize_produces`:**
- single-produces: still uses `output_value`; version stamped from the sole key.
- multi-produces: each file = its disk content; `output_value` never leaks into a
  sibling file (the contamination regression — assert `register.md` keeps register
  content even when decision JSON would pass its schema); each file stamped with
  its own version; `register.md` (no key) gets `mission_id` only.
- fail-soft: missing-disk multi file → skipped, no crash.

---

## 7. Rollback

Both changes are additive and isolated to two functions. Revert the two commits
(`grounding.py`, `hooks.py`) to restore prior behaviour; no data migration, no
schema/JSON changes to undo.
