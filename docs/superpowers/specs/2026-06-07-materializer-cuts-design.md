# Materializer cuts — multi-produces contamination hardening (Cut #2); `_schema_version` stamping deferred (Cut #1)

**Date:** 2026-06-07
**Status:** Design approved → revised after adversarial review (Cut #1 deferred). Spec for implementation of Cut #2.
**Follows:** `2026-06-05-deterministic-materializer-design.md` (§7 deferred non-goals)

---

## 0. Context

`src/workflows/engine/hooks.py::materialize_produces` is the **sole writer** of a
workflow step's declared `produces` paths (shipped `4a15fbec`, 2026-06-05). Per
file it gathers candidates (agent's on-disk write + the LLM `output_value`),
picks the schema-best via `coulson.grounding.select_canonical`, stamps
`mission_id` front-matter idempotently, and writes the canonical path last
(`hooks.py:272-342`).

Two follow-ups were cut at ship time (design §7): (1) `_schema_version`
stamping, (2) multifile `result→N-artifact` mapping. This spec covers both —
but an adversarial review (2026-06-07) showed **Cut #1 is unsafe to build now**
and **Cut #2 should be reframed from a splitter (YAGNI) to a contamination fix**.

---

## 1. Cut #1 — `_schema_version` stamping — DEFERRED (do not implement)

Original intent: have the materializer deterministically stamp `_schema_version`
into each artifact (like `mission_id`), reading the version from the step's
`artifact_schema` (84/239 steps author it per-key, values `{"1","2"}`).

**Why deferred (two independent blockers):**

1. **JSON stamp would DLQ correct artifacts.** ADR decision files are validated
   by `verify_adr_shape`, which reads `_schema_version` *from the artifact* and
   fails when it `!= expected_schema_version`. For step `4.4` the **same file**
   `database_schema_decision.json` has `artifact_schema.database_schema_decision
   ._schema_version = "2"` but `checks.verify_adr_shape.payload
   .expected_schema_version = "1"` (i2p_v3.json). Today this works only because
   the LLM authors `"1"` and a stamp is no-op-when-present. If the materializer
   stamps the schema's `"2"` whenever the LLM omits it, `verify_adr_shape` DLQs a
   correct artifact. The `artifact_schema` is therefore **not** a trustworthy
   single source of truth for ADR JSON versions — the i2p data is internally
   inconsistent (`2` vs `1`).
2. **MD stamp has no live consumer.** The only artifact-version checker that
   reads `.md`, `verify_schema_version`, is **wired nowhere** in `i2p_v3.json`
   (0 occurrences as `post_hooks` or `checks`). `verify_adr_shape` only targets
   `.json` (8 paths, 0 `.md`). So stamping `_schema_version` into markdown is
   pure speculative metadata no gate reads.

**Reopen criteria:** build Cut #1 only once (a) `verify_schema_version` (or an
equivalent) is actually wired so a stamped version is read, AND (b) the i2p
`_schema_version` data is reconciled so `artifact_schema` agrees with each
`verify_adr_shape.expected_schema_version` (resolve `4.4`'s `2` vs `1`). Until
both hold, deterministic stamping is either dead metadata or an active
regression. Tracked here; not in this implementation.

---

## 2. Cut #2 — multi-produces contamination hardening (implement)

### 2.1 The bug

For a multi-produces step (e.g. ADR `4.1`: `[<decision>.json, register.md]`),
`materialize_produces` builds the same candidate list **for every file**:
`select_canonical([disk, output_value], _schema_ok)` (`hooks.py:324`).
`output_value` is the step's single LLM result (the decision content). So for
`register.md` the candidates are `[register_disk, decision_content]`, and
`select_canonical` can return the decision content — **overwriting `register.md`
with the decision artifact**:

- **Single-key ADR schema (4.1, 4.2):** `_schema_ok(register_md)` fails the
  object schema (one-line register text), `_schema_ok(decision_json)` passes →
  `select_canonical` returns the decision JSON (first passing form). Direct hit.
- **Multi-key ADR schema (4.4, 4.6, 4.8, 4.9, 4.10):** the decision JSON also
  fails (`validate_artifact_schema` checks *all* keys; the decision lacks
  `database_schema`/`tables` etc.), so neither passes — `select_canonical` falls
  through to the **most-substantial form** (`grounding.py:238`), i.e. the longer
  of register vs decision. The decision JSON is typically longer → `register.md`
  is still clobbered, via the length fallback.

Either way, `output_value` from one logical artifact can overwrite a sibling
file. Severity is masked today only by `validate_artifact_schema` looseness and
the absence of a phase-4 mission on disk to observe it, but the data-flow bug is
real for all 13 multi-produces steps.

### 2.2 The fix

In `materialize_produces`, make `output_value` a candidate **only for
single-produces steps**. `single` is already computed
(`hooks.py:308-309`). Change the per-file candidate list:

```python
candidates = [disk, output_value] if single else [disk]
chosen = select_canonical(candidates, _schema_ok)
```

- **Single (`single == True`):** unchanged — `[disk, output_value]`, and
  `canonical_out` becomes `chosen` (existing behaviour, `:340-341`).
- **Multi (`single == False`):** candidate list is `[disk]` only. `output_value`
  cannot leak into any file. Each file keeps its own agent-written disk content
  (a lone candidate is always returned by `select_canonical`, whether or not it
  passes `schema_ok`), is `mission_id`-stamped as today, and written back.
  `canonical_out` stays `output_value` (unchanged multi return contract).

No path↔schema-key mapping is needed (that primitive was only for Cut #1).
`select_canonical` with `[disk]` still unwraps a narration fence when the
unwrapped form passes schema, matching single-path parity.

**Fail-soft edge (unchanged):** a multi file with no disk content → disk is
`None` → `select_canonical([None])` returns `None` → `continue`; no file
written, no crash.

### 2.3 Why this is safe for `register.md`

`register.md` is appended to across sibling ADR steps (4.1, 4.2, 4.4, …). With
disk-only candidates, each step re-materializes `register.md` from the agent's
current on-disk content (which already contains the cumulative rows) and
re-stamps `mission_id` idempotently — never substituting another file's result.

---

## 3. Files touched

| File | Change |
|------|--------|
| `src/workflows/engine/hooks.py` | `materialize_produces`: candidate list becomes `[disk, output_value]` for single, `[disk]` for multi |

No changes to `grounding.py`, the expander, the schema gate, or i2p JSON.
(Cut #1 would have touched `grounding.py`; deferred.)

---

## 4. Testing (`src/workflows/engine` tests)

- **single-produces unchanged:** `output_value` still competes with disk; a
  richer/valid `output_value` is still selectable; `canonical_out` returns the
  chosen content. (regression lock)
- **multi-produces contamination fixed (core):** two produces files where
  `output_value` is the first file's artifact; assert the **second** file
  (`register.md`) retains its disk content even when `output_value` would pass
  the first file's schema (single-key case) AND when neither passes but
  `output_value` is longer (length-fallback / multi-key case). `output_value`
  must never appear in the second file.
- **multi-produces stamping:** each file still gets idempotent `mission_id`
  front-matter; cumulative `register.md` rows survive re-materialization.
- **fail-soft:** a multi file with no disk content is skipped, no crash, other
  files still written.
- **multi return contract:** `materialize_produces` returns `output_value`
  unchanged for the multi case.

---

## 5. Rollback

Single-function, single-line-region change in `hooks.py`. Revert the commit to
restore prior behaviour; no data migration, no schema/JSON changes.
