# Handoff — Deterministic Materializer shipped + LIVE-ERRORS triage after restart

**Date:** 2026-06-05
**Author:** prior session (materializer build)
**Status:** Materializer fully implemented + tests green; **KutAI restarted by founder → MANY live errors observed → investigation deferred to next session.**

---

## 0. READ FIRST — the live errors are probably NOT (only) the materializer

A **large batch** of restart-gated work from *several* sessions all went live in this one restart. Any of it could be the source of "many errors." Do **subsystem triage before touching code**. Today's landed-and-now-live commits (newest first):

| Commit | Subsystem | Live-risk |
|--------|-----------|-----------|
| `4a15fbec` `5602e259` `585a950c` | **Materializer** (this session) — sole writer of produces paths | MED — new write path, removed 3 writers |
| `0ea97a15` | **Beckman schema-gate** — deterministic artifact-schema gate before LLM grade | **HIGH — "#1 broad tightening: 240 schemas now hard-checked" (memory). Previously-unenforced schemas now hard-fail → could DLQ many tasks live. Memory explicitly says "live smoke test required post-restart."** |
| `1bd6dece` | emit — fire constrained re-emit on incomplete drafts (not just missing keys) | MED — changes emit firing frequency |
| `4847aa01` | retry — escalate failed model on ALL quality re-pends | LOW/MED |
| `789ebac4` | grading — removed all LLM-I/O truncation | MED — larger prompts now |
| `19f23aa1`+`3d9ac045`/`33aafb52`/`bed84231`/`94a42f86` | **CPS SP4a** — vision/brand_voice_lint/copy_compliance/yalayut now `husam.run` inline | MED — needs `husam` importable at runtime (see §1) |
| `3749a31a` | dispatcher Phase-4 de-accretion (evicted telemetry/dead code) | MED — touched the hot dispatch path |
| `cd51894a`/`63b11459`/`3566f595`/… | **nerd-herd S7/S6 continuity** — selection signal reshape | MED — could shift/break model selection live |

**First triage step (do this before any fix):** pull the live error signatures and bucket them.
```
rtk read logs/orchestrator.jsonl   # or: grep -E "Traceback|ERROR|DLQ|schema validation|materialize|husam|ModuleNotFound" logs/*.jsonl | tail -100
```
Then match the dominant error class to the table above. **Do not assume it's the materializer.** Likely-first suspects by blast radius: **schema-gate 240-tightening** (mass DLQs with "schema validation" / "missing sections"), then SP4a (`husam` import / raw_dispatch), then materializer (artifact write/disk issues), then S7/S6 (selection / "no model candidates").

**husam check already done:** `.venv/.../python -c "import husam"` → OK (resolves to `packages/husam/src`). So "husam missing" is *probably* ruled out — but confirm the **running orchestrator** can import it (the live process may use a different interpreter/path than the test venv). If you see `ModuleNotFoundError: husam` in the live log, run `.venv\Scripts\pip install -e packages\husam` and restart.

---

## 1. What this session shipped (the materializer)

Goal: make ONE deterministic engine pass the **sole writer** of declared `produces` paths, so narration-to-file / fence-burial / wrong-path / missing-front-matter become structurally impossible. Closes the mission-81 #289715 class.

**Spec:** `docs/superpowers/specs/2026-06-05-deterministic-materializer-design.md`
**Plan:** `docs/superpowers/plans/2026-06-05-deterministic-materializer.md`

### Commits (this session, in order)
| Commit | What |
|--------|------|
| `056dc597` | **§4 fix** (the "first brick", was uncommitted from the prior session): `autopersist_candidate` gained `schema_ok` to prefer a fence-buried artifact over a narration wrapper; bundled with the §2 `recanonicalize_candidate`/`unwrap_fenced_artifact` + `test_recanonicalize.py`. |
| `aa6d6e79` | Materializer design spec |
| `0726f206` | **pytest 120s per-test timeout** (`pytest-timeout`, `thread` method) in `pytest.ini` — so a slow test fails fast with a stack instead of looking like an infinite hang |
| `10adb19b` | Implementation plan |
| `585a950c` | `grounding.py`: `stamp_front_matter()` + `select_canonical()` (pure, TDD — `test_materializer_helpers.py`, 10 tests) |
| `5602e259` | `hooks.py`: `materialize_produces()` orchestrator + `test_materialize_produces.py` (5 tests incl. mission-81 #289715 regression) |
| `4a15fbec` | **Wire it + collapse writers** (the risky one — see §3 rollback) |

### The mechanism (what changed in the live write path)
- `src/workflows/engine/hooks.py` `post_execute_workflow_step`: the old **legacy root-`<name>.md` write** + **fill-missing produces block** are replaced by a single `output_value = await materialize_produces(ctx, task, result, output_value)` call, run **before** the schema gate. A root-`<name>.md` write is **kept only as a fallback for steps with NO declared `produces`**.
- `materialize_produces`: for each declared `.md`/`.json` produces path, gather candidates `[disk_content, output_value]`, pick the schema-best via `select_canonical` (disk outranks output_value; unwrapped-fence form preferred within a source), stamp `mission_id` front-matter idempotently via `stamp_front_matter`, write the canonical path **last**, and **return the canonical content as the new `output_value`** so the in-memory schema gate validates exactly what is on disk. Fail-soft (never raises). Reads `WORKSPACE_DIR` dynamically from `src.tools.workspace`.
- `_produces_file_is_stale()` **deleted** (subsumed by schema-based selection); its test file deleted.
- `packages/coulson/src/coulson/react.py`: the in-loop **AUTO-PERSIST** + **CANONICALIZE OVERRIDE** blocks removed (engine owns it now); dead imports dropped. Replaced with a comment pointing to `materialize_produces`.

### Tests (all green at ship)
- coulson + workflows: **217 passed** (single run, 120s-bounded).
- `test_materializer_helpers.py` (10), `test_materialize_produces.py` (5), `test_produces_persist.py` (2, the intake-#73 guard — still green).

---

## 2. Deferred items (materializer)

**Intentional scope cuts (spec §7):**
1. **`_schema_version` stamping** — materializer stamps only `mission_id`. Needs a version threaded into each step's `artifact_schema` via the expander first (touches i2p step defs).
2. **Multifile produces** — v1 stamps/validates each declared path but cannot map one `result` to N distinct artifacts. A result→multi-artifact splitter is later.

**Risks flagged but NOT audited (spec §8) — these could relate to live errors:**
3. **Non-workflow coulson paths** (ad-hoc `/task`, `single_shot`) that relied on the removed in-loop auto-persist. Workflow steps are fine (all hit the engine hook); a non-workflow producer would have **lost** its inline artifact recovery. **If live errors mention missing artifacts / grounding DLQ on ad-hoc tasks, this is the cause.**
4. **`output_value` rewrite consumers** — every reader of the post-gate `output_value` (summary path, `result_router`, post-hook chain) now receives the *canonicalized* content. Not audited. Mitigated (only changes when materialization actually changed it).
5. **Legacy root-`.md` deviation** — produces-having steps **no longer** get a mission-root `<name>.md` (only produces-less steps do). **If any downstream consumer reads `mission_<id>/<name>.md` for a step that declares a produces path, it now finds nothing.** Highest-suspicion materializer regression — check this first if errors point at the materializer.

**Cleanup left open:**
6. `autopersist_candidate` + `recanonicalize_candidate` in `grounding.py` are now **dead** (react blocks removed) — kept with their tests (`test_autopersist_candidate.py`, `test_recanonicalize.py`). Remove-or-keep is an open call (they're pure + tested + potentially reusable by a future in-loop fallback for risk #3).

**Pre-existing, out of materializer scope:**
7. `validate_artifact_schema` markdown branch still finds `## Section` *inside* fences (handoff §4 weakness). Materializer dodges it by unwrapping; the validator itself stays loose. A future fold could make it fence-aware.
8. `hooks.py:~365` DeprecationWarning (invalid escape `\`` in a docstring) — trivial, pre-existing.

---

## 3. If the materializer IS the culprit — how to confirm + roll back

**Confirm:** live errors that name produces paths, `materialize_produces`, missing `mission_<id>/<name>.md`, schema validation on artifacts that used to pass, or grounding/verify DLQ on artifacts that exist on disk.

**Surgical rollback (keeps the pure helpers + tests, restores old write behavior):**
```
git revert 4a15fbec    # un-wires the materializer; restores react auto-persist/canonicalize + engine fill-missing
```
`4a15fbec` is the only commit that changed live behavior by removing writers. `585a950c`/`5602e259` only ADD pure functions + a non-wired orchestrator + tests — harmless to leave. Reverting `4a15fbec` returns to the §2/§4 state (which was itself a real improvement over the original).

**Partial mitigation (keep materializer, restore root write):** if risk #5 is the issue, re-add the unconditional root-`<name>.md` write alongside `materialize_produces` (drop the `not (ctx.get("produces") or [])` guard in `hooks.py`).

---

## 4. Other deferred / known-failing (NOT materializer)

- **`general_beckman/tests/test_posthook_llm_child.py::test_emit_child_spec_is_raw_dispatch`** — FAILS (`enqueue` awaited 0 times). Caused by the prior schema-gate / `should_skip_emit` change (`0ea97a15`/`1bd6dece`), **not** the materializer (my commits never touch `apply.py`). general_beckman is otherwise **243 passed**. Either the test is stale vs the new skip behavior, or emit genuinely regressed — worth confirming since `1bd6dece` is live.
- **Schema-gate 240-schema tightening** (`0ea97a15`) — memory flagged "live smoke test required post-restart." Top suspect for mass DLQs. See `[Schema-gate shipped 2026-06-05]` memory.
- **CPS SP4b** (`1ae7d003` kickoff) — extract LLM out of 6 mr_roboto executors; pump-gated; not started.

---

## 5. Validation once stable

After the live errors are triaged/fixed, validate the materializer end-to-end:
- Re-run i2p phase 1 (or retry DLQ #289715).
- Confirm `materialize_produces -> <path>` log lines appear, and that NO `auto_persist` / `recanonicalized` synthetic `write_file` audit entries remain (those code paths are gone).
- `SELECT lane,status,COUNT(*) FROM tasks GROUP BY 1,2` — watch for a DLQ spike correlated with the restart timestamp; bucket by error_category to attribute to the right subsystem from §0.
