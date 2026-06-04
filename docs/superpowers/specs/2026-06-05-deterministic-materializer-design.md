# Deterministic Materializer — sole writer of `produces` paths

**Date:** 2026-06-05
**Status:** Design (approved for planning)
**Predecessor:** `docs/handoff/2026-06-04-competitive-positioning-narration-to-file-FIXED.md` §3 (the moonshot). The §2 canonicalize-override + §4 autopersist-unwrap fixes (commit `056dc597`) are the first brick; this spec is the full structure.

---

## 1. Problem

The LLM owns two jobs today: **compose** artifact content (good at) and **materialize** it into the right file / shape / path / front-matter (flaky at). Every observed false-DLQ in this family is job #2 leaking to the model:

- **narration-to-file** — agent writes a "Findings / Recommendations" report to the produces path instead of the artifact (mission 81 #289715).
- **fence-burial** — the real document sits inside a ```` ```yaml ```` fence in `final_answer.result`; raw narration lands on disk.
- **wrong-path drift** — agent writes a sibling path, not the declared one.
- **forgotten front-matter** — `---` / `mission_id` missing, or buried inside a fence so `verify_*` (which needs `---` at file start) fails.

These survive because **four uncoordinated writers** touch the produces path and **two gates read different copies**:

| Writer | Location |
|--------|----------|
| Agent `write_file` (side-effect tool) | `coulson/react.py:1271-1407` |
| Auto-persist (rescue unwritten) | `coulson/react.py:820-864` |
| Canonicalize override (fix written-wrong) | `coulson/react.py:866-938` |
| Engine fill-missing + legacy root `.md` | `src/workflows/engine/hooks.py:1500-1557` |

And the divergence that closed mission 81 only by luck: **`validate_artifact_schema` validates the in-memory `output_value`** (`hooks.py:1631`), while **grounding / `verify_*` read the disk file** (`mr_roboto/verify_artifacts.py`). The loose markdown header scan in `validate_artifact_schema` finds `## Section` *inside a fence*, so a narration-wrapped result passes the in-memory gate yet fails the strict on-disk front-matter gate.

## 2. Goal

One **deterministic, no-new-LLM** pass that produces a **single canonical artifact** per declared produces path, used for **both** the on-disk file and the `output_value` the schema gate validates. After it, narration-to-file / fence-burial / wrong-path / forgotten-front-matter are **structurally impossible** for the covered artifact surface, because nothing else writes the produces path.

Non-goals (explicit follow-ups, §7): `_schema_version` stamping, multifile result→N-files mapping, tool-layer interception of `write_file`.

## 3. Placement

Replace engine hook blocks (1) legacy root-`.md` write and (2) fill-missing (`hooks.py:1500-1557`) with a single call:

```
materialize_produces(ctx, task, result, output_value) -> canonical_output_value
```

It runs at task completion in the workflow engine hook, **after** the agent finishes and **before**:
- the schema gate (`hooks.py:1559`), which will now validate the returned canonical content; and
- the independent grounding / `verify_*` post-hooks, which read the now-canonical disk file.

This is the only chokepoint every workflow producer passes through — coulson agents *and* mechanical executors. `mission_id`, `ctx["produces"]`, `ctx["artifact_schema"]`, `task`, `result`, `output_value` are all already in scope there.

## 4. Algorithm (per declared `.md` / `.json` produces path)

1. **Gather candidates.** The on-disk content (whatever the agent's `write_file` left), and `result`/`output_value`. For each, also compute its fence-unwrapped form via `unwrap_fenced_artifact` (already in `coulson/grounding.py`).
2. **Select.** Build `schema_ok` from `ctx["artifact_schema"]` via `validate_artifact_schema`. Prefer a candidate that **passes** `schema_ok`; among passers prefer the unwrapped / cleanest (front-matter or header at file start). If none pass, keep the most-substantial candidate.
3. **Stamp front-matter (idempotent).** If the chosen `.md` lacks `mission_id` in front-matter, inject it; if front-matter is absent, prepend a minimal `---\nmission_id: <id>\n---` block. **Skip entirely when correct front-matter already present** (handoff Q(c) — no double-stamp). `.json`: stamp `mission_id` as a top-level key only if absent. `_schema_version` deferred (§7).
4. **Write the canonical path last** — the sole write to the produces path. `os.makedirs` + UTF-8 write, fail-soft (log, never raise).
5. **Return the canonical content** as the new `output_value`, so the in-memory schema gate validates exactly what is on disk — closing the divergence.
6. **Best-effort guarantee.** A file always exists after this pass. If no candidate conformed, the schema gate / grade / `verify_*` fail normally and feed the retry rail with their precise, gate-specific feedback. The materializer is never a quality judge and never hard-fails.

### JSON specifics
Deterministic unwrap + parse + stamp happen here. If a JSON artifact **still** fails its schema after materialization, `constrained_emit` (LLM reshape) runs after, exactly as today — **deterministic-first, LLM-rescue**. `constrained_emit` is not folded in (it would put model dispatch inside the deterministic chokepoint).

## 5. What collapses

- **Remove** `react.py` auto-persist (`820-864`) and canonicalize override (`866-938`) blocks. Their behavior is subsumed by the engine materializer; the pure helpers (`unwrap_fenced_artifact`, `autopersist_candidate`, `recanonicalize_candidate`) are reused by `materialize_produces` (kept in `coulson/grounding.py`, imported by the engine — they are already pure and schema-injected).
- **Remove** the legacy mission-root `<name>.md` write (`hooks.py:1500-1513`).
- **Retire** `_produces_file_is_stale` (`hooks.py:272-300`) — schema-based candidate selection subsumes the junk-vs-rich heuristic.

Four writers → one. grounding / `verify_*` always read a canonical file; the schema gate always validates the same bytes.

## 6. Components & boundaries

| Unit | Responsibility | Depends on | Pure? |
|------|----------------|------------|-------|
| `materialize_produces(ctx, task, result, output_value)` | orchestrate gather→select→stamp→write→return for each declared path | grounding helpers, `validate_artifact_schema`, `WORKSPACE_DIR` | no (disk I/O) |
| `select_canonical(candidates, schema_ok)` | pick best candidate (pure) | grounding helpers | yes |
| `stamp_front_matter(content, mission_id, kind)` | idempotent mission_id stamp (pure) | — | yes |
| existing grounding helpers | unwrap / candidate decisions (pure) | — | yes |

The two new pure functions live alongside the grounding helpers (`coulson/grounding.py`) so they stay unit-testable without disk or engine imports; `materialize_produces` (the impure orchestrator) lives in the engine hook module and is the only piece touching disk.

## 7. Follow-ups (out of scope for v1)

1. **`_schema_version` stamping** — prerequisite: thread a version into each step's `artifact_schema` via the expander; then `stamp_front_matter` adds it. Tracked separately (touches i2p step defs).
2. **Multifile produces** — v1 validates+stamps each declared path *in place* (no unwrap-from-single-result, since one `result` can't map to N files). A result→multi-artifact splitter is a later effort.
3. **Tool-layer interception** — demoting `write_file` to write a staging path was considered and rejected for v1 (more plumbing, risk to non-workflow callers). "On-disk write = candidate" achieves sole-writer semantics without it.

## 8. Risks & mitigations

- **`output_value` rewrite breaks a downstream consumer.** The hook returns/propagates the canonical content; audit every reader of `output_value` after the gate (summary path, result router) before cutover. Mitigate: return canonical content only when materialization actually changed it; otherwise pass through unchanged.
- **Removing coulson in-loop recovery.** All workflow steps pass the engine hook, so coverage holds for them. Confirm no *non-workflow* coulson path (ad-hoc `/task`, single_shot) relied on auto-persist; if one does, keep a thin in-loop fallback or route it through the engine hook.
- **Mechanical producers.** Most mechanicals don't emit artifacts (bookkeeping), and the schema gate already skips them (`_is_producer`). Materializer must likewise no-op when `output_value` is empty / executor is mechanical — mirror the existing `_is_producer` guard.
- **Idempotent stamp correctness.** Front-matter detection must handle: no front-matter, correct front-matter, front-matter missing only `mission_id`. Unit-cover all three; never double-`---`.

## 9. Testing

- Pure functions (`select_canonical`, `stamp_front_matter`): TDD unit tests — narration-vs-artifact selection, unwrap preference, idempotent stamp (all three front-matter states), json key stamp, no-candidate-conforms → most-substantial.
- `materialize_produces`: integration test against a temp workspace — narration on disk + artifact in result → canonical file written, returned output_value matches disk, schema gate + a `verify_*`-style front-matter check both pass.
- Regression: the mission-81 #289715 fixture must pass end-to-end through the engine hook (not just the helper).
- Re-run full coulson + general_beckman suites; the pre-existing `test_z6_polish_detect_bail_e2e.py::test_detect_and_bail_no_llm_call` hang is unrelated (torch/asyncio) and remains deselected until separately fixed.

## 10. Rollout

Not live until KutAI `/restart` via Telegram. After cutover, re-run i2p phase 1 (or retry #289715) and confirm a single canonical write per produces path in logs (`materialize_produces -> <path>`), with no `auto_persist` / `recanonicalized` synthetic writes remaining (those code paths are gone).
