# Handoff — Grade gate validates canonical produces artifact (not result narration)

**Date:** 2026-06-25
**Status:** ✅ DONE — committed + pushed `origin/main 773127e4`, live-verified, mission 90 flowing.
**Trigger:** `❌ Task #567373 → DLQ [0.1] product_charter — degenerate repeat: identical output across attempts, not converging`

---

## 1. The bug (false-reject loop — the artifact was CORRECT)

Task 567373 (mission 90, i2p step `[0.1] product_charter`, agent `writer`) DLQ'd with
"degenerate repeat" at `apply.py:553` (the T7 `_retry_or_dlq` detector that hashes
`tasks.result` across attempts). But the produced artifact was **complete and correct** —
the DLQ was a false-reject the writer could never satisfy.

Two validators judged **different representations of the same artifact**:

| Validator | Reads | Verdict |
|---|---|---|
| `verify_charter_shape` (mr_roboto) | **disk** `workspace/mission_90/.charter/product_charter.md` | ✅ 7868 chars, all 5 `## ` sections |
| grade schema gate `apply.py` + LLM grader | **`tasks.result`** = writer prose narration | ❌ "missing all 5 sections" |

`tasks.result` was a 570-char narration ("Wrote `mission_90/.charter/product_charter.md`
(7.8KB) containing all five required sections…") with **zero `##` headers**.

## 2. Root cause

Schema'd `produces` markdown steps auto-strip `write_file` (`hooks.py:372`,
`_apply_auto_strip`). The writer therefore **cannot write disk** and instead **narrates**
its `final_answer`. `materialize_produces` writes the correct artifact to disk
**unconditionally** (`hooks.py:410`) — which `verify_charter_shape` and downstream
consumers (`coulson/context.py`) read. But the **grade chain** read `source.get("result")`
(the narration) at two points:
- deterministic schema gate: `apply.py` `_draft = source.get("result")`
- LLM grader: `grading.py:263` `result_text = source.get("result")`

→ both saw narration → "missing all 5 sections" → feedback the writer can't act on (its
file IS correct) → byte-identical re-emit → degenerate-repeat DLQ.

This is a recurrence of the canonical seam (`docs/handoff/.../produces-result-canonical-seam`,
memory `project_produces_result_canonical_seam_20260621`). That earlier fix **pushed**
canonical → `tasks.result` (`hooks.py:1706`, gated `_single_produces` AND
`output_value != _pre_mat`). Push is fragile — the grade chain refetches `source` fresh
from the DB (`apply.py:1628`) and saw the narration; the pushed value didn't survive every
persistence path.

## 3. The fix — pull, not push (uniform single source of truth)

**Principle:** for a step that declares `produces`, the materialized **disk canonical** is
the single source of truth; `tasks.result` is raw agent output, never the artifact. Every
artifact validator PULLS from the canonical. `verify_charter_shape` + downstream already
did — the grade chain was the lone push-dependent outlier. Repoint it to pull.

**Commit `773127e4` (3 files, +223, additive only):**

1. `src/workflows/engine/hooks.py` — new `resolve_produces_artifact(source, source_ctx)`:
   single-`produces` → read the produces path from disk (fence-unwrapped via
   `coulson.grounding.unwrap_fenced_artifact`, `{mission_id}`-substituted); else `None`.
2. `packages/general_beckman/src/general_beckman/apply.py` — grade branch (`if kind == "grade"`)
   resolves the canonical once and overrides the in-memory `source = {**source, "result": _canon}`
   → **one override** covers the schema gate + `build_grading_spec` LLM grader + the degenerate
   check. `None` → non-produces step, keeps `source["result"]` (JSON/array path unchanged).
3. `packages/general_beckman/tests/test_grade_canonical_produces.py` — TDD (RED→GREEN), 4 tests.

**Wiring verified:** grade chain `_advance_posthook_chain_locked` does `ctx = _parse_ctx(source)`
(full context, carries `produces`) → `_enqueue_posthook_llm_child(head, source, ctx)`
(`apply.py:1665`). 567373's stored ctx confirmed `produces=["mission_90/.charter/product_charter.md"]`.

**Tests:** 4 new + 24 beckman grade-gate + 73 root (grading/degenerate/charter/clarify) +
25 workflow-engine — all green. Import smoke OK.

## 4. Restart incident (same session — important operational note)

The fix is restart-gated. The user's soft `/restart` **did not cycle the orchestrator**
(it stayed the 10:36 PID; fix never loaded). Separately, **Nerd Herd** (a separate
`python -m nerd_herd --port 9881` process) had been **orphaned-dead since ~16:22** — no
crash trace; its restart-sidecar runs **inside** the orchestrator event loop, which had
wedged, so it stopped respawning Nerd Herd.

- **Stall mechanism:** after 567373 finished (16:25 → `ungraded`) and spawned the
  `self_reflect` grade child `570756`, the main LLM-dispatch loop produced **zero**
  `rank_candidates`/`begin_call`. The pump kept admitting only **mechanical** (no-LLM)
  tasks; `570756` sat in every Ready list but was never admitted; 213 dependents blocked.
- **Selection was fine without Nerd Herd** (16:22:41 ranked 45 candidates) — Nerd Herd was
  noise/`in_flight`-POST failures, not a selection block.
- **Recovery (user chose "force-kill orchestrator PID"):** process tree was
  wrapper(`30076`, already dead) → guard(`82320`, idle, CPU 0.03s) → orchestrator(`29840`,
  CPU 2294s, held the 9881 nerd_herd client). Killed leaf `29840`; **the guard `82320`
  cascaded down too** (its own parent was already dead) → whole tree down → manual relaunch
  `start_kutai.bat` (`.venv\Scripts\python.exe kutai_wrapper.py`; stale `logs/guard.lock`
  PID `30076` was reclaimed automatically). Fresh tree `60196→68492→85976→84072`, Nerd Herd
  respawned (9881, PID 70072), boot 17:38:55.

**GOTCHA for next time:** force-killing just the orchestrator leaf did **not** get it
respawned — the guard exited too (its supervisor parent was already dead). A full
`start_kutai.bat` relaunch was required anyway. If the tree looks half-supervised
(parent of the guard already gone), skip the leaf-kill and go straight to a full wrapper
relaunch.

## 5. Live verification

Fresh boot 17:38:55 → boot reconcile re-drove `ungraded` 567373 → grade read the canonical
disk charter → **PASSED** → `567373 = completed` (worker_attempts=1, no re-DLQ). Mission 90:
completed 44 → 57, 2 processing, former blocked dependents (567374 analyst, 567379 writer)
dispatching to gpt-oss. Nerd Herd `/api/*` calls succeeding, `rank_candidates`/`begin_call`
flowing.

## 6. Deferred (not scope-crept)

- **Rewriter seam (same root, different kind):** `self_reflect` / `constrained_emit`
  rewriters still read `source.result` (narration), not the canonical. It didn't bite 567373
  because the **terminal grade** gate now reads canonical, but a future markdown step whose
  rewriter mangles the narration (and re-materializes it) could resurface. Candidate: route
  the rewrite kinds through `resolve_produces_artifact` too.
- **Store-write divergence:** `materialize_produces` artifact-**store** write (`hooks.py:1708`)
  is still gated on `_single_produces` AND `output_value != _pre_mat`; downstream store
  readers could get a pre-materialize narration in edge cases. Candidate: make the store-write
  unconditional so store == disk == canonical.
- **Nerd Herd supervision:** the in-orchestrator restart-sidecar can't respawn Nerd Herd when
  the orchestrator event loop itself wedges. Consider supervising Nerd Herd from the guard
  (out-of-process), not from inside the orchestrator.

## 7. Quick references

- DLQ detector: `packages/general_beckman/src/general_beckman/apply.py:553` (`_retry_or_dlq`, T7)
- Grade branch (the fix): `apply.py` `if kind == "grade":` (~`_enqueue_posthook_llm_child`)
- Resolver: `src/workflows/engine/hooks.py::resolve_produces_artifact`
- Schema gate markdown check: `hooks.py:885` (`validate_artifact_schema`)
- Step def: `src/workflows/i2p/i2p_v3.json` step `0.1` product_charter
- Memory: `project_grade_gate_canonical_pull_20260625`
- Read-only DB inspect pattern: `sqlite3.connect("file:<DB>?mode=ro", uri=True, timeout=3)` + `PRAGMA busy_timeout=3000`
