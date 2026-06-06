> **RESOLVED 2026-06-06.** All 5 fixed, pushed to `main`, full yalayut+intersect
> suites green (261 passed). #3 and #5 were **not product bugs** (doc drift / stale
> test from intentional Phase-3 changes); #1 dead-code delete; #2 + #4 real fixes.
> #2 (`/yalayut mcp kill|restart`) is live only after KutAI restart. See memory
> `project_readme_sweep_bugs_fixed_20260605` for the per-bug breakdown.

# Handoff — bugs surfaced during the README gold-standard sweep (2026-06-05)

While writing gold-standard READMEs for 14 packages (commit `06762ef0`), the
per-package analysis agents read each package's source against its tests and
surfaced 5 latent code bugs. The README work was read-only — **none of these were
fixed**. This handoff is the investigation queue.

Confidence note: findings #1, #2, #4 are confirmed at the cited file:line in this
session. #3 and #5 are agent-reported and need first-hand confirmation.

Suggested order: **#1 → #2** (functional dead wiring) → **#3** (latent
correctness) → **#5** (test/env) → **#4** (typing only).

---

## #1 — `workflow_engine`: `next_subtasks` is a permanent dead no-op  [HIGH]

**Symptom.** `advance()` never emits next-phase subtasks. The phase-transition
emission path is silently disabled; phase logic falls back to the caller forever.

**Evidence (confirmed).** `packages/workflow_engine/src/workflow_engine/advance.py:98`
```python
from src.workflows.engine.recipe import advance_recipe   # <- target does not exist
next_subs = await advance_recipe(mission_id, completed_task_id, ...)
...
out.error = f"advance_recipe: {e}"[:300]   # ImportError is swallowed here
```
`advance_recipe` is **not defined anywhere** — grep across
`packages/workflow_engine/src` and `src/workflows/engine` finds only this import
site, no definition. So the `try` raises `ImportError` every call, the `except`
sets `out.error` and returns empty `next_subs`. `next_subtasks` is always `[]`.

**Hypothesis.** Either (a) `advance_recipe` was never implemented / was removed and
the import is stale, or (b) phase-transition emission was deliberately moved to the
caller and this branch is dead code that should be deleted. The swallowed
ImportError hides which.

**Investigation.**
1. `git log -S advance_recipe -- src/workflows/engine` — did it ever exist? renamed?
2. Find who consumes `AdvanceResult.next_subtasks` — does any caller depend on it
   being populated, or does the real phase-transition happen elsewhere (expander /
   `general_beckman`)? If the latter, this is dead code → delete the branch.
3. If phase emission is genuinely expected here, `advance_recipe` must be
   implemented (or the import repointed to the live primitive).

**Open question.** Is multi-phase subtask emission supposed to live in this package
at all? The package is a ~250-line façade; the engine internals live in
`src/workflows/engine/*`. The module docstring hints at a future migration.

---

## #2 — `yalayut.admin`: MCP kill/restart import non-existent functions  [HIGH]

**Symptom.** `admin.mcp_kill` and `admin.mcp_restart` raise `ImportError` at call
time. 4 tests fail (`packages/yalayut/tests/test_admin_phase3.py`).

**Evidence (confirmed).** `packages/yalayut/src/yalayut/admin.py:318` & `:329`
```python
from yalayut.plugins.mcp import restart_process   # :318
from yalayut.plugins.mcp import kill_process       # :329
```
`yalayut/plugins/mcp.py` defines `_slug`, `_embed`, `_cosine`,
`rank_tools_by_intent`, `enforce_step_budget`, `to_application`, `bind_args` — and
**neither `kill_process` nor `restart_process`**.

**Hypothesis.** The MCP process-control helpers were renamed, moved, or never
implemented; the admin handlers were wired against the intended names. Likely the
process-lifecycle logic lives in a different module (an `mcp_manager` /
`mcp_executor`?) and the import path is wrong.

**Investigation.**
1. `grep -rn "def .*\(kill\|restart\|stop\|terminate\).*process\|mcp.*kill" packages/yalayut/src` — find the real process-control functions.
2. If they exist under another name/module → fix the two imports in `admin.py`.
3. If they don't exist → implement them in `plugins/mcp.py` (the admin handlers and
   tests already define the expected contract — read `test_admin_phase3.py` for the
   intended signature: `await kill_process(artifact_id)` / `await restart_process(artifact_id)`).

---

## #3 — `dallama`: `load_timeout` raises the ceiling but docstring says floor  [MEDIUM]

**Symptom (agent-reported, confirm first).** `load_timeout` is applied as a `max`
(raises the timeout ceiling) but `server.py`'s docstring says it should be a `min`
(floor). Behavior and doc disagree.

**Evidence (agent).** `packages/dallama/src/dallama/server.py` — roughly
`if load_timeout > timeout: timeout = load_timeout`, while the docstring describes
a minimum. Also reached via `infer()` / `swap()`.

**Investigation.**
1. Read `server.py` `load_timeout` handling + its docstring; confirm the
   max-vs-min contradiction first-hand.
2. Decide intent: is `load_timeout` meant to **extend** time for slow first-loads
   (ceiling, current behavior) or **guarantee a floor**? Pick one, fix the other
   side (code or doc). The README documents current (`max`) behavior — update it if
   the code changes.

---

## #5 — `intersect`: prebind seed-convention test fails  [MEDIUM-LOW]

**Symptom (agent-reported, confirm first).** `test_binding.py::test_flash_produces_prebind_envelope_with_seed_convention`
fails: "expected at least one skills entry." Working tree was clean (not caused by
the README work).

**Evidence (agent).** `packages/intersect/tests/test_binding.py`. The seed-convention
prebind path produced an empty skills envelope.

**Investigation.**
1. Run in isolation: `& .\.venv\Scripts\python.exe -m pytest packages\intersect\tests\test_binding.py::test_flash_produces_prebind_envelope_with_seed_convention -q`.
2. **Rule out env first** — `intersect` → `yalayut.query` → embeddings. A
   wrong/missing embedding model makes `_query_engine._cosine` return `0.0` on
   length mismatch → empty catalog → empty envelope (this is also `intersect`'s
   documented #1 gotcha). Confirm the embedding model is loaded in the test env.
3. If env is fine, it's a live bug in the prebind/seed-convention binding path.

---

## #4 — `yasar_usta`: `GuardConfig.on_exit` type annotation is wrong  [LOW / cosmetic]

**Symptom.** Misleading type annotation; no runtime effect.

**Evidence (confirmed).** `packages/yasar_usta/src/.../config.py:131`
```python
on_exit: None = None  # callable(exit_code: int) -> None, called after process exits
```
Annotated `None`, but `guard.py` calls `self.cfg.on_exit(exit_code)` and
`kutai_wrapper.py` passes a callable.

**Fix.** `on_exit: Callable[[int], None] | None = None` (add `from typing import
Callable`). Trivial; do it when touching this file for anything else.

---

*Source: README gold-standard sweep, 2026-06-05. See
`docs/superpowers/specs/2026-05-29-readme-standard-design.md` and memory
`project_readme_gold_standard_20260605`.*
