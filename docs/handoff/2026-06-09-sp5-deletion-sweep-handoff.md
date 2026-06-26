# Handoff — SP5 deletion sweep (delete `LLMDispatcher.request()` + `await_inline`)

**Date:** 2026-06-09
**For:** the session that finishes SP5 — deletes the deprecated `request()` shim and the `await_inline` blocking primitive.
**From:** the shopping-intelligence CPS migration session (Group 2 re-homed; `request()` reach in `src/shopping` now 0).
**Read first:** memory `project_shopping_cps_migration_20260609`, `project_shopping_sp5_group_findings_20260608`, `project_cps_sp4b_plan1_plan3_merged_20260608`; prior handoff `docs/handoff/2026-06-09-shopping-intelligence-wiring-handoff.md`.

---

## What just shipped (this session) — MERGED, NOT pushed, restart-gated

`src/shopping/intelligence/_llm.py::_llm_call` is now **inert** (`return ""`). It no longer calls `LLMDispatcher.request()`. All 13 shopping intelligence modules degrade to their existing rule-based fallback on `""` (verified per-module; see spec Appendix A). No tool dropped, no module deleted.

- Merge commit: **`fdc25a94`** (`merge(sp5): shopping intelligence CPS migration`). 4 commits. Worktree+branch removed.
- Spec: `docs/superpowers/specs/2026-06-09-shopping-intelligence-cps-migration-design.md`.
- Verified: `rg "\.request\(" src/shopping` → **0**. 760 shopping tests pass.
- **Two reviews:** review #1 FIX-FIRST caught that dropping the 4 LLM tools would strip a LIVE capability (they're in `deal_analyst`/`product_researcher`, reachable via `/compare`→`combo_research`) — a ruling-#1 violation; spec corrected to inert-seam-only. Review #2 SHIP.
- **NOT pushed; needs founder `/restart`** to load (live KutAI runs from `main`).

---

## What this session must do — the deletion order

`request()` can be deleted only when it has **zero callers** tree-wide. `src/shopping` is now clear; the remaining blockers are Group 3 + `single_shot.py`. Execute in this order:

### Step 1 — delete Group 3 (confirmed DEAD, posthook-child path is live)
- `packages/coulson/.../reflection.py::self_reflect` (was ~:88) — SP3b Task 7 moved self-reflection to a Beckman posthook child; zero live callers.
- `packages/coulson/.../constrained_emit.py::maybe_apply` (was ~:147) — same, moved to posthook child.
- **Re-grep callers before deleting** (line numbers are from 2026-06-08, may have drifted): `rg "self_reflect|maybe_apply" packages src` and confirm only defs + tests reference them.

### Step 2 — delete `single_shot.py` (dead, TEST-ENTANGLED)
- Dead, but **5 tests patch `coulson._single_shot_run`** — they must be rewritten/removed, not left dangling. Find them: `rg "_single_shot_run|single_shot" tests packages` .

### Step 3 — delete `request()` + helpers (the goal)
- `LLMDispatcher.request()` in `src/core/llm_dispatcher.py`.
- `_request_kwargs_to_spec` (its helper).
- `_task_result_to_request_response` — **re-grep for 0 callers first**, then delete.
- After this, `src/core/llm_dispatcher.py` drops ~190 LOC (the SP5-gated bulk noted in CLAUDE.md).
- **Re-grep tree-wide before deleting:** `rg "\.request\(|_request_kwargs_to_spec|_task_result_to_request_response" src packages` — must be 0 live callers (docstring mentions OK). If ANY live caller remains, STOP and characterize it — do not force the delete.

### Step 4 — delete `await_inline` (the SP5 finale)
- The blocking primitive. Confirm zero callers first (`rg "await_inline" src packages`). SP4a/SP4b migrated the known sites (reviews/crisis/incident/press_kit/demo_storyboard) off it; this session's shopping work removed the last shopping reach. **Re-grep — there may be carve-outs.** Memory `project_cps_sp4b_plan1_plan3_merged_20260608` notes "SP5 ledger = 2 carve-outs + shopping shim left"; the shopping shim is now gone — identify and resolve the **2 carve-outs** before deleting `await_inline`.

---

## Gotchas / discipline

- **Re-grep every symbol before deleting** — line numbers in this handoff are stale snapshots. Audit call sites, not docstrings (memory `feedback_audit_call_sites`).
- **Work on a git worktree.** Multiple parallel agent sessions advance `main` (today: `reviewer-failure-routing`, `resource-signals`, `prompt-foundry` all in flight). `main` was at `fdc25a94` after this merge — re-check divergence before your own merge.
- `timeout 120 pytest` always; **never** concurrent pytest (SQLite WAL lock crash-loops live KutAI); run `tests/` and `packages/*/tests/` in **separate** invocations.
- Bare `python -c` misses editable packages (coulson/husam/fatih_hoca) — validate via pytest, or use the venv python with cwd on the worktree.
- **Never `taskkill`** llama-server or KutAI. Live KutAI loads new code only on founder `/restart` (Telegram).
- The 2 things still pending from THIS session before SP5 is truly clear: (a) founder `/restart` to load `fdc25a94`; (b) optional `git push`.

---

## Out of scope (residuals, deliberately deferred — not SP5 blockers)

- **7 dead shopping intelligence modules** (`search_planner`, `return_analyzer`, `installment_calculator`, `combo_builder`, `substitution`, `special/complementary`, `special/used_market`): kept intact, inert LLM path, rule fallback live. Founder: "no need to wire them now." Wire as v3 producer triads if/when a real trigger exists.
- **`src/tools/vision.py`** calls `husam.run()` directly — non-compliant with ruling #1 (Beckman admission). It does **not** touch `request()`, so it does **not** block SP5. Separate compliance fix; deferred.
