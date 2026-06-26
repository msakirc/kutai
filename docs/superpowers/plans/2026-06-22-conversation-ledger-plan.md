# Implementation Plan — Rejection Ledger + Bounded Conversation Reset (v2)

**Spec:** `docs/superpowers/specs/2026-06-22-conversation-ledger-design.md`
**Resolves spec-review F1–F7 AND plan-review M1–M6.** Each task is TDD (failing test → watch fail → minimal impl → green). Foreground tests only, `timeout`, package tests `-o addopts="" -p no:aiohttp`, reap by PID (no background pytest — SQLite-orphan hazard). Restart-gated; commit per task; merge+push only after final review GREEN.

**Anchor hygiene (F7):** `retry.py` = `packages/general_beckman/src/general_beckman/retry.py`. Degenerate rejection = `src/workflows/engine/hooks.py:1535-1543`.

**KEY DESIGN DECISION (plan-review M1): the dispatch discriminator is `worker_attempts`, not exit-enumeration.**
`worker_attempts` increments once per quality re-dispatch (`apply.py:515` `attempts = worker_attempts+1`) and is UNCHANGED on a crash-resume of the same attempt. The checkpoint records the `worker_attempts` it was saved under. On `run()` entry:
- `checkpoint.saved_attempts == current worker_attempts` → SAME attempt interrupted mid-loop → **restore messages** (crash-resume, C4).
- `checkpoint.saved_attempts < current worker_attempts` → checkpoint is from a COMPLETED prior dispatch (a new attempt started) → **do NOT restore** (fresh + ledger + durable draft).
This needs NO terminal-exit enumeration (any new attempt mismatches) and NO new counter. It errs toward reset, which is safe: durable artifacts (C3) survive; only the (bloated) conversation is dropped.

---

## PHASE 1 — Ledger + durable same-step continuation (additive; does NOT touch the restore branch)

### T1 — Rejection ledger: append, bounded, persisted on every quality path
**Files:** `src/workflows/engine/hooks.py` (schema-fail write ~1774-1794; **degenerate branch ~1536**), `packages/general_beckman/src/general_beckman/apply.py` (verdict appliers 2903/2929/3032/3058/3165; `_retry_or_dlq` 502).
**Change:** add a shared helper `append_rejection(ctx, attempt, reason, out_hash)` that does `ctx.setdefault("_rejection_ledger", []).append({"attempt": int(attempt), "category": "quality", "reason": str(reason)[:500], "out_hash": out_hash})` (**M3/F5:** reason capped 500). Call it wherever a quality reason is written today.
- **M3 (degenerate persist):** the degenerate branch (`hooks.py:1536-1543`) returns BEFORE the `update_task(context=…)` persist at ~1789 — an in-memory append is LOST. Add an explicit `await update_task(task["id"], context=json.dumps(ctx))` (or route through the existing ctx-persist) at the degenerate site so the ledger entry survives. reason = `"degenerate: <cq.signature>"`. Degenerate stores no artifact → **ledger-only, no durable draft (F3)**.
- **Empty-completed** (`result_router.py:138`): append a ledger entry `reason="empty result"` too (M-note: this path also never sets `_schema_error`).
**Test (`tests/test_rejection_ledger.py`, pure ctx in/out, no DB):** `append_rejection` 3× (schema, grade, degenerate) → 3 stamped entries, each reason ≤500, survives `json.dumps`/parse; availability path appends none. RED first.

### T2 — Render the ledger, OUTSIDE the `_schema_error` guard (M4)
**Files:** `packages/coulson/src/coulson/context.py` (build_user_context; **NOT** nested under the `if task_context.get("_schema_error"):` block at ~952 — the target bloat paths never set it).
**Change:** new independent block, gated on `ctx.get("_rejection_ledger")` with ≥2 entries: render `## Prior attempts (do not repeat):` + one whole line per entry `- attempt N: <reason>`; if the block would exceed ~2000 chars, drop OLDEST whole entries (no byte-slice — C1/F5). Single-entry ledger → no header (the existing checklist covers it).
**Test:** ctx with 3-entry ledger (no `_schema_error`) → context contains all 3 reasons + header, whole-line; 1-entry → no header.

### T3 — Same-step durable draft read-back (M2/F4), OUTSIDE the `_schema_error` guard
**Files:** `packages/coulson/src/coulson/context.py`.
**Change:** on retry (`worker_attempts>0`) of a workflow step, inject the step's OWN prior artifact(s) WHOLE:
- import `from src.workflows.engine.hooks import get_artifact_store, extract_output_artifact_names`.
- names = `extract_output_artifact_names(task_context)` (**M2: artifact NAMES, not `ctx["produces"]` paths**). For each name, `store.retrieve(mission_id, f"{name}_summary")` then fallback `store.retrieve(mission_id, name)` (mirror `_fetch_deps` `_summary`/bare fallback, context.py:577-586).
- inject `## Your prior draft (continue/fix — do NOT restart):\n<val>` WHOLE (no truncation, C1).
- **Dedup (M-Phase1 regression guard):** skip any name already injected by the deps block (`input_artifacts`) — same-step produces are not upstream deps, so normally disjoint, but guard anyway.
- **Scope:** only when `extract_output_artifact_names` is non-empty AND mission_id present → non-workflow tasks and degenerate (no stored artifact) get NO draft block (fall back to ledger / `_prev_output`).
**Test (mock `get_artifact_store`):** retry with stored `competitive_positioning` (8k) → full 8k draft, no `[truncated]`; no stored artifact → no draft block; name collision with deps → injected once.

### T4 — C1 truncation-site audit; demote `_prev_output` to fallback
**Files:** `hooks.py:1784` `[:6000]`; `apply.py` `[:6000]` ×~12 (2906,2932,3035,3061,3168,3194,3323,3345,3617,3639,3823,3845); `context.py:1059` `[:4000]`.
**Change:** these now feed ONLY the no-durable-artifact fallback. Comment each `# fallback only — artifact-backed continuation reads full draft (T3)`. Deliver the site table in the commit message.
**Test:** artifact-backed retry never hits `[:N]` (full draft present); pure-conversation output still uses the capped fallback.

**Phase-1 gate:** additive only; restore branch untouched. Commit T1–T4. Regression: `test_prompt_noise_reduction.py`, `test_recency_reorder.py`, new ledger/render/draft tests.

---

## PHASE 2 — Attempt-stamped checkpoint + bounded reset (the bloat kill; dedicated review)

### T5 — Stamp `saved_attempts` on the checkpoint (M1)
**Files:** `packages/coulson/src/coulson/checkpoint.py` (`save_checkpoint` signature + serialized `state` dict + `load_task_checkpoint` surfaces it), `packages/coulson/src/coulson/react.py` (pass `worker_attempts` at every `save_checkpoint` call — the value is already in scope as `reqs`/task context).
**Change:** add `saved_attempts: int = 0` to the checkpoint state; every save stamps the CURRENT `worker_attempts`. `task_state` is a JSON column → round-trips, **no migration**. No terminal-exit enumeration needed (the attempt value already discriminates).
**Test (`tests/test_dispatch_boundary.py`, pure):** a saved checkpoint carries `saved_attempts`; load surfaces it.

### T6 — Pure `should_restore_messages` helper + gate the restore (M5 + M1)
**Files:** `packages/coulson/src/coulson/react.py` (extract a pure helper, mirror `reqs_for_run` at :204; apply at the restore branch 260-298).
**Change:**
```python
def should_restore_messages(checkpoint: dict | None, current_attempts: int) -> bool:
    if not checkpoint: return False
    # same attempt interrupted mid-loop → resume; older attempt → fresh
    return int(checkpoint.get("saved_attempts", 0)) >= int(current_attempts)
```
Restore `messages` only when `should_restore_messages(...)` is True; else rebuild fresh (system + context + ledger + durable draft). Keep `reqs` always-fresh (:291). The existing `_schema_error` skip (:260) becomes redundant but stays (harmless).
**Test (pure unit, no `run()`/DB — M5):** completed-prior (`saved_attempts < current`) → False (fresh); same-attempt interrupt (`==`) → True (resume); no checkpoint → False. Plus an integration-lite assertion that the F1 trigger shapes (degenerate/empty/post-availability, all with incremented worker_attempts) → False.
**Success:** C4 (crash-resume preserved when attempt matches); the 775k accumulation cannot recur (any re-attempt mismatches → fresh).

**Phase-2 gate:** commit T5–T6. Run dispatch_boundary + restore-helper tests, full coulson react suite, `test_prompt_noise_reduction.py`, `test_recency_reorder.py`. **Dedicated review before merge.**

---

## PHASE 3 — Repeat-detector + escalation

### T7 — Identical re-attempt → escalate via `_dlq_write` (F6)
**Files:** `packages/general_beckman/src/general_beckman/apply.py` (`_retry_or_dlq` ~502).
**Change:** compute `out_hash` of the current attempt's output (the refetched `task` result, hashed); compare to the immediately-prior `_rejection_ledger` entry's `out_hash` (stored in T1). If identical → `_dlq_write(task, error="degenerate repeat: identical output, not converging", category="quality", attempts=attempts)` (apply.py:685, founder notice) instead of re-dispatch. **Limit (F6):** exact-hash only; semantic near-dupes out of scope.
**Test:** 2 identical outputs → `_dlq_write` called, no re-pend; 2 different → normal retry.

---

## PHASE 4 — Live verification (M6)

### T8 — Prove prompt size stays flat across re-dispatches
**No code.** After Phase 2 is live (post-restart, user-driven), query the existing `messages state` telemetry (react.py:464) for a re-dispatched task: assert `assistant` char total does NOT grow across dispatches (vs the 709,922c baseline). Document the before/after in the final report. (Pre-merge proxy: a test that re-runs the restore decision N times and asserts the rebuilt message set is bounded.)

---

## Cross-cutting
- Keep any conversation bound < 64k (nest under parallel `est_in` clamp + rollup filter).
- After Phase 2: `packages/fatih_hoca/tests/sim/run_scenarios.py` (no estimate regression).
- Commit per task, conventional commits, Co-Authored-By footer.
- **Rollback discipline:** P1 (hooks/context) and P3 (`_retry_or_dlq`) are independent of P2 (checkpoint/restore). If Phase-2 review is not clean → commit P1+P3, HOLD P2, do not push P2.

## Success criteria
1. Re-dispatch prompt bounded — no cross-dispatch assistant accumulation (T6 unit + T8 telemetry).
2. Distinct prior reasons (X,Y) visible to attempt 3 (T2 ledger).
3. Full prior draft carried whole on same-step retry; no byte-slice on the live path (T3/T4).
4. Crash-resume (same `worker_attempts`) still restores messages (T6).
5. Identical-repeat escalates, not re-dispatches (T7).
6. All existing react/context/prompt-noise tests green.
