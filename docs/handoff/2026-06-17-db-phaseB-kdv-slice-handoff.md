# Handoff: DB Phase B §4 — kdv_state slice DONE + remaining-work reassessment

**Date:** 2026-06-17. **Predecessor:** `docs/handoff/2026-06-17-db-phaseB-deferred-remaining-handoff.md`.

## What shipped (MERGED to `main`, NOT pushed — restart-gated)

| Slice | Commit | Notes |
|---|---|---|
| §4 kdv_state → kuleden_donen_var | `21abfee3` (merge `7b677b4d`) | Full §0-pattern relocation. Reviewed GO. |

`kdv_state` (KDV persistent rate-limit state) relocated out of the `dabidabi` engine into its owner package `kuleden_donen_var`:
- NEW `kuleden_donen_var/schema.py` — owns `kdv_state` DDL + `create_kdv_schema`, registered via `dabidabi.register_schema("kuleden_donen_var_kdv", ...)`.
- NEW `kuleden_donen_var/persistence.py` — `save`/`load`/`load_sync` relocated from the **deleted** `src/infra/kdv_persistence.py`; now imports `from dabidabi import connect_aux, connect_aux_sync` (off the `src.infra.db` alias shim) + stdlib `logging` (drops a package→src up-reach). Logic byte-equivalent to the old module (reviewer diffed line-by-line).
- `dabidabi/__init__.py` — inline `CREATE TABLE kdv_state` **removed** (single source now in owner), pointer comment left.
- `kuleden_donen_var/__init__.py` — `from . import schema, persistence` so importing the package registers the schema.
- 4 callers repointed → `from kuleden_donen_var import persistence as kdv_persistence` (run.py, rate_limiter.py, mr_roboto/kdv_persist.py, tests/infra/test_kdv_persistence_outcomes.py).
- Cold-init imports added (`import kuleden_donen_var`) to run.py top + dlq_feedback.py + mission_lessons.py → fresh DB still creates kdv_state via init_db's registration loop.

**Live-restart safe:** prod kutai.db already has kdv_state; removing inline DDL only affects fresh DBs, which the cold-init imports cover. A fresh-DB init_db guard test proves the registration path materialises the table.

**Verified green:** 189 (kuleden + db-pkg + relocated test, incl. main's new phantom-remaining test). Adversarial subagent review = **GO**, no blockers/should-fixes. Merge into main (`cf669ad6` → `7b677b4d`) conflict-free (zero file overlap with main's 6 concurrent commits: FC-gate, GGUF-scan, surface-derive, kdv phantom-remaining −1.0).

### Optional nits (NOT blockers, NOT done — pre-existing)
- `tests/infra/test_kdv_persistence_outcomes.py` inlines a 3rd copy of the kdv_state DDL (`_create_kdv_state_table`). Could import `create_kdv_schema`, but its fixture is sync sqlite3 and `create_kdv_schema` is async → conversion adds churn for no functional gain. Left as-is.
- The two new in-package tests assert on `dabidabi._registered_schemas` (a private). Acceptable for an ownership test.

## USER restart-gate
`/restart` + verify: boot smoke shows NO `no such table: kdv_state`; KDV state restores on boot (log line `kdv state restored (sync): models=... providers=...`); then push the stack.

---

## Remaining work — REASSESSED this session (important: the clean-leaf well is DRY)

After kdv, I audited ALL ~80 engine tables + every §4/§5 item. Findings:

### §5 engine up-reaches — ALREADY DONE
No real `from src.` imports remain in `dabidabi/__init__.py` (grep clean; only comments). §5a hooks (`dabidabi.hooks` + `db_hooks.wire`) already inverted all 5 service reaches. Nothing left in §5 tail.

### §2 reads — DONE (predecessor `4b743458`). §3 cold-init, §6 doc — DONE.

### §4 domain splits — only ledger substantive, and it's NOT a clean extraction
- `kdv_state` → ✅ DONE.
- **`ledger`** (`cost_budgets` + `model_call_tokens` + `cost_by_iteration` VIEW, 19 helpers ~1979 LOC) — **DEFERRED BY USER 2026-06-17.** Reassessment corrected two earlier claims:
  - **No `_get_tx_lock` coupling** (the scout/earlier-handoff claim was a mis-read; all helpers use plain `get_db()` + `commit()`).
  - BUT it is **NOT a clean leaf** — it crosses 3 beckman boundaries: (1) `cost_by_iteration` VIEW `JOIN tasks`; (2) helpers `UPDATE tasks SET estimated_cost_usd/actual_cost_usd` → would **violate beckman's AST write-guard** (beckman owns tasks writes); (3) `mission_budget_alerts` is in `MISSION_SCOPED_TABLES` (green-tag rollback atomicity that must stay in beckman's one file). **→ A new `ledger` package is WRONG (breaches all 3). If it ever moves, the only coherent owner is `general_beckman`.** It is also hot-path (every LLM call records tokens/cost via hallederiz_kadir + husam) → supervised + restart-verified, NOT an AFK move.
- `yalayut` — schema already owned by `yalayut/schema.py`; init_db calls it via a guarded lazy import + does seed-load. Converting to `register_schema` is **net-negative** (forces a heavy `import yalayut` at run.py boot for registration, and the seed-load up-reach stays regardless). Leave as-is.
- `core.db` spine (~60 tables: missions/tasks/comms/email/marketing/growth/…) — intentionally STAYS one file (beckman rollback atomicity).
- `product.db` (comms) — BLOCKED by user (no owner, unvalidated).
- **Owner-ambiguous log tables** (`paraflow_diff_log`, `admission_violations`, `critic_log`, `streaming_guard_log`, `step_token_stats`, …) — each would need a heavy owner package imported at boot just to register a trivial table (same net-negative as yalayut). Checked `paraflow_diff_log`: engine holds ONLY its `CREATE`; all access already external (mr_roboto INSERT + telegram SELECT, inline SQL); owner would be mr_roboto (heavy). **Not worth it — this is why §4 never listed them.**

**Net: the safe, clean §0-pattern extractions are exhausted with kdv.** The real remaining work is all big/gated.

### The actual remaining big items (all supervised, none AFK-safe)
1. **§1 registry.db physical file-split** — the original moonshot payoff (actually splits a file out of core.db). Prereqs unchanged from predecessor §1: build the `attach_db()`/`attached()` engine primitive, unify the 3 connection openers (`fatih_hoca/registry_store.py` sync singleton, `connect_aux[_sync]` per-call, async `get_db()` singleton) to all ATTACH the split file, qualify registry CREATEs with `registry.` prefix, then a crash-safe per-table guarded migration (cross-file atomicity dies under WAL — no single transaction). Migration + restart gated.
2. **Ledger → general_beckman** — see above. Hot-path, supervised.
3. **§5b-db shim delete** (`src/infra/db.py`) — **RE-MEASURED: 304 importer files** (predecessor said ~100), incl. **22 relative-form landmines** across ~18 files (telegram_bot god-file ×2, base.py, orchestrator.py, state_machine.py, the `src/infra/` siblings using bare `from .db import get_db`: audit/metrics/progress/tracing/projects/artifacts_register, and `src/security/*` ×7) + 3 string/sys.modules stubs. **Pure hygiene, HIGH boot-break risk. Both the handoff and this session recommend AGAINST** unless the churn is deemed worth it; if done, grep ALL forms first (the times slice proved relative imports are invisible to an absolute grep) and compile-all + import-test telegram_bot/orchestrator/run before deleting.

---

## CARRY-FORWARD GOTCHAS (unchanged, bit prior sessions)
1. **NEVER run pytest via `run_in_background` on Windows** — orphans a child holding the prod kutai.db SQLite lock for HOURS. Run FOREGROUND + hard shell `timeout`; pytest.ini `timeout=` is INERT (pytest-timeout not installed). (This session a Bash call auto-backgrounded a pytest once — it completed clean, but watch for it.)
2. **Worktree pytest: keep ini addopts** (`--import-mode=importlib`), add `-p no:aiohttp` ONLY. `-o addopts=""` drops importlib → collection HANG.
3. **Raw `python -c` loads MAIN, not the worktree** — only pytest (conftest path-prepend, incl. `packages/*/src`) resolves worktree code. Verify via pytest.
4. **`dabidabi.configure()` requires an ABSOLUTE path.**
5. **Relative-import landmine** — grep `from ..infra.X`, `from .X import` before any shim delete.
6. **Concurrent agent sessions cross `main`** — many worktrees live (`git worktree list` showed 8). Use a worktree; integrate via real 3-way merge (never force/reset). main advanced 6 commits during this session; merge stayed conflict-free.
7. **NEVER bulk-kill `python.exe`** — live wrapper runs on system Python310 (same as pytest). ~9 orphaned pytest procs from OTHER sessions were observed and left untouched. Kill only by exact PID after matching `pytest` in CommandLine; exclude kutai_wrapper/run.py/guard.lock PID.
8. **Merge-from-worktree mechanics:** main is checked out at root with `receive.denyCurrentBranch=refuse` → cannot push/FF main from a linked worktree. This session merged main INTO the branch (conflict-free), then landed via ExitWorktree → root FF.
