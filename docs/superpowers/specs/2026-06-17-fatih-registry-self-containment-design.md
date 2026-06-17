# Design: decouple `model_pick_log` from core `tasks` (fatih_hoca independence — slice 1)

**Date:** 2026-06-17
**Supersedes the §1 framing in:** `docs/handoff/2026-06-17-db-phaseB-kdv-slice-handoff.md`
**Status:** approved design (Opus-reviewed, FIX-FIRST findings folded in), pre-plan

## Reframe: the goal is package independence, not file isolation

Prior handoffs framed §1 as a *physical* `registry.db` file-split via an `ATTACH DATABASE` primitive + a crash-safe live data migration. That framing is wrong for the goal.

**The goal (user, 2026-06-17):** *"DB files don't matter. Packages matter. Each package should own its tables to be able for independent releases."*

A physical file-split does **not** serve that goal: ATTACH leaves fatih_hoca still reading core's tables — file location is irrelevant to the coupling. The file-split only adds permanent cost (ATTACH on every connection open incl. per-call `connect_aux`; lost cross-file write atomicity as a forever-footgun; a risky one-shot migration on the live prod DB) for operational isolation nobody asked for.

## Scope of THIS slice (slice 1 of an independence roadmap)

This slice removes exactly **one** coupling: the `model_pick_log → tasks` JOIN in `fatih_hoca/db.py::get_latest_model_for_mission`. It does **NOT** make fatih_hoca independently releasable on its own — see "Remaining blockers" below. It is the first, cleanest, fully live-safe step.

### What is already done (Phase B `e523f889`)
- fatih_hoca owns the registry DDL (`fatih_hoca/schema.py`, single source).
- fatih_hoca owns the sync connection (`fatih_hoca/registry_store.py`) + `ensure_registry_schema_sync`.
- fatih_hoca registers its schema with the engine via `dabidabi.register_schema("fatih_hoca_registry", ...)`.
- All selection/scoring logic lives in fatih_hoca.
- Dependency on `dabidabi` (engine: `connect`, `get_db`, `register_schema`, `configure`) is acceptable — it is the shared storage engine, like depending on a DB driver. Independent release = fatih_hoca + dabidabi-as-dependency.

### The coupling removed by this slice
`fatih_hoca/db.py::get_latest_model_for_mission` (lines ~345-395) resolves a mission's latest pick via two JOINs into the core-owned `tasks` table:
- Tier-0: `model_pick_log mpl JOIN tasks t ON mpl.task_id = t.id WHERE t.mission_id = ?`
- Tier-1: `model_pick_log mpl JOIN tasks t ON t.title = mpl.task_name WHERE t.mission_id = ?` (legacy, for rows where `task_id IS NULL`)
- Tier-2: global most-recent non-reinforce pick (no JOIN).

`get_latest_pick_for_task` (same file) is JOIN-free. The sync selector reads (`model_stats`/`providers`/`models` via `registry_store`) are JOIN-free and side-effect-free w.r.t. `model_pick_log` (`selector.py:27`).

## The fix: denormalize `mission_id` onto `model_pick_log`

`model_pick_log` currently has `task_id` but no `mission_id` (schema.py:36-57). The sibling table `registry_events` **already** carries `mission_id` (schema.py:119) — proving the write path has mission_id available and that this denormalization is an established pattern in the same package.

Store `mission_id` on each pick row at write time; backfill existing rows once; rewrite the read to filter by `mission_id` and drop the `tasks` JOINs entirely.

## Components (6)

### 1. Schema — `packages/fatih_hoca/src/fatih_hoca/schema.py`
- Add `mission_id INTEGER` to the `model_pick_log` CREATE TABLE block.
- Add to `REGISTRY_ALTERS`: `"ALTER TABLE model_pick_log ADD COLUMN mission_id INTEGER"` (idempotent; "duplicate column name" swallowed by `_is_dup_col`).
- Add index: `"CREATE INDEX IF NOT EXISTS idx_pick_log_mission ON model_pick_log(mission_id, timestamp DESC)"`. Verified non-redundant — no existing index leads with `mission_id`; serves the Tier-0 `WHERE mission_id=? ORDER BY timestamp DESC LIMIT 1` exactly.
- Live-safe: additive ALTER on a table prod already has; CREATE covers fresh DBs. Both `create_registry_schema` (async) and `ensure_registry_schema_sync` (sync) run `REGISTRY_DDL` + `REGISTRY_ALTERS`, so both paths pick it up.

### 2. Context — `src/core/heartbeat.py`
- Add `current_mission_id: ContextVar[int | None] = ContextVar("current_mission_id", default=None)` beside `current_task_id`.
- Set/clear it at the dispatch site **verified at `src/core/orchestrator.py:253`** — the dispatched `task` dict is in scope and `task.get("mission_id")` is already used there (lines 218, 245). `current_task_id.set()` runs at 253 **before** `asyncio.create_task(_run_with_audit())` at 255, so the child task inherits a copy of the context (contextvars copy-on-task-create). Set `current_mission_id` at the same point; clear in the same teardown as `current_task_id`.
- Rationale for a ContextVar (not a write-time `SELECT mission_id FROM tasks WHERE id=?`): the pick-write path peaks ~44 writes/sec (`src/infra/pick_log.py` docstring). A per-write core lookup would re-introduce the exact hot-path core read we are removing. The ContextVar carries mission_id for free, mirroring task_id.
- Propagation verified: `record_pick` is `await`ed inline within the dispatcher child (`llm_dispatcher.py:253/263/274` → `pick_recorder.py:62`), all inside the context-copied child task — not re-`create_task`'d. task_id reliably lands on rows today (proven by `idx_pick_log_task_id` + existing tests), so mission_id will too.

### 3. Write path — thread `mission_id` through
- `fatih_hoca/db.py::insert_pick_log_row`: add `mission_id: int | None = None` kwarg; include `mission_id` column + value in the INSERT.
- `src/infra/pick_log.py::write_pick_log_row`: add `mission_id: int | None = None`; pass through.
- `src/telemetry/pick_recorder.py::record_pick`: read `current_mission_id.get()` (defensive try/except like the existing `current_task_id` read) and pass through.
- All three keep their fire-and-forget / never-raise contracts.
- **Second writer (do NOT change):** `fatih_hoca/db.py::record_reinforce_nudge` (~line 189) INSERTs `call_category='reinforce'` rows with no task_id/mission_id. This is intentional: reinforce rows are excluded from every read tier (`call_category != 'reinforce'`), so leaving `mission_id` NULL is correct. Documented here so a future maintainer does not "fix" it or remove the reinforce exclusion.

### 4. Read path — `fatih_hoca/db.py::get_latest_model_for_mission`
- Tier-0: `SELECT picked_model, provider FROM model_pick_log WHERE mission_id = ? AND call_category != 'reinforce' ORDER BY timestamp DESC LIMIT 1`. **No JOIN, no `tasks` reference.**
- Tier-1 (title-JOIN): **removed** (the backfill in Component 5 makes it unnecessary — pre-change rows get a real `mission_id`, so they match Tier-0).
- Tier-2 (global most-recent non-reinforce, `mission_id is None` / no match): unchanged.
- Result: the function references only `model_pick_log` — genuinely `tasks`-free.
- **Docstring updates (same edit):** rewrite the three-tier docstring (db.py:351-356) to two tiers; remove the stale "§1's ATTACH-split will qualify with the `registry.` prefix" line in `get_latest_pick_for_task` (db.py:318-320) — that ATTACH approach is abandoned.

### 5. Backfill — one-shot, idempotent, app/engine-level (NOT package code)
- `UPDATE model_pick_log SET mission_id = (SELECT t.mission_id FROM tasks t WHERE t.id = model_pick_log.task_id) WHERE mission_id IS NULL AND task_id IS NOT NULL`.
- Runs once at boot inside `dabidabi.init_db`'s existing migration block (the engine already performs domain data migrations like goals→missions there; a cross-table backfill is consistent). It must NOT live in `fatih_hoca/*` — the package stays free of any `tasks` read.
- Idempotent + cheap on every boot: after the first run, `WHERE mission_id IS NULL AND task_id IS NOT NULL` matches 0 rows. Single connection, both tables in core.db — no cross-file concern.
- Eliminates the deploy-window regression: active missions whose earlier picks were written pre-change get a real `mission_id`, so their reinforce lookups still resolve via Tier-0 (no silent fall-through to a cross-mission Tier-2 answer).
- Rows with `task_id IS NULL` (overhead/legacy) keep `mission_id` NULL — they never matched a mission lookup before either.

### 6. Tests (TDD — write first)
1. **Schema (fresh + existing):** fresh DB via `create_registry_schema` has `mission_id`; a pre-existing DB without it gets it via `REGISTRY_ALTERS`; re-run is a no-op.
2. **Write:** with `current_mission_id` set, a recorded pick persists the correct `mission_id`; unset → NULL, no error.
3. **Read no longer JOINs `tasks` (regression guard):** seed **only** `model_pick_log` rows (NO `tasks` table in the test DB); `get_latest_model_for_mission(mid)` returns the latest pick for that mission. Stated purpose: *guards that `get_latest_model_for_mission` no longer references `tasks`* — NOT a proof of package independence (the package still reads `missions` elsewhere; see Remaining blockers).
4. **Tier-2 fallback:** `get_latest_model_for_mission(None)` and a mission with no rows both return the global most-recent non-reinforce pick.
5. **Reinforce exclusion:** `call_category='reinforce'` rows excluded from Tier-0 and Tier-2.
6. **Backfill:** seed legacy rows (task_id set, mission_id NULL) + a `tasks` table mapping task_id→mission_id; run the migration; assert mission_id populated; re-run → no change (idempotent).

## Data flow (after change)
```
orchestrator dispatch (orchestrator.py:253)
  → current_task_id.set(tid); current_mission_id.set(mid)
    → asyncio.create_task(_run_with_audit())   # inherits context copy
      → LLM pick fires
        → pick_recorder.record_pick reads current_task_id + current_mission_id
          → write_pick_log_row(..., task_id, mission_id)
            → insert_pick_log_row → INSERT INTO model_pick_log(..., task_id, mission_id)
  ...later, on verdict (record_verdict.py:204)...
  → get_latest_model_for_mission(mid)
    → SELECT ... FROM model_pick_log WHERE mission_id = mid ...   (no tasks JOIN)
```

## Live-restart safety
- Schema: additive ALTER + new index on an existing table → no `no such table`, no destructive change.
- Backfill: idempotent UPDATE in init_db; first boot populates, later boots no-op. Bounded by row count of `model_pick_log` (single indexed UPDATE).
- Write: new column populated forward; paths not passing mission_id default NULL (nullable).
- Read: with the backfill, active-mission reinforce behavior is preserved across the restart (no transition regression). Post-change rows carry mission_id natively.
- Verification after `/restart`: boot smoke (no schema error); `SELECT COUNT(*) FROM model_pick_log WHERE mission_id IS NOT NULL` > 0 (backfill ran); new picks carry mission_id; reinforce resolves a model for an active mission.

## Remaining blockers to fatih_hoca independence (OUT OF SCOPE — roadmap for later slices)
Removing the `tasks` JOIN is necessary but not sufficient. After this slice, fatih_hoca still cannot run against a DB/runtime lacking core:
- **`requirements_builder.py:215`** — `SELECT context FROM missions WHERE id = ?` (reads core-owned `missions`) + **`:230`** `from src.workflows.engine.loader import load_workflow`. Slice 2 candidate (mission context is not ContextVar-trivial; needs its own design).
- **`src.*` import surface (dependency inversions):** `selector.py:106` `src.core.in_flight`, `selector.py:161` `src.core.router`, `requirements_builder.py:155` `src.core.retry`, `auto_tuner.py:190/315` `src.models.model_registry`, `__init__.py:364` `src.models.benchmark`, `registry_store.py:122` `src.app.config`. Each is a DI refactor (inject the dependency or invert via a port), its own slice.
- `src.infra.logging_config` (used throughout) — benign shared infra; lowest priority.

## Out of scope (this slice)
- ATTACH primitive / physical `registry.db` file-split (abandoned — wrong tool for the goal).
- `src/infra/db.py` shim delete (§5b-db, 304 importers).
- ledger → general_beckman.
- The `missions` read + `src.*` inversions above (separate slices).

## Carry-forward gotchas (apply during implementation)
- Use a git worktree — concurrent agent sessions cross `main`; integrate via real 3-way merge, never force/reset.
- NEVER run pytest via `run_in_background` on Windows (orphans hold prod SQLite lock for hours). Foreground + hard shell `timeout`. Keep pytest.ini `--import-mode=importlib`; add `-p no:aiohttp` only.
- Raw `python -c` loads MAIN, not the worktree — verify worktree code via pytest only.
- `dabidabi.configure()` requires an ABSOLUTE path.
