# Design: fatih_hoca registry self-containment (decouple `model_pick_log` from core `tasks`)

**Date:** 2026-06-17
**Supersedes the §1 framing in:** `docs/handoff/2026-06-17-db-phaseB-kdv-slice-handoff.md`
**Status:** approved design, pre-plan

## Reframe: the goal is package independence, not file isolation

Prior handoffs framed §1 as a *physical* `registry.db` file-split via an `ATTACH DATABASE` primitive + a crash-safe live data migration. That framing is wrong for the actual goal.

**The goal (user, 2026-06-17):** *"DB files don't matter. Packages matter. Each package should own its tables to be able for independent releases."*

A physical file-split does **not** serve that goal: ATTACH leaves fatih_hoca still reading core's `tasks` table — the file location is irrelevant to the coupling. The file-split only adds permanent cost (ATTACH on every connection open incl. per-call `connect_aux`; lost cross-file write atomicity as a forever-footgun; a risky one-shot migration on the live prod DB) for operational isolation nobody asked for.

What actually blocks fatih_hoca from being independently releasable is **one logical coupling**: a single async read that JOINs the registry table `model_pick_log` to the core-owned table `tasks`.

### What is already done (Phase B `e523f889`)
- fatih_hoca owns the registry DDL (`fatih_hoca/schema.py`, single source).
- fatih_hoca owns the sync connection (`fatih_hoca/registry_store.py`) + `ensure_registry_schema_sync`.
- fatih_hoca registers its schema with the engine via `dabidabi.register_schema("fatih_hoca_registry", ...)`.
- All selection/scoring logic lives in fatih_hoca.
- Dependency on `dabidabi` (engine: `connect`, `get_db`, `register_schema`, `configure`) is acceptable — it is the shared storage engine, like depending on a DB driver. Independent release = fatih_hoca + dabidabi-as-dependency.

### The single remaining coupling
`fatih_hoca/db.py::get_latest_model_for_mission` (lines ~361-377) resolves a mission's latest pick via two JOINs into core `tasks`:
- Tier-0: `model_pick_log mpl JOIN tasks t ON mpl.task_id = t.id WHERE t.mission_id = ?`
- Tier-1: `model_pick_log mpl JOIN tasks t ON t.title = mpl.task_name WHERE t.mission_id = ?` (legacy, for rows where `task_id IS NULL`)

`tasks` is owned by general_beckman/core. This JOIN means fatih_hoca cannot run against a DB that lacks the core `tasks` schema → not independently releasable.

`get_latest_pick_for_task` (same file) is JOIN-free (pure `model_pick_log`). The sync selector reads (`model_stats`/`providers`/`models` via `registry_store`) are JOIN-free. A repo-wide JOIN grep confirms `get_latest_model_for_mission` is the **only** registry↔core JOIN.

## The fix: denormalize `mission_id` onto `model_pick_log`

`model_pick_log` currently has `task_id` but no `mission_id` (schema.py:36-57). The sibling table `registry_events` **already** carries `mission_id` (schema.py:119) — proving the write path has mission_id available and that this denormalization is an established pattern in the same package.

Store `mission_id` on each pick row at write time; rewrite the read to filter by it directly. The JOIN disappears; fatih_hoca reads only its own tables.

## Components (5)

### 1. Schema — `packages/fatih_hoca/src/fatih_hoca/schema.py`
- Add `mission_id INTEGER` to the `model_pick_log` CREATE TABLE block.
- Add to `REGISTRY_ALTERS`: `"ALTER TABLE model_pick_log ADD COLUMN mission_id INTEGER"` (idempotent; "duplicate column name" swallowed by existing `_is_dup_col`).
- Add index: `"CREATE INDEX IF NOT EXISTS idx_pick_log_mission ON model_pick_log(mission_id, timestamp DESC)"`.
- Live-safe: additive ALTER on a table the prod DB already has; CREATE path covers fresh DBs. Both the async (`create_registry_schema`) and sync (`ensure_registry_schema_sync`) registration paths pick this up since both run `REGISTRY_DDL` + `REGISTRY_ALTERS`.

### 2. Context — `src/core/heartbeat.py`
- Add `current_mission_id: ContextVar[int | None] = ContextVar("current_mission_id", default=None)` beside the existing `current_task_id`.
- The orchestrator already sets `current_task_id` at dispatch and clears it after; set/clear `current_mission_id` at the same site (the dispatched task carries `mission_id`). Exact site to be confirmed in the plan (search for `current_task_id.set(`).
- Rationale for a ContextVar (not a write-time `SELECT mission_id FROM tasks WHERE id=?`): the pick-write path peaks ~44 writes/sec (per `src/infra/pick_log.py` docstring). A per-write core lookup would re-introduce a hot-path core read — exactly the coupling we are removing. The ContextVar carries mission_id for free, mirroring how task_id already flows.

### 3. Write path — thread `mission_id` through
- `fatih_hoca/db.py::insert_pick_log_row`: add `mission_id: int | None = None` kwarg; include `mission_id` column + value in the INSERT. (Owns the raw SQL.)
- `src/infra/pick_log.py::write_pick_log_row`: add `mission_id: int | None = None` kwarg; pass through to `insert_pick_log_row`.
- `src/telemetry/pick_recorder.py::record_pick`: read `current_mission_id.get()` (defensive try/except like the existing `current_task_id` read) and pass to `write_pick_log_row`.
- All three keep their fire-and-forget / never-raise contracts unchanged.

### 4. Read path — `fatih_hoca/db.py::get_latest_model_for_mission`
- Tier-0: `SELECT picked_model, provider FROM model_pick_log WHERE mission_id = ? AND call_category != 'reinforce' ORDER BY timestamp DESC LIMIT 1`. **No JOIN, no `tasks` reference.**
- Tier-1 (title-JOIN): **dropped** (user decision 2026-06-17). Old rows with `task_id IS NULL` / `mission_id IS NULL` fall through to Tier-2.
- Tier-2 (global most-recent non-reinforce pick, `mission_id is None` or no match): unchanged.
- Result: the function references only `model_pick_log`. `get_db()` (dabidabi shared engine) stays — that is the engine dependency, not a core-schema dependency.

### 5. No backfill (YAGNI)
- Existing pre-change rows keep `mission_id = NULL`. They will not match Tier-0 mission lookups and fall to Tier-2.
- This is acceptable: mission-latest lookups are about *recent* picks (the reinforce loop right after a verdict); historical rows being unmatched only affects long-past missions, and Tier-2 still returns a sensible global recent pick.
- A backfill would require a one-shot `UPDATE model_pick_log SET mission_id = (SELECT mission_id FROM tasks WHERE id = task_id)` — a core read. Out of scope unless a concrete need for mission-matched historical telemetry appears.

## Data flow (after change)
```
orchestrator dispatch
  → current_task_id.set(tid); current_mission_id.set(mid)
    → LLM pick fires
      → pick_recorder.record_pick reads current_task_id + current_mission_id
        → write_pick_log_row(..., task_id, mission_id)
          → insert_pick_log_row → INSERT INTO model_pick_log(..., task_id, mission_id)
  ...later, on verdict...
  → mr_roboto record_verdict → get_latest_model_for_mission(mid)
    → SELECT ... FROM model_pick_log WHERE mission_id = mid ...   (no tasks JOIN)
```

## Testing (TDD — write tests first)
1. **Schema (fresh + existing):** fresh DB via `create_registry_schema` has `mission_id` column; a pre-existing DB without the column gets it via `REGISTRY_ALTERS` (and re-running is a no-op).
2. **Write:** with `current_mission_id` set, a recorded pick persists the correct `mission_id`; with it unset (overhead/test), persists NULL without error.
3. **Self-containment read (the key proof):** seed **only** `model_pick_log` rows (NO `tasks` table created in the test DB); `get_latest_model_for_mission(mid)` returns the latest pick for that mission. This test fails today (JOIN needs `tasks`) and is the regression guard that fatih_hoca no longer reaches into core.
4. **Tier-2 fallback:** `get_latest_model_for_mission(None)` and a mission with no matching rows both return the global most-recent non-reinforce pick.
5. **Reinforce exclusion:** rows with `call_category = 'reinforce'` are excluded from Tier-0 and Tier-2.

## Live-restart safety
- Schema change is an additive ALTER + new index on a table prod already has → no `no such table` / no destructive change.
- Write path: new column populated going forward; old code paths that don't pass mission_id default to NULL (column is nullable).
- Read path: behavior for current missions is preserved for rows written after the change; pre-change rows degrade to Tier-2 (a documented, benign change). No boot dependency.
- Restart-gated verification: after `/restart`, confirm boot smoke (no schema errors), confirm new picks carry `mission_id` (`SELECT mission_id, COUNT(*) FROM model_pick_log GROUP BY mission_id IS NULL`), confirm reinforce still resolves a model for an active mission.

## Out of scope
- ATTACH primitive / physical `registry.db` file-split (abandoned — wrong tool for the goal).
- `src/infra/db.py` shim delete (§5b-db, 304 importers — separate hygiene slice, recommended against).
- ledger → general_beckman.
- Any other package's table-ownership (this spec is fatih_hoca/registry only).

## Carry-forward gotchas (apply during implementation)
- Use a git worktree — concurrent agent sessions cross `main`; integrate via real 3-way merge, never force/reset.
- NEVER run pytest via `run_in_background` on Windows (orphans hold prod SQLite lock for hours). Foreground + hard shell `timeout`. Keep pytest.ini `--import-mode=importlib`; add `-p no:aiohttp` only.
- Raw `python -c` loads MAIN, not the worktree — verify worktree code via pytest only.
- Grep relative import forms too if touching imports.
