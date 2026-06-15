# Handoff: DB ownership — Phase B (domain tables out of `dabidabi` → owners + file split)

**Date:** 2026-06-15. **Status:** Phase A DONE + merged to main + pushed (see below). Phase B = NOT started. This is the moonshot-landing phase.

## Where we are (Phase A recap)

The DB engine was lifted out of the app tree into a standalone package: `packages/db` (import name **`dabidabi`**). Chain so far:
- Eval (`docs/handoff/2026-06-11-db-ownership-split-eval-handoff.md`) → write-consolidation Step 1 (`docs/handoff/2026-06-11-step1-write-consolidation-worklist.md`, beckman = sole write-owner of missions/tasks/growth) → **Phase A** (`docs/handoff/2026-06-14-db-package-phaseA-scope.md`).
- Phase A commits on main: `20b7bf7b` (sever db.py from src.app.config + logging_config) · `abff8ea7` (lift db.py + times.py into `dabidabi`; `src/infra/db.py`+`src/infra/times.py` are now `sys.modules` ALIAS shims) · `734ea53e` (repoint all 11 packages off `src.infra.db` → `dabidabi`).
- **`dabidabi` today = the WHOLE former db.py** (9k LOC, engine + all ~95 query helpers + the 3,800-LOC `init_db` monolith + all 97 tables' schema). That is the Phase-A compromise: one wholesale lift so packages stop reaching into `src/`. It is NOT yet "engine only".

## What Phase B is

Phase A made packages *importable-standalone* (no `src.*` reach). It did NOT make them *data-independent* — every package still shares one `dabidabi` that knows all 97 tables. Phase B is the split that delivers the moonshot:

> Move each domain's **schema (DDL) + query helpers** OUT of `dabidabi` INTO its owning package; give that package its own `.db` file. `dabidabi` keeps only the **engine** — connection lifecycle (WAL, autocommit), per-mission tx-lock sharding, the migration runner, ATTACH orchestration, and generic helpers. The 3,800-LOC `init_db` monolith decomposes into per-domain schema modules that live with their owners.

When done: `pip install fatih_hoca` drags only the model tables, not comms/shopping/spine. That is the independence the user asked for.

## The hard constraints (from the eval — do NOT relitigate)

1. **FK cannot cross SQLite files.** Splitting a table out of the file makes its FKs unenforceable → the app must enforce (orphan sweeps). Only split tables with no cross-file integrity need.
2. **Cross-file atomicity dies under WAL.** A multi-file transaction is atomic per-file, NOT across the set. KutAI is WAL (load-bearing — concurrent readers + non-blocking writer).
3. ⇒ **The spine + its rollback cascade MUST stay in one file** (`core.db`). `restore_mission_db_rows` (green-tag rollback) is atomic across missions/tasks/artifacts/growth — that atomicity is a SAFETY feature and breaks the instant any of those tables crosses a file boundary. Spine schema moves INTO the `general_beckman` package, but stays one file.
4. **JOINs across files need ATTACH** — re-couples the "independent" files at query time. Cross-domain reads are handled by ATTACH or a thin read-API on the owner, not by giving everyone the schema.

## Split order (cleanest first — readiness from the eval matrix)

1. **`registry.db` → `fatih_hoca`** ✅ start here. Model/provider/model_stats/model_pick_log/kdv tables. Leaf, single-owner, zero spine FK. Cross-boundary READS only (~3): `fatih_hoca/counterfactual.py`, `mr_roboto/executors/analytics_digest.py`, `src/app/telegram_bot.py` (model_pick_log). Resolve via ATTACH or a `fatih_hoca` read-API. **This is the pattern-proving vertical slice — do it end-to-end first, then the rest repeat it.**
2. **`ledger.db`** ✅ cheap. cost_budgets/model_call_tokens. Zero atomicity loss (cost is ALREADY excluded from `restore_mission_db_rows`'s `MISSION_SCOPED_TABLES`). 3 read edges (beckman briefing ×2, src/app daily_briefing). Owner: likely `general_beckman` or a small new ledger module — decide.
3. **`yalayut.db`** ✅ leaf, yalayut already owns its tables.
4. **`core.db` (spine) → `general_beckman`** — schema relocates into beckman, stays ONE file (never split per constraint 3). Includes the 17 spine-bound tables + rollback cascade + growth + artifacts. Beckman already is the write-owner; this just moves the DDL/SQL home.
5. **`product.db` (comms) → ❌ BLOCKED.** No single owner (mr_roboto + src/app split it) AND deferred by the user (fresh subsystem, zero missions have exercised it — don't consolidate ownership of unvalidated tables). Revisit after first live mission runs through comms. `launches`/`email_sequences` = zero-writer-but-unfinished, NOT dead (don't sweep).
6. **artifacts + app tables** = src/infra, not packages → stay app infra, not materialized.

## Mechanics per domain extraction (repeat per owner)

1. **Schema:** pull that domain's `CREATE TABLE` / migrations out of `dabidabi`'s `init_db` into a `<owner>/schema.py`; register it with the engine's migration runner (new API — see below).
2. **Queries:** move the domain's helper functions from `dabidabi/__init__.py` into `<owner>/db.py` (or similar), built on `dabidabi`'s connection/lock primitives.
3. **File:** point the owner's connection at its own `.db` (or keep it in `core.db` and split the FILE last — file-split is the cheap, reversible final move once code-ownership is clean).
4. **Cross-domain reads:** ATTACH the owner's db, or expose a read-API on the owner. Never re-share the schema.
5. **Guards:** the beckman write-guards (`packages/general_beckman/tests/test_{task,mission,growth}_*_api.py`, helper in `tests/conftest.py`) currently treat `packages/db/src/dabidabi/__init__.py` as the SQL owner. As spine SQL moves into beckman, update the allowed-owner path; add per-domain guards as other domains gain their own write APIs.

## Engine work `dabidabi` needs (to enable the above)

- **Migration-runner API:** let each owner package register its DDL/migrations so the engine runs them at init (instead of the monolithic `init_db`). Decompose `init_db` (3,800 LOC) into per-domain schema modules that register.
- **ATTACH orchestration:** a clean way to attach sibling `.db` files for the cross-domain read JOINs (estimate_task_cost, get_artifact_provenance, estimate_conversation_cost — all read-only, ~4 total).
- **Multi-path connection management:** today one singleton on one `DB_PATH`. Splitting files means N connections (or ATTACH on one). Decide: one connection + ATTACH (simpler, keeps single-writer) vs per-file connections.

## Phase A tail to fold in (cleanups that make `dabidabi` truly independent)

- **Relocate the ~70 `src/*` in-app callers** off the `src.infra.db` / `src.infra.times` aliases → `dabidabi`. Then the two alias shims can be deleted. (Not a publish concern, so deferred out of Phase A; do it opportunistically.)
- **Invert the 5 lazy/optional engine UP-reaches** (`src.tools.shell`, `src.memory.vector_store` ×3, `src.infra.dead_letter`) inside `dabidabi/__init__.py` — they make a package reach back into the app. Replace with injected callbacks/hooks so `dabidabi` depends on nothing in `src/`.

## Gotchas / rules to respect

- **WAL is load-bearing** — do not casually drop it.
- **`restore_mission_db_rows` atomicity** = green-tag rollback safety. Anything crossing a file boundary with a rollback-cascade table breaks it.
- **Don't trust `src/core/router.py`** scoring copy (stale/dead). Live selection = fatih_hoca.
- **Concurrent agent sessions on `main` cross work** — use a git worktree for implementation.
- **Windows pytest hang:** the repo default `addopts` makes the **pytest-aiohttp plugin** orphan a PATH-`python` (Python310) child that deadlocks, holds SQLite locks, and survives `pytest-timeout`. **Run package tests with `-o addopts="" -p no:aiohttp`.** Output still lands in a `Tee` file when the job lingers; kill stragglers with `Stop-Process` (pytest only — never llama-server/orchestrator).
- **Editable install path:** `dabidabi` is editable-installed; if you implement in a worktree, re-`pip install -e ./packages/db` from the MAIN path before `/restart`.

## NEXT STEP for new session

Do the **`registry.db` → `fatih_hoca` vertical slice** end-to-end (constraint-light leaf, proves the whole pattern): build the engine's migration-runner + ATTACH API on the way, move fatih's model-table DDL + query helpers into the package, point at `registry.db`, resolve the 3 cross-boundary reads via ATTACH/read-API, update guards. Then repeat for ledger.db, yalayut.db. Spine (core.db) and comms (product.db, blocked) come later.
