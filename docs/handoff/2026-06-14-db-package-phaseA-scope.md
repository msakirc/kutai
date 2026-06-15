# Phase A scope — `db` engine package extraction (2026-06-14)

Continues `2026-06-11-db-ownership-split-eval-handoff.md` + `2026-06-11-step1-write-consolidation-worklist.md` (Step 1 DONE/merged/pushed `337d655c`).

**Decision locked (user, 2026-06-14):** moonshot = packages publishable independently. `src.infra.db` imports make every package unpublishable (can't `pip install` a unit that reaches into the app tree). yazbunu precedent applies — **but only to the engine.** One central package holding all 97 tables' schema would force `pip install fatih_hoca` to drag the whole app schema → defeats independence. So:

- **`db` package = ENGINE ONLY** (connection/WAL/locks/migration-runner/primitives) — the yazbunu-equivalent, lean, every package depends on it.
- **Domain tables (DDL + queries) eventually live in their owning package** (Phase B). Spine stays in beckman/`core.db`.

**Phase A goal:** get `db.py` out of `src/` into `packages/db/` so packages stop importing `src.infra.db` → unblocks publishing. Wholesale lift (engine + all 95 business fns + `init_db`) so the ~100 package-side imports repoint cleanly. Phase B later pulls domain tables back out into owners, leaving `db` pure engine.

---

## Scale (scout-verified)

- `src/infra/db.py` = **9,051 LOC**, ~95 async fns + helpers, 18 domain sections, `init_db()` = **3,800-LOC monolith**.
- **213 importing modules** (177 source + 36 test), **715 import statements**.
- **11 publishable packages blocked** by `src.infra.db`: general_beckman (26 files), mr_roboto (47), yalayut (14), fatih_hoca (5), coulson (4), intersect (3), hallederiz_kadir (2), husam (2), kuleden_donen_var (1), nerd_herd (1), workflow_engine (1).
- src-side importers (lower priority, can sit behind a shim): src/app (24), src/infra (13), src/tools (7), src/memory (6), src/core (5), src/workflows (4), src/ops (4), + scattered (~70 total).

## House packaging convention (from yazbunu + existing packages)

- src-layout: `packages/db/src/db/`, `pyproject.toml` (setuptools, PEP 621), `where=["src"]`, version `0.1.0`.
- Install: `-e ./packages/db` in root `requirements.txt` → editable `.pth`.
- `__init__.py` = clean facade exporting public API; import sites do `from db import X`.
- Precedent for engine-as-package: **`logging_config` is already a re-export of the external `yazbunu` package** — same move, one layer down.

---

## THE BLOCKER: db.py's 3 outbound `src/` deps (must sever before lift)

These are why it's not a pure copy-paste. Found at db.py module top:

| dep | line | severity | fix |
|---|---|---|---|
| `from src.infra.logging_config import get_logger` | 9 | **trivial** | `logging_config` is *itself* a re-export of `yazbunu`. Replace with `from yazbunu import get_logger`. yazbunu is already an external installed package → zero new coupling. |
| `from src.infra.times import utc_now, db_now, to_db, DB_FMT` | 10 | **easy** | small pure-util module. Either (a) vendor `times.py` into `packages/db/src/db/times.py`, or (b) extract `times` to a tiny shared package. Recommend (a) for Phase A (times is DB-timestamp formatting — belongs with the DB engine), leave `src/infra/times.py` as a shim re-exporting from `db` for the ~N app callers. Verify caller count before choosing. |
| `from src.app.config import DB_PATH` | 8 | **the real inversion** | db package must not import `src.app`. Fix: `db` reads `DB_PATH` from env (`os.environ["DB_PATH"]`) or accepts `db.configure(db_path=...)` at startup. `.env` already sets `DB_PATH` (per CLAUDE.md). Recommend env-read with optional `configure()` override → no caller changes, app already populates env. |

### 5 lazy/optional outbound imports (Phase A: leave as-is; Phase A.5 debt)
All inside function bodies, `try/except`-wrapped, so they do NOT block import-time packaging:
- `add_mission()` → `src.tools.shell` (docker sandbox, optional)
- `store_memory()` / `semantic_recall()` / `purge_mission_chroma_collections_via_db()` → `src.memory.vector_store` (vector embed, optional)
- `recover_startup_tasks()` → `src.infra.dead_letter` (quarantine, optional)

These are db.py reaching UP into app code — a layering smell. They survive the lift (lazy, graceful). Invert via injected callbacks/hooks in a later pass. Flag, don't block.

---

## EXECUTION LOG

**Step 1 — sever in-place (worktree `dabidabi-phaseA`, branch off `337d655c`), 2026-06-14:**
- `db.py`: `from src.infra.logging_config import get_logger` → `from yazbunu import get_logger` (logging_config is a pure yazbunu re-export — verified).
- `db.py`: `from src.app.config import DB_PATH` → self-contained resolution block (dotenv-load + absolute-default + abs-guard, mirrors config.py contract) + `configure(db_path)` runtime override. `DB_PATH` kept as a reassignable module global (test fixtures monkeypatch `db.DB_PATH`; `setenv` alone no-ops post-import — 2026-05-04).
- **`times` NOT severed in Step 1** — deliberate. Moving `src/infra/times.py` without a destination package = a duplicated copy = band-aid (golden rule). `times` is also imported by yalayut (×5) + workflow_engine, so it belongs in the shared package with a shim. Severs in Step 2 alongside `dabidabi` creation. db.py's only remaining top-level `src.` import is `src.infra.times` (the 5 lazy/optional ones unchanged).
- Decisions locked: package name **`dabidabi`**; DB_PATH = env + `configure()`.
- Tests: root DB_PATH-monkeypatch suite 29 passed; beckman write-guards + lifecycle 47 passed.
- Committed `20b7bf7b` on branch `dabidabi-phaseA`.

**Step 2a — create package + lift db.py/times.py (committed `abff8ea7`):**
- `packages/db/` created (import name `dabidabi`, src-layout, house pyproject; deps aiosqlite/python-dotenv/yazbunu). `git mv` src/infra/db.py → `dabidabi/__init__.py`, src/infra/times.py → `dabidabi/times.py` (history kept).
- Engine de-app-coupled: `from dabidabi.times import …`; DB_PATH resolution made install-location-independent (find_dotenv from cwd; repo-root default 4-up for editable; standalone must set env/configure()).
- `src/infra/db.py` + `src/infra/times.py` → **sys.modules ALIAS** shims (`sys.modules[__name__] = dabidabi[.times]`). This (not star re-export) preserves `monkeypatch.setattr(<db module>, "DB_PATH", …)` — verified `src.infra.db IS dabidabi` and monkeypatch reaches the engine global.
- `-e ./packages/db` in requirements.txt; editable-installed (points at WORKTREE path → re-install from main path post-merge). 2 beckman SQL guards updated to allow the engine's new home.
- Verified: smoke (aliases+monkeypatch) + 37 root + 44 beckman guard/API green.

**Step 2b — repoint 11 packages (committed `734ea53e`, 5 parallel subagents):**
- `from src.infra.db import X` → `from dabidabi import X` (+ times) across mr_roboto(57), beckman(17), yalayut(13), workflow_engine(1), fatih_hoca(5), coulson(4), intersect(3), hallederiz_kadir(1), husam(1), kuleden_donen_var(1). nerd_herd = no real import (docstring only). `dabidabi` added to each touched pyproject.
- Guards made dabidabi-aware: shared AST import-guard matches `dabidabi`|`src.infra.db`; 3 raw-SQL/call guards (task/mission/growth) allow `packages/db/src/dabidabi/__init__.py`.
- Verified: all 11 packages import clean vs worktree; 55 beckman guard/API green. Full beckman suite [running].

### Windows test-harness lesson
The repo's default pytest `addopts` makes the **pytest-aiohttp plugin** orphan a PATH-`python` (system Python310) child that deadlocks, holds SQLite locks, and survives pytest-timeout. **Always run package tests with `-o addopts="" -p no:aiohttp`.** Cost ~6h of zombie pytest this session before isolating it; kill stragglers with `Stop-Process` (pytest only — never llama/orchestrator).

### Remaining (Phase A tail + B)
- src/* in-app callers still use the `src.infra.*` aliases (NOT a publish concern; optional relocation).
- 5 lazy/optional UP-reaches in the engine (shell, vector_store, dead_letter) — invert via injected hooks (Phase A.5 debt).
- **Pre-merge**: re-`pip install -e ./packages/db` from the MAIN path (editable currently points at the worktree) before `/restart`.
- **Phase B**: domain tables out of `dabidabi` into owners (spine→beckman/core.db, models→fatih/registry.db, cost→ledger.db); file-split per package.

## Phase A execution order (worktree per chunk; reversible)

1. **Sever the 3 outbound deps in-place** (still in `src/infra/db.py`, ship + test before moving anything):
   - swap logging import → `yazbunu` direct.
   - decide times: vendor vs shared; wire `DB_PATH` via env+`configure()`.
   - Verify full suite green with db.py still at its current path but src-import-free (except the lazy 5).
2. **Create `packages/db/`** (house convention), move `db.py` → `packages/db/src/db/__init__.py` (or split engine vs business modules — see below), add to `requirements.txt`, `pip install -e`. Verify prod import: `python -c "from db import get_db, add_task"`.
3. **Leave `src/infra/db.py` as a shim**: `from db import *` (+ explicit re-exports) → src-side 70 importers keep working untouched, incremental.
4. **Repoint the 11 packages** off `src.infra.db` → `from db import` (this is the publish-unblock; ~100 files, mechanical). Do biggest last or per-package PRs: mr_roboto (47) and general_beckman (26) are the bulk.
5. **Update the Step-1 AST write-guard tests** (`packages/general_beckman/tests/test_{task,mission}_write_api.py`): allowed-writer set changes from `src.infra.db` → `db` package. The guards enforce "only beckman + the db owner write tasks/missions" — the owner module path moves. Must update or they false-fail.

### Optional within Phase A: pre-split the module
Instead of one 9k `__init__.py`, lay it out so Phase B is cheaper:
- `packages/db/src/db/engine.py` — connection singleton, `get_db`/`close_db`, `_get_tx_lock`/`_CombinedLock`, PRAGMAs, `connect_aux`, `configure`.
- `packages/db/src/db/schema.py` — `init_db()` (the 3,800-LOC monolith; Phase B splits per-domain).
- `packages/db/src/db/<domain>.py` — business fns grouped by the existing 18 sections (missions, tasks, models, cost, growth, comms, …).
- `__init__.py` re-exports all → flat `from db import X` API preserved.

## How Step-1 (beckman write-ownership) folds in

- Step 1 made beckman the sole **caller** with a write API; the **SQL bodies** (`add_task`, `update_mission_fields`, `insert_growth_event`, `_build_mission_update`, reset/recovery helpers) live in db.py. After the lift they live in `packages/db` and beckman imports `from db import ...` — ownership semantics unchanged, just relocated.
- Phase B: spine SQL (`missions`/`tasks`/`continuations` + rollback cascade + growth) migrates OUT of `db` INTO `general_beckman` (it already owns the write API). `db` package keeps only the engine. fatih's model tables → fatih, cost → ledger, etc. File split (`core.db`/`registry.db`/`ledger.db`) falls out per package.
- AST guards travel with the owner: in Phase A they point at `db`; in Phase B at `general_beckman` once spine SQL lands there.

## Effort

- Step 1 (sever 3 deps): **S** (2 trivial, 1 config inversion).
- Steps 2–3 (create package + shim): **S**, mechanical.
- Step 4 (repoint 11 packages): **M–L**, ~100 files, mostly find/replace `src.infra.db`→`db`; mr_roboto + beckman are the bulk.
- Step 5 (guard tests): **XS**.
- Net Phase A: **M–L**, fully reversible, each chunk ships independently. Unblocks publishing for all 11 packages.

## Open decisions for user
1. **Package name** — `db` (generic, matches request) vs a Turkish-persona nickname like the other packages (yasar_usta/fatih_hoca/…). House style is personas; `db` breaks the pattern but is honest. Pick.
2. **`times.py`** — vendor into `db` (recommended, it's DB-timestamp util) vs extract to a shared micro-package (if many non-DB callers depend on it). Decide after counting `src.infra.times` callers.
3. **`DB_PATH`** — env-read (recommended, `.env` already sets it) vs explicit `db.configure()` at app startup. Can do both (env default + override).

## Gotchas to respect
- WAL + per-mission tx-lock sharding is load-bearing (`restore_mission_db_rows` atomicity = green-tag rollback safety). The engine package must preserve the singleton + lock semantics exactly — do NOT change connection handling during the lift.
- `get_db()` is `isolation_level=None` (autocommit) — known, canonicalized in db.py. Don't "fix" it during the move.
- Concurrent agent sessions on `main` have crossed prior work — use a git worktree.
- Leave the shim at `src/infra/db.py` until src-side callers are repointed; deleting it early breaks ~70 modules.
