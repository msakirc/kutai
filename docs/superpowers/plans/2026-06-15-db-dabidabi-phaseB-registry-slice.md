# DB Phase B — registry domain → `fatih_hoca` (code-ownership slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the model-registry domain (DDL + query helpers + the sync `registry_store`) out of the shared `dabidabi` engine and into the `fatih_hoca` package, proving the per-domain extraction pattern — **with the physical file-split deferred to a later slice**.

**Architecture:** `dabidabi` stays the engine (one singleton aiosqlite connection, WAL, per-mission lock sharding, migration runner). Each domain package owns its `schema.py` (DDL) + query helpers and **registers** its schema with the engine at init. **This slice does NOT split files** — all five registry tables stay physically in `core.db`. Because `registry_store`'s own sync connection and `connect_aux`/`connect_aux_sync` all open `core.db` by path, a code-ownership move needs **zero data-path change** (no ATTACH). The physical `registry.db` split + ATTACH is a separate, later slice (see "Deferred" section) once every connection opener is unified.

**Tech Stack:** Python 3.10, aiosqlite (WAL) for the async engine path, raw `sqlite3` for the sync `registry_store` hot path, pytest (`-o addopts="" -p no:aiohttp` on Windows — the aiohttp plugin orphans a PATH-python child that deadlocks + holds SQLite locks), editable installs.

---

## What changed from the first draft (adversarial review caught these)

The first draft assumed registry access flowed only through the `get_db()` singleton + planned an ATTACH file-split. Two scouts proved otherwise. Corrections baked into this plan:

1. **File-split DEFERRED** (was Task 8). Registry is read/written through `registry_store.py`'s own sync `sqlite3` singleton + `connect_aux`/`connect_aux_sync` — ATTACH on the singleton is invisible to them, so a file-split would break every non-singleton reader and the cross-file data migration is **non-atomic under WAL**. Deferring kills both risks. This slice keeps tables in `core.db`.
2. **`update_model_stats` is DEAD** — its inline "Schema B" `CREATE` is shadowed by init_db's "Schema A" and its `INSERT` is silently `try/except`ed. Its sole consumer `src/memory/self_improvement.py:88-115` has been silently failing (queries Schema-B columns that don't exist). **Schema A is canonical.** Delete `update_model_stats`; fix/delete the dead self_improvement query — do NOT move verbatim.
3. **Three DDL copies** of providers/models/registry_events (init_db, `registry_store._ensure_schema`, + the dual model_stats) → collapse to ONE: a DDL **string list** in `fatih_hoca/schema.py`, executed by both an async callback (engine registration) and the sync `registry_store`.
4. **Missed writers**: `src/infra/pick_log.py:67` (live model_pick_log insert), `mr_roboto/.../sms_send.py:61` (registry_events), all `registry_store.py` writes. The write-guard (Task 7) must account for all of them.
5. **`registry_events` has 3 writers / 2 column profiles** — `record_action_event` + `sms_send` use migrated columns (mission_id/task_id/verb/reversibility); `registry_store._emit_event` uses the base 8. The canonical DDL must include ALL columns; the guard allowlists all three writers.
6. **No `record_verdict.py` change** — its 2 cross-table JOINs work as plain SQL while everything is in `core.db` (no ATTACH needed this slice).

---

## Scope

**Tables → owned by `fatih_hoca` after this slice (all stay in `core.db`):**
`models`, `providers`, `registry_events`, `model_stats`, `model_pick_log`.

**Canonical schemas (verified):**
- `model_stats` = **Schema A** (aggregate/upsert: avg_grade/success_rate/total_* + `UNIQUE(model, agent_type)`). Schema B deleted.
- `registry_events` = **full shape**: base 8 cols (id, timestamp, scope, target, event, cause, actor, payload_json) **+ migrated** (mission_id, task_id, verb, reversibility).

**OUT of scope (own later slices):** `kdv_state` → `kuleden_donen_var`; `model_call_tokens` → ledger slice; the physical `registry.db` file-split.

**Public API that MUST keep working** (callers depend on it): the 13 `registry_store` functions + `CAUSE_POLICY` + `set_db_path`/`close`/`_get_conn`; the async helpers `record_model_call`, `get_model_stats`, `get_model_performance_ranking`, `record_reinforce_nudge`, `record_action_event`.

---

## File Structure

- `packages/db/src/dabidabi/__init__.py` — add `register_schema()` + `_run_registered_schemas()`; DELETE moved helpers (replace with lazy re-export shims); DELETE registry DDL from `init_db`; DELETE dead `update_model_stats`.
- `packages/fatih_hoca/src/fatih_hoca/schema.py` (new) — `REGISTRY_DDL` + `REGISTRY_ALTERS` (sole DDL source) + async `create_registry_schema(db)` that registers with the engine.
- `packages/fatih_hoca/src/fatih_hoca/db.py` (new) — moved async helpers + cross-domain read-API.
- `packages/fatih_hoca/src/fatih_hoca/registry_store.py` (new, relocated from `src/infra`) — sync owner of providers/models/registry_events; `_ensure_schema` runs the shared `REGISTRY_DDL`.
- `src/infra/registry_store.py` — becomes a thin re-export shim of the relocated module.
- `src/memory/self_improvement.py:88-115` — fix the dead Schema-B query → Schema A (or delete the proposal).
- `src/infra/pick_log.py:67`, `packages/mr_roboto/.../analytics_digest.py:207`, `src/app/telegram_bot.py:2815` — repoint to fatih helpers/read-API.
- Guards: `packages/fatih_hoca/tests/test_registry_write_guard.py` (new); beckman guard allowed-sets unchanged (they never matched registry SQL — verified).

---

## Pre-flight (once, not a task)

Isolated worktree (concurrent `main` sessions cross work — memory + handoff warn). Use `superpowers:using-git-worktrees`. After engine/fatih edits in the worktree: `pip install -e ./packages/db ./packages/fatih_hoca` from the **worktree** path before testing; from the **MAIN** path before `/restart`.

Baseline:
```
python -m pytest packages/fatih_hoca/tests packages/general_beckman/tests tests/infra/test_registry_store.py -o addopts="" -p no:aiohttp -q
```

---

### Task 1: Engine — `register_schema()` registration layer

**Files:**
- Modify: `packages/db/src/dabidabi/__init__.py` (state near line ~95; call site near end of `init_db` ~line 4178)
- Test: `packages/db/tests/test_schema_registration.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# packages/db/tests/test_schema_registration.py
import pytest
import dabidabi


@pytest.mark.asyncio
async def test_register_schema_runs_registered_ddl(tmp_path):
    dabidabi.configure(str(tmp_path / "t.db"))
    calls = []

    async def _my_schema(db):
        calls.append("ran")
        await db.execute("CREATE TABLE IF NOT EXISTS widget (id INTEGER PRIMARY KEY)")

    dabidabi.register_schema("test_widget", _my_schema)
    await dabidabi.init_db()
    assert calls == ["ran"]
    db = await dabidabi.get_db()
    cur = await db.execute("SELECT name FROM sqlite_master WHERE name='widget'")
    assert await cur.fetchone() is not None
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_register_schema_dedupes_by_name(tmp_path):
    dabidabi.configure(str(tmp_path / "t2.db"))
    n = []

    async def _s(db):
        n.append(1)
        await db.execute("CREATE TABLE IF NOT EXISTS w2 (id INTEGER PRIMARY KEY)")

    dabidabi.register_schema("dup", _s)
    dabidabi.register_schema("dup", _s)
    await dabidabi.init_db()
    assert len(n) == 1
    await dabidabi.close_db()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/db/tests/test_schema_registration.py -o addopts="" -p no:aiohttp -q`
Expected: FAIL — `module 'dabidabi' has no attribute 'register_schema'`

- [ ] **Step 3: Implement `register_schema` + runner**

Add to module state (after other globals, ~line 95):

```python
# Per-domain schema registration. Owner packages call register_schema() at
# import time; init_db() runs each registered DDL callback after the engine's
# own core schema. Keyed by name so a module imported twice registers once.
_registered_schemas: "dict[str, callable]" = {}


def register_schema(name: str, fn) -> None:
    """Register an ``async fn(db)`` schema callback run by init_db()."""
    _registered_schemas[name] = fn


async def _run_registered_schemas(db) -> None:
    for name, fn in list(_registered_schemas.items()):
        await fn(db)
        logger.info(f"Ran registered schema: {name}")
    await db.commit()
```

- [ ] **Step 4: Call the runner near the end of `init_db`**

Immediately before the Yalayut integration block (~line 4178):

```python
    # Per-domain registered schemas (owner packages register their own DDL).
    await _run_registered_schemas(db)
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest packages/db/tests/test_schema_registration.py -o addopts="" -p no:aiohttp -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add packages/db/src/dabidabi/__init__.py packages/db/tests/test_schema_registration.py
git commit -m "feat(db): dabidabi per-domain schema registration API (Phase B)"
```

---

### Task 2: `fatih_hoca/schema.py` — single canonical DDL source

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/schema.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py` (import schema for self-registration)
- Test: `packages/fatih_hoca/tests/test_registry_schema.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_registry_schema.py
import pytest
import dabidabi
import fatih_hoca  # noqa: F401  side-effect registers registry schema

REGISTRY_TABLES = {"models", "providers", "registry_events", "model_stats", "model_pick_log"}


@pytest.mark.asyncio
async def test_registry_tables_created_via_registration(tmp_path):
    dabidabi.configure(str(tmp_path / "reg.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in await cur.fetchall()}
    assert REGISTRY_TABLES.issubset(names)
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_model_stats_is_schema_A(tmp_path):
    dabidabi.configure(str(tmp_path / "reg2.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("PRAGMA table_info(model_stats)")
    cols = {r[1] for r in await cur.fetchall()}
    # Schema A (aggregate) — NOT the dead Schema B (recorded_at/cost/success)
    for c in ("avg_grade", "success_rate", "total_calls", "updated_at"):
        assert c in cols
    assert "recorded_at" not in cols


@pytest.mark.asyncio
async def test_registry_events_has_migrated_columns(tmp_path):
    dabidabi.configure(str(tmp_path / "reg3.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("PRAGMA table_info(registry_events)")
    cols = {r[1] for r in await cur.fetchall()}
    for c in ("scope", "target", "event", "mission_id", "task_id", "verb", "reversibility"):
        assert c in cols
    await dabidabi.close_db()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_schema.py -o addopts="" -p no:aiohttp -q`
Expected: FAIL — `No module named 'fatih_hoca.schema'`

- [ ] **Step 3: Create `schema.py` (DDL string list = single source)**

Copy each `CREATE`/`ALTER` byte-for-byte from the cited dabidabi source lines. The `registry_events` CREATE here is the **full** shape (base + migrated cols inline) so fresh DBs never need the ALTERs; the ALTERs cover existing prod DBs.

```python
# packages/fatih_hoca/src/fatih_hoca/schema.py
"""Model-registry schema, owned by fatih_hoca. SINGLE source of truth for the
registry tables' DDL. Both the async engine registration (create_registry_schema)
and the sync registry_store._ensure_schema execute REGISTRY_DDL/REGISTRY_ALTERS,
so there is exactly one place the schema is defined."""
import dabidabi

# Full CREATE statements (idempotent). model_stats = Schema A (canonical).
REGISTRY_DDL = [
    # model_stats — dabidabi lines 911-928 (Schema A; Schema B in update_model_stats is DEAD)
    """CREATE TABLE IF NOT EXISTS model_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT NOT NULL, agent_type TEXT NOT NULL,
        avg_grade REAL DEFAULT 0.0, avg_cost REAL DEFAULT 0.0,
        avg_latency REAL DEFAULT 0.0, success_rate REAL DEFAULT 1.0,
        total_calls INTEGER DEFAULT 0, total_successes INTEGER DEFAULT 0,
        total_grade_sum REAL DEFAULT 0.0, total_cost_sum REAL DEFAULT 0.0,
        total_latency_sum REAL DEFAULT 0.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model, agent_type)
    )""",
    # providers — dabidabi 1366-1374 / registry_store 151-159
    """CREATE TABLE IF NOT EXISTS providers (
        name TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active',
        cause TEXT, marked_at TIMESTAMP, revived_at TIMESTAMP, key_hash TEXT
    )""",
    # models — dabidabi 1376-1387 / registry_store 160-172
    """CREATE TABLE IF NOT EXISTS models (
        litellm_name TEXT PRIMARY KEY, provider TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active', cause TEXT, marked_at TIMESTAMP,
        revived_at TIMESTAMP, expires_at TIMESTAMP, source TEXT,
        first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_models_status ON models(status, provider)",
    # registry_events — FULL shape: base 8 (dabidabi 1389-1398) + migrated 4 (1524-1537)
    """CREATE TABLE IF NOT EXISTS registry_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        scope TEXT NOT NULL, target TEXT NOT NULL, event TEXT NOT NULL,
        cause TEXT, actor TEXT, payload_json TEXT,
        mission_id INTEGER, task_id INTEGER, verb TEXT, reversibility TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_registry_events_target_ts ON registry_events(target, timestamp DESC)",
    # model_pick_log — dabidabi 1233-1252
    """CREATE TABLE IF NOT EXISTS model_pick_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        task_name TEXT NOT NULL, agent_type TEXT, difficulty INTEGER,
        call_category TEXT, picked_model TEXT NOT NULL, picked_score REAL NOT NULL,
        picked_reasons TEXT, candidates_json TEXT NOT NULL, failures_json TEXT,
        snapshot_summary TEXT, pool TEXT, urgency REAL, success INTEGER,
        error_category TEXT, provider TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_provider ON model_pick_log(provider)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_task ON model_pick_log(task_name, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_model ON model_pick_log(picked_model, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_pick_log_task_id ON model_pick_log(task_id)",
]

# ALTERs for EXISTING DBs whose CREATE predates the added columns. Each is
# attempted individually; "duplicate column name" is the expected/ignored error.
REGISTRY_ALTERS = [
    # registry_events migrated cols (dabidabi 1524-1527) — no-op on fresh DBs (full CREATE above)
    "ALTER TABLE registry_events ADD COLUMN mission_id INTEGER",
    "ALTER TABLE registry_events ADD COLUMN task_id INTEGER",
    "ALTER TABLE registry_events ADD COLUMN verb TEXT",
    "ALTER TABLE registry_events ADD COLUMN reversibility TEXT",
    # model_pick_log added cols (dabidabi 1256-1265)
    "ALTER TABLE model_pick_log ADD COLUMN pool TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN urgency REAL",
    "ALTER TABLE model_pick_log ADD COLUMN success INTEGER",
    "ALTER TABLE model_pick_log ADD COLUMN error_category TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN provider TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN outcome TEXT",
    "ALTER TABLE model_pick_log ADD COLUMN reinforce REAL",
    "ALTER TABLE model_pick_log ADD COLUMN task_id INTEGER",
]


def _is_dup_col(err: Exception) -> bool:
    return "duplicate column name" in str(err).lower()


async def create_registry_schema(db) -> None:
    """Async executor (engine registration path)."""
    for sql in REGISTRY_DDL:
        await db.execute(sql)
    for sql in REGISTRY_ALTERS:
        try:
            await db.execute(sql)
        except Exception as e:
            if not _is_dup_col(e):
                raise


def ensure_registry_schema_sync(conn) -> None:
    """Sync executor (registry_store path). conn = sqlite3.Connection."""
    for sql in REGISTRY_DDL:
        conn.execute(sql)
    for sql in REGISTRY_ALTERS:
        try:
            conn.execute(sql)
        except Exception as e:
            if not _is_dup_col(e):
                raise


dabidabi.register_schema("fatih_hoca_registry", create_registry_schema)
```

> Note: the original init_db ALTER loop re-raises on non-duplicate errors (lines 1268-1270); this matches that (only "duplicate column name" is swallowed) — strictly correct, unlike the first draft's bare `except: pass`.

- [ ] **Step 4: Wire the import side-effect**

In `packages/fatih_hoca/src/fatih_hoca/__init__.py` (after existing imports):

```python
from . import schema as _schema  # noqa: F401  registers registry DDL with dabidabi
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_schema.py -o addopts="" -p no:aiohttp -q`
Expected: PASS (3 passed). (DDL still ALSO present in init_db at this point — `IF NOT EXISTS` makes the overlap safe; Task 5 removes the duplicate.)

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/schema.py packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_registry_schema.py
git commit -m "feat(fatih_hoca): single-source registry DDL via dabidabi registration (Phase B)"
```

---

### Task 3: `fatih_hoca/db.py` — move async helpers; delete dead `update_model_stats`

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/db.py`
- Modify: `packages/db/src/dabidabi/__init__.py` (delete moved helpers + the dead `update_model_stats`; add lazy re-export shims)
- Test: `packages/fatih_hoca/tests/test_registry_db_api.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_registry_db_api.py
import pytest
import dabidabi
import fatih_hoca  # registers schema
from fatih_hoca import db as fdb


@pytest.mark.asyncio
async def test_record_and_get_model_stats(tmp_path):
    dabidabi.configure(str(tmp_path / "s.db"))
    await dabidabi.init_db()
    # match real signature: record_model_call(model, agent_type, success, cost, latency, grade)
    await fdb.record_model_call("m1", "coder", True, cost=0.01, latency=1.0, grade=0.8)
    rows = await fdb.get_model_stats(model="m1")
    assert rows and rows[0]["model"] == "m1"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_recent_picks_read_api(tmp_path):
    dabidabi.configure(str(tmp_path / "p.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, picked_score, candidates_json) "
        "VALUES ('t', 'm1', 0.9, '[]')"
    )
    await db.commit()
    picks = await fdb.get_recent_picks(limit=10)
    assert any(p["picked_model"] == "m1" for p in picks)
    await dabidabi.close_db()


def test_update_model_stats_is_gone():
    # dead Schema-B writer removed; not resurrected by any shim
    assert not hasattr(dabidabi, "update_model_stats")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_db_api.py -o addopts="" -p no:aiohttp -q`
Expected: FAIL — `No module named 'fatih_hoca.db'`

- [ ] **Step 3: Create `db.py` by moving the helpers VERBATIM**

Move these from `dabidabi/__init__.py` into `fatih_hoca/db.py` (cited ranges): `record_model_call` (6348-6416), `get_model_stats` (6455-6474), `get_model_performance_ranking` (6477-6492), `record_reinforce_nudge` (8008-8049), `record_action_event` (7650-7690). Header:

```python
# packages/fatih_hoca/src/fatih_hoca/db.py
"""Async model-registry query helpers, owned by fatih_hoca. Built on dabidabi's
connection primitives. Schema A is canonical for model_stats."""
from dabidabi import get_db
from src.infra.logging_config import get_logger

logger = get_logger("fatih_hoca.db")

# <verbatim moved functions here — ranges cited above>

# --- cross-domain READ-API (replaces raw SQL in outside async callers) ---
async def get_recent_picks(limit: int = 100, since_days: int | None = None):
    db = await get_db()
    if since_days is not None:
        cur = await db.execute(
            "SELECT * FROM model_pick_log WHERE timestamp > datetime('now', ?) "
            "ORDER BY timestamp DESC LIMIT ?", (f"-{since_days} days", limit))
    else:
        cur = await db.execute(
            "SELECT * FROM model_pick_log ORDER BY timestamp DESC LIMIT ?", (limit,))
    return [dict(r) for r in await cur.fetchall()]


async def get_model_stats_rows():
    db = await get_db()
    cur = await db.execute("SELECT model, total_calls, success_rate FROM model_stats")
    return [dict(r) for r in await cur.fetchall()]
```

> `record_confidence_claim` (8862-8957) READS model_pick_log but WRITES confidence_outcomes (not registry) — leave in dabidabi; same connection, same file.

- [ ] **Step 4: Delete dead `update_model_stats` + add lazy shims in dabidabi**

Delete `update_model_stats` (6913-6943) entirely — it is dead (Schema B; `INSERT` silently try/excepted; sole consumer is the broken self_improvement query fixed in Task 6). Delete the moved helper bodies. At the END of `dabidabi/__init__.py` add lazy delegating shims (back-compat for any `from dabidabi import record_model_call` callers until repointed). Lazy import avoids the package→package cycle (fatih_hoca imports dabidabi at module load):

```python
# Back-compat: registry helpers now live in fatih_hoca.db. Lazy shims delegate
# without an eager dabidabi->fatih_hoca import (which would cycle).
def _registry_shim(_name):
    async def _f(*a, **k):
        from fatih_hoca import db as _fdb
        return await getattr(_fdb, _name)(*a, **k)
    _f.__name__ = _name
    return _f

for _n in ("record_model_call", "get_model_stats", "get_model_performance_ranking",
           "record_reinforce_nudge", "record_action_event"):
    globals()[_n] = _registry_shim(_n)
```

> All five are `async def`, so the shim is `async` and `await`s the target — fixes the first draft's sync-shim-around-async bug. Verify the internal caller `record_vendor_cost` (dabidabi ~8534) still does `await record_action_event(...)` — it will, via the async shim.

- [ ] **Step 5: Run targeted + broad suites**

Run:
```
python -m pytest packages/fatih_hoca/tests/test_registry_db_api.py -o addopts="" -p no:aiohttp -q
python -m pytest packages/fatih_hoca/tests packages/general_beckman/tests -o addopts="" -p no:aiohttp -q
```
Expected: PASS, same baseline count (no import errors from moved helpers).

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/db.py packages/db/src/dabidabi/__init__.py packages/fatih_hoca/tests/test_registry_db_api.py
git commit -m "refactor(db): move registry helpers to fatih_hoca.db, drop dead update_model_stats (Phase B)"
```

---

### Task 4: Relocate `registry_store.py` → fatih (sync owner) + back-compat shim

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/registry_store.py` (moved from `src/infra/registry_store.py`)
- Modify: the moved file's `_ensure_schema` → call `ensure_registry_schema_sync`
- Modify: `src/infra/registry_store.py` → thin re-export shim
- Test: `packages/fatih_hoca/tests/test_registry_store_relocation.py` (create); existing `tests/infra/test_registry_store.py` must still pass via the shim

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_registry_store_relocation.py
def test_relocated_module_importable():
    from fatih_hoca import registry_store as rs
    for fn in ("register_model", "mark_dead", "is_dead", "revive", "list_dead",
               "register_provider", "mark_provider_dead", "is_provider_dead",
               "list_dead_providers", "revive_provider", "get_provider_key_hash",
               "recent_events", "get_model_cause"):
        assert hasattr(rs, fn)
    assert hasattr(rs, "CAUSE_POLICY")


def test_legacy_import_path_still_works():
    # the src/infra shim must re-export the same objects (identity)
    from src.infra import registry_store as legacy
    from fatih_hoca import registry_store as new
    assert legacy.register_model is new.register_model
    assert legacy.CAUSE_POLICY is new.CAUSE_POLICY


def test_ensure_schema_uses_shared_ddl(tmp_path):
    from fatih_hoca import registry_store as rs
    rs.set_db_path(str(tmp_path / "rs.db"))
    try:
        conn = rs._get_conn()  # triggers _ensure_schema
        cur = conn.execute("PRAGMA table_info(registry_events)")
        cols = {r[1] for r in cur.fetchall()}
        # shared DDL → migrated cols present even when registry_store creates first
        assert {"mission_id", "task_id", "verb", "reversibility"}.issubset(cols)
    finally:
        rs.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_store_relocation.py -o addopts="" -p no:aiohttp -q`
Expected: FAIL — `No module named 'fatih_hoca.registry_store'`

- [ ] **Step 3: Move the file + point `_ensure_schema` at shared DDL**

`git mv src/infra/registry_store.py packages/fatih_hoca/src/fatih_hoca/registry_store.py`. In the moved file, replace the body of `_ensure_schema` (the inline CREATEs, ~lines 146-192) with:

```python
def _ensure_schema(conn):
    from fatih_hoca.schema import ensure_registry_schema_sync
    ensure_registry_schema_sync(conn)
```

This deletes the 3rd DDL copy (now stale-proof: same source as the engine). Keep everything else (the sync `_get_conn`, `set_db_path`, `close`, `_resolve_db_path`, the 13 public fns, `CAUSE_POLICY`) unchanged. `_resolve_db_path` still falls back to `from src.app.config import DB_PATH` — fine (core.db; no file-split).

- [ ] **Step 4: Replace `src/infra/registry_store.py` with a re-export shim**

```python
# src/infra/registry_store.py
"""Back-compat shim — registry_store relocated to fatih_hoca.registry_store.
Existing `from src.infra import registry_store` callers keep working."""
from fatih_hoca.registry_store import *  # noqa: F401,F403
from fatih_hoca.registry_store import (  # explicit re-export for `_`-prefixed test hooks
    _get_conn, _ensure_schema, _resolve_db_path, set_db_path, close, CAUSE_POLICY,
)
```

> `import *` won't pull underscore names; `tests/infra/test_registry_store.py` reaches `_get_conn`/`set_db_path`/`close` — the explicit line re-exports them. Confirm against that test file's actual hook usage.

- [ ] **Step 5: Run relocation + legacy + infra suites**

Run:
```
python -m pytest packages/fatih_hoca/tests/test_registry_store_relocation.py tests/infra/test_registry_store.py packages/fatih_hoca/tests -o addopts="" -p no:aiohttp -q
```
Expected: PASS. (Callers: fatih_hoca/registry.py, fatih_hoca/__init__.py:275, telegram_bot, migrate script — all import paths satisfied by shim + relocated module.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(fatih_hoca): relocate sync registry_store as owner; src/infra shim (Phase B)"
```

---

### Task 5: Delete registry DDL from `init_db`

**Files:**
- Modify: `packages/db/src/dabidabi/__init__.py`
- Test: reuse `packages/fatih_hoca/tests/test_registry_schema.py`

- [ ] **Step 1: Confirm registration is the only source (negative test)**

Temporarily comment `from . import schema` in `fatih_hoca/__init__.py`; run `test_registry_schema.py`. Expected: FAIL (tables missing) — proves registration is load-bearing. Re-enable the import.

- [ ] **Step 2: Delete registry DDL from `init_db`**

Remove from `init_db`: `model_stats` Schema-A CREATE (911-928), `model_pick_log` CREATE + its ALTERs (1256-1265) + backfill `UPDATE ... provider='local'` (1274-1276) + 4 indexes (1278-1288), `providers`/`models`/`registry_events` CREATEs + indexes (1366-1407). The registry_events migrated columns live inside an `apply_migration("2026-05-10-registry-events-action-scope", ...)` **call** at 1521-1539 (NOT a bare ALTER loop) — delete the whole `apply_migration(...)` call cleanly (those columns are now in the canonical CREATE + `REGISTRY_ALTERS`). **Do NOT touch** `kdv_state` (1345-1352) or `model_call_tokens`.

> The `provider='local'` backfill (1274-1276) is now redundant on fresh DBs and irrelevant after this slice; existing prod rows already backfilled at first boot post-Phase-A. Safe to drop.

- [ ] **Step 3: Make schema registration an explicit boot dependency**

After Task 5, tables are created ONLY if `fatih_hoca/__init__` (→ `from . import schema`) executed before `init_db()`. Today that fires transitively (`run.py → src.app.config → src.models.model_registry → fatih_hoca.registry → fatih_hoca/__init__`), but `model_registry` is a "thin shim" slated for deletion — a future slice could silently break table creation. Make it explicit: add near the top of `src/app/run.py`, BEFORE the `init_db()` call (~line 187):

```python
import fatih_hoca  # noqa: F401  registers registry schema with dabidabi (Phase B)
```

- [ ] **Step 4: Schema parity test**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_schema.py -o addopts="" -p no:aiohttp -q`
Expected: PASS (tables created only via registration).

- [ ] **Step 5: Fresh-init smoke**

```
python -c "import asyncio, dabidabi, fatih_hoca; dabidabi.configure('data/_pb_smoke.db'); asyncio.run(dabidabi.init_db()); print('init ok')"
```
Expected: `init ok`. Then delete `data/_pb_smoke.db*`.

- [ ] **Step 6: Commit**

```bash
git add packages/db/src/dabidabi/__init__.py src/app/run.py
git commit -m "refactor(db): registry DDL out of init_db (fatih_hoca owns it) (Phase B)"
```

---

### Task 6: Fix dead self_improvement query + repoint external async reads

**Files:**
- Modify: `src/memory/self_improvement.py:88-115`
- Modify: `packages/mr_roboto/src/mr_roboto/executors/analytics_digest.py:207-212`
- Modify: `src/app/telegram_bot.py:2815-2821`
- Modify: `src/infra/pick_log.py:67`
- Test: `packages/fatih_hoca/tests/test_self_improvement_query.py` (create)

- [ ] **Step 1: Write the failing test for the self_improvement fix**

```python
# packages/fatih_hoca/tests/test_self_improvement_query.py
import pytest
import dabidabi
import fatih_hoca
from fatih_hoca import db as fdb


@pytest.mark.asyncio
async def test_get_model_stats_rows_returns_schema_A(tmp_path):
    dabidabi.configure(str(tmp_path / "si.db"))
    await dabidabi.init_db()
    await fdb.record_model_call("m1", "coder", True, cost=0.01, latency=1.0, grade=0.8)
    rows = await fdb.get_model_stats_rows()
    assert rows and "success_rate" in rows[0] and "total_calls" in rows[0]
    await dabidabi.close_db()
```

- [ ] **Step 2: Run to verify it fails (or passes trivially if Task 3 landed)**

Run: `python -m pytest packages/fatih_hoca/tests/test_self_improvement_query.py -o addopts="" -p no:aiohttp -q`
Expected: PASS if `get_model_stats_rows` exists from Task 3; this test pins the contract self_improvement will consume.

- [ ] **Step 3: Fix the dead self_improvement query**

`self_improvement.py:88-115` currently queries dead Schema-B columns (`success`, `cost`, `recorded_at`) wrapped in `try/except: pass` (silently broken). Replace the raw SQL with the canonical read:

```python
# src/memory/self_improvement.py  (inside the proposal builder, ~line 88)
from fatih_hoca.db import get_model_stats_rows
rows = await get_model_stats_rows()  # Schema A: model, total_calls, success_rate
underperformers = [r for r in rows if r["total_calls"] >= 10 and r["success_rate"] < 0.5]
# ...build proposal from underperformers (was: AVG(cost), HAVING calls>=10)
```

If `avg_cost` is needed, add it to `get_model_stats_rows`'s SELECT (`avg_cost` exists in Schema A). Keep the surrounding `try/except` but remove the silent swallow of a now-valid query (let real errors surface or log them).

- [ ] **Step 4: Repoint the plain external reads**

- `analytics_digest.py:207-212` (`SELECT picked_model, COUNT(*), AVG(picked_score) ... GROUP BY`): add a purpose helper `get_pick_summary(since_days)` to `fatih_hoca/db.py` returning the aggregate, and call it. (analytics_digest is async — `from dabidabi import get_db` already → switch to `from fatih_hoca.db import get_pick_summary`.)
- `telegram_bot.py:2815-2821` (7-day pick summary via `connect_aux`): replace with `await get_pick_summary(since_days=7)`.
- `src/infra/pick_log.py:67` (`INSERT INTO model_pick_log`): move ONLY the raw INSERT into a `fatih_hoca/db.py` helper (e.g. `_insert_pick_log_row(**row)`); **keep `src.infra.pick_log.write_pick_log_row` as the public callable** that delegates to it. `write_pick_log_row` is imported/patched by `src/telemetry/pick_recorder.py:67` + ~6 test files (`tests/infra/test_pick_log*.py`, `packages/husam/tests/*`) — do NOT delete the name or the file. Only the SQL relocates; fatih becomes the SQL owner while the public entry point stays put.

> Leave `counterfactual.py` / `grading.py` as-is — they are in-package (fatih_hoca) and use `connect_aux_sync` deliberately (sync CLI / hot path). They already point at core.db; ownership is satisfied (same package).

- [ ] **Step 5: Run targeted + broad suites**

Run:
```
python -m pytest packages/fatih_hoca/tests packages/mr_roboto/tests packages/general_beckman/tests -o addopts="" -p no:aiohttp -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fix(registry): repair dead self_improvement query; repoint external reads to fatih (Phase B)"
```

---

### Task 7: Write-guard — fatih owns registry SQL (allowlist all sanctioned writers)

**Files:**
- Create: `packages/fatih_hoca/tests/test_registry_write_guard.py`

> Beckman guards are NOT touched — verified they regex only `tasks`/`missions`/`growth_events`, never registry tables. No change needed there.

- [ ] **Step 1: Write the guard (covers ALL writers found by the scout)**

```python
# packages/fatih_hoca/tests/test_registry_write_guard.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SQL = re.compile(
    r'(INSERT\s+INTO\s+(model_stats|model_pick_log|providers|models|registry_events)\b'
    r'|UPDATE\s+(model_stats|model_pick_log|providers|models|registry_events)\s+SET'
    r'|DELETE\s+FROM\s+(model_stats|model_pick_log|providers|models|registry_events)\b)',
    re.IGNORECASE,
)
# Sanctioned registry writers after the slice:
ALLOWED = {
    ROOT / "packages/fatih_hoca/src/fatih_hoca/db.py",
    ROOT / "packages/fatih_hoca/src/fatih_hoca/schema.py",
    ROOT / "packages/fatih_hoca/src/fatih_hoca/registry_store.py",
    # sms_send writes a distinct scope='sms_send' registry_events row (cap counter) —
    # cannot repoint to record_action_event (forces scope='action'); sanctioned 2nd writer.
    ROOT / "packages/mr_roboto/src/mr_roboto/executors/sms_send.py",
}
ALLOWED = {p.resolve() for p in ALLOWED}


def test_no_registry_writes_outside_fatih():
    violations = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "packages").rglob("*.py")):
        if "tests" in p.parts or p.resolve() in ALLOWED:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if SQL.search(line):
                violations.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert violations == [], "Registry writes outside fatih_hoca:\n" + "\n".join(violations)
```

- [ ] **Step 2: Run the guard**

Run: `python -m pytest packages/fatih_hoca/tests/test_registry_write_guard.py -o addopts="" -p no:aiohttp -q`
Expected: PASS. If it lists a site: either repoint it to a fatih helper or, if a genuinely sanctioned distinct writer, add it to `ALLOWED` with a one-line justification comment. (after Task 6 Step 4 `pick_log.py` holds no raw INSERT — the SQL moved to `fatih_hoca/db.py` while `write_pick_log_row` stays as a delegating public entry; `record_action_event` is now in `fatih_hoca/db.py` = allowed; `dabidabi` shims delegate, contain no raw table SQL.)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(registry): write-guard pins fatih_hoca as registry SQL owner (Phase B)"
```

---

## Final verification (before claiming done)

Run the full affected surface (per verification-before-completion skill — evidence before assertion):
```
python -m pytest packages/fatih_hoca/tests packages/general_beckman/tests packages/mr_roboto/tests packages/db/tests tests/infra/test_registry_store.py -o addopts="" -p no:aiohttp -q
```
Grep — only sanctioned registry SQL remains outside fatih:
```
rg -n "INSERT INTO (model_stats|model_pick_log|providers|models|registry_events)|UPDATE (model_stats|model_pick_log|providers|models|registry_events) SET" src packages --glob '!**/tests/**' --glob '!packages/fatih_hoca/**'
```
Expected: only `mr_roboto/.../sms_send.py` (sanctioned).

## Post-implementation (NOT tasks)

- Re-install editable from MAIN path before `/restart`: `pip install -e ./packages/db ./packages/fatih_hoca`.
- **Live verify (USER, restart-gated):** `/restart`; run a task → `/benchpicks` (model_pick_log writes land); `/dead` + `/revive` (registry_store via shim); confirm no DLQ from registry reads. All DB moonshot work is restart-gated (memory).

## Deferred — the physical `registry.db` file-split (its own later slice)

Prereqs before it is safe: (1) unify ALL connection openers onto one access path OR have each (`registry_store` sync conn, every `connect_aux`/`connect_aux_sync` caller) ATTACH `registry.db` + qualify names; (2) a WAL-safe data migration (cross-attach commit is NOT atomic under WAL — copy with per-table existence guards + idempotent verify, NOT a single cross-file transaction). Build the `attach_db()`/`attached()` engine primitive then (designed + reviewed-sound in draft 1; omitted here since unused without the split).

## Other Phase B follow-ups
`kdv_state` → `kuleden_donen_var`; `ledger.db` (cost_budgets/model_call_tokens); `yalayut.db`; spine (`core.db` → general_beckman, never split); invert the 5 lazy engine UP-reaches; relocate the ~70 `src/*` callers off the alias shims; consider generalizing `record_action_event` to accept `scope`/`target` so `sms_send` can consolidate (optional).

## Self-review
- Connection model decided (one conn + ATTACH) but **ATTACH unused this slice** (no file-split) → no multi-connection breakage.
- Schema A canonical; Schema B (`update_model_stats`) + its dead consumer fixed, not carried.
- ONE DDL source (`schema.py` string list), executed async + sync → kills all 3 copies + drift.
- All writers inventoried (record_model_call, record_reinforce_nudge, pick_log, record_action_event, sms_send, registry_store) → guard allowlists exactly the sanctioned set.
- Shims: async (matches async targets), lazy (no import cycle), identity-preserving for registry_store.
- No `record_verdict`/ATTACH/file-split changes → B1/B5/B6 from review removed entirely.
