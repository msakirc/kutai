# Decouple model_pick_log from core `tasks` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the only registry→core JOIN (`model_pick_log JOIN tasks`) so `fatih_hoca.db.get_latest_model_for_mission` reads only its own table — slice 1 of the fatih_hoca independence roadmap.

**Architecture:** Denormalize `mission_id` onto `model_pick_log`. A new `current_mission_id` ContextVar (set beside `current_task_id` at dispatch) carries it to the fire-and-forget pick writer — no hot-path core read. A one-shot idempotent backfill in `dabidabi.init_db` populates pre-existing rows so Tier-1 (the legacy title-JOIN) can be deleted without a transition regression. The read filters by `mission_id`; no `tasks` reference remains.

**Tech Stack:** Python 3.10 (async), aiosqlite/sqlite3, pytest (`-p no:aiohttp`, keep pytest.ini `--import-mode=importlib`). Spec: `docs/superpowers/specs/2026-06-17-fatih-registry-self-containment-design.md`.

**Pre-flight (do once before Task 1):**
- Work in a git worktree (concurrent agent sessions cross `main`). The using-git-worktrees skill should have created one.
- All pytest runs: FOREGROUND, with a hard shell `timeout`, never `run_in_background`.

---

## File Structure

- `packages/fatih_hoca/src/fatih_hoca/schema.py` — add `mission_id` column, ALTER, index (Task 1).
- `src/core/heartbeat.py` — add `current_mission_id` ContextVar (Task 2).
- `src/core/orchestrator.py:253` — set `current_mission_id` at dispatch (Task 2).
- `packages/fatih_hoca/src/fatih_hoca/db.py` — `insert_pick_log_row` mission_id param + INSERT (Task 3); `get_latest_model_for_mission` rewrite + docstrings (Task 4).
- `src/infra/pick_log.py::write_pick_log_row` — passthrough (Task 3).
- `src/telemetry/pick_recorder.py::record_pick` — read ContextVar (Task 3).
- `packages/db/src/dabidabi/__init__.py` (after line 4050) — backfill (Task 5).
- `packages/fatih_hoca/tests/test_pick_log_mission_decouple.py` — new test file (Tasks 1,3,4,5).
- `tests/unit/test_heartbeat_mission_ctx.py` — new test file (Task 2).

---

## Task 1: Schema — `mission_id` on `model_pick_log`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/schema.py` (CREATE block ~36-57, indexes ~60-63, `REGISTRY_ALTERS` ~115-132)
- Test: `packages/fatih_hoca/tests/test_pick_log_mission_decouple.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_pick_log_mission_decouple.py
import pytest
import dabidabi
import fatih_hoca  # noqa: F401 — registers schema
from fatih_hoca import db as fdb


async def _col_names(db, table):
    cur = await db.execute(f"PRAGMA table_info({table})")
    rows = await cur.fetchall()
    await cur.close()
    return {r[1] for r in rows}


@pytest.mark.asyncio
async def test_model_pick_log_has_mission_id_on_fresh_db(tmp_path):
    dabidabi.configure(str(tmp_path / "fresh.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    assert "mission_id" in await _col_names(db, "model_pick_log")
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_mission_id_alter_is_idempotent_on_existing_db(tmp_path):
    # Simulate a pre-change DB: create model_pick_log WITHOUT mission_id,
    # then run the schema (ALTER adds it; re-run is a no-op).
    from fatih_hoca.schema import create_registry_schema
    dabidabi.configure(str(tmp_path / "existing.db"))
    db = await dabidabi.get_db()
    await db.execute(
        "CREATE TABLE model_pick_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "task_name TEXT NOT NULL, picked_model TEXT NOT NULL, "
        "picked_score REAL NOT NULL, candidates_json TEXT NOT NULL)")
    await db.commit()
    await create_registry_schema(db)
    assert "mission_id" in await _col_names(db, "model_pick_log")
    await create_registry_schema(db)  # idempotent — must not raise
    assert "mission_id" in await _col_names(db, "model_pick_log")
    await dabidabi.close_db()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v`
Expected: both FAIL — `assert "mission_id" in {...}` (column absent).

- [ ] **Step 3: Implement the schema change**

In `packages/fatih_hoca/src/fatih_hoca/schema.py`, in the `model_pick_log` CREATE block, add `mission_id` after the `task_id INTEGER` line (keep it the last column; mind the trailing comma on the now-not-last `task_id` line):

```python
            task_id INTEGER,
            mission_id INTEGER
        )
    """,
```

Add a new index alongside the other `idx_pick_log_*` indexes:

```python
    "CREATE INDEX IF NOT EXISTS idx_pick_log_mission ON model_pick_log(mission_id, timestamp DESC)",
```

In `REGISTRY_ALTERS`, append after the `task_id` ALTER:

```python
    "ALTER TABLE model_pick_log ADD COLUMN mission_id INTEGER",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/schema.py packages/fatih_hoca/tests/test_pick_log_mission_decouple.py
git commit -m "feat(fatih_hoca): add mission_id column + index to model_pick_log"
```

---

## Task 2: `current_mission_id` ContextVar + dispatch wiring

**Files:**
- Modify: `src/core/heartbeat.py` (after line 23, the `current_task_id` definition)
- Modify: `src/core/orchestrator.py:253`
- Test: `tests/unit/test_heartbeat_mission_ctx.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_heartbeat_mission_ctx.py
def test_current_mission_id_exists_and_defaults_none():
    from src.core import heartbeat as hb
    assert hb.current_mission_id.get() is None


def test_current_mission_id_set_get():
    from src.core import heartbeat as hb
    tok = hb.current_mission_id.set(99)
    try:
        assert hb.current_mission_id.get() == 99
    finally:
        hb.current_mission_id.reset(tok)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 python -m pytest tests/unit/test_heartbeat_mission_ctx.py -p no:aiohttp -v`
Expected: FAIL — `AttributeError: module 'src.core.heartbeat' has no attribute 'current_mission_id'`.

- [ ] **Step 3: Add the ContextVar**

In `src/core/heartbeat.py`, directly after the `current_task_id` definition (line 23):

```python
current_mission_id: ContextVar[int | None] = ContextVar("current_mission_id", default=None)
```

- [ ] **Step 4: Wire the dispatch site**

In `src/core/orchestrator.py`, immediately after line 253 (`_hb.current_task_id.set(...)`), add (mirrors the existing pattern — single set, no token reset; the var is overwritten on the next dispatch exactly as `current_task_id` is):

```python
            _hb.current_mission_id.set(task.get("mission_id"))
```

- [ ] **Step 5: Run test + import smoke**

Run: `timeout 60 python -m pytest tests/unit/test_heartbeat_mission_ctx.py -p no:aiohttp -v`
Expected: PASS.
Run: `timeout 60 python -c "import src.core.orchestrator"`
Expected: no error (orchestrator still imports).

- [ ] **Step 6: Commit**

```bash
git add src/core/heartbeat.py src/core/orchestrator.py tests/unit/test_heartbeat_mission_ctx.py
git commit -m "feat(core): carry current_mission_id ContextVar from dispatch"
```

---

## Task 3: Write path — persist `mission_id`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/db.py::insert_pick_log_row` (~258-304)
- Modify: `src/infra/pick_log.py::write_pick_log_row` (~32-83)
- Modify: `src/telemetry/pick_recorder.py::record_pick` (~58-80)
- Test: `packages/fatih_hoca/tests/test_pick_log_mission_decouple.py` (append)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_insert_pick_log_row_persists_mission_id(tmp_path):
    dabidabi.configure(str(tmp_path / "w.db"))
    await dabidabi.init_db()
    await fdb.insert_pick_log_row(
        task_name="t", agent_type="coder", difficulty=1,
        picked_model="m1", picked_score=0.9, category="MAIN_WORK",
        candidates_json="[]", snapshot_summary="", success=True,
        error_category="", provider="local", outcome="success",
        task_id=42, mission_id=7,
    )
    db = await dabidabi.get_db()
    cur = await db.execute(
        "SELECT mission_id FROM model_pick_log WHERE task_id = 42")
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 7
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_insert_pick_log_row_mission_id_defaults_null(tmp_path):
    dabidabi.configure(str(tmp_path / "wn.db"))
    await dabidabi.init_db()
    await fdb.insert_pick_log_row(
        task_name="t", agent_type=None, difficulty=None,
        picked_model="m1", picked_score=0.9, category="OVERHEAD",
        candidates_json="[]", snapshot_summary="", success=True,
        error_category="", provider="local", outcome="success",
        task_id=None,
    )
    db = await dabidabi.get_db()
    cur = await db.execute("SELECT mission_id FROM model_pick_log LIMIT 1")
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is None
    await dabidabi.close_db()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k mission_id`
Expected: `test_insert_pick_log_row_persists_mission_id` FAILS — `insert_pick_log_row()` got an unexpected keyword argument `mission_id`.

- [ ] **Step 3a: Modify `insert_pick_log_row`**

In `packages/fatih_hoca/src/fatih_hoca/db.py`, add the kwarg to the signature (after `task_id: int | None,`):

```python
    task_id: int | None,
    mission_id: int | None = None,
) -> None:
```

Update the INSERT to include the column + bind value:

```python
    await db.execute(
        "INSERT INTO model_pick_log "
        "(task_name, agent_type, difficulty, picked_model, picked_score, "
        " call_category, candidates_json, snapshot_summary, success, "
        " error_category, provider, outcome, task_id, mission_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_name,
            agent_type,
            difficulty,
            picked_model,
            picked_score,
            category,
            candidates_json,
            snapshot_summary,
            1 if success else 0,
            error_category,
            provider,
            outcome,
            task_id,
            mission_id,
        ),
    )
```

- [ ] **Step 3b: Modify `write_pick_log_row`**

In `src/infra/pick_log.py`, add `mission_id: int | None = None` to the signature (after `task_id: int | None = None,`) and pass it through to `insert_pick_log_row`:

```python
    task_id: int | None = None,
    mission_id: int | None = None,
) -> None:
```

```python
        await insert_pick_log_row(
            task_name=task_name,
            agent_type=agent_type or None,
            difficulty=difficulty,
            picked_model=picked_model,
            picked_score=picked_score,
            category=category,
            candidates_json="[]",
            snapshot_summary=snapshot_summary,
            success=success,
            error_category=error_category,
            provider=provider,
            outcome=outcome,
            task_id=task_id,
            mission_id=mission_id,
        )
```

- [ ] **Step 3c: Modify `record_pick`**

In `src/telemetry/pick_recorder.py`, where `_active_task_id` is resolved from the ContextVar (~lines 58-65), add a sibling mission_id resolution:

```python
        _active_mission_id: int | None = None
        try:
            from src.core.heartbeat import current_mission_id as _cmid
            _active_mission_id = _cmid.get()
        except Exception:
            pass
```

Then pass it in the `write_pick_log_row(...)` call (after `task_id=_active_task_id,`):

```python
            task_id=_active_task_id,
            mission_id=_active_mission_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k mission_id`
Expected: both write tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/db.py src/infra/pick_log.py src/telemetry/pick_recorder.py packages/fatih_hoca/tests/test_pick_log_mission_decouple.py
git commit -m "feat(fatih_hoca): persist mission_id on every pick row"
```

---

## Task 4: Read path — filter by `mission_id`, drop the `tasks` JOIN

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/db.py::get_latest_model_for_mission` (~345-391) + docstrings at `get_latest_pick_for_task` (~318-320)
- Test: `packages/fatih_hoca/tests/test_pick_log_mission_decouple.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_get_latest_model_for_mission_no_tasks_dependency(tmp_path):
    # The hard regression guard: DROP the tasks table, then the query must
    # still resolve the latest pick by mission_id alone.
    dabidabi.configure(str(tmp_path / "r.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    # Target mission 7 (older) + a different mission 9 (newer). Old JOIN code
    # would miss mission 7 (no tasks) and fall to Tier-2 → return mission 9's
    # model. New code returns mission 7's model.
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "picked_score, candidates_json, mission_id, call_category, timestamp) "
        "VALUES ('a','mA','local',0.9,'[]',7,'MAIN_WORK','2026-06-16 10:00:00')")
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "picked_score, candidates_json, mission_id, call_category, timestamp) "
        "VALUES ('b','mB','gemini',0.9,'[]',9,'MAIN_WORK','2026-06-16 11:00:00')")
    await db.commit()
    await db.execute("DROP TABLE tasks")
    await db.commit()
    model, provider = await fdb.get_latest_model_for_mission(7)
    assert model == "mA" and provider == "local"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_latest_model_for_mission_tier2_fallback(tmp_path):
    dabidabi.configure(str(tmp_path / "r2.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "picked_score, candidates_json, mission_id, call_category, timestamp) "
        "VALUES ('a','mA','local',0.9,'[]',7,'MAIN_WORK','2026-06-16 10:00:00')")
    await db.commit()
    # mission None → global most-recent; unknown mission → also Tier-2.
    assert (await fdb.get_latest_model_for_mission(None))[0] == "mA"
    assert (await fdb.get_latest_model_for_mission(999))[0] == "mA"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_get_latest_model_for_mission_excludes_reinforce(tmp_path):
    dabidabi.configure(str(tmp_path / "r3.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "picked_score, candidates_json, mission_id, call_category, timestamp) "
        "VALUES ('a','mReal','local',0.9,'[]',7,'MAIN_WORK','2026-06-16 10:00:00')")
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, provider, "
        "picked_score, candidates_json, mission_id, call_category, timestamp) "
        "VALUES ('a','mReinf','local',0.9,'[]',7,'reinforce','2026-06-16 12:00:00')")
    await db.commit()
    model, _ = await fdb.get_latest_model_for_mission(7)
    assert model == "mReal"  # reinforce row (newer) excluded
    await dabidabi.close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k get_latest_model_for_mission`
Expected: `..._no_tasks_dependency` FAILS — old code references `tasks` (after DROP → `no such table: tasks`) or returns `mB`.

- [ ] **Step 3: Rewrite `get_latest_model_for_mission`**

Replace the body (the `if mission_id is not None:` block with its two `tasks` JOINs, ~360-382) with a single JOIN-free Tier-0 lookup; keep the Tier-2 global fallback unchanged:

```python
async def get_latest_model_for_mission(
    mission_id: int | None,
) -> tuple[str | None, str]:
    """Resolve (picked_model, provider) for a mission's latest non-reinforce pick.

    Two tiers:
      * Tier-0 — most-recent non-reinforce ``model_pick_log`` row whose
        denormalized ``mission_id`` matches (no JOIN — fatih_hoca owns this read
        end-to-end; see the 2026-06-17 registry-decouple spec).
      * Tier-2 — global most-recent non-reinforce pick (mission None / no match).
    Reinforce nudges (``call_category = 'reinforce'``) are excluded so we never
    reinforce a model based on a prior reinforce row. Returns ``(None, 'local')``
    when nothing matches.
    """
    db = await get_db()
    if mission_id is not None:
        cur = await db.execute(
            "SELECT picked_model, provider FROM model_pick_log "
            "WHERE mission_id = ? AND call_category != 'reinforce' "
            "ORDER BY timestamp DESC LIMIT 1",
            (mission_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row and row[0]:
            return row[0], row[1] or "local"
    cur = await db.execute(
        "SELECT picked_model, provider FROM model_pick_log "
        "WHERE call_category != 'reinforce' ORDER BY timestamp DESC LIMIT 1"
    )
    row = await cur.fetchone()
    await cur.close()
    if row and row[0]:
        return row[0], row[1] or "local"
    return None, "local"
```

- [ ] **Step 4: Remove the stale ATTACH docstring line**

In `get_latest_pick_for_task` (~318-320), delete the paragraph beginning "Owns the registry-table READ..." through the ATTACH-split sentence, OR replace it with:

```python
    Pure registry read (no core-table JOIN). Owned by fatih_hoca.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k get_latest_model_for_mission`
Expected: all three PASS.

- [ ] **Step 6: Run the existing registry db-api suite (regression)**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_registry_db_api.py -p no:aiohttp -v`
Expected: PASS. Note: if a pre-existing test asserts the old Tier-1 title-JOIN behavior of `get_latest_model_for_mission`, update it to the new two-tier contract (Tier-1 is intentionally removed).

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/db.py packages/fatih_hoca/tests/test_pick_log_mission_decouple.py
git commit -m "feat(fatih_hoca): read latest mission pick by mission_id, drop tasks JOIN"
```

---

## Task 5: One-shot backfill in `init_db`

**Files:**
- Modify: `packages/db/src/dabidabi/__init__.py` (insert right after line 4050, `await _run_registered_schemas(db)`)
- Test: `packages/fatih_hoca/tests/test_pick_log_mission_decouple.py` (append)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_backfill_populates_mission_id_idempotently(tmp_path):
    dabidabi.configure(str(tmp_path / "bf.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    # Legacy pick row: task_id set, mission_id NULL. Matching tasks row maps it.
    await db.execute("INSERT INTO tasks (id, mission_id, title) VALUES (42, 7, 't')")
    await db.execute(
        "INSERT INTO model_pick_log (task_name, picked_model, picked_score, "
        "candidates_json, task_id) VALUES ('t','m1',0.9,'[]',42)")
    await db.commit()
    await dabidabi.close_db()

    # Re-run init_db on the SAME file → backfill runs.
    dabidabi.configure(str(tmp_path / "bf.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("SELECT mission_id FROM model_pick_log WHERE task_id = 42")
    assert (await cur.fetchone())[0] == 7
    await cur.close()
    await dabidabi.close_db()

    # Third run → still 7, no error (idempotent).
    dabidabi.configure(str(tmp_path / "bf.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("SELECT mission_id FROM model_pick_log WHERE task_id = 42")
    assert (await cur.fetchone())[0] == 7
    await cur.close()
    await dabidabi.close_db()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k backfill`
Expected: FAIL — `mission_id` stays NULL (`assert None == 7`).

- [ ] **Step 3: Add the backfill**

In `packages/db/src/dabidabi/__init__.py`, immediately after line 4050 (`await _run_registered_schemas(db)` — registry schema has now created `model_pick_log` + its `mission_id` column), insert:

```python
    # One-shot idempotent backfill: denormalize mission_id onto legacy
    # model_pick_log rows so fatih_hoca's get_latest_model_for_mission can
    # resolve by mission_id without a JOIN into core `tasks` (registry-decouple
    # slice 1, 2026-06-17 spec). After first run this matches 0 rows. Must run
    # AFTER _run_registered_schemas (which adds the mission_id column).
    try:
        await db.execute(
            "UPDATE model_pick_log SET mission_id = ("
            "SELECT t.mission_id FROM tasks t WHERE t.id = model_pick_log.task_id"
            ") WHERE mission_id IS NULL AND task_id IS NOT NULL"
        )
        await db.commit()
    except Exception as e:
        logger.debug(f"model_pick_log mission_id backfill skipped: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_pick_log_mission_decouple.py -p no:aiohttp -v -k backfill`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/db/src/dabidabi/__init__.py packages/fatih_hoca/tests/test_pick_log_mission_decouple.py
git commit -m "feat(dabidabi): one-shot backfill mission_id onto legacy pick rows"
```

---

## Task 6: Full-suite regression + import smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the touched test packages**

Run: `timeout 300 python -m pytest packages/fatih_hoca/tests/ tests/unit/test_heartbeat_mission_ctx.py -p no:aiohttp -q`
Expected: green (pre-existing unrelated failures noted in the kdv/deferred handoffs — e.g. `test_bench_picks_command.py` — are NOT introduced here; confirm they fail identically on the base commit if any appear).

- [ ] **Step 2: Import smoke (boot-critical modules)**

Run: `timeout 120 python -m pytest --co -q tests/ -p no:aiohttp` then:
Run: `timeout 60 python -c "import src.core.orchestrator, src.telemetry.pick_recorder, src.infra.pick_log, fatih_hoca.db, dabidabi"`
Expected: no import error.

- [ ] **Step 3: Verify no remaining `tasks` reference in the read fn**

Run: `git grep -n "JOIN tasks" packages/fatih_hoca/src/fatih_hoca/db.py`
Expected: no output (the JOIN is gone).

- [ ] **Step 4: Final commit (if any test fixups were needed)**

```bash
git add -A
git commit -m "test(fatih_hoca): regression fixups for mission_id decouple"
```

---

## Done criteria
- `get_latest_model_for_mission` references only `model_pick_log` (no `tasks`).
- New picks carry `mission_id`; legacy rows backfilled idempotently.
- All new tests + existing fatih_hoca suite green.
- Branch ready for 3-way merge into `main` (restart-gated; do NOT push — user `/restart` + verify per spec "Live-restart safety").

## Out of scope (roadmap — separate slices)
`requirements_builder.py:215` core `missions` read + `src.workflows` import; `src.core/src.models/src.app` import inversions. See spec "Remaining blockers".
