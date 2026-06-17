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
    # Minimal pre-change shape. MUST include every column the REGISTRY_DDL
    # indexes reference (timestamp, provider, task_id) — create_registry_schema
    # runs those CREATE INDEX statements with no try/except, and
    # `CREATE INDEX IF NOT EXISTS` does NOT suppress a missing-column error. Only
    # `mission_id` is intentionally absent (the ALTER must add it).
    await db.execute(
        "CREATE TABLE model_pick_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "task_name TEXT NOT NULL, picked_model TEXT NOT NULL, "
        "picked_score REAL NOT NULL, candidates_json TEXT NOT NULL, "
        "provider TEXT, task_id INTEGER)")
    await db.commit()
    await create_registry_schema(db)
    assert "mission_id" in await _col_names(db, "model_pick_log")
    await create_registry_schema(db)  # idempotent — must not raise
    assert "mission_id" in await _col_names(db, "model_pick_log")
    await dabidabi.close_db()


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
