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
