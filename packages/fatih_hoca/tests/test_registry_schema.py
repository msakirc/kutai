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
    for c in ("avg_grade", "success_rate", "total_calls", "updated_at"):
        assert c in cols
    assert "recorded_at" not in cols
    await dabidabi.close_db()


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


@pytest.mark.asyncio
async def test_model_pick_log_has_expected_columns(tmp_path):
    dabidabi.configure(str(tmp_path / "reg4.db"))
    await dabidabi.init_db()
    db = await dabidabi.get_db()
    cur = await db.execute("PRAGMA table_info(model_pick_log)")
    cols = {r[1] for r in await cur.fetchall()}
    for c in ("picked_model", "picked_score", "candidates_json", "pool",
              "urgency", "success", "error_category", "provider", "task_id"):
        assert c in cols
    await dabidabi.close_db()


def test_ensure_registry_schema_sync_standalone(tmp_path):
    import sqlite3
    from fatih_hoca.schema import ensure_registry_schema_sync
    conn = sqlite3.connect(str(tmp_path / "bare.db"))
    try:
        ensure_registry_schema_sync(conn)  # must not raise (regression: idx on task_id)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r[0] for r in cur.fetchall()}
        assert {"models", "providers", "registry_events", "model_stats", "model_pick_log"}.issubset(names)
        # task_id index present + column exists
        cols = {r[1] for r in conn.execute("PRAGMA table_info(model_pick_log)").fetchall()}
        assert "task_id" in cols
        idxs = {r[1] for r in conn.execute("PRAGMA index_list(model_pick_log)").fetchall()}
        assert "idx_pick_log_task_id" in idxs
    finally:
        conn.close()
