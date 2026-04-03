import asyncio
import aiosqlite
import pytest


@pytest.mark.asyncio
async def test_smart_search_tables_exist(tmp_path):
    """Verify all 4 new tables are created by init_db."""
    db_path = str(tmp_path / "test.db")

    import src.infra.db as db_mod

    original = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    db_mod._db = None
    try:
        await db_mod.init_db()
        async with aiosqlite.connect(db_path) as db:
            for table in [
                "api_keywords",
                "api_category_patterns",
                "smart_search_log",
                "api_reliability",
            ]:
                cur = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                row = await cur.fetchone()
                assert row is not None, f"Table {table} not created"
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None
