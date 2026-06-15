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
