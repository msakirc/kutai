import aiosqlite
import pytest


@pytest.mark.asyncio
async def test_load_mode_from_db_normalizes_legacy(tmp_path):
    from nerd_herd.__main__ import _load_mode_from_db
    db = str(tmp_path / "t.db")
    async with aiosqlite.connect(db, isolation_level=None) as c:
        await c.execute(
            "CREATE TABLE load_mode (id INTEGER PRIMARY KEY, mode TEXT NOT NULL, "
            "auto_managed INTEGER NOT NULL DEFAULT 1, updated_at TEXT)"
        )
        await c.execute("INSERT INTO load_mode (id, mode) VALUES (1, 'shared')")
        await c.commit()
    assert await _load_mode_from_db(db) == "balanced"


@pytest.mark.asyncio
async def test_load_mode_from_db_missing_returns_full(tmp_path):
    from nerd_herd.__main__ import _load_mode_from_db
    assert await _load_mode_from_db(str(tmp_path / "none.db")) == "full"
