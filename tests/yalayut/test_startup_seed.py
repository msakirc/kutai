"""Fix 3 — yalayut seed manifests must be loaded into yalayut_index at
startup (init_db), not only from tests.

Before the fix, init_db() only ran ensure_yalayut_schema (schema, no data),
so a fresh deploy had an empty yalayut_index and yalayut.query() returned
nothing — the whole catalog->intersect pipeline carried zero items.
"""
import asyncio

import pytest

import src.infra.db as _db


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_init_db_loads_yalayut_seed_catalog(loop, monkeypatch, tmp_path):
    """A fresh init_db() must populate yalayut_index with the seed manifests
    (so query() has something to return)."""
    async def _run():
        # Deterministic embedding — no sentence-transformers in tests.
        async def fake_embed(text, is_query=False):
            return [1.0] + [0.0] * 767

        monkeypatch.setattr("yalayut.migration._embed", fake_embed)

        # Point the DB singleton at a fresh temp file.
        fresh = tmp_path / "fresh_kutai.db"
        monkeypatch.setattr(_db, "DB_PATH", str(fresh))
        # Drop any cached connection so get_db reopens against the temp path.
        monkeypatch.setattr(_db, "_db_connection", None)
        monkeypatch.setattr(_db, "_db_connection_path", None)

        await _db.init_db()
        db = await _db.get_db()

        cur = await db.execute(
            "SELECT COUNT(*) c FROM yalayut_index WHERE source != 'internal'")
        seed_count = (await cur.fetchone())["c"]
        await cur.close()
        assert seed_count >= 20, (
            f"init_db must load the seed manifests; got {seed_count} rows")

        # Owners + sources seeded too.
        cur = await db.execute("SELECT COUNT(*) c FROM yalayut_sources")
        assert (await cur.fetchone())["c"] >= 4
        await cur.close()

    loop.run_until_complete(_run())


def test_init_db_seed_load_is_idempotent(loop, monkeypatch, tmp_path):
    """A second init_db() on an already-seeded DB must not duplicate seed
    rows — the empty-index gate makes the expensive path run once only."""
    async def _run():
        async def fake_embed(text, is_query=False):
            return [1.0] + [0.0] * 767

        monkeypatch.setattr("yalayut.migration._embed", fake_embed)

        fresh = tmp_path / "fresh_kutai2.db"
        monkeypatch.setattr(_db, "DB_PATH", str(fresh))
        monkeypatch.setattr(_db, "_db_connection", None)
        monkeypatch.setattr(_db, "_db_connection_path", None)

        await _db.init_db()
        await _db.init_db()
        db = await _db.get_db()

        cur = await db.execute(
            "SELECT COUNT(*) c FROM yalayut_index WHERE source != 'internal'")
        count = (await cur.fetchone())["c"]
        await cur.close()
        assert count >= 20
        # No duplication: UNIQUE(source,name,version) holds, and the gate
        # means the second init_db skipped run_full_migration entirely.
        cur = await db.execute(
            "SELECT source, name, version, COUNT(*) c FROM yalayut_index "
            "GROUP BY source, name, version HAVING c > 1")
        dups = await cur.fetchall()
        await cur.close()
        assert not dups, f"seed rows duplicated on second init_db: {dups}"

    loop.run_until_complete(_run())
