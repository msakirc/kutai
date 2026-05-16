"""Migration: existing skills rows -> yalayut_index."""
import pytest

from yalayut.migration import migrate_skills_to_yalayut, run_full_migration

pytestmark = pytest.mark.asyncio


async def _make_legacy_skills(db):
    """Build the legacy skills table + 2 rows on the same connection."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            skill_type TEXT DEFAULT 'auto',
            strategies TEXT DEFAULT '[]',
            injection_count INTEGER DEFAULT 0,
            injection_success INTEGER DEFAULT 0,
            created_at TEXT, updated_at TEXT
        )
    """)
    await db.execute(
        "INSERT INTO skills (name, description, strategies) VALUES (?, ?, ?)",
        ("route-shopping", "Route shopping queries to advisor",
         '["use shopping_advisor"]'),
    )
    await db.execute(
        "INSERT INTO skills (name, description, strategies) VALUES (?, ?, ?)",
        ("debug-imports", "Fix circular import errors", "[]"),
    )
    await db.commit()


async def test_migration_copies_rows(yalayut_db, monkeypatch):
    await _make_legacy_skills(yalayut_db)

    # embedding stub — deterministic, no sentence-transformers in tests
    async def fake_embed(text, is_query=True):
        return [float(len(text) % 7)] + [0.0] * 767

    monkeypatch.setattr(
        "yalayut.migration._embed", fake_embed,
    )
    result = await migrate_skills_to_yalayut(yalayut_db)
    assert result["migrated"] == 2

    cur = await yalayut_db.execute(
        "SELECT name, kind, artifact_type, exposure_class, vet_tier, source, "
        "       embedding FROM yalayut_index"
    )
    rows = await cur.fetchall()
    assert len(rows) == 2
    for r in rows:
        assert r["kind"] == "internal_hint"
        assert r["artifact_type"] == "skill"
        assert r["exposure_class"] == "inject"
        assert r["vet_tier"] == 0
        assert r["source"] == "internal"
        assert r["embedding"] is not None        # real embedding stored


async def test_migration_idempotent(yalayut_db, monkeypatch):
    await _make_legacy_skills(yalayut_db)

    async def fake_embed(text, is_query=True):
        return [1.0] + [0.0] * 767

    monkeypatch.setattr("yalayut.migration._embed", fake_embed)
    await migrate_skills_to_yalayut(yalayut_db)
    second = await migrate_skills_to_yalayut(yalayut_db)
    assert second["migrated"] == 0   # UNIQUE(source,name,version) -> no dups
    cur = await yalayut_db.execute("SELECT COUNT(*) c FROM yalayut_index")
    assert (await cur.fetchone())["c"] == 2


async def test_migration_handles_no_skills_table(yalayut_db, monkeypatch):
    async def fake_embed(text, is_query=True):
        return [1.0] + [0.0] * 767
    monkeypatch.setattr("yalayut.migration._embed", fake_embed)
    # no skills table at all
    result = await migrate_skills_to_yalayut(yalayut_db)
    assert result["migrated"] == 0
    assert result["skipped_no_table"] is True


async def test_run_full_migration_seeds_and_migrates(yalayut_db, monkeypatch):
    await _make_legacy_skills(yalayut_db)

    async def fake_embed(text, is_query=True):
        return [1.0] + [0.0] * 767
    monkeypatch.setattr("yalayut.migration._embed", fake_embed)
    result = await run_full_migration(yalayut_db)
    assert result["owners_seeded"] == 7
    assert result["sources_seeded"] == 4
    assert result["policy_seeded"] is True
    assert result["skills_migrated"] == 2


# ── H1 regression ──────────────────────────────────────────────────────────────

async def test_run_full_migration_indexes_20_seed_manifests(
    yalayut_db, monkeypatch
):
    """H1: run_full_migration() must install all 20 seed manifests into the
    index so they are queryable.  Before the fix this yields seeds_indexed=0."""
    async def fake_embed(text, is_query=True):
        return [1.0] + [0.0] * 767
    monkeypatch.setattr("yalayut.migration._embed", fake_embed)

    result = await run_full_migration(yalayut_db)
    assert result.get("seeds_indexed", 0) == 20, (
        f"expected 20 seed manifests in index, got {result}"
    )

    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_index WHERE source != 'internal'"
    )
    count = (await cur.fetchone())["c"]
    assert count == 20, f"expected 20 seed rows in yalayut_index, got {count}"


async def test_run_full_migration_seed_index_is_idempotent(
    yalayut_db, monkeypatch
):
    """H1 idempotency: running run_full_migration() twice must not duplicate
    seed rows — UNIQUE(source, name, version) must hold."""
    async def fake_embed(text, is_query=True):
        return [1.0] + [0.0] * 767
    monkeypatch.setattr("yalayut.migration._embed", fake_embed)

    await run_full_migration(yalayut_db)
    await run_full_migration(yalayut_db)

    cur = await yalayut_db.execute(
        "SELECT COUNT(*) c FROM yalayut_index WHERE source != 'internal'"
    )
    count = (await cur.fetchone())["c"]
    assert count == 20, f"double-run must not duplicate seeds; got {count} rows"
