"""Migration — install schema/seed and copy legacy skills rows.

Per spec Migration section: existing `skills` rows become yalayut_index rows
with kind='internal_hint', exposure_class='inject', vet_tier=0,
source='internal'. The embedding is built from description + strategies so the
hint is searchable by query() exactly like a fetched skill.

run_full_migration() is the single boot-time entry: schema -> policy seed ->
owner/source/disabled-import seed -> skills copy. Idempotent throughout.
"""
from __future__ import annotations

import json

import aiosqlite

from yalayut.index import embedding_to_blob
from yalayut.schema import ensure_yalayut_schema
from yalayut.seed.seed_data import (
    seed_disabled_imports, seed_owners, seed_sources,
)
from yalayut.vetting.policy import seed_policy


async def _embed(text: str, is_query: bool = False) -> list[float] | None:
    """Embed via KutAI's shared utility. Lazy import keeps yalayut light and
    lets tests monkeypatch this symbol directly."""
    from src.memory.embeddings import get_embedding
    return await get_embedding(text, is_query=is_query)


async def _skills_table_exists(db: aiosqlite.Connection) -> bool:
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills'"
    )
    return await cur.fetchone() is not None


async def migrate_skills_to_yalayut(db: aiosqlite.Connection) -> dict:
    """Copy every legacy `skills` row into yalayut_index. Idempotent —
    UNIQUE(source, name, version) makes a re-run a no-op."""
    if not await _skills_table_exists(db):
        return {"migrated": 0, "skipped_no_table": True}

    cur = await db.execute(
        "SELECT name, description, strategies FROM skills"
    )
    rows = await cur.fetchall()
    migrated = 0
    for r in rows:
        name = r["name"]
        description = r["description"] or ""
        strategies = r["strategies"] or "[]"
        try:
            strat_list = json.loads(strategies)
        except (json.JSONDecodeError, TypeError):
            strat_list = []
        strat_text = " ".join(str(s) for s in strat_list)
        embed_text = f"{description} {strat_text}".strip()
        emb = await _embed(embed_text, is_query=False)
        emb_blob = embedding_to_blob(emb) if emb else None
        excerpt = embed_text[:500]
        ins = await db.execute(
            """
            INSERT OR IGNORE INTO yalayut_index
              (artifact_type, kind, source, owner, name, name_original,
               version, manifest_path, body_excerpt, embedding, vet_tier,
               exposure_class, applies_to, vet_state, mechanizable,
               env_status, enabled, created_at, vetted_at)
            VALUES
              ('skill', 'internal_hint', 'internal', 'kutai', ?, ?, '1.0.0',
               NULL, ?, ?, 0, 'inject', 'execution', 'migrated', 0, 'ready',
               1, strftime('%Y-%m-%d %H:%M:%S','now'),
               strftime('%Y-%m-%d %H:%M:%S','now'))
            """,
            (name, name, excerpt, emb_blob),
        )
        migrated += ins.rowcount or 0
    await db.commit()
    return {"migrated": migrated, "skipped_no_table": False}


async def run_full_migration(db: aiosqlite.Connection) -> dict:
    """Boot-time entry: schema + all seeds + skills copy. Idempotent."""
    await ensure_yalayut_schema(db)
    await seed_policy(db)
    owners = await seed_owners(db)
    sources = await seed_sources(db)
    disabled = await seed_disabled_imports(db)
    skills = await migrate_skills_to_yalayut(db)
    return {
        "owners_seeded": owners,
        "sources_seeded": sources,
        "disabled_imports_seeded": disabled,
        "policy_seeded": True,
        "skills_migrated": skills["migrated"],
    }
