"""Daily discovery cron — pull every trusted cron source, fetch, synthesize,
vet, tier-classify, and index.

This is the REAL body behind daily_discovery() — not a stub. It walks
yalayut_sources WHERE discovery_mode IN ('cron','both') AND trusted=1, runs
the matching SourceAdapter end-to-end, and writes vetted rows to yalayut_index.
Phase 1 ships the github_path adapter; the adapter registry is keyed on
source_type so Phase 3 adds public_apis_md / cookiecutter_template by
registering more entries.
"""
from __future__ import annotations

import aiosqlite

from yalayut.contracts import SourceConfig
from yalayut.discovery.sources.github_path import GithubPathAdapter
from yalayut.discovery.synthesize import synthesize
from yalayut.index import store
from yalayut.tier_classifier import classify
from yalayut.trust import owner_max_tier, source_max_tier
from yalayut.vetting.auto_checks import run_all

# adapter registry — source_type -> adapter instance. Phase 3 extends this.
_ADAPTERS = {
    "github_path": GithubPathAdapter(),
}


async def _embed(text: str, is_query: bool = False) -> list[float] | None:
    """Embed via KutAI's shared utility. Lazy import; monkeypatched in tests."""
    from src.memory.embeddings import get_embedding
    return await get_embedding(text, is_query=is_query)


async def _is_disabled(
    db: aiosqlite.Connection, source: str, name: str
) -> bool:
    cur = await db.execute(
        "SELECT 1 FROM yalayut_disabled_imports "
        "WHERE source = ? AND artifact_name = ?",
        (source, name),
    )
    return await cur.fetchone() is not None


async def run_cron_discovery(db: aiosqlite.Connection) -> dict:
    """Pull all trusted cron sources end-to-end. Returns a result summary
    dict (the mechanical-task body daily_discovery() returns)."""
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env, trusted, "
        "       discovery_mode, min_interval_s FROM yalayut_sources "
        "WHERE discovery_mode IN ('cron','both') AND trusted = 1 "
        "AND enabled = 1"
    )
    source_rows = await cur.fetchall()

    sources_run = 0
    artifacts_indexed = 0
    skipped_disabled = 0
    errors: list[str] = []

    for sr in source_rows:
        adapter = _ADAPTERS.get(sr["source_type"])
        if adapter is None:
            errors.append(f"no adapter for {sr['source_type']}")
            continue
        cfg = SourceConfig(
            source_id=sr["source_id"], source_type=sr["source_type"],
            endpoint=sr["endpoint"] or "", auth_env=sr["auth_env"],
            trusted=bool(sr["trusted"]), discovery_mode=sr["discovery_mode"],
            min_interval_s=sr["min_interval_s"],
        )
        sources_run += 1
        try:
            refs = await adapter.discover(cfg)
        except Exception as e:  # adapter network failure — isolate per source
            errors.append(f"discover {cfg.source_id}: {type(e).__name__}: {e}")
            continue

        for ref in refs:
            if await _is_disabled(db, ref.source_id, ref.name):
                skipped_disabled += 1
                continue
            try:
                body_path = await adapter.fetch(ref)
                raw = body_path.read_bytes()
                manifest, body = synthesize(ref, raw)
            except Exception as e:
                errors.append(f"fetch {ref.name}: {type(e).__name__}: {e}")
                continue

            check_maxes = await run_all(db, manifest, body_path)
            src_max = await source_max_tier(db, manifest.source)
            own_max = await owner_max_tier(db, manifest.owner)
            tier, audit = classify(src_max, own_max, check_maxes)

            emb = await _embed(
                f"{manifest.name} {manifest.name_original} "
                f"{' '.join(manifest.intent_keywords)} {body[:500]}",
                is_query=False,
            )
            await store(
                db, manifest, body, tier, audit, emb or [0.0] * 768,
                manifest_path=str(body_path.parent / "manifest.yaml"),
            )
            artifacts_indexed += 1

        await db.execute(
            "UPDATE yalayut_sources SET "
            "last_run_at = strftime('%Y-%m-%d %H:%M:%S','now') "
            "WHERE source_id = ?",
            (cfg.source_id,),
        )
        await db.commit()

    return {
        "sources_run": sources_run,
        "artifacts_indexed": artifacts_indexed,
        "skipped_disabled": skipped_disabled,
        "errors": errors,
    }
