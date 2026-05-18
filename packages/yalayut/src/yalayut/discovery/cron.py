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
import yaml
from datetime import timedelta

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db, from_db
from yalayut.contracts import SourceConfig
from yalayut.discovery.sources.github_path import GithubPathAdapter
from yalayut.discovery.synthesize import synthesize
from yalayut.index import store
from yalayut.manifest import validate_manifest
from yalayut.tier_classifier import classify
from yalayut.trust import owner_max_tier, source_max_tier
from yalayut.vetting.auto_checks import run_all

logger = get_logger("yalayut.cron")

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


async def _ingest_one_source(
    db: aiosqlite.Connection,
    cfg: SourceConfig,
    *,
    artifact_cap: int | None = None,
) -> dict:
    """Run the full per-source ingest pipeline for ONE source.

    discover -> fetch -> synthesize -> validate_manifest -> run_all ->
    source/owner tier -> classify -> embed -> store.

    This is the REAL per-source pipeline — extracted from run_cron_discovery
    so that the Phase 4 ``daily_discovery``/``on_demand_discovery`` loops can
    reuse it instead of calling a non-existent ``fetch.ingest_all``.

    Returns ``{ingested, enabled, quarantined, skipped_disabled, errors}``.
    ``ingested`` counts rows written to yalayut_index; ``enabled`` counts those
    with tier in (0,1); ``quarantined`` counts tier in (2,3). Never raises —
    adapter failures are isolated into ``errors``.

    ``artifact_cap`` (if set) limits how many artifacts a single source run
    will fetch+index — the on-demand first-flood guard.
    """
    result = {
        "ingested": 0,
        "enabled": 0,
        "quarantined": 0,
        "skipped_disabled": 0,
        "errors": [],
    }
    adapter = _ADAPTERS.get(cfg.source_type)
    if adapter is None:
        result["errors"].append(f"no adapter for {cfg.source_type}")
        return result

    try:
        refs = await adapter.discover(cfg)
    except Exception as e:  # adapter network failure — isolate per source
        result["errors"].append(
            f"discover {cfg.source_id}: {type(e).__name__}: {e}"
        )
        return result

    for ref in refs:
        if artifact_cap is not None and result["ingested"] >= artifact_cap:
            break
        if await _is_disabled(db, ref.source_id, ref.name):
            result["skipped_disabled"] += 1
            continue
        try:
            body_path = await adapter.fetch(ref)
            raw = body_path.read_bytes()
            manifest, body = synthesize(ref, raw)
            manifest_errs = validate_manifest(manifest)
            if manifest_errs:
                result["errors"].append(
                    f"validate {ref.name}: " + "; ".join(manifest_errs)
                )
                continue
            # H3 fix: write manifest.yaml next to SKILL.md so that
            # _to_artifact can load intent_keywords/inputs_schema via
            # _load_manifest_fields().  The YAML shape mirrors
            # parse_manifest_yaml()'s expected keys (round-trip safe).
            manifest_yaml_path = body_path.parent / "manifest.yaml"
            manifest_dict = {
                "name": manifest.name,
                "name_original": manifest.name_original,
                "version": manifest.version,
                "artifact_type": manifest.artifact_type,
                "kind": manifest.kind,
                "source": manifest.source,
                "owner": manifest.owner,
                "license": manifest.license,
                "mechanizable": manifest.mechanizable,
                "model_hint": manifest.model_hint,
                "applies_to": manifest.applies_to,
                "intent_keywords": list(manifest.intent_keywords),
                "inputs_schema": dict(manifest.inputs_schema),
            }
            manifest_yaml_path.write_text(
                yaml.safe_dump(manifest_dict, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception as e:
            result["errors"].append(
                f"fetch {ref.name}: {type(e).__name__}: {e}"
            )
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
        result["ingested"] += 1
        # store() sets enabled=1 for tier in (0,1), 0 for (2,3).
        if tier in (0, 1):
            result["enabled"] += 1
        else:
            result["quarantined"] += 1

    await db.execute(
        "UPDATE yalayut_sources SET "
        "last_run_at = strftime('%Y-%m-%d %H:%M:%S','now') "
        "WHERE source_id = ?",
        (cfg.source_id,),
    )
    await db.commit()
    return result


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
        cfg = SourceConfig(
            source_id=sr["source_id"], source_type=sr["source_type"],
            endpoint=sr["endpoint"] or "", auth_env=sr["auth_env"],
            trusted=bool(sr["trusted"]), discovery_mode=sr["discovery_mode"],
            min_interval_s=sr["min_interval_s"],
        )
        if cfg.source_type not in _ADAPTERS:
            errors.append(f"no adapter for {cfg.source_type}")
            continue
        sources_run += 1
        res = await _ingest_one_source(db, cfg)
        artifacts_indexed += res["ingested"]
        skipped_disabled += res["skipped_disabled"]
        errors.extend(res["errors"])

    return {
        "sources_run": sources_run,
        "artifacts_indexed": artifacts_indexed,
        "skipped_disabled": skipped_disabled,
        "errors": errors,
    }


# ── Phase 4: daily_discovery() — demand-aware cron loop ─────────────────────

async def _due_cron_sources() -> list[dict]:
    """Trusted, enabled, cron/both sources whose min_interval has elapsed."""
    db = await get_db()
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env, "
        "       min_interval_s, last_run_at "
        "FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 1 "
        "  AND discovery_mode IN ('cron', 'both')",
    )
    rows = await cur.fetchall()
    await cur.close()
    now = utc_now()
    due: list[dict] = []
    for sid, stype, endpoint, auth_env, min_iv, last_run in rows:
        if last_run and min_iv:
            try:
                last_dt = from_db(str(last_run))
                if now - last_dt < timedelta(seconds=int(min_iv)):
                    continue
            except (ValueError, TypeError):
                pass
        due.append({
            "source_id": sid, "source_type": stype, "endpoint": endpoint,
            "auth_env": auth_env, "min_interval_s": min_iv,
        })
    return due


async def _ingest_source(source_row: dict) -> dict:
    """Run the real per-source ingest pipeline for one source row.

    Builds a SourceConfig from the source_row dict, runs the shared
    ``_ingest_one_source`` helper (discover -> fetch -> synthesize -> vet ->
    classify -> embed -> store), and returns ``{ingested, enabled,
    quarantined}``. Never raises — a missing adapter yields a zero-count
    dict with an error string. The Phase 4 cron test monkeypatches this
    function as the per-source seam.
    """
    cfg = SourceConfig(
        source_id=source_row["source_id"],
        source_type=source_row["source_type"],
        endpoint=source_row.get("endpoint") or "",
        auth_env=source_row.get("auth_env"),
        trusted=bool(source_row.get("trusted", True)),
        discovery_mode=source_row.get("discovery_mode", "cron"),
        min_interval_s=source_row.get("min_interval_s"),
    )
    db = await get_db()
    res = await _ingest_one_source(db, cfg)
    return {
        "ingested": res["ingested"],
        "enabled": res["enabled"],
        "quarantined": res["quarantined"],
        "errors": res["errors"],
    }


async def _mark_ran(source_id: str) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_sources SET last_run_at = ? WHERE source_id = ?",
        (to_db(utc_now()), source_id),
    )
    await db.commit()


async def daily_discovery() -> dict:
    """Pull all due trusted cron-mode sources. Returns a summary dict."""
    due = await _due_cron_sources()
    summary = {
        "sources_scanned": 0,
        "sources_skipped": 0,
        "artifacts_ingested": 0,
        "artifacts_enabled": 0,
        "artifacts_quarantined": 0,
        "errors": [],
    }
    # Count rows that exist but are not due as "skipped".
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 1 "
        "  AND discovery_mode IN ('cron', 'both')")
    total = (await cur.fetchone())[0]
    await cur.close()
    summary["sources_skipped"] = max(0, total - len(due))

    for src in due:
        try:
            res = await _ingest_source(src)
            summary["sources_scanned"] += 1
            summary["artifacts_ingested"] += int(res.get("ingested", 0))
            summary["artifacts_enabled"] += int(res.get("enabled", 0))
            summary["artifacts_quarantined"] += int(res.get("quarantined", 0))
            await _mark_ran(src["source_id"])
        except Exception as e:  # noqa: BLE001 — one bad source must not abort
            logger.warning("cron ingest failed for %s: %s",
                            src["source_id"], e)
            summary["errors"].append(f"{src['source_id']}: {e}")
    logger.info("daily_discovery complete", **{
        k: v for k, v in summary.items() if k != "errors"})
    return summary
