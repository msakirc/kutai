"""Yalayut Phase 4 — on-demand discovery.

``on_demand_discovery(demand)`` is the need-driven path: one ``DemandSignal``
fires, this fetches against *untrusted* sources whose intent overlaps the
demand's keywords. Volume-dangerous catalogs (public-apis ~1.4k, awesome-mcp
~1k) are only pulled here, with a per-source artifact cap.
"""
from __future__ import annotations

import inspect

from dabidabi import get_db
from src.infra.logging_config import get_logger
from yalayut.discovery import demand as _demand

logger = get_logger("yalayut.on_demand")

#: Per-source artifact cap for an on-demand run — first-flood guard.
ON_DEMAND_ARTIFACT_CAP: int = 10


async def _untrusted_sources_for(intent_keywords: list[str]) -> list[dict]:
    """Untrusted, enabled, on_demand/both sources. The intent filter is
    coarse — adapters apply per-artifact keyword matching downstream."""
    db = await get_db()
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env "
        "FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 0 "
        "  AND discovery_mode IN ('on_demand', 'both')",
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"source_id": s, "source_type": t, "endpoint": e, "auth_env": a,
         "intent_keywords": intent_keywords}
        for s, t, e, a in rows
    ]


async def _ingest_source_capped(source_row: dict, *,
                                artifact_cap: int = ON_DEMAND_ARTIFACT_CAP) -> dict:
    """Real per-source ingest pipeline with a per-run artifact cap.

    Builds a SourceConfig from the source_row dict and runs the shared
    ``_ingest_one_source`` helper, honoring ``artifact_cap`` as the
    first-flood guard. Never raises — a missing adapter yields a zero-count
    dict with an error string.
    """
    from yalayut.contracts import SourceConfig
    from yalayut.discovery.cron import _ingest_one_source

    cfg = SourceConfig(
        source_id=source_row["source_id"],
        source_type=source_row["source_type"],
        endpoint=source_row.get("endpoint") or "",
        auth_env=source_row.get("auth_env"),
        trusted=bool(source_row.get("trusted", False)),
        discovery_mode=source_row.get("discovery_mode", "on_demand"),
        min_interval_s=source_row.get("min_interval_s"),
    )
    db = await get_db()
    res = await _ingest_one_source(db, cfg, artifact_cap=artifact_cap)
    return {
        "ingested": res["ingested"],
        "enabled": res["enabled"],
        "quarantined": res["quarantined"],
        "errors": res["errors"],
    }


async def on_demand_discovery(demand: dict) -> dict:
    """Fetch untrusted sources for one demand signal. Returns a summary dict."""
    pattern = demand.get("source_step_pattern", "")
    keywords = list(demand.get("intent_keywords") or [])
    summary = {
        "pattern": pattern,
        "sources_scanned": 0,
        "artifacts_ingested": 0,
        "artifacts_enabled": 0,
        "artifacts_quarantined": 0,
        "errors": [],
    }
    _sources_result = _untrusted_sources_for(keywords)
    sources = await _sources_result if inspect.isawaitable(_sources_result) else _sources_result
    for src in sources:
        try:
            res = await _ingest_source_capped(
                src, artifact_cap=ON_DEMAND_ARTIFACT_CAP)
            summary["sources_scanned"] += 1
            summary["artifacts_ingested"] += int(res.get("ingested", 0))
            summary["artifacts_enabled"] += int(res.get("enabled", 0))
            summary["artifacts_quarantined"] += int(res.get("quarantined", 0))
        except Exception as e:  # noqa: BLE001
            logger.warning("on-demand ingest failed for %s: %s",
                            src["source_id"], e)
            summary["errors"].append(f"{src['source_id']}: {e}")

    # Consume the demand: every signal on this pattern is marked discovered
    # so the next pending_signals() sweep skips it.
    if pattern:
        await _demand.mark_discovered(pattern)
    logger.info("on_demand_discovery complete", pattern=pattern,
                ingested=summary["artifacts_ingested"])
    return summary
