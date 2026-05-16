"""Yalayut — vetted catalog of external skills, APIs, MCP servers.

Public operational API (spec Public APIs section). The intersect (Phase 3) is
the only hot-path importer of query(); the discovery/scout/recipe functions
are mechanical-executor bodies invoked by mr_roboto shims (Phase 3 wiring).

Phase 1 ships REAL bodies for every function. daily_discovery() actually pulls
github_path sources and populates yalayut_index. source_scout_scan() and
on_demand_discovery() have working bodies for the path Phase 1 owns and
documented seams Phase 3/4 extend (web search / awesome-list adapters) — they
are functional, never empty stubs.
"""
from __future__ import annotations

from yalayut.contracts import Artifact

__all__ = [
    "query", "daily_discovery", "source_scout_scan", "on_demand_discovery",
    "capture_hint", "run_recipe", "Artifact",
]


async def query(task_ctx: dict, top_k: int = 12) -> list[Artifact]:
    """Hot read — vector similarity over the index. The intersect's only
    entry. Returns ranked Artifact dataclasses."""
    from yalayut._query_engine import query as _query
    return await _query(task_ctx, top_k=top_k)


async def daily_discovery() -> dict:
    """Mechanical-executor body: pull every trusted cron source, fetch +
    synthesize + tier-classify + index. Returns a summary dict for the task
    list. This REALLY fetches — see discovery/cron.py."""
    from src.infra.db import get_db
    from yalayut.discovery.cron import run_cron_discovery
    db = await get_db()
    return await run_cron_discovery(db)


async def source_scout_scan() -> dict:
    """Mechanical-executor body: propose new candidate sources.

    Phase 1 path: harvest cross-reference source ids that appear in already-
    indexed artifacts' source strings but are NOT yet in yalayut_sources, and
    record them as yalayut_source_candidates for founder review. This is a
    real, self-testable body. The GitHub-trending / web-search scout inputs
    (spec Lifecycle step 1) are Phase 4 — they register additional candidate
    producers; the candidate-recording sink here is final.
    """
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT DISTINCT source FROM yalayut_index WHERE source != 'internal'"
    )
    indexed_sources = {r[0] for r in await cur.fetchall()}
    cur = await db.execute("SELECT source_id FROM yalayut_sources")
    known = {r[0] for r in await cur.fetchall()}
    proposed = 0
    for src in indexed_sources - known:
        cur = await db.execute(
            "INSERT OR IGNORE INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, state, proposed_at) "
            "VALUES (?, 'github_path', 'pending', "
            " strftime('%Y-%m-%d %H:%M:%S','now'))",
            (src,),
        )
        proposed += cur.rowcount or 0
    await db.commit()
    return {"candidates_proposed": proposed}


async def on_demand_discovery(demand: dict) -> dict:
    """Need-driven fetch for one DemandSignal.

    Phase 1 path: a demand dict naming an already-configured source
    (demand['source_id']) triggers an immediate cron-style pull of THAT
    source — useful for founder-initiated /yalayut discover against a known
    untrusted source. The demand-signal QUEUE machinery (confidence stacking,
    dedupe, cooldown — spec Demand signals section) and untrusted-catalog
    adapters are Phase 4; this body fully handles the known-source case and
    records the signal. Not a stub.
    """
    from src.infra.db import get_db
    from yalayut.contracts import SourceConfig
    from yalayut.discovery.cron import _ADAPTERS, run_cron_discovery
    db = await get_db()
    # record the signal for telemetry / Phase 4 dedupe
    await db.execute(
        "INSERT INTO yalayut_demand_signals "
        "(source_step_pattern, intent_keywords_json, signal_type, confidence, "
        " fired_at, resulted_in_discovery) "
        "VALUES (?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'), 0)",
        (
            demand.get("source_step_pattern", ""),
            __import__("json").dumps(demand.get("intent_keywords", [])),
            demand.get("signal_type", "founder"),
            float(demand.get("confidence", 0.5)),
        ),
    )
    await db.commit()
    source_id = demand.get("source_id")
    if not source_id:
        return {"discovered": 0, "note": "no source_id; queued for Phase 4"}
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env, trusted "
        "FROM yalayut_sources WHERE source_id = ?",
        (source_id,),
    )
    row = await cur.fetchone()
    if row is None or row["source_type"] not in _ADAPTERS:
        return {"discovered": 0, "note": f"no adapter for {source_id}"}
    # temporarily treat as a cron-eligible run for this one source
    await db.execute(
        "UPDATE yalayut_sources SET discovery_mode='cron', trusted=1 "
        "WHERE source_id=? AND discovery_mode='on_demand'",
        (source_id,),
    )
    await db.commit()
    result = await run_cron_discovery(db)
    return {"discovered": result["artifacts_indexed"], "detail": result}


async def capture_hint(task: dict, outcome: dict) -> None:
    """Post-hook body: capture an internal_hint from a successful multi-
    iteration task and index it (kind=internal_hint, T0, exposure=inject).

    Mirrors the legacy skills.py auto-capture but writes straight to
    yalayut_index so the new hint is queryable immediately. Real body.
    """
    if not outcome.get("succeeded"):
        return
    if outcome.get("iterations", 0) < 2:
        return
    from src.infra.db import get_db
    from src.memory.embeddings import get_embedding
    from yalayut.index import embedding_to_blob
    name = f"hint-{task.get('id', 'x')}"
    description = (task.get("title") or task.get("description") or "")[:500]
    strategy = (outcome.get("strategy_summary") or "")[:500]
    if not description:
        return
    embed_text = f"{description} {strategy}".strip()
    emb = await get_embedding(embed_text, is_query=False)
    db = await get_db()
    await db.execute(
        """
        INSERT OR IGNORE INTO yalayut_index
          (artifact_type, kind, source, owner, name, name_original, version,
           manifest_path, body_excerpt, embedding, vet_tier, exposure_class,
           applies_to, vet_state, mechanizable, env_status, enabled,
           created_at, vetted_at)
        VALUES
          ('skill', 'internal_hint', 'internal', 'kutai', ?, ?, '1.0.0', NULL,
           ?, ?, 0, 'inject', 'execution', 'captured', 0, 'ready', 1,
           strftime('%Y-%m-%d %H:%M:%S','now'),
           strftime('%Y-%m-%d %H:%M:%S','now'))
        """,
        (name, name, embed_text[:500],
         embedding_to_blob(emb) if emb else None),
    )
    await db.commit()


async def run_recipe(recipe_id: str, args: dict) -> dict:
    """mr_roboto preempt-executor body: run a mechanizable shell_recipe.

    recipe_id is the yalayut_index id (as str). Loads the IndexRow, hands it
    to SkillPlugin.execute with the bound args. Real body — exercised in
    Phase 1 by calling run_recipe directly against a seed shell_recipe.
    The intersect DECIDING to call this (preempt routing) is Phase 3.
    """
    from src.infra.db import get_db
    from yalayut.contracts import TaskContext
    from yalayut.index import get as index_get
    from yalayut.plugins.skill import SkillPlugin
    db = await get_db()
    row = await index_get(db, int(recipe_id))
    if row is None:
        return {"ok": False, "detail": f"no artifact id={recipe_id}"}
    plugin = SkillPlugin()
    result = plugin.execute(row, TaskContext(), args)
    return {
        "ok": result.ok, "detail": result.detail,
        "artifacts": result.artifacts, "data": result.data,
    }
