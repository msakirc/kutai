"""Yalayut Phase 4 — autonomous source scout.

``source_scout_scan()`` proposes candidate sources from four signals:
  1. GitHub trending in relevant topics
  2. cross-refs in approved artifacts' READMEs
  3. web search on accumulated demand signals
  4. founder-mentioned URLs (queued via /yalayut scout <url>)

Candidates are deduped against existing ``yalayut_sources`` and pending
``yalayut_source_candidates`` rows, capped per day, and written as
``pending`` rows for founder review. The scout NEVER auto-adds a source.
"""
from __future__ import annotations

import json

import httpx

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.source_scout")

#: Spec — per-day candidate cap (default 5).
DAILY_CANDIDATE_CAP: int = 5

#: GitHub topics the scout watches for trending skill/agent repos.
SCOUT_TOPICS: tuple[str, ...] = (
    "claude-skill", "agent-skill", "mcp-server", "cookiecutter-template",
)


async def _scan_github_trending() -> list[dict]:
    """Signal 1 — GitHub trending repos in skill/agent topics."""
    out: list[dict] = []
    for topic in SCOUT_TOPICS:
        url = (f"https://api.github.com/search/repositories"
               f"?q=topic:{topic}&sort=stars&per_page=5")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url, headers={"Accept": "application/vnd.github+json"})
                if resp.status_code != 200:
                    continue
                items = resp.json().get("items", [])
        except Exception as e:  # noqa: BLE001
            logger.debug("github trending scan failed (%s): %s", topic, e)
            continue
        for repo in items:
            full = repo.get("full_name", "")
            if not full:
                continue
            out.append({
                "candidate_source_id": f"github:{full}",
                "source_type": "github_topic",
                "endpoint": topic,
                "metadata_json": json.dumps({
                    "description": repo.get("description") or "",
                    "stars": repo.get("stargazers_count"),
                    "topic": topic,
                }),
            })
    return out


async def _scan_readme_crossrefs() -> list[dict]:
    """Signal 2 — GitHub URLs referenced in approved artifacts' READMEs."""
    # Approved artifacts' bodies are on disk under vendor/skills. Phase 1/3
    # stored body_excerpt in yalayut_index — scan it for github.com refs.
    import re
    db = await get_db()
    cur = await db.execute(
        "SELECT DISTINCT body_excerpt FROM yalayut_index "
        "WHERE enabled = 1 AND body_excerpt IS NOT NULL")
    rows = await cur.fetchall()
    await cur.close()
    ref_re = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")
    seen: set[str] = set()
    out: list[dict] = []
    for (excerpt,) in rows:
        for m in ref_re.finditer(excerpt or ""):
            full = m.group(1).rstrip(".")
            if full in seen:
                continue
            seen.add(full)
            out.append({
                "candidate_source_id": f"github:{full}",
                "source_type": "github_path",
                "endpoint": f"https://github.com/{full}",
                "metadata_json": json.dumps({"via": "readme_crossref"}),
            })
    return out


async def _scan_demand_websearch() -> list[dict]:
    """Signal 3 — web search on accumulated high-confidence demand signals.

    Reads the top demand patterns and (best-effort) searches for source
    repos. Web search is delegated to vecihi; when vecihi is unavailable
    this returns []. The scout degrades gracefully — never crashes.
    """
    from yalayut.discovery import demand as _demand
    pending = await _demand.pending_signals(limit=3)
    out: list[dict] = []
    for sig in pending:
        if sig["stacked_confidence"] < 0.5:
            continue
        query = " ".join(sig["intent_keywords"][:4]) + " skill OR mcp github"
        try:
            from packages.vecihi import search as _vsearch  # type: ignore
        except Exception:
            try:
                import vecihi as _vsearch  # type: ignore
            except Exception:
                logger.debug("vecihi unavailable — demand websearch skipped")
                return out
        try:
            results = await _vsearch.search(query, limit=3)
        except Exception as e:  # noqa: BLE001
            logger.debug("demand websearch failed: %s", e)
            continue
        for r in results or []:
            u = r.get("url", "") if isinstance(r, dict) else ""
            if "github.com/" not in u:
                continue
            full = u.split("github.com/", 1)[1].strip("/").split("/")[:2]
            if len(full) != 2:
                continue
            out.append({
                "candidate_source_id": f"github:{'/'.join(full)}",
                "source_type": "github_path",
                "endpoint": u,
                "metadata_json": json.dumps({
                    "via": "demand_websearch",
                    "demand_pattern": sig["source_step_pattern"],
                }),
            })
    return out


async def _scan_founder_urls() -> list[dict]:
    """Signal 4 — founder-mentioned URLs queued by /yalayut scout <url>.

    They are written by admin.queue_scout_url() with state='founder_queued';
    this promotes them into the candidate proposal flow.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, candidate_source_id, source_type, endpoint, metadata_json "
        "FROM yalayut_source_candidates WHERE state = 'founder_queued'")
    rows = await cur.fetchall()
    await cur.close()
    out: list[dict] = []
    for cid, csid, stype, endpoint, meta in rows:
        out.append({
            "candidate_source_id": csid,
            "source_type": stype,
            "endpoint": endpoint,
            "metadata_json": meta or "{}",
            "_promote_row_id": cid,
        })
    return out


async def _already_known(candidate_source_id: str) -> bool:
    """True when this id is already an approved source or a pending/decided
    candidate (deduped — no double proposal)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_sources WHERE source_id = ? LIMIT 1",
        (candidate_source_id,))
    if await cur.fetchone():
        await cur.close()
        return True
    await cur.close()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_source_candidates "
        "WHERE candidate_source_id = ? AND state IN "
        "('pending', 'approved', 'rejected') LIMIT 1",
        (candidate_source_id,))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def source_scout_scan() -> dict:
    """Run all four scout signals; write up to DAILY_CANDIDATE_CAP new
    ``yalayut_source_candidates`` rows. Returns a summary dict."""
    summary = {"candidates_proposed": 0, "candidates_deduped": 0,
               "telegram_cards": []}
    raw: list[dict] = []
    for scanner in (_scan_github_trending, _scan_readme_crossrefs,
                    _scan_demand_websearch, _scan_founder_urls):
        try:
            raw.extend(await scanner())
        except Exception as e:  # noqa: BLE001
            logger.warning("scout scanner %s failed: %s",
                           scanner.__name__, e)

    db = await get_db()
    now = to_db(utc_now())
    proposed = 0
    for cand in raw:
        if proposed >= DAILY_CANDIDATE_CAP:
            break
        csid = cand["candidate_source_id"]
        promote_id = cand.get("_promote_row_id")
        if promote_id is None and await _already_known(csid):
            summary["candidates_deduped"] += 1
            continue
        if promote_id is not None:
            # founder-queued row → flip it to pending (it IS the candidate).
            await db.execute(
                "UPDATE yalayut_source_candidates SET state = 'pending', "
                "proposed_at = ? WHERE id = ?", (now, promote_id))
        else:
            await db.execute(
                "INSERT INTO yalayut_source_candidates "
                "(candidate_source_id, source_type, endpoint, metadata_json, "
                " state, proposed_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (csid, cand["source_type"], cand["endpoint"],
                 cand.get("metadata_json") or "{}", now))
        proposed += 1
        summary["telegram_cards"].append(csid)
    await db.commit()
    summary["candidates_proposed"] = proposed
    logger.info("source_scout_scan complete", proposed=proposed,
                deduped=summary["candidates_deduped"])
    return summary
