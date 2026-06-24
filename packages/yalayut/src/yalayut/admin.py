"""Yalayut Phase 4 — founder-ops module.

Backs the ``/yalayut`` Telegram command group. Imported only by
``src/app/telegram_bot.py``. Every function is async and operates directly
on the yalayut tables — no LLM, no exposure logic.
"""
from __future__ import annotations

import json
import os

from dabidabi import get_db
from yazbunu import get_logger
from dabidabi.times import utc_now, to_db

logger = get_logger("yalayut.admin")


# ─── Artifact vetting ───────────────────────────────────────────────────

async def pending_artifacts() -> list[dict]:
    """T2 artifacts awaiting founder promotion (quarantined-until-promoted)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, name, name_original, source, owner, kind, vet_tier "
        "FROM yalayut_index WHERE vet_tier = 2 AND enabled = 0 "
        "ORDER BY created_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"id": r[0], "name": r[1], "name_original": r[2], "source": r[3],
         "owner": r[4], "kind": r[5], "vet_tier": r[6]}
        for r in rows
    ]


async def approve_artifact(artifact_id: int) -> dict:
    """Promote a T2 artifact: enable it (founder accepts the risk)."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1, vetted_at = ? WHERE id = ?",
        (to_db(utc_now()), artifact_id))
    await db.commit()
    logger.info("artifact approved", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def reject_artifact(artifact_id: int) -> dict:
    """Reject an artifact: disable it. Never deleted (spec — re-enableable)."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,))
    await db.commit()
    logger.info("artifact rejected", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def requeue(artifact_id: int) -> dict:
    """Re-enable a T3-quarantined artifact for re-vetting."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1, vet_state = 'requeued' "
        "WHERE id = ?", (artifact_id,))
    await db.commit()
    logger.info("artifact requeued", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def disable(artifact_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,))
    await db.commit()
    return {"ok": True, "artifact_id": artifact_id}


async def enable(artifact_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1 WHERE id = ?", (artifact_id,))
    await db.commit()
    return {"ok": True, "artifact_id": artifact_id}


# ─── Source candidates ──────────────────────────────────────────────────

async def pending_sources() -> list[dict]:
    """Source-scout proposals awaiting founder decision."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, candidate_source_id, source_type, endpoint, metadata_json "
        "FROM yalayut_source_candidates WHERE state = 'pending' "
        "ORDER BY proposed_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    out = []
    for r in rows:
        try:
            meta = json.loads(r[4] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        out.append({"id": r[0], "candidate_source_id": r[1],
                    "source_type": r[2], "endpoint": r[3], "metadata": meta})
    return out


async def approve_source(candidate_id: int, *, trusted: bool) -> dict:
    """Approve a candidate → create a ``yalayut_sources`` row.

    Trusted sources get ``discovery_mode='cron'`` (daily pull); untrusted
    get ``discovery_mode='on_demand'`` (only pulled on a demand signal)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT candidate_source_id, source_type, endpoint "
        "FROM yalayut_source_candidates WHERE id = ?", (candidate_id,))
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return {"ok": False, "reason": "candidate not found"}
    source_id, source_type, endpoint = row
    mode = "cron" if trusted else "on_demand"
    trust_score = 0.7 if trusted else 0.3
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_sources "
        "(source_id, source_type, endpoint, trust_score, discovery_mode, "
        " trusted, enabled, min_interval_s) "
        "VALUES (?, ?, ?, ?, ?, ?, 1, 86400)",
        (source_id, source_type, endpoint, trust_score, mode,
         1 if trusted else 0))
    await db.execute(
        "UPDATE yalayut_source_candidates SET state = 'approved', "
        "decided_at = ? WHERE id = ?", (to_db(utc_now()), candidate_id))
    await db.commit()
    logger.info("source approved", source_id=source_id, trusted=trusted)
    return {"ok": True, "source_id": source_id, "trusted": trusted}


async def reject_source(candidate_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_source_candidates SET state = 'rejected', "
        "decided_at = ? WHERE id = ?", (to_db(utc_now()), candidate_id))
    await db.commit()
    return {"ok": True, "candidate_id": candidate_id}


async def promote_source(source_id: str, tier: int) -> dict:
    """Manual source trust promotion (spec — promotion always manual)."""
    db = await get_db()
    new_score = {0: 0.9, 1: 0.6, 2: 0.3}.get(int(tier), 0.3)
    await db.execute(
        "UPDATE yalayut_sources SET trust_score = ?, trusted = ? "
        "WHERE source_id = ?",
        (new_score, 1 if int(tier) == 0 else 0, source_id))
    await db.commit()
    logger.info("source promoted", source_id=source_id, tier=tier)
    return {"ok": True, "source_id": source_id, "tier": tier}


async def promote_owner(owner_name: str) -> dict:
    """Trust an owner — all future sources from them inherit the elevation."""
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_owners WHERE owner_id = ?", (owner_name,))
    exists = await cur.fetchone()
    await cur.close()
    if exists:
        await db.execute(
            "UPDATE yalayut_owners SET trust_score = 0.9 WHERE owner_id = ?",
            (owner_name,))
    else:
        await db.execute(
            "INSERT INTO yalayut_owners (owner_id, trust_score, notes) "
            "VALUES (?, 0.9, 'founder-promoted')", (owner_name,))
    await db.commit()
    logger.info("owner promoted", owner=owner_name)
    return {"ok": True, "owner": owner_name}


async def queue_scout_url(url: str) -> dict:
    """Founder-mentioned candidate source (/yalayut scout <url>).

    Written with state='founder_queued'; the next source_scout_scan()
    promotes it into the pending proposal flow."""
    db = await get_db()
    owner = url.split("//", 1)[-1].split("/", 1)[0]
    cand_id = f"web:{url}"
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_source_candidates "
        "(candidate_source_id, source_type, endpoint, metadata_json, state, "
        " proposed_at) VALUES (?, 'web_markdown', ?, ?, 'founder_queued', ?)",
        (cand_id, url, json.dumps({"via": "founder", "owner": owner}),
         to_db(utc_now())))
    await db.commit()
    return {"ok": True, "candidate_source_id": cand_id}


# ─── Policy proposals ───────────────────────────────────────────────────

async def policy_proposals() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT id, check_name, key, proposed_value, evidence_json "
        "FROM yalayut_policy_proposals WHERE state = 'pending' "
        "ORDER BY proposed_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    out = []
    for r in rows:
        try:
            ev = json.loads(r[4] or "{}")
        except (json.JSONDecodeError, TypeError):
            ev = {}
        out.append({"id": r[0], "check_name": r[1], "key": r[2],
                    "proposed_value": r[3], "evidence": ev})
    return out


async def propose_policy(check_name: str, key: str,
                         value: str = "allow") -> dict:
    """Founder-initiated policy proposal (/yalayut policy add)."""
    db = await get_db()
    await db.execute(
        "INSERT INTO yalayut_policy_proposals "
        "(check_name, key, proposed_value, evidence_json, state, proposed_at) "
        "VALUES (?, ?, ?, '{\"via\": \"founder\"}', 'pending', ?)",
        (check_name, key, value, to_db(utc_now())))
    await db.commit()
    return {"ok": True, "check_name": check_name, "key": key}


async def decide_policy(proposal_id: int, *, approve: bool) -> dict:
    """Approve → write the yalayut_policy row. Reject → just mark decided."""
    db = await get_db()
    cur = await db.execute(
        "SELECT check_name, key, proposed_value FROM yalayut_policy_proposals "
        "WHERE id = ?", (proposal_id,))
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return {"ok": False, "reason": "proposal not found"}
    check_name, key, value = row
    state = "approved" if approve else "rejected"
    if approve:
        await db.execute(
            "INSERT OR REPLACE INTO yalayut_policy "
            "(check_name, key, value, added_by, added_at) "
            "VALUES (?, ?, ?, 'auto_proposal', ?)",
            (check_name, key, value, to_db(utc_now())))
    await db.execute(
        "UPDATE yalayut_policy_proposals SET state = ?, decided_at = ? "
        "WHERE id = ?", (state, to_db(utc_now()), proposal_id))
    await db.commit()
    logger.info("policy decided", proposal_id=proposal_id, approve=approve)
    return {"ok": True, "proposal_id": proposal_id, "state": state}


# ─── Auth / secrets ─────────────────────────────────────────────────────

async def missing_auth() -> list[dict]:
    """Artifacts blocked by a missing env var (env_status != 'ready')."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, name, env_status FROM yalayut_index "
        "WHERE env_status IS NOT NULL AND env_status != 'ready'")
    rows = await cur.fetchall()
    await cur.close()
    return [{"id": r[0], "name": r[1], "env_status": r[2]} for r in rows]


def _fernet():
    """Build a Fernet from the .env key. Raises if YALAYUT_SECRET_KEY unset."""
    from cryptography.fernet import Fernet
    key = os.getenv("YALAYUT_SECRET_KEY")
    if not key:
        raise RuntimeError("YALAYUT_SECRET_KEY not set in .env")
    return Fernet(key.encode() if isinstance(key, str) else key)


async def set_secret(key_name: str, value: str) -> dict:
    """Encrypt + store a secret; re-vet artifacts that were missing it."""
    enc = _fernet().encrypt(value.encode())
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO yalayut_secrets "
        "(key_name, encrypted_value, added_at) VALUES (?, ?, ?)",
        (key_name, enc, to_db(utc_now())))
    # flip artifacts that were blocked solely on this env var → ready.
    await db.execute(
        "UPDATE yalayut_index SET env_status = 'ready' "
        "WHERE env_status = ?", (f"missing_{key_name}",))
    await db.commit()
    logger.info("secret set", key_name=key_name)
    return {"ok": True, "key_name": key_name}


# ─── MCP process control ────────────────────────────────────────────────

async def mcp_status() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT p.artifact_id, i.name, p.pid, p.port, p.health, "
        "       p.last_probe_at "
        "FROM yalayut_mcp_processes p "
        "JOIN yalayut_index i ON i.id = p.artifact_id")
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"artifact_id": r[0], "name": r[1], "pid": r[2], "port": r[3],
         "health": r[4], "last_probe_at": r[5]}
        for r in rows
    ]


async def mcp_restart(artifact_id: int) -> dict:
    """Restart an MCP process: shut it down so it re-spawns lazily on next use.

    The MCP manager has no eager-respawn primitive, and KutAI's
    ``no_auto_connect`` rule forbids re-spawning a server at rest. A shutdown
    forces a fresh start the next time intersect matches the artifact — which
    is exactly how the manager's own probe-failure restart path behaves
    (``McpManager.reprobe_if_due`` → ``shutdown``).
    """
    try:
        from yalayut.mcp_manager import get_manager
        await get_manager().shutdown(artifact_id)
        return {"ok": True, "artifact_id": artifact_id,
                "note": "stopped; re-spawns on next use"}
    except Exception as e:  # noqa: BLE001
        logger.warning("mcp_restart failed: %s", e)
        return {"ok": False, "reason": str(e)}


async def mcp_kill(artifact_id: int) -> dict:
    """Kill an MCP process via the live MCP manager."""
    try:
        from yalayut.mcp_manager import get_manager
        await get_manager().shutdown(artifact_id)
        return {"ok": True, "artifact_id": artifact_id}
    except Exception as e:  # noqa: BLE001
        logger.warning("mcp_kill failed: %s", e)
        return {"ok": False, "reason": str(e)}


# ─── Stats ──────────────────────────────────────────────────────────────

async def stats() -> dict:
    """Overview for /yalayut: counts by tier/type/exposure, queue depths."""
    db = await get_db()

    async def _counts(col: str) -> dict:
        cur = await db.execute(
            f"SELECT {col}, COUNT(*) FROM yalayut_index "
            f"WHERE enabled = 1 GROUP BY {col}")
        rows = await cur.fetchall()
        await cur.close()
        return {str(r[0]): r[1] for r in rows}

    tier_counts = await _counts("vet_tier")
    type_counts = await _counts("artifact_type")
    exposure_counts = await _counts("exposure_class")

    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_index WHERE vet_tier = 2 AND enabled = 0")
    vet_queue = (await cur.fetchone())[0]
    await cur.close()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_source_candidates WHERE state = 'pending'")
    source_queue = (await cur.fetchone())[0]
    await cur.close()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_demand_signals "
        "WHERE resulted_in_discovery = 0")
    demand_backlog = (await cur.fetchone())[0]
    await cur.close()

    # exposure-class A/B from yalayut_usage.
    cur = await db.execute(
        "SELECT exposure_class, "
        "       SUM(CASE WHEN succeeded THEN 1 ELSE 0 END), COUNT(*) "
        "FROM yalayut_usage WHERE exposure_class IS NOT NULL "
        "GROUP BY exposure_class")
    ab_rows = await cur.fetchall()
    await cur.close()
    exposure_ab = {
        r[0]: {"succeeded": r[1] or 0, "total": r[2] or 0}
        for r in ab_rows
    }

    return {
        "tier_counts": tier_counts,
        "type_counts": type_counts,
        "exposure_class_counts": exposure_counts,
        "vet_queue_depth": vet_queue,
        "source_candidate_queue_depth": source_queue,
        "demand_signal_backlog": demand_backlog,
        "exposure_ab": exposure_ab,
    }
