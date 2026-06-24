"""Yalayut Phase 4 — policy observer.

KutAI proposes allowlist additions from observation. ``observe_and_propose``
scans ``yalayut_index.check_max_json`` audit data for unknown shell tokens
(and domains) that capped 3+ artifacts at T2, and writes pending
``yalayut_policy_proposals`` rows. The founder approves via Telegram
(``/yalayut policy review``). The observer NEVER mutates ``yalayut_policy``.
"""
from __future__ import annotations

import json
from collections import Counter

from dabidabi import get_db
from yazbunu import get_logger
from dabidabi.times import utc_now, to_db

logger = get_logger("yalayut.policy_observer")

#: How many artifacts must hit the same unknown token before we propose it.
PROPOSE_THRESHOLD: int = 3


async def _proposal_exists(check_name: str, key: str) -> bool:
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_policy_proposals "
        "WHERE check_name = ? AND key = ? AND state = 'pending' LIMIT 1",
        (check_name, key))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def _already_policy(check_name: str, key: str) -> bool:
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_policy WHERE check_name = ? AND key = ? LIMIT 1",
        (check_name, key))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def observe_and_propose() -> int:
    """Scan vetting audit data; write policy proposals. Returns the count of
    new proposals written."""
    db = await get_db()
    cur = await db.execute(
        "SELECT name, check_max_json FROM yalayut_index "
        "WHERE check_max_json IS NOT NULL")
    rows = await cur.fetchall()
    await cur.close()

    # (check_name, key) -> [artifact names that hit it]
    hits: dict[tuple[str, str], list[str]] = {}
    for name, cmj in rows:
        try:
            checks = json.loads(cmj or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        for check_name, detail in (checks or {}).items():
            if not isinstance(detail, dict):
                continue
            token = detail.get("unknown_token") or detail.get("unknown_domain")
            if not token:
                continue
            key = (check_name, str(token))
            hits.setdefault(key, []).append(name)

    written = 0
    now = to_db(utc_now())
    for (check_name, token), artifacts in hits.items():
        if len(artifacts) < PROPOSE_THRESHOLD:
            continue
        if await _already_policy(check_name, token):
            continue
        if await _proposal_exists(check_name, token):
            continue
        await db.execute(
            "INSERT INTO yalayut_policy_proposals "
            "(check_name, key, proposed_value, evidence_json, state, "
            " proposed_at) VALUES (?, ?, 'allow', ?, 'pending', ?)",
            (check_name, token,
             json.dumps({"artifacts": artifacts,
                         "occurrences": len(artifacts)}),
             now))
        written += 1
        logger.info("policy proposal written", check=check_name,
                    key=token, occurrences=len(artifacts))
    await db.commit()
    return written
