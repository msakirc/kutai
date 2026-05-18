"""Z7 T3C — press_kit_freshness posthook handler.

Fires monthly (wired via mr_roboto cron / beckman posthook). Checks whether
the most-recently published press kit for a product is stale:

  stale = kit is >90 days old AND spec has changed since assembly

When stale, emits a founder_action (kind='generic') surfacing:
    "Regenerate press kit? Spec has changed since v{N} was published."

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:

- ``{"status": "ok", "reason": "fresh"}``     — kit is <90 days old
- ``{"status": "ok", "reason": "unchanged"}`` — spec hash matches (no change)
- ``{"status": "flagged", "kit_id": N, ...}`` — stale, founder_action emitted
- ``{"status": "skip", "reason": str}``        — no kit or no product_id
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.press_kit_freshness")

# A kit older than this threshold is considered stale (if spec also changed)
STALE_DAYS: int = 90


# ---------------------------------------------------------------------------
# Injected DB helpers (testable via monkeypatch)
# ---------------------------------------------------------------------------

async def _get_latest_kit(product_id: str) -> dict | None:
    """Return the latest press_kit row for product_id, or None."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT kit_id, product_id, version, manifest_json, created_at "
            "FROM press_kits "
            "WHERE product_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (product_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "kit_id": row[0],
            "product_id": row[1],
            "version": row[2],
            "manifest_json": row[3],
            "created_at": row[4],
        }
    except Exception as exc:
        logger.warning(
            "press_kit_freshness: _get_latest_kit failed", error=str(exc)
        )
        return None


async def _get_spec_hash(product_id: str) -> str | None:
    """Return the current spec hash for the product, or None if unavailable.

    In production: query the mission spec artifact or a product_specs table.
    Lightweight fallback: hash the product_id itself so tests can control the
    result via monkeypatching.
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        # Try product_specs table if it exists (Z7 future table)
        cur = await db.execute(
            "SELECT spec_hash FROM product_specs WHERE product_id = ?",
            (product_id,),
        )
        row = await cur.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    # Fallback: return None (caller treats None as "unknown = assume changed")
    return None


async def _emit_founder_action(
    *,
    mission_id: int,
    product_id: str,
    version: int,
    kit_age_days: int,
) -> Any:
    """Emit a founder_action requesting press kit regeneration."""
    try:
        from src.founder_actions import create as create_founder_action

        title = (
            f"Press kit v{version} for '{product_id}' is {kit_age_days}d old "
            f"and the spec has changed — regenerate?"
        )
        why = (
            f"The current press kit (v{version}) was assembled {kit_age_days} days ago. "
            "The product spec has been updated since then. "
            "Journalists, investors, and partners may receive outdated information."
        )
        instructions = [
            "Run press_kit/assemble to regenerate all 4 audience variants.",
            "Review the assembled kit and approve press_kit/publish.",
            "Or dismiss this card if the changes aren't material enough to republish.",
        ]
        return await create_founder_action(
            mission_id=mission_id,
            kind="generic",
            title=title,
            why=why,
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=False,
        )
    except Exception as exc:
        logger.warning(
            "press_kit_freshness: _emit_founder_action failed", error=str(exc)
        )
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def handle(task: dict, result: dict) -> dict[str, Any]:
    """press_kit_freshness posthook handler."""
    task_id = task.get("id")
    mission_id = task.get("mission_id") or 0

    # Parse task context for product_id
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    product_id = ctx.get("product_id") or ""
    if not product_id:
        logger.debug(
            "press_kit_freshness: no product_id in task context — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no product_id in task context"}

    # Fetch the latest kit
    kit = await _get_latest_kit(product_id)
    if kit is None:
        logger.debug(
            "press_kit_freshness: no press_kits row for product — skip",
            task_id=task_id,
            product_id=product_id,
        )
        return {"status": "skip", "reason": "no press kit found for product"}

    # Check age
    created_at_str = kit.get("created_at") or ""
    try:
        created_at = datetime.strptime(created_at_str[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # Unparseable timestamp — treat as stale
        created_at = datetime.utcnow() - timedelta(days=STALE_DAYS + 1)

    kit_age = datetime.utcnow() - created_at
    kit_age_days = kit_age.days

    if kit_age_days < STALE_DAYS:
        logger.info(
            "press_kit_freshness: kit is fresh",
            product_id=product_id,
            version=kit["version"],
            age_days=kit_age_days,
        )
        return {"status": "ok", "reason": "fresh", "age_days": kit_age_days}

    # Check spec hash
    manifest_str = kit.get("manifest_json") or "{}"
    try:
        manifest_data = json.loads(manifest_str)
    except Exception:
        manifest_data = {}

    stored_spec_hash = manifest_data.get("spec_hash") or ""
    current_spec_hash = await _get_spec_hash(product_id)

    if current_spec_hash is not None and current_spec_hash == stored_spec_hash:
        logger.info(
            "press_kit_freshness: kit is old but spec unchanged",
            product_id=product_id,
            version=kit["version"],
            age_days=kit_age_days,
        )
        return {"status": "ok", "reason": "unchanged", "age_days": kit_age_days}

    # Kit is stale and spec has changed → emit founder_action
    logger.warning(
        "press_kit_freshness: stale kit + spec changed",
        product_id=product_id,
        version=kit["version"],
        age_days=kit_age_days,
    )

    fa = await _emit_founder_action(
        mission_id=mission_id,
        product_id=product_id,
        version=kit["version"],
        kit_age_days=kit_age_days,
    )

    return {
        "status": "flagged",
        "kit_id": kit["kit_id"],
        "product_id": product_id,
        "version": kit["version"],
        "age_days": kit_age_days,
        "founder_action_id": getattr(fa, "id", None) if fa else None,
    }
