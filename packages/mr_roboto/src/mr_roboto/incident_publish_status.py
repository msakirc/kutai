"""Z7 T3D — B3: incident/publish_status mr_roboto verb.

Publishes a reviewed status update to the incidents / status_updates tables.
This verb runs ONLY after a founder_action review has been approved —
the caller is responsible for gating on founder approval.

After publishing:
  - Inserts a row into status_updates.
  - Updates incidents.current_status_md.
  - Invalidates the /status page cache (via cache-busting sentinel in DB).
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("mr_roboto.incident_publish_status")


async def run(payload: dict) -> dict:
    """Execute incident/publish_status.

    Expected payload keys:
      - incident_id  (int, required)
      - product_id   (str, required)
      - body_md      (str, required) — founder-approved status update text
      - status_kind  ('investigating'|'identified'|'monitoring'|'resolved')
      - founder_action_id (int, optional) — the FA that approved this publish

    Returns:
      {"status": "ok", "update_id": int}
    """
    import json as _json
    from dabidabi import get_db
    from dabidabi.times import db_now

    incident_id = payload.get("incident_id")
    product_id = payload.get("product_id") or ""
    body_md = payload.get("body_md") or ""
    status_kind = payload.get("status_kind") or "investigating"
    founder_action_id = payload.get("founder_action_id")

    VALID_STATUS_KINDS = {"investigating", "identified", "monitoring", "resolved"}
    if status_kind not in VALID_STATUS_KINDS:
        return {
            "status": "error",
            "error": f"status_kind must be one of {sorted(VALID_STATUS_KINDS)}, got {status_kind!r}",
        }

    if not incident_id:
        return {"status": "error", "error": "incident_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    if not body_md.strip():
        return {"status": "error", "error": "body_md must not be empty"}

    now = db_now()
    db = await get_db()

    # Verify incident exists and belongs to this product.
    async with db.execute(
        "SELECT incident_id FROM incidents WHERE incident_id = ? AND product_id = ?",
        (incident_id, product_id),
    ) as cur:
        if not await cur.fetchone():
            return {
                "status": "error",
                "error": f"incident {incident_id} not found for product {product_id!r}",
            }

    # Insert status_update row.
    cur = await db.execute(
        "INSERT INTO status_updates "
        "(product_id, incident_id, posted_at, body_md, status_kind) "
        "VALUES (?, ?, ?, ?, ?)",
        (product_id, incident_id, now, body_md.strip(), status_kind),
    )
    await db.commit()
    update_id = int(cur.lastrowid or 0)

    # Update incidents.current_status_md (and resolved_at if resolved).
    if status_kind == "resolved":
        await db.execute(
            "UPDATE incidents SET current_status_md = ?, resolved_at = ? "
            "WHERE incident_id = ?",
            (body_md.strip(), now, incident_id),
        )
    else:
        await db.execute(
            "UPDATE incidents SET current_status_md = ? WHERE incident_id = ?",
            (body_md.strip(), incident_id),
        )
    await db.commit()

    # Invalidate the /status page in-memory cache so next render is fresh.
    try:
        from src.app.status_page import invalidate_cache
        invalidate_cache()
    except Exception as exc:
        # Best-effort — cache TTL will expire naturally — but log so a real
        # import/call regression is visible instead of silently swallowed.
        logger.debug(
            "incident_publish_status: status-page cache invalidation skipped",
            error=str(exc),
        )

    logger.info(
        "incident_publish_status: published",
        update_id=update_id,
        incident_id=incident_id,
        product_id=product_id,
        status_kind=status_kind,
        founder_action_id=founder_action_id,
    )

    return {
        "status": "ok",
        "update_id": update_id,
        "incident_id": incident_id,
        "product_id": product_id,
        "status_kind": status_kind,
        "posted_at": now,
    }
