"""Audit-log recipe — append-only event ledger.

RECIPE_PARAM markers:
  # RECIPE_PARAM:TABLE_NAME=audit_events
  # RECIPE_PARAM:RETENTION_DAYS=365
  # RECIPE_PARAM:ACTOR_FIELD=actor_user_id
  # RECIPE_PARAM:EMIT_TO_STDOUT=false

Routes:
  POST /audit/events            — record an event
  GET  /audit/events/{rt}/{rid} — list events for a resource (paginated, cursor-style)
  POST /audit/sweep             — admin-only retention sweep
"""
from __future__ import annotations

import json as _json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

TABLE_NAME = "audit_events"  # RECIPE_PARAM:TABLE_NAME=audit_events
RETENTION_DAYS = 365  # RECIPE_PARAM:RETENTION_DAYS=365
ACTOR_FIELD = "actor_user_id"  # RECIPE_PARAM:ACTOR_FIELD=actor_user_id
EMIT_TO_STDOUT = False  # RECIPE_PARAM:EMIT_TO_STDOUT=false

router = APIRouter(prefix="/audit", tags=["audit"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AuditEventCreate(BaseModel):
    actor_user_id: Optional[int] = None
    action: str
    resource_type: str
    resource_id: str
    payload: dict = {}


class AuditEventOut(BaseModel):
    id: int
    actor_user_id: Optional[int]
    action: str
    resource_type: str
    resource_id: str
    payload: dict
    created_at: str


class AuditEventListResponse(BaseModel):
    events: list[AuditEventOut]
    next_cursor: Optional[int] = None
    has_more: bool


class SweepResponse(BaseModel):
    deleted: int


# ---------------------------------------------------------------------------
# Core helpers (also usable directly without HTTP layer)
# ---------------------------------------------------------------------------

async def record_event(
    actor_user_id: Optional[int],
    action: str,
    resource_type: str,
    resource_id: str,
    payload: dict,
    db: Any = None,
) -> int:
    """Append a single audit event. Returns inserted row id.

    Parameters
    ----------
    actor_user_id:
        User performing the action. None for system events.
    action:
        Verb describing the operation (e.g. "user.updated", "order.placed").
    resource_type:
        Entity type (e.g. "user", "order").
    resource_id:
        Entity primary key as string.
    payload:
        Arbitrary context dict — stored as JSON TEXT.
    db:
        aiosqlite connection. If None, fetched from project db getter.
    """
    if db is None:
        from src.infra.db import get_db  # T6: swap for project db getter
        db = await get_db()

    payload_str = _json.dumps(payload)
    cur = await db.execute(
        f"INSERT INTO {TABLE_NAME} (actor_user_id, action, resource_type, resource_id, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (actor_user_id, action, resource_type, resource_id, payload_str),
    )
    await db.commit()
    row_id: int = cur.lastrowid

    if EMIT_TO_STDOUT:
        import sys
        print(
            _json.dumps({
                "audit": {
                    "id": row_id, "actor": actor_user_id, "action": action,
                    "rt": resource_type, "rid": resource_id,
                }
            }),
            file=sys.stdout,
            flush=True,
        )

    return row_id


async def sweep_retention(db: Any = None) -> int:
    """Delete events older than RETENTION_DAYS. Returns rows deleted."""
    if db is None:
        from src.infra.db import get_db  # T6: swap for project db getter
        db = await get_db()

    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    cur = await db.execute(
        f"DELETE FROM {TABLE_NAME} WHERE created_at < ?",
        (cutoff_str,),
    )
    await db.commit()
    return cur.rowcount


async def list_events_for_resource(
    resource_type: str,
    resource_id: str,
    cursor: Optional[int] = None,
    limit: int = 25,
    db: Any = None,
) -> tuple[list[dict], Optional[int]]:
    """Paginated lookup of events for a (resource_type, resource_id) pair.

    Uses id-based cursor (integer) for stable ordering on append-only tables.

    Returns
    -------
    (rows, next_cursor) where next_cursor is None on last page.
    """
    if db is None:
        from src.infra.db import get_db  # T6: swap for project db getter
        db = await get_db()

    fetch_limit = limit + 1
    if cursor is not None:
        cur = await db.execute(
            f"SELECT id, actor_user_id, action, resource_type, resource_id, "
            f"payload, created_at FROM {TABLE_NAME} "
            "WHERE resource_type = ? AND resource_id = ? AND id > ? "
            "ORDER BY id ASC LIMIT ?",
            (resource_type, resource_id, cursor, fetch_limit),
        )
    else:
        cur = await db.execute(
            f"SELECT id, actor_user_id, action, resource_type, resource_id, "
            f"payload, created_at FROM {TABLE_NAME} "
            "WHERE resource_type = ? AND resource_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (resource_type, resource_id, fetch_limit),
        )

    rows = await cur.fetchall()
    cols = [d[0] for d in cur.description]
    dicts = [dict(zip(cols, row)) for row in rows]

    has_more = len(dicts) > limit
    page = dicts[:limit]
    next_cursor: Optional[int] = page[-1]["id"] if has_more and page else None

    # Deserialize payload JSON
    for row in page:
        if isinstance(row.get("payload"), str):
            try:
                row["payload"] = _json.loads(row["payload"])
            except Exception:
                row["payload"] = {}

    return page, next_cursor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/events", response_model=AuditEventOut, status_code=201)
async def create_event(body: AuditEventCreate) -> AuditEventOut:
    """Record a single audit event. Append-only — no UPDATE/DELETE."""
    row_id = await record_event(
        actor_user_id=body.actor_user_id,
        action=body.action,
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        payload=body.payload,
    )
    from src.infra.db import get_db  # T6: swap for project db getter
    db = await get_db()
    cur = await db.execute(
        f"SELECT id, actor_user_id, action, resource_type, resource_id, payload, created_at "
        f"FROM {TABLE_NAME} WHERE id = ?",
        (row_id,),
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Insert succeeded but row not found")
    cols = [d[0] for d in cur.description]
    data = dict(zip(cols, row))
    if isinstance(data.get("payload"), str):
        try:
            data["payload"] = _json.loads(data["payload"])
        except Exception:
            data["payload"] = {}
    return AuditEventOut(
        id=data["id"],
        actor_user_id=data.get("actor_user_id"),
        action=data["action"],
        resource_type=data["resource_type"],
        resource_id=data["resource_id"],
        payload=data["payload"],
        created_at=str(data.get("created_at", "")),
    )


@router.get("/events/{resource_type}/{resource_id}", response_model=AuditEventListResponse)
async def list_events(
    resource_type: str,
    resource_id: str,
    cursor: Optional[int] = Query(default=None, description="Cursor id for pagination"),
    limit: int = Query(default=25, ge=1, le=100),
) -> AuditEventListResponse:
    """List audit events for a resource, paginated by id cursor."""
    events, next_cursor = await list_events_for_resource(
        resource_type=resource_type,
        resource_id=resource_id,
        cursor=cursor,
        limit=limit,
    )
    out = [
        AuditEventOut(
            id=e["id"],
            actor_user_id=e.get("actor_user_id"),
            action=e["action"],
            resource_type=e["resource_type"],
            resource_id=e["resource_id"],
            payload=e.get("payload") or {},
            created_at=str(e.get("created_at", "")),
        )
        for e in events
    ]
    return AuditEventListResponse(
        events=out,
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )


@router.post("/sweep", response_model=SweepResponse)
async def admin_sweep() -> SweepResponse:
    """Admin-only retention sweep. Deletes events older than RETENTION_DAYS.

    Auth: # RECIPE_PARAM:ADMIN_CHECK=raise NotImplementedError
    """
    # RECIPE_PARAM:ADMIN_CHECK=raise NotImplementedError
    # T6: replace the line below with your admin auth dependency
    # e.g.: current_user = Depends(require_admin_user)
    deleted = await sweep_retention()
    return SweepResponse(deleted=deleted)
