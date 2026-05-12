"""Pagination recipe — cursor + offset paged list endpoints.

RECIPE_PARAM markers:
  # RECIPE_PARAM:PAGE_SIZE_DEFAULT=25
  # RECIPE_PARAM:MAX_PAGE_SIZE=100
  # RECIPE_PARAM:CURSOR_FIELD=created_at
  # RECIPE_PARAM:USE_CURSOR=true

Routes:
  GET /items                 — list with cursor pagination (USE_CURSOR=true)
  GET /items?page=N&limit=L  — offset pagination (USE_CURSOR=false, opt-in)

Cursor token encodes (cursor_field_value, id) as base64 to stay opaque and
URL-safe. Offset variant adds COUNT(*) per request — suitable for low-cardinality
admin tables only.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

PAGE_SIZE_DEFAULT = 25  # RECIPE_PARAM:PAGE_SIZE_DEFAULT=25
MAX_PAGE_SIZE = 100  # RECIPE_PARAM:MAX_PAGE_SIZE=100
CURSOR_FIELD = "created_at"  # RECIPE_PARAM:CURSOR_FIELD=created_at
USE_CURSOR = True  # RECIPE_PARAM:USE_CURSOR=true

router = APIRouter(prefix="/items", tags=["items"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CursorPage(BaseModel):
    """Response shape for cursor-based pagination."""
    items: list[dict]
    next_cursor: Optional[str] = None
    has_more: bool


class OffsetPage(BaseModel):
    """Response shape for offset-based pagination."""
    items: list[dict]
    page: int
    total: int
    pages: int


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def _encode_cursor(cursor_field_value: Any, row_id: int) -> str:
    """Encode (cursor_field_value, id) as an opaque base64 token."""
    payload = json.dumps([str(cursor_field_value), row_id])
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(token: str) -> tuple[str, int]:
    """Decode a cursor token. Raises HTTPException 400 on invalid input."""
    try:
        payload = base64.urlsafe_b64decode(token.encode()).decode()
        parts = json.loads(payload)
        if not isinstance(parts, list) or len(parts) != 2:
            raise ValueError("bad shape")
        return str(parts[0]), int(parts[1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cursor: {exc}") from exc


# ---------------------------------------------------------------------------
# DB helpers (aiosqlite — adapts to Postgres by swapping ? -> $N)
# ---------------------------------------------------------------------------

async def _fetch_cursor_page(
    db: Any,
    limit: int,
    cursor_value: Optional[str],
    cursor_id: Optional[int],
) -> list[dict]:
    """Fetch next page using cursor-style WHERE.

    SQLite: (cursor_field > ? OR (cursor_field = ? AND id > ?))
    Postgres: ROW(cursor_field, id) > ROW($1, $2)  — swap at instantiation.
    """
    # T6 WILL FILL — replace with real table name + column list
    if cursor_value is None:
        sql = (
            "SELECT id, <<CURSOR_FIELD>>, title FROM <<TABLE_NAME>> "  # T6: real columns
            "ORDER BY <<CURSOR_FIELD>> ASC, id ASC "
            "LIMIT ?"
        )
        params: list = [limit + 1]
    else:
        sql = (
            "SELECT id, <<CURSOR_FIELD>>, title FROM <<TABLE_NAME>> "  # T6: real columns
            "WHERE (<<CURSOR_FIELD>> > ? OR (<<CURSOR_FIELD>> = ? AND id > ?)) "
            "ORDER BY <<CURSOR_FIELD>> ASC, id ASC "
            "LIMIT ?"
        )
        params = [cursor_value, cursor_value, cursor_id, limit + 1]
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


async def _fetch_offset_page(
    db: Any,
    page: int,
    limit: int,
) -> tuple[list[dict], int]:
    """Fetch page N (1-indexed) + total count."""
    offset = (page - 1) * limit
    # T6 WILL FILL — replace with real table name + column list
    count_cur = await db.execute("SELECT COUNT(*) FROM <<TABLE_NAME>>")  # T6: add WHERE if filtered
    total = (await count_cur.fetchone())[0]
    data_cur = await db.execute(
        "SELECT id, <<CURSOR_FIELD>>, title FROM <<TABLE_NAME>> "  # T6: real columns
        "ORDER BY <<CURSOR_FIELD>> ASC, id ASC LIMIT ? OFFSET ?",
        [limit, offset],
    )
    rows = await data_cur.fetchall()
    cols = [d[0] for d in data_cur.description]
    return [dict(zip(cols, row)) for row in rows], total


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=CursorPage)
async def list_items_cursor(
    cursor: Optional[str] = Query(default=None, description="Opaque cursor token"),
    limit: int = Query(default=PAGE_SIZE_DEFAULT, ge=1, le=MAX_PAGE_SIZE),
) -> CursorPage:
    """List items with cursor-based pagination.

    Returns next_cursor=null on the last page.
    """
    # T6 WILL FILL — inject db dependency via Depends(get_db)
    from src.infra.db import get_db  # T6: swap for project db getter
    db = await get_db()

    cursor_value: Optional[str] = None
    cursor_id: Optional[int] = None
    if cursor:
        cursor_value, cursor_id = _decode_cursor(cursor)

    rows = await _fetch_cursor_page(db, limit=limit, cursor_value=cursor_value, cursor_id=cursor_id)
    has_more = len(rows) > limit
    items = rows[:limit]

    next_cursor: Optional[str] = None
    if has_more and items:
        last = items[-1]
        next_cursor = _encode_cursor(last.get(CURSOR_FIELD), last["id"])

    return CursorPage(items=items, next_cursor=next_cursor, has_more=has_more)


@router.get("/offset", response_model=OffsetPage)
async def list_items_offset(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=PAGE_SIZE_DEFAULT, ge=1, le=MAX_PAGE_SIZE),
) -> OffsetPage:
    """List items with offset-based pagination (opt-in, USE_CURSOR=false variant).

    Adds COUNT(*) per request — suitable for low-cardinality admin tables.
    """
    # T6 WILL FILL — inject db dependency via Depends(get_db)
    from src.infra.db import get_db  # T6: swap for project db getter
    db = await get_db()

    items, total = await _fetch_offset_page(db, page=page, limit=limit)
    import math
    pages = math.ceil(total / limit) if limit else 1
    return OffsetPage(items=items, page=page, total=total, pages=pages)
