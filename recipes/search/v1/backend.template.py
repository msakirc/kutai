"""Search recipe — full-text search via Postgres tsvector or Meilisearch adapter.

RECIPE_PARAM markers:
  # RECIPE_PARAM:SEARCH_BACKEND=postgres_tsvector
  # RECIPE_PARAM:INDEX_NAME=default_search
  # RECIPE_PARAM:MIN_QUERY_LEN=2
  # RECIPE_PARAM:MAX_RESULTS=50

Routes:
  GET /search?q=<query>&limit=N  — full-text search, returns ranked results

SEARCH_BACKEND alternatives:
  postgres_tsvector  — Postgres GIN index + tsvector; zero extra infra
  meilisearch        — external Meilisearch service; typo-tolerant, richer ranking

SQLite FTS5 is used when SEARCH_BACKEND=postgres_tsvector and the DB is SQLite
(the template detects this via the connection driver at runtime — T6 wires).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

SEARCH_BACKEND = "postgres_tsvector"  # RECIPE_PARAM:SEARCH_BACKEND=postgres_tsvector
INDEX_NAME = "default_search"  # RECIPE_PARAM:INDEX_NAME=default_search
MIN_QUERY_LEN = 2  # RECIPE_PARAM:MIN_QUERY_LEN=2
MAX_RESULTS = 50  # RECIPE_PARAM:MAX_RESULTS=50

router = APIRouter(prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    id: int
    title: str
    snippet: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    backend: str


# ---------------------------------------------------------------------------
# Backend: postgres_tsvector / SQLite FTS5
# ---------------------------------------------------------------------------

async def _search_tsvector(db: Any, query: str, limit: int) -> list[dict]:
    """Full-text search via SQLite FTS5 (tsvector path adapts for Postgres at T6).

    SQLite: SELECT ... FROM fts_table WHERE fts_table MATCH ?
    Postgres: SELECT ..., ts_rank(search_vector, q) AS score
              FROM <<TABLE_NAME>>, plainto_tsquery('simple', ?) q
              WHERE search_vector @@ q ORDER BY score DESC LIMIT ?
    """
    # T6 WILL FILL — swap with FTS5 MATCH or Postgres tsvector query
    # SQLite FTS5 example (T6 replaces with real table):
    #   sql = "SELECT rowid AS id, title, snippet(<<FTS_TABLE_NAME>>, 1, '<b>', '</b>', '...', 10) AS snippet FROM <<FTS_TABLE_NAME>> WHERE <<FTS_TABLE_NAME>> MATCH ? LIMIT ?"
    #   params = [query, limit]
    # Postgres tsvector example (T6 replaces):
    #   sql = "SELECT id, title, ts_headline('simple', body, plainto_tsquery('simple', $1)) AS snippet, ts_rank(search_vector, plainto_tsquery('simple', $1)) AS score FROM <<TABLE_NAME>> WHERE search_vector @@ plainto_tsquery('simple', $1) ORDER BY score DESC LIMIT $2"
    raise NotImplementedError(
        "search recipe v1: T6 will fill in _search_tsvector with real table name"
    )


# ---------------------------------------------------------------------------
# Backend: Meilisearch
# ---------------------------------------------------------------------------

async def _search_meilisearch(query: str, limit: int) -> list[dict]:
    """Full-text search via Meilisearch client."""
    # T6 WILL FILL — inject Meilisearch client from app state
    # Example:
    #   from app.state import get_meilisearch_client
    #   client = get_meilisearch_client()
    #   index = client.index(INDEX_NAME)
    #   result = index.search(query, {"limit": limit, "attributesToHighlight": ["title", "body"]})
    #   return result["hits"]
    raise NotImplementedError(
        "search recipe v1: T6 will fill in _search_meilisearch"
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=MIN_QUERY_LEN, description="Full-text search query"),
    limit: int = Query(default=MAX_RESULTS, ge=1, le=MAX_RESULTS),
) -> SearchResponse:
    """Search across indexed records.

    Returns ranked results with optional snippets. Backend is determined by
    SEARCH_BACKEND param at instantiation time.
    """
    if len(q.strip()) < MIN_QUERY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Query too short: minimum {MIN_QUERY_LEN} characters required",
        )

    if SEARCH_BACKEND == "meilisearch":
        raw_hits = await _search_meilisearch(query=q, limit=limit)
        results = [
            SearchResult(
                id=hit.get("id", 0),
                title=hit.get("title", ""),
                snippet=hit.get("_formatted", {}).get("body"),
                score=None,
            )
            for hit in raw_hits
        ]
        return SearchResponse(results=results, total=len(results), backend="meilisearch")

    # Default: postgres_tsvector / SQLite FTS5
    from src.infra.db import get_db  # T6: swap for project db getter
    db = await get_db()
    raw_rows = await _search_tsvector(db, query=q, limit=limit)
    results = [
        SearchResult(
            id=row.get("id", 0),
            title=row.get("title", ""),
            snippet=row.get("snippet"),
            score=row.get("score"),
        )
        for row in raw_rows
    ]
    return SearchResponse(results=results, total=len(results), backend="postgres_tsvector")
