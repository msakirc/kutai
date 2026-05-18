# Pagination Recipe (v1)

## Scope
Cursor-based and offset-based pagination for list endpoints. Cursor variant is
the default (stable ordering, no drift on inserts); offset variant is opt-in via
`USE_CURSOR=false`.

## When to pick
- Any list endpoint that could return more rows than a client should fetch at once
- Cursor variant: feeds with real-time inserts (notifications, activity streams)
- Offset variant: admin tables where users jump to "page 5 of 12" by number

## When NOT to pick
- Single-row lookups or sub-10-row static enumerations (no paging needed)
- Search results that need relevance re-ranking across pages (use search recipe instead)
- Infinite-scroll where cursor stability doesn't matter AND full-text ordering is needed

## Shape

### Cursor variant (default, `USE_CURSOR=true`)
- `GET /items?cursor=<opaque>&limit=N`
- Cursor encodes `(cursor_field_value, id)` as a base64 opaque token
- Response: `{"items": [...], "next_cursor": "<opaque>|null", "has_more": bool}`
- Stable across concurrent inserts: new rows before the cursor don't shift pages

### Offset variant (`USE_CURSOR=false`)
- `GET /items?page=N&limit=L`
- Response: `{"items": [...], "page": N, "total": T, "pages": P}`
- Adds `COUNT(*)` query per request — acceptable for low-cardinality admin tables

## Tradeoffs
- Cursor tokens are opaque; clients cannot jump to arbitrary pages without seeking (acceptable for feeds, not for paginated admin grids → use offset variant)
- `MAX_PAGE_SIZE` cap is enforced server-side; over-limit requests return 400
- Cursor field must be indexed; recipe auto-suggests the index hint template
- Offset `COUNT(*)` is expensive on large tables with complex WHERE clauses
