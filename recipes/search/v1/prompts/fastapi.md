# Search recipe — FastAPI stack-specific notes

- Use `Query(min_length=MIN_QUERY_LEN)` on the `q` parameter to reject short queries at FastAPI layer before the DB call.
- Wrap the search backend in a dependency so tests can swap it without patching globals.
- For postgres_tsvector, use `asyncpg` or `aiosqlite` depending on the stack. The template uses aiosqlite; swap the connection layer for asyncpg on Postgres.
- SQLite FTS5: `SELECT ... FROM fts_table WHERE fts_table MATCH ?` — the MATCH operator, not LIKE. Use `fts5` virtual table.
- Return `"backend"` field in the response so clients can detect which engine served the query (useful for debugging relevance).
- Add `X-Search-Backend: postgres_tsvector` response header as a secondary signal for load-balancer observability.
- Meilisearch client must be initialized once at app startup via `lifespan` context — don't create a new client per request.
