# Search Recipe (v1)

## Scope
Full-text search endpoint with two backend adapters:
- `postgres_tsvector` — Postgres-native, zero extra service, tsvector + GIN index
- `meilisearch` — external Meilisearch service, richer relevance, typo-tolerance

## When to pick
- Users need to find records by free-text query (product names, article titles, notes)
- `postgres_tsvector`: existing Postgres stack, <1M documents, no infra budget for extra services
- `meilisearch`: typo-tolerant search matters, >1M documents, or relevance tuning required

## When NOT to pick
- Exact-match lookups with known keys (use a DB index + WHERE clause instead)
- Semantic / vector similarity search (use a vector store recipe instead)
- Log search / analytics queries (use a dedicated log pipeline)

## Shape
- `GET /search?q=<query>&limit=N` — returns `{"results": [...], "total": N, "backend": "postgres_tsvector|meilisearch"}`
- `MIN_QUERY_LEN` enforced — short queries return 400 to avoid full-table scans
- `MAX_RESULTS` cap enforced server-side

### postgres_tsvector backend
- `search_vector` column (tsvector) maintained via trigger or application-layer update
- GIN index on `search_vector`
- Query uses `to_tsquery` with `plainto_tsquery` fallback for phrase queries

### meilisearch backend
- Index created at startup with `INDEX_NAME`
- Documents synced via `on_record_created/updated` hooks (T6 fills wiring)
- Fallback: if Meilisearch unreachable, falls back to postgres_tsvector if available

## Tradeoffs
- tsvector: no typo tolerance, language-dependent stemming, free if Postgres is already there
- Meilisearch: typo-tolerance + facets, requires a sidecar service and index sync discipline
- `MIN_QUERY_LEN=2` prevents single-character queries from hitting expensive GIN scans
- Trigger-based tsvector maintenance adds write overhead — acceptable for <10k writes/min
