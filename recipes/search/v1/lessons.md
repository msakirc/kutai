# Search recipe — gotchas

- `plainto_tsquery` is safer than `to_tsquery` for user input — `to_tsquery` raises on special chars like `&`, `|`, `!` in the query string.
- Always escape user input before passing to tsvector queries; SQLite FTS5 has its own quoting rules.
- SQLite FTS5 and Postgres tsvector are NOT API-compatible — the template wraps both behind the same `search()` function; don't leak tsvector syntax into shared code.
- GIN index creation on an existing large table can take minutes and locks the table in Postgres. Run `CREATE INDEX CONCURRENTLY` outside a transaction.
- `MIN_QUERY_LEN=2` rejects single-char queries. Some languages (CJK) have meaningful single-character tokens — override to 1 for CJK apps.
- Meilisearch index sync: if you forget to call `add_documents()` on create/update, search returns stale results silently. Add an assertion in tests that a newly created record appears in search.
- `MAX_RESULTS=50` is not a pagination cap — it's a result set ceiling. Wire the pagination recipe if users need more than 50 results across pages.
- Meilisearch task queue is async — `add_documents()` returns a task ID, not a confirmation. In tests, call `client.wait_for_task(task_uid)` before asserting search results.
- tsvector `ts_rank` scores are not normalized across queries — don't display raw rank values to users.
- For multilingual content, set `default_text_search_config = 'simple'` in Postgres to avoid language-specific stemming mismatches.
- The migration template adds the `search_vector` column to an existing table. If the table is large, backfill in batches, not in a single UPDATE.
