# Pagination recipe — gotchas

- Cursor must encode both `(cursor_field_value, id)` — cursor-field alone causes ties when two rows share the same timestamp; add `id` as tiebreaker.
- Base64-encode the cursor token so it's safe in query strings; never expose raw SQL values.
- Always clamp `limit` server-side: `limit = min(requested_limit, MAX_PAGE_SIZE)`. Client-supplied limits of 10000 will OOM small instances.
- `USE_CURSOR=false` adds a `COUNT(*)` per request. On tables >100k rows with complex WHERE filters, this is the bottleneck. Cache the count or skip it.
- Index on `(cursor_field, id)` — not just `cursor_field`. Composite index is critical for the cursor WHERE clause (`WHERE (created_at, id) > (?, ?)` style).
- Postgres uses `ROW(created_at, id) > ROW($1, $2)` syntax; SQLite uses `(created_at > ? OR (created_at = ? AND id > ?))`. Both patterns are in the template.
- `next_cursor=null` signals last page. Clients must handle null; don't return an empty string.
- Cursor field changes (e.g. switching from `created_at` to `updated_at`) invalidate all in-flight cursors. Version the cursor encoding if you might change the field.
- Avoid sorting by a non-unique float column (`price`, `score`). Float ties break stable ordering. Add `id` tiebreaker unconditionally.
- Offset variant `page=0` vs `page=1` — the template uses 1-indexed pages to match UI conventions; document this for frontend consumers.
- `has_more: true` with zero items is impossible — guard against off-by-one in the cursor extraction logic.
- For cursor variant, the response must include the `cursor_field` value in each item so clients can bookmark position manually if needed.
