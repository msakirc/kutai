# Pagination recipe — FastAPI stack-specific notes

- Use `Query(default=None)` for cursor parameter; `Query(default=1, ge=1)` for page.
- Enforce `MAX_PAGE_SIZE` via `Query(default=PAGE_SIZE_DEFAULT, le=MAX_PAGE_SIZE)`.
- Cursor decoding can raise `binascii.Error` or `ValueError` — return HTTP 400 on invalid cursor.
- For Postgres, prefer `ROW(cursor_field, id) > ROW($1, $2)` — reads cleanly in SQL logs.
- For SQLite with aiosqlite, fall back to `(cursor_field > ? OR (cursor_field = ? AND id > ?))`.
- Don't use `OFFSET` for cursor variant — OFFSET scans discard rows, making it O(N) on deep pages.
- Return `next_cursor: null` (JSON null), not the string `"null"` — check serialization.
