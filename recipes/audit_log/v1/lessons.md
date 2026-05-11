# Audit-log recipe — gotchas

- SQLite TEXT-payload + Postgres JSONB-payload: keep the column type swappable. Don't write `payload->>'key'` queries in v1 — they don't work on SQLite.
- Retention sweeper holds a write lock on the table during DELETE — schedule off-hours.
- `actor_user_id NULL` is valid (system events). Don't add NOT NULL.
- Indexes on (resource_type, resource_id) get hot fast. If lookups dominate writes, switch to a covering index that includes `action` + `created_at`.
- `EMIT_TO_STDOUT=true` shouldn't fan out millions of events — gate behind log level.
- Append-only is enforced at the API layer; nothing stops a privileged DB user from UPDATE/DELETE. Add a CHECK trigger if compliance demands.
- Schema migrations against the events table are dangerous (table grows huge). Add columns rather than alter existing ones.
- v1 ships skeleton only — T6 fills the FastAPI router + Pydantic models + actual record_event/sweep_retention implementations.
