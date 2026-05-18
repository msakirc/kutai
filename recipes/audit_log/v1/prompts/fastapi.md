# Audit-log recipe — FastAPI stack-specific notes

- POST route should reject any non-INSERT shape: never UPDATE/DELETE through the API.
- Retention sweep is a separate route (admin-gated) AND a cron entry — both call the same `sweep_retention` fn.
- Payload JSON column: SQLite stores as TEXT (use `json.loads` on read). Postgres can upgrade to JSONB by post-instantiation edit; document this in the mission's lessons.md.
- Indexes are write-amplifying — three composite indexes on the events table. Skip the actor index if no `actor_user_id`-keyed lookup queries appear in the spec.
- For high-volume events, batch INSERT inside a transaction wrapper. v1 single-row INSERT is fine for <500 events/sec.
