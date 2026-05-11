# Audit Log Recipe (v1)

## Scope
Append-only audit-event ledger. Records who did what to which resource, when. Optimised for write-heavy + occasional point-in-time query.

## When to pick
- Compliance requirement: SOC2, HIPAA, GDPR data-access logging
- Need to answer "what changed and who changed it" for any resource
- Want retention policy enforcement at the storage layer

## When NOT to pick
- High-throughput streaming events (use a log pipeline + warehouse instead)
- Free-text application logs (use loguru/structlog + a log aggregator)
- Real-time alerting on events (this is a ledger, not a stream — add a pub/sub layer separately)

## Shape
- `audit_events` table: id, actor_user_id, action, resource_type, resource_id, payload (JSON), created_at
- INSERT-only API; UPDATE/DELETE rejected at the route layer
- Retention sweeper: cron deletes rows older than `RETENTION_DAYS` (default 365)
- Indexed lookup by (resource_type, resource_id, created_at DESC) for resource-history queries
- Optional `EMIT_TO_STDOUT=true` flag mirrors every event to stdout for log shipping

## Tradeoffs
- No write batching: one row per event. Buy a queue if you need >1000 events/sec.
- JSON payload is opaque to the DB — can't query inside it without app-layer joins. SQLite stack: stored as TEXT. Postgres: switch column to JSONB by post-instantiation edit if needed.
- Retention sweeper is destructive by design; if you need legal-hold, gate the sweeper behind a flag set per resource_type.
