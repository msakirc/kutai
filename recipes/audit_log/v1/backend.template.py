"""Audit-log recipe — append-only event ledger.

RECIPE_PARAM markers:
  # RECIPE_PARAM:TABLE_NAME=audit_events
  # RECIPE_PARAM:RETENTION_DAYS=365
  # RECIPE_PARAM:ACTOR_FIELD=actor_user_id
  # RECIPE_PARAM:EMIT_TO_STDOUT=false

Routes:
  POST /audit/events           — record an event
  GET  /audit/events/{rt}/{rid} — list events for a resource (paginated)
  POST /audit/sweep            — admin-only retention sweep

T6 will fill in the FastAPI router, Pydantic models, and SQL helpers.
v1 ships the skeleton + RECIPE_PARAM marker surface so pick_recipe +
instantiate_recipe routes against this recipe in dry-run scenarios.
"""
from __future__ import annotations

# T6 WILL FILL — FastAPI app + router wiring goes here.

TABLE_NAME = "audit_events"  # RECIPE_PARAM:TABLE_NAME=audit_events
RETENTION_DAYS = 365  # RECIPE_PARAM:RETENTION_DAYS=365
ACTOR_FIELD = "actor_user_id"  # RECIPE_PARAM:ACTOR_FIELD=actor_user_id
EMIT_TO_STDOUT = False  # RECIPE_PARAM:EMIT_TO_STDOUT=false


async def record_event(actor_user_id: int, action: str, resource_type: str,
                       resource_id: str, payload: dict) -> int:
    """Append a single audit event. Returns inserted row id."""
    # T6 WILL FILL — INSERT INTO {TABLE_NAME}(...) RETURNING id
    raise NotImplementedError("audit_log recipe v1: T6 will fill in record_event")


async def sweep_retention() -> int:
    """Delete events older than RETENTION_DAYS. Returns rows deleted."""
    # T6 WILL FILL — DELETE FROM {TABLE_NAME} WHERE created_at < now - RETENTION_DAYS
    raise NotImplementedError("audit_log recipe v1: T6 will fill in sweep_retention")
