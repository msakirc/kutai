"""Helper for writing model_pick_log rows from dispatcher post-iteration.

Fire-and-forget: never raises to the caller. The dispatcher invokes this
after each LLM iteration with the outcome (success/failure), capturing
what was actually dispatched rather than only what the selector picked.

Schema reference: src/infra/db.py :: model_pick_log table.
Columns written:
    task_name, picked_model, picked_score, call_category,
    candidates_json (NOT NULL — "[]" placeholder; dispatcher has no
    candidate list post-select), snapshot_summary, success, error_category.
Timestamp column uses CURRENT_TIMESTAMP default so string ordering in
the idx_pick_log_task(timestamp DESC) index keeps working.

Routing: production triage 2026-05-01 found pick_log was firing 44+/sec
during heavy selector activity. Each fresh ``connect_aux`` conn opened
its own aiosqlite worker thread + 3 PRAGMA setups + INSERT + close, all
while contending for the WAL writer slot against the singleton's
update_task / hook writes. The singleton starved out (lock errors on
update_task >60s busy_timeout). Routing pick_log through ``get_db()``
singleton: same connection's worker thread serializes the INSERT
naturally, no cross-conn WAL contention, INSERT is ~3ms vs ~50ms via
fresh conn. ``db_path`` parameter is now ignored (kept for back-compat).
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("infra.pick_log")


async def write_pick_log_row(
    db_path: str,
    task_name: str,
    picked_model: str,
    picked_score: float,
    category: str,
    success: bool,
    error_category: str = "",
    snapshot_summary: str = "",
    provider: str = "local",
    agent_type: str = "",
    difficulty: int | None = None,
) -> None:
    """Fire-and-forget write of a pick outcome.

    Never raises. If the row cannot be persisted (missing DB, schema
    mismatch, whatever) it logs a warning and returns.

    ``db_path`` is ignored — INSERT goes through the singleton get_db()
    connection. See module docstring for rationale.
    """
    # Derive `outcome` (TEXT semantic label) from success+error_category.
    # success → "success"; failure → error_category if non-empty, else
    # generic "failed". Lets queries filter / group by outcome without
    # joining the int success column with the string error_category.
    outcome = "success" if success else (error_category or "failed")
    try:
        from src.infra.db import get_db
        db = await get_db()
        await db.execute(
            "INSERT INTO model_pick_log "
            "(task_name, agent_type, difficulty, picked_model, picked_score, "
            " call_category, candidates_json, snapshot_summary, success, "
            " error_category, provider, outcome) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_name,
                agent_type or None,
                difficulty,
                picked_model,
                picked_score,
                category,
                "[]",
                snapshot_summary,
                1 if success else 0,
                error_category,
                provider,
                outcome,
            ),
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("pick_log write failed: %s", e)
