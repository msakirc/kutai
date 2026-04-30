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
"""
from __future__ import annotations

import aiosqlite

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
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, agent_type, difficulty, picked_model, picked_score, "
                " call_category, candidates_json, snapshot_summary, success, "
                " error_category, provider) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
            await db.commit()
    except Exception as e:  # noqa: BLE001 — telemetry must never propagate
        logger.warning("pick_log write failed: %s", e)
