# infra/tracing.py
"""
Phase 9.2 — Task Execution Tracing

Per-task ordered trace stored in DB as JSON. Records tool calls and model
calls with timing and cost.
"""
from __future__ import annotations
import json
import time
from typing import Any, Optional

from .logging_config import get_logger
from .times import db_now
from .db import get_db

logger = get_logger("infra.tracing")


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS task_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL UNIQUE,
            trace JSON NOT NULL DEFAULT '[]',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        await db.execute("CREATE INDEX IF NOT EXISTS idx_traces_task ON task_traces(task_id)")
        await db.commit()
    except Exception:
        pass


async def append_trace(
    task_id: int,
    entry_type: str,
    input_summary: str = "",
    output_summary: str = "",
    cost: float = 0.0,
    duration_ms: float = 0.0,
) -> None:
    """Append a trace entry for a task. Silently ignores failures."""
    try:
        db = await get_db()
        await _ensure_table(db)

        # Load existing trace
        cursor = await db.execute(
            "SELECT trace FROM task_traces WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        trace = json.loads(row[0]) if row else []

        # Append new entry
        trace.append({
            "type": entry_type,
            "timestamp": db_now(),
            "input": input_summary[:200],
            "output": output_summary[:200],
            "cost": round(cost, 6),
            "duration_ms": round(duration_ms, 1),
        })

        # Keep last 100 entries
        if len(trace) > 100:
            trace = trace[-100:]

        await db.execute(
            """INSERT INTO task_traces (task_id, trace, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(task_id) DO UPDATE SET
               trace = excluded.trace, updated_at = CURRENT_TIMESTAMP""",
            (task_id, json.dumps(trace)),
        )
        await db.commit()
    except Exception as exc:
        logger.debug(f"Trace append failed (non-critical): {exc}")


async def get_trace(task_id: int) -> list[dict]:
    """Get the execution trace for a task."""
    try:
        db = await get_db()
        await _ensure_table(db)
        cursor = await db.execute(
            "SELECT trace FROM task_traces WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        return json.loads(row[0]) if row else []
    except Exception:
        return []


def format_trace(entries: list[dict]) -> str:
    """Format a trace as a readable string."""
    if not entries:
        return "_No trace entries._"
    lines = []
    total_cost = 0.0
    for e in entries:
        ts = e.get("timestamp", "")
        etype = e.get("type", "?")
        inp = e.get("input", "")
        out = e.get("output", "")
        cost = e.get("cost", 0)
        dur = e.get("duration_ms", 0)
        total_cost += cost
        line = f"`{ts}` [{etype}]"
        if inp:
            line += f" ← {inp[:60]}"
        if out:
            line += f" → {out[:60]}"
        if cost > 0:
            line += f" (${cost:.4f})"
        if dur > 0:
            line += f" {dur:.0f}ms"
        lines.append(line)
    lines.append(f"\n💰 Total cost: ${total_cost:.4f}")
    return "\n".join(lines)
