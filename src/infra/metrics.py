# infra/metrics.py
"""
Phase 9.1 — Metrics Collection

In-memory counters with periodic DB persistence. Tracks task completion,
model calls, costs, latency, tool usage, and token counts.
"""
from __future__ import annotations
import asyncio
import time
from collections import defaultdict
from typing import Any

from .logging_config import get_logger
from .db import get_db

logger = get_logger("infra.metrics")

# ── In-memory counters ───────────────────────────────────────────────────────

_counters: dict[str, float] = defaultdict(float)
_last_persisted: float = 0.0
PERSIST_INTERVAL_SECS = 3600  # persist hourly


def increment(key: str, value: float = 1.0) -> None:
    """Increment an in-memory counter."""
    _counters[key] += value


def get_counter(key: str) -> float:
    """Get current value of a counter."""
    return _counters.get(key, 0.0)


def get_all_counters() -> dict[str, float]:
    """Get all current counters."""
    return dict(_counters)


def record_task_complete(model: str = "", cost: float = 0.0) -> None:
    increment("tasks_completed")
    if cost > 0:
        increment("cost_total")
        increment(f"cost:{model}", cost)


def record_task_failed(model: str = "") -> None:
    increment("tasks_failed")


def track_model_call_metrics(model: str, cost: float = 0.0, latency_ms: float = 0.0, tokens: int = 0) -> None:
    """Update in-memory counters for a model call (cost, latency, tokens).

    This is the single entry point for in-memory metric tracking.  The DB-level
    ``record_model_call`` in ``infra.db`` calls this automatically, so most
    callers should go through that function instead.
    """
    increment(f"model_calls:{model}")
    if cost > 0:
        increment(f"cost:{model}", cost)
    if latency_ms > 0:
        # Store sum for average calculation
        increment(f"latency_sum:{model}", latency_ms)
        increment(f"latency_count:{model}")
    if tokens > 0:
        increment(f"tokens:{model}", tokens)


def record_tool_call(tool: str) -> None:
    increment(f"tool_calls:{tool}")


def record_queue_depth(depth: int) -> None:
    _counters["queue_depth"] = float(depth)  # not cumulative


async def _ensure_table(db) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS metrics_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            counters JSON NOT NULL
        )
    """)
    await db.commit()


async def persist_metrics() -> None:
    """Persist current counters to DB as a snapshot."""
    global _last_persisted
    try:
        import json
        db = await get_db()
        await _ensure_table(db)
        await db.execute(
            "INSERT INTO metrics_snapshots (counters) VALUES (?)",
            (json.dumps(dict(_counters)),),
        )
        await db.commit()
        _last_persisted = time.monotonic()
        logger.debug("Metrics snapshot persisted")
    except Exception as exc:
        logger.debug(f"Metrics persist failed: {exc}")


async def maybe_persist() -> None:
    """Persist metrics if enough time has passed."""
    if time.monotonic() - _last_persisted > PERSIST_INTERVAL_SECS:
        await persist_metrics()


def format_metrics_summary() -> str:
    """Format current metrics as a readable string."""
    c = dict(_counters)
    if not c:
        return "_No metrics collected yet._"

    lines = ["📊 *Metrics Summary*\n"]
    lines.append(f"✅ Tasks completed: {int(c.get('tasks_completed', 0))}")
    lines.append(f"❌ Tasks failed: {int(c.get('tasks_failed', 0))}")
    lines.append(f"💰 Total cost: ${c.get('cost_total', 0):.4f}")
    lines.append(f"📥 Queue depth: {int(c.get('queue_depth', 0))}")

    # Model breakdown
    model_calls = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("model_calls:")}
    if model_calls:
        lines.append("\n*Model calls:*")
        for model, count in sorted(model_calls.items(), key=lambda x: -x[1])[:5]:
            cost = c.get(f"cost:{model}", 0)
            lines.append(f"  • {model}: {int(count)} calls (${cost:.4f})")

    # Top tools
    tool_calls = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("tool_calls:")}
    if tool_calls:
        lines.append("\n*Top tools:*")
        for tool, count in sorted(tool_calls.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  • {tool}: {int(count)}")

    return "\n".join(lines)
