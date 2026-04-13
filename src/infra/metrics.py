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


# ── Detection / scraper metrics (Phase 10.2) ────────────────────────────────

async def sync_detection_metrics() -> None:
    """Pull per-domain detection metrics into the central counters.

    Call this before ``persist_metrics()`` or ``format_metrics_summary()``
    to ensure the snapshot includes up-to-date scraper health data.
    """
    try:
        from src.shopping.resilience.detection_monitor import get_detection_metrics
        detection = await get_detection_metrics()
        for domain, stats in detection.items():
            _counters[f"scraper_requests:{domain}"] = float(stats["total_requests"])
            _counters[f"scraper_success:{domain}"] = float(stats["successful_requests"])
            _counters[f"scraper_success_rate:{domain}"] = stats["rolling_success_rate"]
            if stats["in_cooldown"]:
                _counters[f"scraper_cooldown:{domain}"] = stats["seconds_remaining"]
            else:
                _counters.pop(f"scraper_cooldown:{domain}", None)
    except Exception as exc:
        logger.debug("Failed to sync detection metrics: %s", exc)


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
        await sync_detection_metrics()
    except Exception:
        pass  # best-effort; don't block persistence
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

    # Scraper / detection health (Phase 10.2)
    scraper_domains = {
        k.split(":", 1)[1] for k in c if k.startswith("scraper_requests:")
    }
    if scraper_domains:
        lines.append("\n*Scraper health:*")
        for domain in sorted(scraper_domains):
            reqs = int(c.get(f"scraper_requests:{domain}", 0))
            rate = c.get(f"scraper_success_rate:{domain}", 1.0)
            cooldown_key = f"scraper_cooldown:{domain}"
            status = f"⏸ cooldown {int(c[cooldown_key])}s" if cooldown_key in c else "✓"
            lines.append(f"  • {domain}: {reqs} reqs, {rate:.0%} success {status}")

    return "\n".join(lines)


# ── NerdHerd collector ──────────────────────────────────────────────

from prometheus_client import Gauge as _Gauge

_g_tasks_ok = _Gauge("kutay_tasks_completed_total", "Total tasks completed")
_g_tasks_fail = _Gauge("kutay_tasks_failed_total", "Total tasks failed")
_g_queue = _Gauge("kutay_queue_depth", "Current task queue depth")
_g_cost = _Gauge("kutay_cost_total_usd", "Total inference cost USD")
_g_model_calls = _Gauge("kutay_model_calls_total", "Model calls", ["model"])
_g_tokens = _Gauge("kutay_tokens_total", "Tokens by model", ["model"])
_orch_gauges = [_g_tasks_ok, _g_tasks_fail, _g_queue, _g_cost, _g_model_calls, _g_tokens]


class OrchestratorCollector:
    """Exposes orchestrator counters via NerdHerd's collector protocol."""

    name = "orchestrator"

    def collect(self) -> dict:
        return dict(_counters)

    def prometheus_metrics(self) -> list:
        _g_tasks_ok.set(_counters.get("tasks_completed", 0))
        _g_tasks_fail.set(_counters.get("tasks_failed", 0))
        _g_queue.set(_counters.get("queue_depth", 0))
        _g_cost.set(_counters.get("cost_total", 0.0))

        for k, v in _counters.items():
            if k.startswith("model_calls:"):
                model = k.split(":", 1)[1]
                _g_model_calls.labels(model=model).set(v)
            elif k.startswith("tokens:"):
                model = k.split(":", 1)[1]
                _g_tokens.labels(model=model).set(v)
        return _orch_gauges
