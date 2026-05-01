"""Build current QueueProfile from queue tables and push to nerd_herd.

Latency target: <5 ms per push. Profile build runs every 2-3 seconds.
Filters to unblocked + pending + dep-resolved. Projects tokens / calls
from estimate_for() per task.
"""
from __future__ import annotations

import json
import os
import time

import aiosqlite

from nerd_herd.types import QueueProfile


# In-process completed-id cache (TTL 30s; invalidated on task completion via on_finish)
_COMPLETED_IDS: set[int] = set()
_COMPLETED_AT: float = 0.0
_CACHE_TTL_SECS = 30.0


def _reset_cache_for_tests() -> None:
    """Clear the completed_ids cache. Tests only — do not call in prod."""
    global _COMPLETED_IDS, _COMPLETED_AT
    _COMPLETED_IDS = set()
    _COMPLETED_AT = 0.0


async def _refresh_completed_ids(db_path: str) -> set[int]:
    global _COMPLETED_IDS, _COMPLETED_AT
    now = time.time()
    if now - _COMPLETED_AT < _CACHE_TTL_SECS and _COMPLETED_IDS:
        return _COMPLETED_IDS
    try:
        from src.infra.db import connect_aux
        async with connect_aux(db_path, _label="queue_profile_completed_ids") as db:
            async with db.execute(
                "SELECT id FROM tasks WHERE status='completed' "
                "AND (completed_at IS NULL OR completed_at > datetime('now', '-7 days'))"
            ) as cur:
                rows = await cur.fetchall()
        _COMPLETED_IDS = {int(r[0]) for r in rows}
        _COMPLETED_AT = now
    except Exception:
        # On error, retain previous cache (don't blow away)
        pass
    return _COMPLETED_IDS


def invalidate_completed_id_cache(task_id: int) -> None:
    """Hook for on_task_finished: add id without forcing a DB read."""
    _COMPLETED_IDS.add(int(task_id))


_NEEDS_VISION_AGENTS = {"visual_reviewer"}
_NEEDS_THINKING_AGENTS = {"analyst", "architect", "planner", "reviewer"}
_NEEDS_TOOLS_AGENTS = {
    "analyst", "implementer", "executor", "researcher",
    "test_generator", "fixer", "coder",
}


class _TaskShim:
    """Duck-typed wrapper passed to estimate_for()."""
    def __init__(self, agent_type: str, ctx: dict):
        self.agent_type = agent_type
        self.context = ctx


async def build_profile(db_path: str | None = None) -> QueueProfile:
    db_path = db_path or os.environ.get("DB_PATH", "kutai.db")
    from src.infra.db import connect_aux
    async with connect_aux(db_path, _label="queue_profile_build") as db:
        async with db.execute(
            """SELECT id, agent_type, depends_on, context
               FROM tasks
               WHERE status='pending'
                 AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))"""
        ) as cur:
            ready_rows = await cur.fetchall()

    completed = await _refresh_completed_ids(db_path)

    # Lazy import to avoid module-load circular
    from fatih_hoca.estimates import estimate_for

    unblocked: list[_TaskShim] = []
    for tid, agent_type, deps_json, ctx_json in ready_rows:
        try:
            deps = json.loads(deps_json or "[]")
        except Exception:
            deps = []
        if not all(int(d) in completed for d in deps):
            continue
        try:
            ctx = json.loads(ctx_json or "{}")
        except Exception:
            ctx = {}
        unblocked.append(_TaskShim(agent_type or "", ctx if isinstance(ctx, dict) else {}))

    # cloud_only: tasks that local models cannot serve (vision today; future
    # capabilities like long-context >local-ctx, tool-form-only models also
    # belong here). QuotaPlanner reads this to reserve paid quota for tasks
    # that have no fallback path.
    by_difficulty: dict[int, int] = {}
    by_capability: dict[str, int] = {
        "vision": 0, "thinking": 0, "function_calling": 0, "cloud_only": 0,
    }
    projected_tokens = 0
    projected_calls = 0
    hard = 0

    for shim in unblocked:
        ctx = shim.context if isinstance(shim.context, dict) else {}
        d = ctx.get("difficulty")
        if d is None and "classification" in ctx and isinstance(ctx["classification"], dict):
            d = ctx["classification"].get("difficulty")
        d = int(d or 5)
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
        if d >= 7:
            hard += 1
        if shim.agent_type in _NEEDS_VISION_AGENTS:
            by_capability["vision"] += 1
            by_capability["cloud_only"] += 1
        if shim.agent_type in _NEEDS_THINKING_AGENTS:
            by_capability["thinking"] += 1
        if shim.agent_type in _NEEDS_TOOLS_AGENTS:
            by_capability["function_calling"] += 1
        e = estimate_for(shim, btable={})
        projected_tokens += e.total_tokens
        projected_calls += e.iterations

    return QueueProfile(
        total_ready_count=len(unblocked),
        hard_tasks_count=hard,
        by_difficulty=by_difficulty,
        by_capability=by_capability,
        projected_tokens=projected_tokens,
        projected_calls=projected_calls,
    )


async def build_and_push(db_path: str | None = None) -> None:
    """Fire-and-forget: build profile and push to nerd_herd. Swallows exceptions."""
    try:
        profile = await build_profile(db_path)
    except Exception:
        return
    try:
        import nerd_herd
        nerd_herd.push_queue_profile(profile)
    except Exception:
        return
