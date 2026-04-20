"""Dispatch-time metrics push.

Captures the side-effect telemetry that used to live inline in the deleted
Orchestrator._handle_complete (mission cost tracking, episodic memory,
model_stats, preference feedback, skill A/B metrics). Runs after
beckman.on_task_finished so the task row is already terminal when we record.

Any subroutine failure is logged and swallowed — telemetry must not
block the dispatch pump.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("core.metrics_push")


async def push_metrics(task: dict, result: dict | None) -> None:
    """Best-effort: record model_stats, mission cost, episodic memory,
    metrics counters, and skill injection outcomes.

    All failures are caught and logged at DEBUG level.
    """
    for fn in (
        _push_model_stats,
        _push_mission_cost,
        _push_episodic_memory,
        _push_metrics_counter,
        _push_preference_feedback,
        _push_skill_injection,
    ):
        try:
            await fn(task, result)
        except Exception as e:
            logger.debug(f"{fn.__name__} failed: {e}")


# ── Subroutines ──────────────────────────────────────────────────────────────

async def _push_model_stats(task: dict, result: dict | None) -> None:
    if result is None:
        return
    status = result.get("status", "")
    if status not in ("completed", "ungraded"):
        return
    model = result.get("model") or ""
    if not model:
        return
    try:
        from src.infra.db import update_model_stats
        agent_type = task.get("agent_type", "executor")
        grade = result.get("grade", 3.0)
        cost = result.get("cost", 0.0)
        # latency_ms: use iterations as a rough proxy (iterations * avg_ms),
        # matching the original _handle_complete logic.
        iterations = result.get("iterations") or 1
        latency_ms = int(iterations * 2000)
        await update_model_stats(
            model=model,
            agent_type=agent_type,
            success=True,
            cost=cost,
            latency_ms=latency_ms,
            grade=grade,
        )
    except ImportError:
        pass


async def _push_mission_cost(task: dict, result: dict | None) -> None:
    """Update the mission cost blackboard — mirrors old Fix #8 logic."""
    mission_id = task.get("mission_id")
    if not mission_id or result is None:
        return
    cost = result.get("cost", 0.0)
    if cost <= 0:
        return
    try:
        from src.collaboration.blackboard import read_blackboard, write_blackboard
        from src.core.task_context import parse_context

        current = await read_blackboard(mission_id, "cost_tracking")
        if not isinstance(current, dict):
            current = {"total_cost": 0.0, "task_count": 0, "by_phase": {}}
        current["total_cost"] = current.get("total_cost", 0.0) + cost
        current["task_count"] = current.get("task_count", 0) + 1
        task_ctx = parse_context(task)
        phase = task_ctx.get("workflow_phase", "unknown") if isinstance(task_ctx, dict) else "unknown"
        phase_costs = current.get("by_phase", {})
        phase_costs[phase] = phase_costs.get(phase, 0.0) + cost
        current["by_phase"] = phase_costs
        await write_blackboard(mission_id, "cost_tracking", current)
    except (ImportError, TypeError):
        pass


async def _push_episodic_memory(task: dict, result: dict | None) -> None:
    """Store task outcome in episodic memory for future RAG retrieval."""
    if result is None or result.get("status") != "completed":
        return
    try:
        from src.memory.episodic import store_task_result
        result_text = result.get("result") or result.get("content") or ""
        model = result.get("model") or "unknown"
        cost = result.get("cost", 0.0)
        await store_task_result(
            task=task,
            result=result_text,
            model=model,
            cost=cost,
            duration=0.0,
            success=True,
        )
    except ImportError:
        pass


async def _push_metrics_counter(task: dict, result: dict | None) -> None:
    """Record in-memory metrics counters (record_task_complete)."""
    if result is None:
        return
    status = result.get("status", "")
    if status not in ("completed", "ungraded"):
        return
    try:
        from src.infra.metrics import record_task_complete
        model = result.get("model") or ""
        cost = result.get("cost", 0.0)
        record_task_complete(model=model, cost=cost)
    except ImportError:
        pass


async def _push_preference_feedback(task: dict, result: dict | None) -> None:
    """Record implicit acceptance feedback for the completed task."""
    if result is None or result.get("status") != "completed":
        return
    try:
        from src.memory.preferences import record_feedback
        await record_feedback(task, "accepted")
    except (ImportError, TypeError):
        pass


async def _push_skill_injection(task: dict, result: dict | None) -> None:
    """Record which injected skills were successfully used."""
    if result is None or result.get("status") == "ungraded":
        return
    try:
        from src.core.task_context import parse_context
        from src.memory.skills import record_injection_success
        task_ctx = parse_context(task)
        injected = task_ctx.get("injected_skills", [])
        if injected:
            await record_injection_success(injected)
    except ImportError:
        pass
